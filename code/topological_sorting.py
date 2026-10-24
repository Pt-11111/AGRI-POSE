"""Core biomass-moment and potential-guided inflorescence reconstruction.

Boxes use two opposite corners in a shared 2-D coordinate system. Each spikelet
has a unique integer ``shape_id`` and ``points``. Base and Tip are semantic
endpoints supplied by the caller. No images, annotations or model are loaded.
"""

import numpy as np


def bounding_box(points):
    """Return [xmin, ymin, xmax, ymax] for a two-corner rectangle."""
    p = np.asarray(points, dtype=float)
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError("points must contain two finite 2-D rectangle corners")
    return np.r_[p.min(axis=0), p.max(axis=0)]


def box_area(points):
    b = bounding_box(points)
    a = float(np.prod(b[2:] - b[:2]))
    if a <= 0:
        raise ValueError("Box area must be positive")
    return a


def assign_spikelets(inflorescences, spikelets):
    """Assign each spikelet to the smallest eligible inflorescence by IoU.

    Eligibility is 0.5 * (A_spikelet/A_inflorescence) <= IoU <=
    1.01 * (A_spikelet/A_inflorescence), as in the evaluated implementation.
    Equal-area candidates are resolved by shape_id. Unmatched nodes are returned
    separately. Annotation descriptions and group memberships are not read.
    Returns (mapping from inflorescence IDs to spikelets, unmatched IDs).
    """
    for records in (inflorescences, spikelets):
        ids = [r['shape_id'] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError("shape_id values must be unique within each list")
    flowers = sorted(inflorescences,
                     key=lambda f: (box_area(f['points']), f['shape_id']))
    assigned = {f['shape_id']: [] for f in flowers}
    unmatched = []
    for s in spikelets:
        sb, sa = bounding_box(s['points']), box_area(s['points'])
        for f in flowers:
            fb, fa = bounding_box(f['points']), box_area(f['points'])
            inter = float(np.prod(np.maximum(
                0, np.minimum(sb[2:], fb[2:]) - np.maximum(sb[:2], fb[:2]))))
            iou = inter / (sa + fa - inter)
            if 0.5 * sa / fa <= iou <= 1.01 * sa / fa:
                assigned[f['shape_id']].append(s)
                break
        else:
            unmatched.append(s['shape_id'])
    return assigned, unmatched


def reconstruct(spikelets, base, tip, epsilon=0.05, secondary_weight=0.1):
    """Return ordered spikelets, normalized moment and the geometric sequence.

    The local origin is Base; the x-axis points from Base to Tip and the y-axis
    is (-x_y, x_x). Continuous biomass weights are A_i/min(A_i). The moment is
    the weighted lateral centroid offset divided by the Base-Tip length.
    ``path`` follows the evaluated convention: Base, ordered centers, then Tip.
    The returned records contain geometry and weights only; inputs are unchanged.
    """
    if not spikelets:
        raise ValueError("At least one assigned spikelet is required")
    if epsilon < 0 or not np.isfinite([epsilon, secondary_weight]).all():
        raise ValueError("Invalid threshold or secondary weight")
    origin, endpoint = np.asarray(base, dtype=float), np.asarray(tip, dtype=float)
    if origin.shape != (2,) or endpoint.shape != (2,) or not np.isfinite([origin, endpoint]).all():
        raise ValueError("Base and Tip must be finite 2-D points")
    axis_length = float(np.linalg.norm(endpoint - origin))
    if axis_length == 0:
        raise ValueError("Base and Tip must be distinct")
    x_axis = (endpoint - origin) / axis_length
    y_axis = np.array([-x_axis[1], x_axis[0]])
    ids = [s['shape_id'] for s in spikelets]
    if len(set(ids)) != len(ids):
        raise ValueError("Spikelet shape_id values must be unique")
    minimum_area = min(box_area(s['points']) for s in spikelets)
    nodes, weighted_y, total_weight = [], 0.0, 0.0
    for s in spikelets:
        b = bounding_box(s['points'])
        # Preserve a supplied detection center, otherwise use the box center.
        center = np.asarray(s.get('center', (b[:2] + b[2:]) / 2), dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Spikelet centers must be finite 2-D points")
        local = [float(np.dot(center-origin, x_axis)),
                 float(np.dot(center-origin, y_axis))]
        weight = box_area(s['points']) / minimum_area
        weighted_y += weight * (local[1] / axis_length)
        total_weight += weight
        nodes.append(dict(shape_id=s['shape_id'], points=np.asarray(s['points']).tolist(),
                          center=center.tolist(), weight=weight, new_center=local))
    moment = weighted_y / total_weight
    bending = 'down' if moment < -epsilon else 'up' if moment > epsilon else 'neutral'
    regions = [[], [], []]
    for s in nodes:
        x = s['new_center'][0]
        regions[0 if x < 0 else 1 if x <= axis_length else 2].append(s)

    def energy(s, region):
        x, y = s['new_center']
        if bending == 'neutral':
            return x
        if region == 1:
            primary, secondary = x, -y if bending == 'down' else y
        elif region == 0:
            primary, secondary = -y if bending == 'down' else y, x
        else:
            primary, secondary = y if bending == 'down' else -y, x
        return primary + secondary_weight * secondary

    ordered = []
    for index, region in enumerate(regions):
        ordered.extend(sorted(region, key=lambda s: energy(s, index)))
    for i, s in enumerate(ordered, 1):
        s['order'] = i
    return dict(spikelets=ordered, ordered_ids=[s['shape_id'] for s in ordered],
                path=[origin.tolist()] + [s['center'] for s in ordered] + [endpoint.tolist()],
                M_y=moment, bending_direction=bending, axis_length=axis_length)


def geometric_metrics(path, image_shape, step=3):
    """Return M-PZR and M-MAD in percent; image_shape is (height, width).

    Adjacent zero trend vectors contribute no reversal but retain their place
    in the denominator. Short sequences are rejected rather than scored as zero.
    """
    p = np.asarray(path, dtype=float)
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < step + 2 or not np.isfinite(p).all():
        raise ValueError("At least step + 2 finite 2-D nodes are required")
    h, w = image_shape[:2]
    if not np.isfinite([h, w]).all() or min(h, w) <= 0:
        raise ValueError("Image height and width must be positive")
    vectors = p[step:] - p[:-step]
    reversals = 0
    for a, b in zip(vectors[:-1], vectors[1:]):
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            continue
        if np.dot(a, b) < 0:
            reversals += 1
    pzr = 100 * reversals / (len(vectors) - 1)
    mad = 100 * (np.linalg.norm(vectors, axis=1).mean() / step) / np.sqrt(h*h + w*w)
    return dict(m_pzr_pct=float(pzr), m_mad_pct=float(mad))
