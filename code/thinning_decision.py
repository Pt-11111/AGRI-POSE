"""Biomass-weighted retention and independent manual-count evaluation.

Input spikelets are already ordered from the reconstruction stage and contain
``shape_id``, ``points`` and ``order``. Selection reads geometry and order only.
Strategies are ``tip_first``, ``base_first`` and ``middle_retention``.
"""

import math


def _prepare(spikelets, r_keep, strategy):
    if not 0 < r_keep <= 1:
        raise ValueError("r_keep must be in (0, 1]")
    if strategy not in {'tip_first', 'base_first', 'middle_retention'}:
        raise ValueError("Unknown retention strategy")
    nodes = sorted(spikelets, key=lambda s: s['order'])
    if not nodes or len({s['shape_id'] for s in nodes}) != len(nodes):
        raise ValueError("Nonempty spikelets with unique shape_id values are required")
    return nodes


def select_biomass(spikelets, r_keep, strategy='tip_first', middle_ratio=1.0):
    """Select a contiguous retention set using biomass weights w_i = k A_i.

    Here k=1: the common factor cancels in the fractional weight targets.
    Tip/Base-first minimize absolute target-weight error among nonempty
    suffixes/prefixes. Equal errors keep the first (shorter) candidate.
    Middle-retention initializes removals with a Tip:Base ratio, then searches
    offsets -2..2 at both boundaries; equal errors prefer more retained nodes.
    Returns selected IDs, biomass weight and the target weight.
    """
    nodes = _prepare(spikelets, r_keep, strategy)
    if not math.isfinite(middle_ratio) or middle_ratio <= 0:
        raise ValueError("middle_ratio must be positive")
    weights = []
    for s in nodes:
        (x1, y1), (x2, y2) = s['points']
        a = abs(x2-x1) * abs(y2-y1)
        if not math.isfinite(a) or a <= 0:
            raise ValueError("Spikelet bounding-box areas must be positive")
        weights.append(a)
    n, total = len(nodes), sum(weights)
    # Retain percentage arithmetic used by the evaluated selection rules.
    percentage = r_keep * 100
    target = total * (percentage / 100)
    if strategy in {'tip_first', 'base_first'}:
        current, best_gap, selected = 0.0, float('inf'), []
        candidates = range(n-1, -1, -1) if strategy == 'tip_first' else range(n)
        for i in candidates:
            current += weights[i]
            gap = abs(current-target)
            if gap < best_gap:
                best_gap = gap
                selected = list(range(i, n)) if strategy == 'tip_first' else list(range(i+1))
    else:
        tip_percentage = (100-percentage) * middle_ratio / (middle_ratio+1)
        base_percentage = (100-percentage) / (middle_ratio+1)
        tip_target, base_target = total * (tip_percentage/100), total * (base_percentage/100)
        top_remove = bottom_remove = 0
        current = 0.0
        for weight in reversed(weights):
            if current + weight <= tip_target:
                current += weight
                top_remove += 1
            else:
                break
        current = 0.0
        for weight in weights:
            if current + weight <= base_target:
                current += weight
                bottom_remove += 1
            else:
                break
        selected = list(range(bottom_remove, n-top_remove))
        best_gap = abs(sum(weights[i] for i in selected)-target)
        for tip_adjust in range(-2, 3):
            for base_adjust in range(-2, 3):
                top = max(0, min(top_remove+tip_adjust, n))
                bottom = max(0, min(bottom_remove+base_adjust, n))
                if top + bottom >= n:
                    continue
                candidate = list(range(bottom, n-top))
                gap = abs(sum(weights[i] for i in candidate)-target)
                if gap < best_gap or (gap == best_gap and len(candidate) > len(selected)):
                    best_gap, selected = gap, candidate
    return dict(selected_ids=[nodes[i]['shape_id'] for i in selected],
                retained_weight=sum(weights[i] for i in selected), target_weight=target)


def select_node_count(spikelets, r_keep, strategy='tip_first', rounding='ceil'):
    """Node-count baseline; nearest rounding maps half-integers upward."""
    nodes = _prepare(spikelets, r_keep, strategy)
    count = len(nodes) * ((r_keep*100)/100)
    if rounding == 'ceil':
        count = math.ceil(count)
    elif rounding == 'floor':
        count = math.floor(count)
    elif rounding == 'nearest':
        count = math.floor(count+0.5)
    else:
        raise ValueError("rounding must be ceil, floor or nearest")
    if count == 0:
        retained = []
    elif strategy == 'tip_first':
        retained = nodes[-count:]
    elif strategy == 'base_first':
        retained = nodes[:count]
    else:
        start = max(0, len(nodes)//2 - count//2)
        retained = nodes[start:min(len(nodes), start+count)]
    return [s['shape_id'] for s in retained]


def evaluate_retention(selected_ids, manual_counts, r_keep):
    """Evaluate a retention set against independent visible-bud counts.

    manual_counts maps every assigned spikelet ID in this inflorescence to its
    manual visible-bud count. N_exp = r_keep * sum(N_vis); the deviation is
    100 * abs(N_retained - N_exp) / N_exp. Counts never enter selection.
    """
    if not 0 < r_keep <= 1 or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Invalid retention ratio or duplicate selected IDs")
    values = list(manual_counts.values())
    if not values or any(not math.isfinite(v) or v < 0 or int(v) != v for v in values):
        raise ValueError("Manual counts must be nonnegative integers")
    total = sum(values)
    if total <= 0:
        raise ValueError("Total manual visible-bud count must be positive")
    target = r_keep * total
    retained = sum(manual_counts[i] for i in selected_ids)
    return dict(n_vis_total=total, n_expected=target, n_vis_retained=retained,
                deviation_pct=100*abs(retained-target)/target)


def biomass_weight_cv(spikelets):
    """Population coefficient of variation of w_i=k*A_i (dimensionless)."""
    areas = [abs(s['points'][1][0]-s['points'][0][0]) *
             abs(s['points'][1][1]-s['points'][0][1]) for s in spikelets]
    if not areas or any(not math.isfinite(a) or a <= 0 for a in areas):
        raise ValueError("Positive spikelet areas are required")
    mean = sum(areas)/len(areas)
    std = math.sqrt(sum((a-mean)**2 for a in areas)/len(areas))
    return std/mean
