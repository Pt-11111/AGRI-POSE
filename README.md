# AGRI-Pose: Core Implementation

This repository contains selected core implementations from the paper **A Decision-Making Framework for Non-Rigid Grape Flower Thinning via Visual Keypoints and Biomass-Guided Topological Reconstruction**. It provides the AGRI-Pose model components, potential-guided topological sorting, biomass-weighted thinning decisions, and dataset split lists. It is not the complete project or an end-to-end training pipeline.

## Contents

| File | Description |
|---|---|
| `code/scsa_attention.py` | Spatial-channel synergistic attention (SCSA). |
| `code/scsa_adapter.py` | Adapter for integrating SCSA into the model. |
| `code/pose_dfl_head.py` | PoseDFL keypoint distribution prediction and coordinate decoding. |
| `code/pose_dfl_loss.py` | Joint detection and keypoint losses, including pose distribution supervision. |
| `code/topological_sorting.py` | IoU-based spikelet assignment, normalized biomass moment, potential-guided ordering, and geometric sequence metrics. |
| `code/thinning_decision.py` | Tip-first, Base-first and Middle-retention decisions; node-count comparison; independent manual-count evaluation. |
| `configs/agri_pose.yaml` | AGRI-Pose architecture configuration. |
| `configs/loss_gains.yaml` | Loss coefficients. |
| `dataset/splits/public/` | Public-source training, validation and test image lists. |
| `dataset/splits/self_collected/` | Self-collected split lists, including separate unrotated-original lists. |
| `dataset/excluded_test_images_44.txt` | Filenames of the 44 public-source images excluded from the test split. |
| `requirements.txt` | Direct Python dependencies. |

## Integration and environment

The model modules use PyTorch, NumPy, and einops. The prediction head and loss also depend on Ultralytics components and require integration into a compatible Ultralytics codebase. All other dependencies and environment settings follow the default Ultralytics requirements and setup instructions.

Register `SCSA` and `PoseDFL` with the model parser, connect `v8PoseDFL` to the pose model's loss initialization, and register the `pose_dfl` loss coefficient. The architecture uses `head_num=1` for SCSA; pass this explicitly when constructing the adapter. PoseDFL uses an offset coverage width of 20 grid units and 24 distribution bins.

The downstream core functions operate on supplied geometry and do not load a model. Topological reconstruction requires NumPy; thinning decisions use the Python standard library.

## Dataset splits

Only split lists are included; images and annotations are not distributed here. The public-source and self-collected lists are stored in separate directories. Their union defines the merged splits used in the paper's 163-image test evaluation.

| Source | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Public | 1,084 | 152 | 107 | 1,343 |
| Self-collected | 376 | 108 | 56 | 540 |
| Merged | 1,460 | 260 | 163 | 1,883 |

Each `train.txt`, `val.txt`, and `test.txt` contains one image path per line, relative to the merged dataset root (for example, `./test/images/1111.JPG`). Preserve filename case and extensions. To use these lists with Ultralytics, resolve the entries against your local dataset root and combine the two source lists for the required subset.

The self-collected subset contains 94 training, 27 validation, and 14 test originals. Its fourfold augmentation comprises each original plus its 90°, 180°, and 270° rotations. The `*_originals.txt` lists include only unrotated images; each original and its rotated versions remain in the same split.

The test split contains 107 public images and 56 self-collected images after excluding 44 public-source test images also present in training. Their filenames are listed in `dataset/excluded_test_images_44.txt`, alongside the `splits` directory.

Public dataset source: https://data.mendeley.com/datasets/skwbr5k9gg/1. Access to the public images follows the original source's conditions.

## Topological reconstruction and thinning decisions

Supply bounding boxes as two opposite corners in the same image coordinate system, semantic Base/Tip coordinates, and unique integer `shape_id` values. For multiple inflorescences, `assign_spikelets` assigns each spikelet exclusively to the smallest eligible inflorescence. Eligibility uses the area-scaled IoU interval documented in the function. Unmatched spikelets are returned separately. Endpoints must already be associated with their inflorescences.

`reconstruct` uses continuous biomass weights `A_i / A_min`, a normalized lateral moment, a bending threshold of 0.05, and a secondary ordering coefficient of 0.1. It returns ordered spikelets and a sequence consisting of Base, ordered spikelet centers, and Tip. `geometric_metrics` calculates M-PZR and M-MAD, with cross-step size 3 by default; image shape is supplied as `(height, width)`.

`select_biomass` selects from the ordered spikelets using `w_i = k A_i`, with `k=1` because the common factor cancels in fractional retention targets. Tip-first and Base-first choose the nonempty suffix or prefix closest to the target weight. Middle-retention initializes the two removal boundaries with a Tip:Base ratio of 1:1, then examines offsets from -2 to 2 at each boundary. `select_node_count` provides the comparison rule with ceiling rounding by default and optional floor or nearest-integer rounding.

Manual visible-bud counts are supplied only to `evaluate_retention`. The reference target is `N_exp = r_keep * sum(N_vis)`, and deviation is `100 * abs(N_retained - N_exp) / N_exp`. `biomass_weight_cv` returns the population coefficient of variation of spikelet weights. These functions do not use manual counts to select retained spikelets.

Minimal example, run from the repository root:

```python
import sys
sys.path.insert(0, "code")
from topological_sorting import reconstruct
from thinning_decision import select_biomass, evaluate_retention

# Illustrative geometry only; replace with your assigned spikelet boxes.
spikelets = [
    {"shape_id": 0, "points": [[1, 0], [3, 2]]},
    {"shape_id": 1, "points": [[4, 0], [6, 1]]},
    {"shape_id": 2, "points": [[7, 0], [8, 1]]},
]
structure = reconstruct(spikelets, base=[0, 0], tip=[10, 0])
selection = select_biomass(structure["spikelets"], r_keep=0.5,
                           strategy="tip_first")
print(selection["selected_ids"])

# Optional evaluation with independently obtained manual counts.
manual_counts = {0: 8, 1: 4, 2: 2}
evaluation = evaluate_retention(selection["selected_ids"], manual_counts, 0.5)
print(evaluation["deviation_pct"])
```

## Data availability

**Data availability:** The self-collected data may be made available upon reasonable request after publication, subject to institutional approval.

**Reason:** Institutional restrictions and research group policies prohibit the public release of the self-collected data before publication.

**Comments:**

After publication, requests for access to the self-collected data should be submitted to the corresponding author. A formal application bearing the official stamp of the applicant’s institution and the signature of the responsible project leader or authorized institutional representative is required. The application should include:

1. The applicant’s full name, position, institutional affiliation, and institutional email address, together with the name and contact details of the responsible person.
2. A description of the research project, the intended use of the data, and the specific data requested, including any required annotations or dataset split files.
3. The proposed duration of data use and the individuals who will have access.
4. A commitment to use the data only for the approved research purpose, store them securely, refrain from redistribution or transfer to third parties without prior written permission, and appropriately acknowledge the data source in resulting publications.

Requests will be assessed against the stated research purpose and applicable institutional requirements. Data access will be subject to approval and agreement on the scope and conditions of use.
