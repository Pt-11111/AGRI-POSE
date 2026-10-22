# AGRI-Pose: Core Implementation

This repository contains selected core implementations from the paper **A Decision-Making Framework for Non-Rigid Grape Flower Thinning via Visual Keypoints and Biomass-Guided Topological Reconstruction**. It provides the AGRI-Pose attention module, PoseDFL prediction head and loss, model configuration, and dataset split lists. It is not the complete project or an end-to-end training pipeline.

## Contents

| File | Description |
|---|---|
| `code/scsa_attention.py` | Spatial-channel synergistic attention (SCSA). |
| `code/scsa_adapter.py` | Adapter for integrating SCSA into the model. |
| `code/pose_dfl_head.py` | PoseDFL keypoint distribution prediction and coordinate decoding. |
| `code/pose_dfl_loss.py` | Joint detection and keypoint losses, including pose distribution supervision. |
| `configs/agri_pose.yaml` | AGRI-Pose architecture configuration. |
| `configs/loss_gains.yaml` | Loss coefficients. |
| `dataset/splits/` | Training, validation, and test split lists for the self-collected subset. |
| `requirements.txt` | Direct Python dependencies. |

## Integration

The modules use PyTorch, NumPy, and einops. The prediction head and loss also depend on Ultralytics components and require integration into a compatible Ultralytics codebase.

All other dependencies and environment settings follow the default Ultralytics requirements and setup instructions.

Register `SCSA` and `PoseDFL` with the model parser, connect `v8PoseDFL` to the pose model's loss initialization, and register the `pose_dfl` loss coefficient. The architecture uses `head_num=1` for SCSA; pass this explicitly when constructing the adapter. PoseDFL uses an offset coverage width of 20 grid units and 24 distribution bins.

## Dataset splits

Only split lists are included; dataset images and annotations are not distributed in this repository. These lists cover the self-collected subset, not the complete merged dataset used in the paper.

| Split | Original images | Images including rotation augmentation |
|---|---:|---:|
| Training | 94 | 376 |
| Validation | 27 | 108 |
| Test | 14 | 56 |
| Total | 135 | 540 |

`train.txt`, `val.txt`, and `test.txt` list all samples in each split. The corresponding `*_originals.txt` files list only the unrotated images. Entries are relative image paths within the dataset root. Suffixes `_rot90`, `_rot180`, and `_rot270` identify rotated versions; each original image and its rotated versions belong to the same split.

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
