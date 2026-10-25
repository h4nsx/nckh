# Experiment Record: EXP-04-WARMUP-FREEZE10-DIAGNOSTIC

## 1. Identity
- **Experiment ID:** `EXP-04-WARMUP-FREEZE10-DIAGNOSTIC`
- **Date:** 2026-10-24
- **Research Question:** Diagnostic evaluation of whether freezing the backbone (`freeze=10`) and zero-initializing the SS2D output projection eliminates the cold-block perturbation
- **Git Commit:** Branch `base`
- **Config Path:** `models/yolo11s_ss2d_p5.yaml` + `models/yolo11s_ss2d_warmstart_zeroinit.pt` + `data/smartspaces_micro/smartspaces_micro.yaml`
- **Operator:** Antigravity AI Assistant & Hans

## 2. Hypothesis and Design
- **Hypothesis:** Freezing early backbone layers (0–9) will shield pretrained visual representations from uncalibrated state-space gradients, enabling the SS2D block and detection head to adapt without destroying upstream features.
- **Independent Variable:** Layer freezing (`freeze=10`, layers 0–9 frozen) combined with zero-initialized output projection (`out_proj`).
- **Controlled Variables:** Input resolution (640x640), batch size (4), epochs (2), optimizer (AdamW, auto lr=0.00125), data split (`Warehouse_000` train, `Warehouse_015` val).
- **Outcome:** **EMPIRICALLY DIAGNOSED**. Freezing the backbone stabilized training losses (`cls_loss` dropped from 4.80 to 4.21, `box_loss` from 6.09 to 5.82). However, detection mAP remained 0.0% due to an architectural root cause uncovered during inspection: the detection head's box regression weights were omitted from the warmstart transfer.

## 3. Data & Splits
- **Dataset:** NVIDIA PhysicalAI-SmartSpaces (Targeted Micro-Subset v1.0)
- **Train Set:** `Warehouse_000` (`Camera_0000`, `Camera_0001`), 300 frames, 12,959 instances.
- **Validation Set:** `Warehouse_015` (`Camera`, `Camera_01`), 100 frames, 3,696 instances.
- **Leakage Safeguard:** Strictly independent facilities and disjoint camera streams between train and validation.
- **Classes:** `0: worker`, `1: forklift`, `2: pallet_truck`, `3: amr`.

## 4. Measured Loss Trajectory (Epoch 1 & 2)

| Epoch | Duration (s) | Train Box Loss | Train Cls Loss | Train DFL Loss | Val Box Loss | Val Cls Loss | Val DFL Loss | $\text{mAP}_{50}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 2,958.49 s (~49.3 min) | 6.0937 | 4.8003 | 4.0066 | 5.4081 | 4.3211 | 4.0444 | 0.00% |
| **2** | 3,175.72 s (~52.9 min) | **5.8216** | **4.2149** | **3.7815** | **5.4016** | **4.1203** | **4.0517** | 0.00% |

## 5. Diagnostic Breakthrough & Scientific Analysis

1. **Why mAP Remained Zero Despite Backbone Freezing**:
   - In `EXP-04`, the backbone layers 0–9 were successfully frozen (as confirmed by Ultralytics logging `Freezing layer 'model.0' ... 'model.9'`).
   - However, inspection of the checkpoint state dictionary revealed that in `models/yolo11s_ss2d_warmstart_zeroinit.pt`, **only the backbone and neck layers had been transferred**.
   - The entire `Layer 23` (`Detect` head), including all 42 box regression layers (`model.23.cv2`) and the distribution focal loss layer (`model.23.dfl`), was completely randomized!
   - Because the detection head had no pretrained knowledge of how to regress bounding boxes, and the backbone was completely frozen, the randomized head could not align to tight bounding boxes within just 2 epochs on 300 images.
2. **The Missing Link Confirmed**:
   - The baseline (`EXP-02`) achieved 41.3% $\text{mAP}_{50}$ because Ultralytics' internal loader automatically preserved the 42 `cv2` box regression layers from `yolo11s.pt`.
   - In contrast, Case B was forced to learn box localization from scratch!
3. **Resolution Implemented**:
   - We created `scripts/build_case_b_full_warmstart.py`, which transfers **all 42 `cv2` box regression layers and `dfl`** from `yolo11s.pt` into Case B, generating `models/yolo11s_ss2d_full_warmstart.pt`.
   - Combined with our verified v2 spatially-aligned SS2D block, Case B now has an identical pretrained foundation to the baseline.
