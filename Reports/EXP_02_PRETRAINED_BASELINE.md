# Experiment Record: EXP-02-PRETRAINED-BASELINE

## 1. Identity
- **Experiment ID:** `EXP-02-PRETRAINED-BASELINE`
- **Date:** 2026-10-23
- **Research Question:** Benchmark anchor calibration using transfer learning from COCO pretrained YOLO11s on SmartSpaces warehouse micro-subset
- **Git Commit:** Branch `base`
- **Config Path:** `yolo11s.pt` + `data/smartspaces_micro/smartspaces_micro.yaml`
- **Operator:** Antigravity AI Assistant & Hans

## 2. Hypothesis and Design
- **Hypothesis:** Leveraging pretrained CNN backbone features will produce non-zero true positive detections, populated PR curves, and meaningful multi-class AP within 2 epochs on the laptop CPU.
- **Independent Variable:** Unmodified standard YOLO11s with COCO pretrained transfer weights.
- **Controlled Variables:** Input resolution (640x640), batch size (4), epochs (2), optimizer (AdamW, lr=0.00125), data split (`Warehouse_000` train, `Warehouse_015` val).
- **Outcome:** **VERIFIED**. Full accuracy metrics, confusion matrices, PR curves, and bounding box predictions were successfully populated.

## 3. Data & Splits
- **Dataset:** NVIDIA PhysicalAI-SmartSpaces (Targeted Micro-Subset v1.0)
- **Train Set:** `Warehouse_000` (`Camera_0000`, `Camera_0001`), 300 frames, 12,959 instances.
- **Validation Set:** `Warehouse_015` (`Camera`, `Camera_01`), 100 frames, 3,696 instances.
- **Leakage Safeguard:** Completely independent warehouse facilities and disjoint camera streams between train and validation.
- **Classes:** `0: worker`, `1: forklift`, `2: pallet_truck`, `3: amr`.

## 4. Measured Accuracy Metrics (Best Checkpoint: Epoch 2)

| Class | Targets (Instances) | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall (All Classes)** | **3,696** | **0.478** | **0.380** | **0.413** (41.3%) | **0.222** (22.2%) |
| `worker` | 3,488 | 0.771 | 0.614 | **0.722** (72.2%) | **0.423** (42.3%) |
| `forklift` | 150 | 0.535 | 0.540 | **0.547** (54.7%) | **0.246** (24.6%) |
| `amr` (NovaCarter) | 45 | 0.416 | 0.289 | **0.268** (26.8%) | **0.135** (13.5%) |
| `pallet_truck` (Transporter) | 13 | 0.189 | 0.077 | **0.118** (11.8%) | **0.083** (8.3%) |

## 5. Loss Convergence Trajectory
- **Epoch 1:**
  - `train/box_loss`: 1.566, `train/cls_loss`: 1.308, `train/dfl_loss`: 1.111
  - `val/box_loss`: 1.652, `val/cls_loss`: 3.469, `val/dfl_loss`: 1.638
  - `mAP50`: 0.267 (26.7%)
- **Epoch 2:**
  - `train/box_loss`: 1.300, `train/cls_loss`: 0.969, `train/dfl_loss`: 1.011
  - `val/box_loss`: 1.438, `val/cls_loss`: 1.204, `val/dfl_loss`: 1.395
  - `mAP50`: 0.398 (val after epoch) / 0.413 (best checkpoint)

## 6. Generated Charts & Visual Artifacts
All files are saved in `runs/detect/runs/pretrained_runs/baseline_pretrained/`:
1. **Precision-Recall Curve:** `BoxPR_curve.png` (162.8 KB) — Shows populated, continuous PR curves for each of the 4 classes.
2. **F1-Confidence Curve:** `BoxF1_curve.png` (197.2 KB) — Optimal F1 score curves.
3. **Confusion Matrices:** `confusion_matrix.png` (146.8 KB) and `confusion_matrix_normalized.png` (149.6 KB) — Populated matrix with worker/forklift classification counts.
4. **Validation Visualizations with Bounding Boxes:**
   - `val_batch0_pred.jpg` (822.8 KB) — Detected bounding box predictions.
   - `val_batch1_pred.jpg` (885.7 KB) — Detected bounding box predictions.
   - `val_batch2_pred.jpg` (864.6 KB) — Detected bounding box predictions.
   - Side-by-side ground truth: `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg`.
5. **Trained Weights:** `weights/best.pt` (19.2 MB).

## 7. Next Step: Case B Matching Evaluation
Now that we have the exact benchmark numbers ($\text{mAP}_{50} = 41.3\%$, $\text{mAP}_{50-95} = 22.2\%$) and full charts for the baseline, we will run the matching 2-epoch evaluation on our pretrained **Case B (YOLO11s-SS2D)** model (`models/yolo11s_ss2d_warmstart.pt`).
