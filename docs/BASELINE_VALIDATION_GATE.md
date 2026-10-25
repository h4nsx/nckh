# Stage 3 Gate: Baseline Validation (YOLO11s Reference)

**Gate Date:** 2026-10-25  
**Gate Status:** **`PASS`**  
**Target Reference Model:** Baseline Ultralytics YOLO11s (COCO Pretrained Transfer)  
**Dataset Benchmark:** NVIDIA PhysicalAI-SmartSpaces Zero-Leakage Logistics Benchmark (`data/smartspaces_micro/smartspaces_micro.yaml`)

---

## 1. Executive Summary & Verification Matrix

| Checklist Item | Protocol Specification | Verified Value / Artifact | Status |
| :--- | :--- | :--- | :---: |
| **Model Definition & Checkpoint** | Official Ultralytics YOLO11s backbone/neck/head | `yolo11s.pt` / `yolo11s.yaml` | **PASS** |
| **Checkpoint Provenance & SHA-256** | Official pretrained weight file checksum | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | **PASS** |
| **Trained Best Weight SHA-256** | Checkpoint in `arm1_baseline_control` | `a213d99a40c4f405dd28f8d07670e34d7e4a000fb8fa51f60549b72a63e52870` | **PASS** |
| **Exact Training Hyperparameters** | Optimizer, LR, Schedule, Seed | AdamW, $\text{lr}_0=0.00125$, $\text{lrf}=0.01$, $\text{seed}=0$, batch=4, imgsz=640 | **PASS** |
| **Zero-Leakage Split Guarantee** | Cross-facility camera & scene separation | Train: $300$ frames ($12,959$ instances); Val: $100$ frames ($3,696$ instances) | **PASS** |
| **Full 20-Epoch Training Completion**| Non-trivial schedule beyond sanity test | 20 epochs completed; full loss logs in `arm1_baseline_control/results.csv` | **PASS** |
| **Reproduced Validation Accuracy** | Re-evaluation of `best.pt` on fixed val split | **$\text{mAP}_{50} = 69.80\%$**, **$\text{mAP}_{50-95} = 45.53\%$**, Precision=$81.82\%$, Recall=$57.21\%$ | **PASS** |
| **Per-Class Breakdown** | Verified across all 4 warehouse target classes | Worker ($89.20\%$), Forklift ($70.13\%$), Pallet Truck ($32.78\%$), AMR ($87.07\%$) | **PASS** |
| **Qualitative & Matrix Artifacts** | Confusion matrices, PR curves, batch plots | Saved in `runs/detect/val-9/` (`confusion_matrix.png`, `BoxPR_curve.png`, etc.) | **PASS** |
| **Model Footprint Metrics** | Recomputed on instantiated model | $9,429,340$ parameters, $21.7\text{ GFLOPs}$, $19.18\text{ MB}$ disk weight | **PASS** |

---

## 2. Checkpoint Provenance & Environment

- **Source Checkpoint:** `yolo11s.pt` (Ultralytics official release)
  - File Size: $19,394,667$ bytes ($18.50\text{ MB}$)
  - SHA-256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`
- **Trained Arm 1 Checkpoint:** `runs/detect/runs/two_stage_protocol_study/arm1_baseline_control/weights/best.pt`
  - File Size: $19,176,602$ bytes ($18.29\text{ MB}$)
  - SHA-256: `a213d99a40c4f405dd28f8d07670e34d7e4a000fb8fa51f60549b72a63e52870`
- **Final Epoch 20 Checkpoint:** `runs/detect/runs/two_stage_protocol_study/arm1_baseline_control/weights/last.pt`
  - File Size: $19,176,602$ bytes ($18.29\text{ MB}$)

---

## 3. Training & Validation Protocol

```yaml
# Exact Training Hyperparameters Recorded in Checkpoint Metadata
model: yolo11s.pt
data: data/smartspaces_micro/smartspaces_micro.yaml
epochs: 20
batch: 4
imgsz: 640
device: 0 (NVIDIA RTX 3090)
workers: 2
optimizer: AdamW
seed: 0
lr0: 0.00125
lrf: 0.01
save: true
save_period: -1
```

---

## 4. Re-evaluation & Numerical Verification

Running `model.val(data="data/smartspaces_micro/smartspaces_micro.yaml", batch=4, imgsz=640, device="cpu", split="val", plots=True)` directly from `arm1_baseline_control/weights/best.pt` reproduced the exact reported metrics:

```
                   Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        100       3696      0.818      0.572      0.698      0.455
                worker        100       3488      0.950      0.725      0.892      0.632
              forklift        100        150      0.936      0.433      0.701      0.318
          pallet_truck         13         13      0.449      0.308      0.328      0.224
                   amr         45         45      0.938      0.822      0.871      0.647
```

- **Validation Speed on Host CPU:** $3.0\text{ ms}$ preprocess, $336.6\text{ ms}$ inference, $2.5\text{ ms}$ postprocess per image.
- **Generated Diagnostic Artifacts:** Stored in `runs/detect/val-9/`:
  - `confusion_matrix.png` & `confusion_matrix_normalized.png`
  - `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
  - `val_batch0_labels.jpg` vs. `val_batch0_pred.jpg`
  - `val_batch1_labels.jpg` vs. `val_batch1_pred.jpg`
  - `val_batch2_labels.jpg` vs. `val_batch2_pred.jpg`

---

## 5. Gate Determination: PASS

The baseline reference has been audited against raw training logs, reproduced via independent checkpoint evaluation, verified for parameter and FLOP metrics, and preserved in version control. Stage 3 is officially **`PASSED`**.
