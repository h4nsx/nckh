# TRAINING PIPELINE AUDIT REPORT (Stage 4 Audit)

- **Date:** 2026-11-04
- **Auditor Role:** Autonomous Research Implementation Auditor (SKILLS.md Stage 4)
- **Status:** **STAGE 4 AUDIT COMPLETE (PASS WITH DOCUMENTED DISCLOSURE)**

---

## 1. Multi-Arm Training Comparability Matrix

| Training Parameter / Protocol | Arm 1 (Baseline Control) | Arm 2 (SS2D Single-Stage) | Arm 3 (SS2D Two-Stage) | Arm 4 (SS2D Dual-Scale) | Comparability Assessment |
|---|---|---|---|---|---|
| **Base Architecture** | Standard YOLO11s | YOLO11s-SS2D ($P_5$) | YOLO11s-SS2D ($P_5$) | YOLO11s-SS2D ($P_4 + P_5$) | Valid architectural progression |
| **Dataset & Split** | `smartspaces_micro.yaml` | `smartspaces_micro.yaml` | `smartspaces_micro.yaml` | `smartspaces_micro.yaml` | Identical zero-leakage split across all arms |
| **Input Resolution** | $640 \times 640$ | $640 \times 640$ | $640 \times 640$ | $640 \times 640$ | Identical across all runs |
| **Batch Size** | 4 | 4 | 4 | 4 | Identical |
| **Random Seed** | `seed=0` | `seed=0` | `seed=0` | `seed=0` | Identical |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | Identical |
| **Learning Rate Schedule** | $\text{lr}_0 = 0.00125, \text{lrf} = 0.01$ | $\text{lr}_0 = 0.00125, \text{lrf} = 0.01$ | Stage 1: $0.00125$<br>Stage 2: $0.0005$ | $\text{lr}_0 = 0.00125, \text{lrf} = 0.01$ | **Audited Finding:** Arm 3 uses refined $\text{lr}_0 = 0.0005$ in Stage 2 fine-tuning (TRACE-03). |
| **Epoch Budget** | 20 epochs | 20 epochs | 5 warmup + 15 fine-tune (20 total) | 20 epochs | Equivalent 20-epoch compute envelope |
| **Pretrained Weights** | COCO pretrained | Warmstarted from baseline | Warmstarted from baseline | Warmstarted from baseline | Valid transfer setup |
| **Anchors & DFL Heads** | Pretrained | Preserved from baseline | Preserved from baseline | Preserved from baseline | Regression anchors preserved across all arms |

---

## 2. Training Dynamics & Convergence Findings

1. **Epoch Budget & Training Stability:**
   - Arm 1 (Baseline Control): Converges rapidly; peak $\text{mAP}_{50} = 70.32\%$ at Epoch 6; final epoch settling at $69.75\%$.
   - Arm 2 (Single-Stage SS2D): Progresses steadily without backbone freezing; reaches highest overall peak $\text{mAP}_{50} = 71.81\%$ at Epoch 18; settles at $65.46\%$ at final epoch 20.
   - Arm 3 (Two-Stage Warmup): Stage 1 warms up SSM parameters and detection head with frozen backbone ($P_1-P_4$) for 5 epochs ($60.63\%\text{ mAP}_{50}$), followed by 15 epochs of end-to-end fine-tuning, reaching peak $67.49\%\text{ mAP}_{50}$ (checkpoint evaluation $67.67\%$).
   - Arm 4 (Dual-Scale P4+P5): Reaches peak $67.17\%$ at Epoch 20.
2. **Honest Reporting of Peak vs Final Metrics:**
   - As audited in `scripts/make_tables.py` and `paper/tables/tab_accuracy_results.tex`, Table II explicitly separates **Final Epoch $\text{mAP}_{50}$** from **Peak Checkpoint $\text{mAP}_{50}$**.
   - The paper does not conflate peak with final metrics or hide final-epoch settling numbers.
3. **Training Protocol Disclosure:**
   - The training protocol difference in Arm 3 (Stage 2 learning rate refinement $\text{lr}_0 = 0.0005$) has been explicitly noted in `audit/PAPER_CODE_TRACEABILITY.md` (Issue TRACE-03) and must be disclosed in Section 4.4 of the manuscript.

---

## 3. Exit Gate G4 Verdict: PASS WITH DOCUMENTED DISCLOSURE
All 4 experimental arms are verified on disk with logged training histories. The 20-epoch comparative protocol is confirmed sound.

