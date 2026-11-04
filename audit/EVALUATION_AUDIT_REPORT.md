# EVALUATION AND STATISTICAL AUDIT REPORT (Stage 5 Audit)

- **Date:** 2026-11-04
- **Auditor Role:** Autonomous Research Implementation Auditor (SKILLS.md Stage 5)
- **Checkpoints Audited:**
  - Arm 1: `runs/detect/runs/two_stage_protocol_study/arm1_baseline_control/weights/best.pt`
  - Arm 3: `runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.pt`
- **Dataset:** `data/smartspaces_micro/smartspaces_micro.yaml` (100 validation images, 3,696 ground truth annotations)
- **Status:** **STAGE 5 COMPLETE (PASS WITH DOCUMENTED TEXTUAL FIXES)**

---

## 1. Direct Re-Evaluation & Verification of Reported Tables

The saved checkpoints were re-evaluated end-to-end on the validation split using the Ultralytics evaluation engine. The freshly computed metrics match the manuscript tables with exact precision:

### Table 1: Global Validation Metrics Parity

| Metric | Arm 1 (Baseline Control) | Arm 3 (Case B Two-Stage) | Verified Delta | Status in Paper Table II |
|---|:---:|:---:|:---:|:---:|
| **$\text{mAP}_{50}$** | $69.80\%$ | $67.67\%$ | $-2.13\%$ | **PASS (100% Match)** |
| **$\text{mAP}_{50-95}$** | $45.53\%$ | $43.15\%$ | $-2.38\%$ | **PASS (100% Match)** |
| **Precision ($P$)** | $81.82\%$ ($0.818$) | $74.12\%$ ($0.741$) | $-7.70\%$ | **PASS (100% Match)** |
| **Recall ($R$)** | $57.21\%$ ($0.572$) | $65.06\%$ ($0.651$) | **$+7.85\%$ ($+7.9\%$)** | **PASS (100% Match)** |

---

### Table 2: Per-Class $\text{mAP}_{50}$ and $\text{mAP}_{50-95}$ Parity

| Target Category | Ground Truth ($n$) | Arm 1 $\text{mAP}_{50} / \text{mAP}_{50-95}$ | Arm 3 $\text{mAP}_{50} / \text{mAP}_{50-95}$ | Freshly Evaluated Delta | Status in Paper Table III |
|---|:---:|:---:|:---:|:---:|:---:|
| **`forklift`** | **150** | $70.13\% / 31.80\%$ | **$76.26\% / 38.36\%$** | **$+6.13\% / +6.56\%$** | **PASS (Exact Match)** |
| **`worker`** | 3,488 | $89.20\% / 63.19\%$ | $87.91\% / 60.53\%$ | $-1.29\% / -2.66\%$ | **PASS (Exact Match)** |
| **`amr`** | 45 | $87.07\% / 64.74\%$ | $83.67\% / 57.50\%$ | $-3.40\% / -7.24\%$ | **PASS (Exact Match)** |
| **`pallet_truck`** | **13** | $32.78\% / 22.37\%$ | $22.85\% / 16.21\%$ | $-9.93\% / -6.16\%$ | **PASS (Exact Match)** |

---

## 2. Definitive Problem Resolution: Metric Terminology and Scientific Framing

We performed a line-by-line audit of all instances where "recall" is mentioned in the manuscript, resolving the exact issues flagged in Prompt D & Prompt F:

### Identified Tension Point EVAL-01: Discrepancy in Introduction Line 37
- **Location:** `paper/sections/01_introduction.tex` (Line 37).
- **Text:** *"and increasing macro recall by $+7.3\%$ in two-stage training."*
- **Empirical Reality:** The audited recall values are $57.21\%$ (Arm 1) vs $65.06\%$ (Arm 3), which gives an exact absolute gain of **$+7.85\%$** (rounded to **$+7.9\%$** in Abstract and Section 5).
- **Root Cause:** Line 37 in the Introduction contains an outdated draft typo ($+7.3\%$) instead of $+7.85\%$ ($+7.9\%$).
- **Action:** Fix line 37 to read $+7.9\%$ ($+7.85\%$ absolute).

### Identified Tension Point EVAL-02: Framing of Macro Recall vs Worker Safety
- **Location:** `paper/sections/06_conclusion.tex` (Line 10).
- **Text:** *"alongside a $+7.85\%$ increase in worker safety recall under two-stage warmup fine-tuning."*
- **Audit Rule (SKILLS.md 7.4 & 11.7):** *"Do not label macro recall as worker-specific recall. Never call macro recall 'worker recall' or another class-specific metric."*
- **Empirical Reality:** $65.06\%$ is the **global macro recall** across all warehouse classes. The recall on `worker` specifically is high across all models ($>93\%$). Conflating global macro recall with "worker safety recall" violates research integrity guideline 11.7.
- **Action:** Update Section 6.1 to state *"alongside a $+7.85\%$ absolute increase in global macro recall ($65.1\%$ vs. $57.2\%$), reducing false negatives across warehouse entities under two-stage warmup fine-tuning."*

---

## 3. Exit Gate G5 Verdict: PASS WITH DOCUMENTED CORRECTIONS
All numeric values in Table II and Table III are 100% reproducible directly from on-disk checkpoint evaluations. The two textual phrasing corrections (EVAL-01 and EVAL-02) are clearly isolated and will be applied cleanly during Stage 7/8.

