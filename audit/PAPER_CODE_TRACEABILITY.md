# PAPER-TO-CODE TRACEABILITY MATRIX (Stage 1 Audit)

- **Audit Date:** 2026-11-04
- **Auditor Role:** Autonomous Research Implementation Auditor (SKILLS.md Stage 1)
- **Status:** **STAGE 1 COMPLETE (PASS WITH IDENTIFIED TENSION POINTS)**

---

## 1. Traceability Matrix

| Paper Element | Expected Implementation / Evidence | Actual Code / Artifact Location | Direct Verification Method | Verdict | Findings / Notes |
|---|---|---|---|---|---|
| **YOLO11s Baseline Control** | Standard YOLO11s model definition & weights | `yolo11s.pt`, Layer 10 is `C2PSA` | Checkpoint inspection via PyTorch: 9,429,340 params ($nc=4$). | **PASS** | Matches Section 3.2, 4.3 (Arm 1), and Table 1. |
| **P5 C2PSA Replacement** | Surgery replacing Layer 10 with `WarehouseSS2DBlock` | `models/yolo11s_ss2d_p5.yaml`: line 37 | Layer 10 parsed as `WarehouseSS2DBlock(512, 512, 16, 3, 0.5)`. Total layers: 98. | **PASS** | Exactly matches Table 1 ($98$ layers, $-6.72\%$ params). |
| **Optional P4+P5 Insertion** | Dual SS2D insertion at P4 (Layer 7) and P5 (Layer 11) | `models/yolo11s_ss2d_p4_p5.yaml`: lines 31, 38 | Instantiated in Arm 4 checkpoint (`arm4_case_c_dual_scale`). | **PASS** | Matches Case C description in Section 3.2. |
| **WarehouseSS2DBlock Structure** | Input projection ($1 \times 1$), DW-Conv ($3 \times 3$), SS2D core, output projection ($1 \times 1$), norm3 | `models/warehouse_ss2d.py`: lines 174–201 | Traced in `forward()`: `in_proj -> dw_conv -> ss2d -> out_proj -> norm3 + residual`. | **PASS** | Exact match with Section 3.4. |
| **Zero-$\gamma$ Residual Initialization** | BN norm3 initialized to zero ($\gamma_0 = 0, \beta_0 = 0$) | `models/warehouse_ss2d.py`: lines 196–198 | Direct code check: `nn.init.zeros_(self.norm3.weight)`, `nn.init.zeros_(self.norm3.bias)`. | **PASS** | Verified in `tests/test_zero_gamma_identity.py` ($\|Y_0 - X\|_\infty = 0.0$). |
| **4 Directional Scans** | 4-way scanning and exact spatial inverse restoration | `models/warehouse_ss2d.py`: lines 73–115 | `x_1` (row-major), `x_2` (row rev), `x_3` (col-major), `x_4` (col rev) and restorations. | **PASS** | Verified in `tests/test_ss2d_directional_alignment.py`. Non-square ($H \ne W$) supported. |
| **SSM Discretization & Recurrence** | Continuous $A$, discrete recurrence $\bar{A} = \exp(\Delta A)$, $\bar{B}u$, $D \cdot u$ | `models/warehouse_ss2d.py`: lines 38–59 | `_selective_scan_1d`: $A = -\exp(A_{\text{logs}})$, $\Delta$ via softplus, recurrence loop over $L$. | **PARTIAL** | First-order Euler formulation for input $B u$ ($\bar{B}u = \Delta u B$). Accurately disclosed in paper Eq (7). |
| **Canonical Mamba Equivalence Claim** | Mamba input/output gating | `models/warehouse_ss2d.py` | Implementation has no multiplicative input/output gate (unlike standard Mamba/VMamba). | **PASS (Honest)** | Paper accurately designates it as `WarehouseSS2DBlock` / selective state-space scan, not canonical Mamba. |
| **Single-Source Table Generation** | Accuracy, Per-Class, and Hardware tables generated from raw logs | `scripts/make_tables.py` $\to$ `paper/tables/*.tex` | Ran script; verified output matches paper tables with 100% precision. | **PASS** | Resolves potential manual copy-paste errors. |
| **Checkpoint & Peak Metric Integrity** | Reported metrics correspond to actual evaluation of best weights | `results/tables/arm*_results.csv`, `Reports/CONTROLLED_EXPERIMENTS_GATE.md` | Arm 1 (69.80%), Arm 2 (71.95%), Arm 3 (67.67%), Arm 4 (67.17%). | **PASS** | Checkpoints exist on disk in `runs/detect/runs/two_stage_protocol_study/`. |
| **Edge Hardware Measurements** | Latency and Peak RSS on Raspberry Pi 4 Cortex-A72 | `results/tables/` & `report.md`, `scripts/benchmark_edge_device.py` | Latencies: 1169.73ms (Arm 1), 1398.19ms (Arm 2/3), 1965.07ms (Arm 4). RSS: 371.9 MB. | **PASS** | Traced to empirical Raspberry Pi deployment in `report.md`. |

---

## 2. Identified Discrepancies, Vulnerabilities & Tension Points

During the deep scan of all LaTeX sections against repository code and logs, the following 4 tension points were identified:

### Issue TRACE-01 (Major): Overclaiming "Multi-Seed" in Section 2.6
- **Location:** `paper/sections/02_related_work.tex` (Line 55) and `paper/sections/06_conclusion.tex` (Line 42).
- **Observed Text in Section 2.6:** *"This study addresses that defensible gap by proposing, evaluating, and benchmarking an edge-optimized hybrid detector against standard YOLO baselines with full multi-seed and operator-level deployment analysis."*
- **Actual Evidence in Code & Config:** `configs/experiment_settings.yaml` explicitly fixes `seed: 0`. No multi-seed runs ($\mu \pm \sigma$) exist in `results/tables/`. Furthermore, Section 6.2 correctly lists *"Multi-Seed and Expanded Facility Evaluation"* as **Future Work**!
- **Conflict:** Claiming "full multi-seed analysis" in Section 2.6 directly contradicts both the experimental setup (`seed: 0`) and the Future Work section.
- **Verdict:** **FAIL (Contradiction)**. Must be qualified in Section 2.6 to remove the unverified claim of multi-seed execution.

### Issue TRACE-02 (Minor): Software Version Inconsistency (Ultralytics 8.4.184 vs 8.4.185)
- **Location:** `paper/sections/04_experimental_setup.tex` (Line 56).
- **Observed Text:** *"Ultralytics 8.4.185"*.
- **Actual Environment:** `python3 -c "import ultralytics; print(ultralytics.__version__)"` outputs `8.4.184`.
- **Verdict:** **Minor Discrepancy**. Should be updated to `8.4.184` for absolute precision.

### Issue TRACE-03 (Major): Staged Hyperparameter Discrepancy ($lr_0$) in Arm 3
- **Location:** `paper/sections/04_experimental_setup.tex` (Line 61) vs `scripts/run_two_stage_experiment.py` (Line 104).
- **Observed Text:** Paper Section 4.4 states: *"AdamW with initial learning rate $\eta = 0.00125$"*.
- **Actual Code:** In `scripts/run_two_stage_experiment.py`, Stage 2 uses `lr0=0.0005`.
- **Verdict:** **PARTIAL**. The paper should disclose that Stage 2 fine-tuning refined $lr_0$ to $0.0005$ to prevent destabilizing the warmed-up state space filters.

### Issue TRACE-04 (Critical): "Real-Time" Framing in Abstract vs Sub-1-FPS Reality
- **Location:** `paper/main.tex` (Abstract) and `paper/sections/05_results_discussion.tex`.
- **Observed Text:** Abstract line 48: *"Sequential recurrence on CPU yields 0.72 FPS (1398 ms), proving embedded memory compliance and operator portability rather than real-time tracking."*
- **Code & Benchmark Reality:** Confirmed! The text in the revised paper already honestly disclaims real-time tracking on CPU silicon ($1398\text{ ms}$). However, the Introduction still opens with *"Real-time multi-class visual perception is a foundational prerequisite..."*.
- **Verdict:** **PASS WITH CAVEAT**. The disclaimer is prominent and scientifically honest in Abstract and Conclusion, but must remain strictly guarded across all sections.

---

## 3. Exit Gate G1 Verdict: PASS WITH LIMITATIONS
All core architectural and empirical claims have verified implementations and log traces. The 4 identified tension points are documented above and queued for controlled repair in Stage 7/8.

