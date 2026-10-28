# Final Reproducibility Gate & Artifact Audit Manifest

**Project:** Lightweight 2D Selective State Space (SS2D) Sequence Modeling for Automated Warehouse Object Detection  
**Deliverable Document:** `Reports/FINAL_REPRODUCIBILITY_GATE.md`  
**Manuscript Artifact:** `paper/main.pdf` (Compiled, 9 Pages, IEEE Transactions Format)  
**Date:** October 28, 2026  
**Status:** **`FULLY VERIFIED & REPRODUCIBLE (PASS)`**

---

## 1. Executive Summary

This manifest serves as the final reproducibility gate for our research project. It verifies that every table, equation, architectural diagram, software script, physical edge benchmark measurement, and checkpoint SHA-256 checksum in the repository is strictly aligned and reproducible.

---

## 2. Checkpoint Provenance & Artifact Checksums

Every model evaluated in the paper and benchmarked on edge devices is backed by an on-disk binary checkpoint with an immutable SHA-256 hash:

### Table 1: Checkpoint Provenance & Checksum Manifest

| Model Arm / Artifact | Relative File Path | Format / Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Arm 1: Baseline YOLO11s** | `runs/detect/.../arm1_baseline_control/weights/best.pt` | PyTorch (18.29 MB) | `a213d99a40c4f405dd28f8d07670e34d7e4a000fb8fa51f60549b72a63e52870` |
| **Arm 2: Case B (Single-Stage)** | `runs/detect/.../arm2_case_b_single_stage/weights/best.pt` | PyTorch (17.07 MB) | `16a9ff3914c37e248c881256c768afd67748ea076dccf774eec5cd5d1470f606` |
| **Arm 3: Case B (Two-Stage)** | `runs/detect/.../arm3_stage2_finetune/weights/best.pt` | PyTorch (17.07 MB) | `5b90cc10afc906ec77d5b9257367032ad11172d013f6be172fdd30a70d007f0e` |
| **Arm 4: Case C (Dual-Scale)** | `runs/detect/.../arm4_case_c_dual_scale/weights/best.pt` | PyTorch (17.26 MB) | `22267e09e823cfab58e02b9e15ed976114ad72e1420224509ba0d981d33685c5` |
| **Baseline ONNX (4-Class)** | `runs/detect/.../arm1_baseline_control/weights/best.onnx` | ONNX (36.18 MB) | `ca47c3d8cc3018298010a25db990a613f9a784127b190c8fe8b9f5a92e2c7e23` |
| **Case B ONNX (4-Class)** | `runs/detect/.../arm3_stage2_finetune/weights/best.onnx` | ONNX (38.47 MB) | `677274c7d89a350f1ad812ea1c2cef16428b5301b2f04597d04c301319051e0a` |
| **Case C ONNX (4-Class)** | `runs/detect/.../arm4_case_c_dual_scale/weights/best.onnx` | ONNX (46.62 MB) | `5c29d0596c5636565f52d1f6c34f9ffce392d2a12d3725073e31c99cd5934706` |

---

## 3. Physical Edge Hardware Verification

* **Physical Unit:** Raspberry Pi 4 Model B Rev 1.2
* **CPU:** Quad-Core Broadcom BCM2711 ARM Cortex-A72 @ 1.5 GHz
* **RAM:** 4 GB LPDDR4-3200 (Total Available: 3,795.7 MiB, Swap: 2,048.0 MiB)
* **Operating System:** Linux Debian / Raspberry Pi OS 12 (Bookworm, 64-bit aarch64)
* **Runtime:** ONNX Runtime v1.31 (`CPUExecutionProvider`, Batch Size = 1, Resolution = $640 \times 640$)

### Measured Empirical Hardware Profile:
* **Baseline Control:** $1169.73 \pm 24.68\text{ ms}$ (0.86 FPS), **$231.5\text{ MB}$ peak RAM RSS**.
* **YOLO11s-SS2D (Case B, P5):** $1398.19 \pm 28.04\text{ ms}$ (0.72 FPS), **$371.9\text{ MB}$ peak RAM RSS** (**$62.8\%$ safety headroom** under 1.0 GB).
* **YOLO11s-SS2D (Case C, P4+P5):** $1965.07 \pm 29.67\text{ ms}$ (0.51 FPS), **$714.9\text{ MB}$ peak RAM RSS**.

---

## 4. Software Environment & Execution Seeds

* **Python Environment:** Python 3.12+ (Host PC) / Python 3.13 (Raspberry Pi venv).
* **Deep Learning Frameworks:** PyTorch 2.4.0+cu121, Ultralytics YOLO 8.3+, ONNX 1.17+, ONNX Runtime 1.31+.
* **Constant Execution Seed:** `seed=0` across all data split generation, weight surgery, and training runs.
* **Deterministic Input Shapes:** Input `[1, 3, 640, 640]`, Output `[1, 8, 8400]` (4 box coordinates + 4 class probabilities for `worker`, `forklift`, `pallet_truck`, `amr`).

---

## 5. Paper Alignment & PDF Build Status

* **Manuscript Source:** `paper/main.tex` and `paper/sections/` (01_introduction through 06_conclusion).
* **Compilation Command:** `pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
* **Build Status:** **`0 Errors`**, **`9 Pages`**, IEEE Transactions format.
* **Deliverable PDF:** [`paper/main.pdf`](file:///home/hans/Documents/nckh/paper/main.pdf) (8.85 MB).

---

## 6. Complete Deliverables Directory

All required audit gates and reports have been generated and committed to the repository:
1. [Reports/RESULTS_INTEGRITY_GATE.md](file:///home/hans/Documents/nckh/Reports/RESULTS_INTEGRITY_GATE.md) — Accuracy traceability, +6.2% forklift verification, and training protocol parity.
2. [Reports/EDGE_MODEL_CONFIGURATION_AUDIT.md](file:///home/hans/Documents/nckh/Reports/EDGE_MODEL_CONFIGURATION_AUDIT.md) — Resolution of tensor output shape `(1, 8, 8400)` and class configuration audit.
3. [Reports/CONTROLLED_EXPERIMENTS_GATE.md](file:///home/hans/Documents/nckh/Reports/CONTROLLED_EXPERIMENTS_GATE.md) — Multi-arm experimental control, head weight transfer audit, and tail-class variance analysis.
4. [Reports/ABLATION_GATE.md](file:///home/hans/Documents/nckh/Reports/ABLATION_GATE.md) — Single-scale vs. dual-scale ablation, CPU recurrence latency trade-off, and robustness failure modes.
5. [Reports/NOVELTY_AND_CLAIMS_GATE.md](file:///home/hans/Documents/nckh/Reports/NOVELTY_AND_CLAIMS_GATE.md) — Literature differentiation, technical distinction from canonical VMamba, and claim boundary guardrails.
6. [Reports/FINAL_REPRODUCIBILITY_GATE.md](file:///home/hans/Documents/nckh/Reports/FINAL_REPRODUCIBILITY_GATE.md) — Checkpoint SHA-256 hashes, physical edge telemetry, and full reproducibility protocol.
7. [paper/main.pdf](file:///home/hans/Documents/nckh/paper/main.pdf) — Compiled research paper.

**FINAL GATE SIGN-OFF: COMPLETED.**
