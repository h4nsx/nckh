# EDGE EXPORT AND BENCHMARK AUDIT REPORT (Stage 6 Audit)

- **Date:** 2026-11-04
- **Auditor Role:** Autonomous Research Implementation Auditor (SKILLS.md Stage 6)
- **Artifacts Audited:**
  - `yolo11s.onnx` (Opset 18, 36.2 MB)
  - `models/yolo11s_ss2d_full_warmstart.onnx` (Opset 17, 38.5 MB)
  - `scripts/benchmark_edge_device.py` & `scripts/profile_onnx_operators.py`
  - `report.md` (Raspberry Pi 4 Model B Rev 1.2 empirical measurements)
- **Status:** **STAGE 6 COMPLETE (PASS)**

---

## 1. Export Correctness & PyTorch-to-ONNX Parity

| Validation Dimension | Specification / Expectation | Measured Result | Status |
|---|---|---|---|
| **Export Opset** | Standard standalone ONNX (Opset 17/18) | Baseline: Opset 18; SS2D: Opset 17 | **PASS** |
| **Custom Operator Dependency** | Zero proprietary C++/CUDA extensions | Graph contains only standard ONNX ops (`Gather`, `Mul`, `Add`, `Softplus`, `Conv`) | **PASS** |
| **Input / Output Shapes** | `images: [1, 3, 640, 640]` $\to$ `output0: [1, 84, 8400]` | Verified under `onnxruntime` across both models | **PASS** |
| **Baseline Parity** | Max absolute difference $< 10^{-3}$ | Max diff: $7.17 \times 10^{-4}$, Mean diff: $1.09 \times 10^{-6}$ | **PASS** |
| **SS2D Parity** | Max absolute difference $< 10^{-3}$ | Max diff: $7.32 \times 10^{-4}$, Mean diff: $1.20 \times 10^{-6}$ | **PASS** |

Both models achieve high numerical output fidelity with mean differences $\approx 10^{-6}$, confirming that operator translation and graph optimizations did not alter model predictions.

---

## 2. Graph Complexity & Latency Causality Analysis

A primary scientific contribution of this study is explaining the latency behavior on general-purpose CPUs:

- **Baseline YOLO11s Graph:** **320 dense parallel nodes** exploiting BLAS multi-threaded matrix multiplications (GEMM) across all CPU cores.
- **YOLO11s-SS2D (Case B) Graph:** **14,767 nodes** resulting from unrolling the 4-directional recurrent loop ($L = 400$ spatial tokens $\times$ 4 scan directions).
- **Physical CPU Latency Overhead:** The recurrence updates run sequentially on CPU silicon without fused parallel prefix-scan kernels, explaining why execution time increases by $\approx +20\text{--}24\%$ across both x86 Core i7 ($304.0\text{ ms}$ vs. $244.7\text{ ms}$) and ARM Cortex-A72 ($1398.2\text{ ms}$ vs. $1169.7\text{ ms}$) despite lower theoretical FLOPs.

---

## 3. Physical Embedded Silicon Benchmark Integrity (Raspberry Pi 4)

Measurements on the Raspberry Pi 4 Model B (Quad-Core Cortex-A72 @ 1.5 GHz, 4GB RAM) were re-verified against `report.md` and paper Table IV:

| Architecture | ONNX Size | Measured Latency (ms) | Throughput (FPS) | Peak RAM RSS | Memory Budget Margin ($\le 1.0\text{ GB}$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Control** | 36.18 MB | $1169.73 \pm 24.7$ ms | 0.85 FPS | 231.5 MB | 76.8% Headroom |
| **Case B (P5 SS2D)** | **38.47 MB** | **$1398.19 \pm 28.0$ ms** | **0.72 FPS** | **371.9 MB** | **62.8% Headroom** |
| **Case C (P4+P5 SS2D)**| 46.62 MB | $1965.07 \pm 29.7$ ms | 0.51 FPS | 714.9 MB | 28.5% Headroom |

### Verification of Scientific Claims:
1. **Memory Ceiling Compliance:** Peak RSS for Case B is $371.9\text{ MB}$, leaving $62.8\%$ safety headroom under a $1.0\text{ GB}$ gateway envelope.
2. **Honest Latency Framing:** Sub-1-FPS ($0.72\text{ FPS}$) is prominently and honestly reported as proving **memory compliance and operator portability rather than real-time tracking**, establishing that line-rate speed ($>30\text{ FPS}$) requires GPU/NPU prefix-scan accelerators.

---

## 4. Exit Gate G6 Verdict: PASS
The export graph is valid, numerical parity is verified, and physical embedded measurements are rigorously documented and scientifically framed.

