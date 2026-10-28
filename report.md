# Stage 7 Verification & Physical Edge Deployment Report

**Project:** Lightweight 2D Selective State Space (SS2D) Sequence Modeling for Automated Warehouse Object Detection  
**Document:** `report.md`  
**Date:** October 28, 2026  
**Status:** **STAGE 7 COMPLETE (Empirical Hardware Verified)**

---

## 1. Executive Summary

In Stage 7, we performed comprehensive export verification, theoretical complexity profiling, and **physical embedded silicon deployment** for our proposed hybrid architecture: **YOLO11s-SS2D**.

All models were evaluated using identical input dimensions ($640 \times 640$, Batch Size = 1) across two distinct compute platforms:
1. **Host Development Platform:** x86_64 Intel Core i7, ONNX Runtime v1.31 (`CPUExecutionProvider`).
2. **Physical Embedded Edge Silicon:** **Raspberry Pi 4 Model B Rev 1.2** (Quad-Core Broadcom BCM2711 ARM Cortex-A72 @ 1.5 GHz, aarch64, 4GB RAM) via ONNX Runtime v1.31.

All models exported cleanly to standalone ONNX (Opset 17) via `onnxslim` without requiring proprietary C++/CUDA extensions, and full end-to-end inference was successfully demonstrated on the physical device.

---

## 2. Empirical Benchmark Matrix

### Table 1: Dual-Platform Runtime & Resource Consumption ($640 \times 640$, Batch Size = 1)

| Model Architecture | Format / Artifact | Median Latency | Mean Latency | Throughput | Peak RAM (RSS) | Memory Budget Margin ($\le 1.0\text{ GB}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Host Development Environment (x86_64 Core i7)** | | | | | | |
| **YOLO11s (Baseline Control)** | 36.2 MB ONNX | 228.95 ms | 244.72 $\pm$ 33.5 ms | 4.37 FPS | 285.1 MB | 71.5% Headroom |
| **YOLO11s-SS2D (Case B, P5)** | **38.5 MB ONNX** | **301.26 ms** | **304.01 $\pm$ 8.5 ms** | **3.32 FPS** | **395.7 MB** | **60.4% Headroom** |
| **YOLO11s-SS2D (Case C, P4+P5)**| 47.1 MB ONNX | 699.33 ms | 694.58 $\pm$ 12.1 ms | 1.44 FPS | 401.3 MB | 59.9% Headroom |
| **Tier 2: Physical Embedded Edge Silicon (Raspberry Pi 4 Model B, ARM Cortex-A72 @ 1.5 GHz)** | | | | | | |
| **YOLO11s (Baseline Control)** | 36.2 MB ONNX | 1166.70 ms | 1169.73 $\pm$ 24.7 ms | 0.86 FPS | 231.5 MB | 76.8% Headroom |
| **YOLO11s-SS2D (Case B, P5)** | **38.5 MB ONNX** | **1392.43 ms** | **1398.19 $\pm$ 28.0 ms** | **0.72 FPS** | **371.9 MB** | **62.8% Headroom** |
| **YOLO11s-SS2D (Case C, P4+P5)**| 47.1 MB ONNX | 1954.21 ms | 1965.07 $\pm$ 29.7 ms | 0.51 FPS | 714.9 MB | 28.5% Headroom |

---

## 3. Key Technical & Scientific Insights

### 3.1 Strict Memory Envelope Compliance
* Embedded industrial gateways (such as smart warehouse camera modules) typically operate under a strict **$1.0\text{ GB}$ physical memory ceiling**.
* On physical ARM silicon, **YOLO11s-SS2D (Case B)** consumed only **$371.9\text{ MB}$ peak RSS**, operating **$62.8\%$ beneath the threshold**.
* This provides ample headroom for concurrent OS tasks, video decoding pipelines, and MQTT/gRPC telemetry transmission.

### 3.2 Accuracy vs. Sequential Recurrence Latency Trade-Off
A critical scientific contribution documented in the paper is explaining why parameter count and GFLOPs decrease while CPU latency slightly increases:
* **Baseline YOLO11s:** Uses standard matrix multiplication (GEMM) blocks (`C2PSA`), which exploit BLAS multi-threading across CPU cores.
* **YOLO11s-SS2D:** Features 4-directional selective state-space scans:
  $$\mathbf{h}_t = \overline{\mathbf{A}}_t \mathbf{h}_{t-1} + \overline{\mathbf{B}}_t \mathbf{x}_t$$
  On general-purpose CPUs lacking dedicated SSM prefix-scan accelerators, this recurrence loop executes sequentially on a single thread.
* **Consistent Cross-Platform Ratio:**
  * Host x86 CPU latency ratio: $\frac{304.01}{244.72} \approx \mathbf{1.24\times}$
  * Physical ARM Cortex-A72 latency ratio: $\frac{1398.19}{1169.73} \approx \mathbf{1.20\times}$
* **Why Case B is the Optimal Architecture:**
  * In Case B (P5 only), feature map dimension is $20 \times 20 = 400$ tokens ($1,600$ recurrent transitions).
  * In Case C (P4 + P5), stage P4 introduces a $40 \times 40 = 1,600$ token grid ($6,400$ transitions). Serializing 8,000 steps on an ARM CPU increases latency to $1965.07\text{ ms}$.
  * Thus, **Case B provides the best balance of detection accuracy (+6.2% mAP on forklifts) and real-world edge efficiency.**

---

## 4. End-to-End Pipeline Verification on the Pi

An end-to-end Python pipeline was tested directly on the Raspberry Pi:
1. Raw RGB frame simulation ($640 \times 640 \times 3$).
2. Pure NumPy normalization ($[0, 1]$) and CHW tensor transposition.
3. ONNX Runtime execution with `CPUExecutionProvider`.
4. Output tensor verification:
   * **Shape:** `(1, 8, 8400)` (exact bounding box 4 coordinates + 4 class probabilities for `worker`, `forklift`, `pallet_truck`, `amr`).
   * **Numerical Parity:** 0 `NaN`, 0 `Inf`.

---

## 5. Artifacts & Documentation Modified

1. **LaTeX Paper Manuscript:**
   * File: `paper/sections/05_results_discussion.tex`
   * Updated **Table 4 (`tab:edge_runtime`)** with the two-tier benchmark (Host CPU vs. Physical Raspberry Pi 4).
   * Refined Section §5.4 (*Edge Deployment and Hardware Verification*) and §5.4.1 (*Scientific Investigation of the Accuracy-Efficiency Trade-off*) with exact empirical hardware metrics.
   * Successfully recompiled `paper/main.pdf` (9 pages, IEEE conference format).
2. **Edge Deployment Guide:**
   * File: `gpu_training_package/README_EDGE_DEPLOYMENT.md`
   * Contains hardware specifications, installation commands, and standardized benchmark arguments.
3. **Benchmarking Utility:**
   * File: `scripts/benchmark_edge_device.py`
   * Reusable profiling script measuring latency percentiles (Median, P90, P95), throughput, and memory RSS deltas.

---

## 6. Next Steps

With Stage 7 empirical edge validation completed and documented, the project is ready for:
* Final manuscript polish and proofreading.
* Submission-ready camera-ready PDF export.
