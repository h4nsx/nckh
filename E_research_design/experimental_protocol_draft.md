# Experimental Protocol Draft & Baseline Specifications

**Document ID:** `E_research_design/experimental_protocol_draft.md`  
**Date:** October 2026  
**Status:** Canonical Methodological Protocol  
**Associated Project:** *A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices*  

---

## 1. Experimental Objectives and Integrity Principles

This protocol governs all empirical evaluations in the project to ensure scientific reproducibility, statistical rigor, and fair baseline comparisons.

### Core Protocol Mandates
1. **No Data Leakage:** Frame sequences from video streams must be partitioned strictly by **Scene and Camera ID**, never by random frame slicing.
2. **Fixed Training Budget:** All models (baselines, ablations, proposed hybrids) will be trained under an identical epoch budget, image resolution, optimizer policy, and augmentation pipeline.
3. **Hardware-Anchored Benchmarking:** Latency numbers must never be estimated from FLOPs or reported without explicit runtime, engine version, power mode, and timing boundaries.

---

## 2. Benchmark Baseline Matrix

To evaluate whether the proposed hybrid architecture achieves a competitive accuracy–efficiency trade-off, it will be evaluated against five distinct baseline categories under identical dataset splits and evaluation scripts.

### Table 2.1: Formal Candidate Baseline Suite

| Baseline ID | Model Name | Category | Primary Justification | Official Source / Architecture | Pretrained Weights Policy |
|:---|:---|:---|:---|:---|:---|
| **BASE-01** | **YOLOv8-Small** (Jocher et al., 2023) | Industry Standard CNN | Primary medium-lightweight CNN benchmark (~11.2M params, 28.6 GFLOPs). | CSPDarkNet + PANet decoupled head | COCO pretrained weights fine-tuned on target dataset. |
| **BASE-02** | **YOLO11-Small** (Ultralytics, 2024) | Latest SOTA CNN | State-of-the-art CNN baseline with C3k2 and C2PSA attention (~9.4M params, 21.5 GFLOPs). | C3k2 + C2PSA + RepPANet | COCO pretrained weights fine-tuned on target dataset. |
| **BASE-03** | **YOLOv8-Nano** (Jocher et al., 2023) | Ultra-Lightweight CNN | Lower-bound efficiency benchmark (~3.2M params, 8.7 GFLOPs). | Compact CSPDarkNet | COCO pretrained weights fine-tuned on target dataset. |
| **BASE-04** | **MambaNeXt-YOLO** (Lei et al., 2025) | Closest Published Hybrid | Direct competing hybrid CNN-Mamba detector (~7.1M params, 22.4 GFLOPs, 31.9 FPS on Orin NX). | ConvNeXt + SS2D + ResGate + MAFPN | Initialized from published weights or trained from scratch under matched setup. |
| **BASE-05** | **Pure CNN Matched Control** | Internal Ablation Control | Exact proposed architecture, but with Hybrid Visual Mamba Blocks replaced by standard CNN C3k2 blocks of matched channel width. | Replaces HVMB with C3k2 | Trained from scratch under matched setup to isolate Mamba's net contribution. |

---

## 3. Evaluation Metrics and Operational Definitions

### 3.1 Detection Performance Metrics
Evaluated on the held-out test split at IoU thresholds from 0.5 to 0.95:
* **$\text{mAP@0.5}$ (AP50):** Mean Average Precision at IoU $= 0.50$ (standard industrial acceptance metric).
* **$\text{mAP@0.5:0.95}$ (AP):** Primary ranking metric averaged over 10 IoU intervals ($[0.50 : 0.05 : 0.95]$).
* **Per-Scale AP ($\text{AP}_S, \text{AP}_M, \text{AP}_L$):**
  * $\text{AP}_S$: Small objects (area $< 32^2$ pixels; captures barcodes, labels, small packages).
  * $\text{AP}_M$: Medium objects ($32^2 \le \text{area} < 96^2$ pixels; cartons, totes, small pallets).
  * $\text{AP}_L$: Large objects (area $\ge 96^2$ pixels; forklifts, AGVs, high-bay shelving).
* **Per-Class Breakdown:** Precision, Recall, and AP for every individual warehouse class.

### 3.2 Computational Complexity & Edge Efficiency Metrics
* **Parameters ($M$):** Total trainable and non-trainable parameter count (budget: $\le 7.5\text{ M}$).
* **GFLOPs ($G$):** Multiply-Accumulate operations calculated at $640 \times 640 \times 3$ input resolution using `fvcore` or `thop`.
* **FP16 Model Storage ($MB$):** Serialized engine / weight file size on disk (budget: $< 30\text{ MB}$).
* **Peak GPU/SoC Memory ($MB$):** Maximum active memory allocated during inference execution, measured via `torch.cuda.max_memory_allocated()` and TegraStats.
* **Latency ($ms$) & Throughput ($\text{FPS}$):** Measured under the standardized edge protocol (Section 4).

---

## 4. Standardized Edge Latency Measurement Protocol

To ensure reproducible and unassailable timing figures, all benchmarks will adhere to the following protocol:

```
+----------------------------------------------------------------------------------------------------+
|                                STANDARDIZED TIMING PROTOCOL BREAKDOWN                              |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [STAGE 1: Pre-Processing]                                                                         |
|    - Letterbox image resize (640x640), normalization, FP32 -> FP16 cast, HWC -> CHW layout.       |
|    - Timing: Recorded separately as T_pre.                                                         |
|                                                                                                    |
|  [STAGE 2: Model Inference (TensorRT FP16 Engine)]                                                 |
|    - Hardware: NVIDIA Jetson Orin NX (16GB, 20W MAX-N mode, locked clock frequencies via jetson_clocks)|
|    - Batch size = 1. Synchronized execution with cudaEventSynchronize().                            |
|    - Warm-up: 200 consecutive inference passes discarded.                                         |
|    - Benchmark: 1,000 measured iterations.                                                         |
|    - Reported metrics: Median (p50), 95th percentile (p95), Mean, and Standard Deviation.         |
|    - Timing: Recorded as T_infer (Model-only latency).                                             |
|                                                                                                    |
|  [STAGE 3: Post-Processing & Head Decoding]                                                       |
|    - Bounding box de-normalization, score thresholding (conf >= 0.25), NMS (IoU >= 0.65).          |
|    - Timing: Recorded separately as T_post.                                                        |
|                                                                                                    |
|  [TOTAL PIPELINE THROUGHPUT]                                                                       |
|    - End-to-End Latency: T_total = T_pre + T_infer + T_post                                        |
|    - Real-Time Acceptance Criterion: T_total <= 33.3 ms (>= 30.0 FPS)                              |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

*Note on Hardware Availability:* If the physical Jetson Orin NX unit is pending lab access, all export engines (ONNX, TensorRT) will be compiled and benchmarked on the available workstation GPU (recording driver, CUDA, TensorRT version), with edge deployment explicitly designated as an **unresolved milestone** pending physical hardware availability.

---

## 5. Controlled Ablation Experiment Series

### Table 5.1: Master Ablation Matrix

| Ablation ID | Target Component | Experimental Configuration | Independent Variable | Controlled Conditions | Hypothesized Outcome |
|:---|:---|:---|:---|:---|:---|
| **ABL-01** | Net Mamba Impact | Proposed Hybrid vs. Matched Pure-CNN Control | Presence of HVMB blocks | Identical parameter count, learning rate, training epochs, dataset split. | Hybrid achieves $\ge +1.8\%\text{ mAP@50:95}$ over CNN control, driven by $+3.0\%\text{ AP}_S$ and occluded classes. |
| **ABL-02** | Block Placement | (a) HVMB in $P_2\text{--}P_5$ vs.<br>(b) HVMB in $P_4\text{--}P_5$ (Proposed) vs.<br>(c) HVMB in $P_5$ only | Spatial stage insertion point | Backbone channel widths and neck topology held constant. | Configuration (b) retains $92\%$ of accuracy of (a) while running $1.8\times$ faster due to avoiding high-resolution MAC. |
| **ABL-03** | Asymmetric Neck Gating | (a) Standard PANet vs.<br>(b) BiFPN vs.<br>(c) Proposed Context-Gated Neck | Multi-scale neck feature flow | Backbone held fixed to proposed hybrid. | Context-Gated neck yields highest $\text{AP}_S$ on tiny warehouse tags with $<0.5\text{ ms}$ latency overhead. |
| **ABL-04** | Detection Head & Assigner | (a) Coupled head vs.<br>(b) Decoupled Head + TAL vs.<br>(c) Decoupled + NMS-Free Training | Head architecture and loss assignment | Backbone and neck held constant. | Decoupled head boosts mAP by $+1.2\%$; NMS-free inference cuts $T_{\text{post}}$ by $\ge 3.5\text{ ms}$ on edge CPU. |
| **ABL-05** | SSM Scan Direction | (a) Standard 4-way SS2D vs.<br>(b) 2-way Orthogonal Rectilinear Scan | Sequence traversal order | Model weights and block structure identical. | Test hypothesis whether warehouse shelf geometry benefits from orthogonal rectilinear scanning without kernel slowdown. |

---

## 6. Pre-Registration of Experimental Hypotheses

* **Hypothesis H1 (Accuracy Superiority):** The proposed hybrid architecture will achieve higher $\text{mAP@0.5:0.95}$ than YOLOv8-Small and MambaNeXt-YOLO at a matched parameter budget ($\le 7.5\text{ M}$).
* **Hypothesis H2 (Occlusion Robustness):** On test images exhibiting severe occlusion ($>40\%$ bounding box overlap), the hybrid model will demonstrate an AP gain $\ge 2.5\%$ over YOLOv8-Small due to global context modeling.
* **Hypothesis H3 (Edge Real-Time Feasibility):** Model-only TensorRT FP16 latency on NVIDIA Jetson Orin NX will remain $\le 25.0\text{ ms}$ ($\ge 40\text{ FPS}$), guaranteeing end-to-end pipeline execution $\ge 30\text{ FPS}$.
* **Falsification Criteria:** If the proposed hybrid achieves less than $+0.5\%\text{ mAP}$ improvement over the matched pure-CNN control (ABL-01) or runs at $<25\text{ FPS}$ on Orin NX, the central hypothesis is falsified, and the architecture must be refactored or pivoted.
