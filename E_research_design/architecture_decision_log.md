# Architecture Decision Log & Component Audit

**Document ID:** `E_research_design/architecture_decision_log.md`  
**Date:** October 2026  
**Status:** Canonical Architectural Audit & Component Specification  
**Associated Project:** *A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices*  

---

## 1. Architectural Philosophy and Design Principles

The design of the proposed hybrid detector is governed by three non-negotiable principles:
1. **Hypothesis Separation:** Architectural proposals are hypotheses to be tested, not established truths.
2. **Hardware Realism:** Flops and parameter counts are secondary to physical Memory Access Cost (MAC), kernel fusion, and cache hierarchy on embedded edge SoCs.
3. **Controlled Modularity:** Every proposed block must have a matched CNN-only control counterpart to enable rigorous ablation isolation.

---

## 2. Component-by-Component Independent Audit

```
+----------------------------------------------------------------------------------------------------+
|                                    PROPOSED SYSTEM TOPOLOGY                                        |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Input: 640x640x3]                                                                                |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 1: CNN High-Resolution Stages P1-P3]                                             |  |
|  | - Focus: Preserves local edges, fine gradients, low MAC at large spatial resolutions       |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 2: Hybrid Visual Mamba Blocks at Low-Resolution P4-P5]                            |  |
|  | - Focus: Long-range global context, rack reasoning, linear O(N) sequence scaling            |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 3: Context-Gated Asymmetric Neck]                                                 |  |
|  | - Focus: Modulates P3 fine features with P5 global context vector; avoids heavy P3 SSMs      |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 4 & 5: Anchor-Free Decoupled Head + TAL / Optional NMS-Free Training]            |  |
|  | - Focus: Task-aligned assignment, DFL distribution, minimized edge post-processing latency   |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

### Component 1: CNN-Only High-Resolution Stages (P1–P3)

* **Motivation & Supporting Literature:**
  * At early stages ($P_1$: $320 \times 320$, $P_2$: $160 \times 160$, $P_3$: $80 \times 80$), standard $3 \times 3$ convolutions possess optimal inductive bias (translation invariance, locality) for edge, corner, and texture extraction.
  * *Mittal (2024, P005):* Convolutions exhibit high arithmetic intensity (compute operations per memory byte accessed). High-resolution feature maps processed via standard convs leverage highly optimized cuDNN fused kernels.
* **Expected Benefit:**
  * Prevents the memory explosion and latency penalties suffered by early-stage Mamba/Transformer insertion.
  * Fast feature downsampling with low Memory Access Cost (MAC).
* **Potential Computational & Optimization Cost:**
  * Receptive field remains strictly local at early stages; cannot disambiguate distant semantic cues.
* **Alternative Designs:**
  * *Alternative 1A:* Early Mamba insertion (as in Hybrid-YOLO, Chen & Er, 2024).
  * *Alternative 1B:* Depthwise separable convolution blocks (MobileNetv4 / LEAF-YOLO).
* **Definitive Ablation Experiment (ABL-02A):**
  * Compare identical detector with: (i) Pure CNN $P_1\text{--}P_3$, (ii) Depthwise Conv $P_1\text{--}P_3$, and (iii) Mamba blocks inserted at $P_2$ and $P_3$.
  * *Metrics measured:* Latency on target edge platform (ms), peak memory allocation (MB), and mAP@0.5.

---

### Component 2: Mamba/SSM Blocks at Low-Resolution Stages (P4–P5)

* **Motivation & Supporting Literature:**
  * At $P_4$ ($40 \times 40$, stride 16, $N=1600$) and $P_5$ ($20 \times 20$, stride 32, $N=400$), the spatial sequence length is compact.
  * *Lei et al. (2025, P001, MambaNeXt-YOLO):* Demonstrates that selective scan (SS2D) captures global scene context with linear complexity $\mathcal{O}(N)$ compared to quadratic self-attention $\mathcal{O}(N^2)$.
  * *ResGate Fusion:* Gating linear projections with depthwise convolutions and residual paths stabilizes gradients during selective scan.
* **Expected Benefit:**
  * Enables holistic reasoning across large warehouse structures (e.g., associating an isolated carton with its parent pallet rack 400 pixels away).
  * Linear computational complexity prevents exponential FLOP explosion.
* **Potential Computational & Optimization Cost:**
  * Selective scan requires 4 directional scans (SS2D), requiring tensor transpositions and unrolled prefix sums.
  * **Critical Cache Caveat:** While sequence length is reduced ($N=400$), whether state tensors remain entirely within hardware L2 cache (e.g., 4MB on Jetson Orin NX) is a **hardware-dependent hypothesis** requiring empirical Nsight profiling, not a mathematical certainty.
* **Alternative Designs:**
  * *Alternative 2A:* Multi-Head Self-Attention (MHSA) or Deformable Attention (RT-DETR style).
  * *Alternative 2B:* C2PSA (Pointwise Spatial Attention as in YOLO11).
  * *Alternative 2C:* Standard CNN bottleneck (C3k2 / C2f).
* **Definitive Ablation Experiment (ABL-02B):**
  * Compare: (i) Pure CNN C3k2 at $P_4\text{--}P_5$, (ii) C2PSA spatial attention at $P_4\text{--}P_5$, (iii) Standard MHSA at $P_4\text{--}P_5$, and (iv) Proposed Hybrid Visual Mamba Block (HVMB) at $P_4\text{--}P_5$.
  * *Metrics measured:* mAP@0.5:0.95, GFLOPs, parameter count, and TensorRT FP16 latency on edge hardware.

---

### Component 3: P5-to-P3 Context-Gated Asymmetric Neck

* **Motivation & Supporting Literature:**
  * Tiny objects (bin tags, labels, small packages) are represented primarily in $P_3$ features, but local $P_3$ patches lack the semantic context to distinguish a legitimate package from background rack clutter.
  * *Yang et al. (2026, P003, UAVDet):* Proved that rebalancing multi-scale feature flow toward small targets yields significant AP gains.
* **Expected Benefit:**
  * Globally pooled contextual vectors from $P_5$ (which encode warehouse structural geometry) are projected and used as a channel-wise / spatial gate to modulate $P_3$ features.
  * Enhances tiny target saliency without inserting expensive attention or Mamba operators directly into the high-resolution $P_3$ feature maps.
* **Potential Computational & Optimization Cost:**
  * Additional $1 \times 1$ pointwise projections and sigmoid/GELU activation layers.
  * Risk of over-suppressing rare classes if gating weights are improperly regularized.
* **Alternative Designs:**
  * *Alternative 3A:* Standard symmetric PANet neck (as in YOLOv8).
  * *Alternative 3B:* BiFPN with learned scalar weights.
  * *Alternative 3C:* Full SSM-based neck (MAFPN as in MambaNeXt-YOLO).
* **Definitive Ablation Experiment (ABL-03):**
  * Compare: (i) Standard symmetric PANet neck, (ii) BiFPN neck, (iii) MAFPN neck (SSM across all levels), and (iv) Proposed Context-Gated Asymmetric Neck.
  * *Metrics measured:* mAP_S (small objects), overall mAP@0.5:0.95, and neck latency breakdown.

---

### Component 4: Anchor-Free Decoupled Detection Head

* **Motivation & Supporting Literature:**
  * *Sapkota & Karkee (2026, P004, YOLO Evolution):* Decoupling classification and regression branches prevents conflicting gradient updates between object category discrimination (translation-invariant) and bounding box regression (translation-covariant).
  * Anchor-free formulation eliminates manually tuned anchor boxes, which generalize poorly to wide scale disparities.
* **Expected Benefit:**
  * Improved localization accuracy and cleaner gradient convergence.
  * Shared convolution weights across pyramid heads can reduce parameter footprint.
* **Potential Computational & Optimization Cost:**
  * Decoupled heads introduce two parallel branches per scale, increasing parameter count slightly compared to coupled heads (YOLOv5).
* **Alternative Designs:**
  * *Alternative 4A:* Coupled single-branch detection head (YOLOv5 style).
  * *Alternative 4B:* Query-based Transformer decoder (RT-DETR style).
* **Definitive Ablation Experiment (ABL-04):**
  * Compare: (i) Coupled head vs. (ii) Decoupled anchor-free head under identical backbone and neck.
  * *Metrics measured:* Parameter count, mAP@0.5:0.95, and head forward-pass latency.

---

### Component 5: Task-Aligned Assigner (TAL) and Optional NMS-Free Training

* **Motivation & Supporting Literature:**
  * *Feng et al. (TOOD) & Jocher et al. (YOLOv8):* Task-Aligned Assigner dynamically computes matching metric $t = s^\alpha \times \text{IoU}^\beta$ aligning classification score $s$ and spatial bounding box overlap.
  * *Wang et al. (YOLOv10) & Sapkota & Karkee (2026, P004):* Dual-label assignment during training (one-to-many branch for rich supervisory signal + one-to-one branch for inference) eliminates post-processing NMS latency.
* **Expected Benefit:**
  * Eliminating NMS removes dynamic CPU/GPU synchronization and eliminates the variable latency tail caused by dense overlapping candidate boxes.
* **Potential Computational & Optimization Cost:**
  * Dual-label assignment requires slightly more VRAM and compute during training (no impact on inference).
  * If one-to-one assignment converges sub-optimally, recall on heavily clustered small targets may suffer.
* **Alternative Designs:**
  * *Alternative 5A:* Standard NMS post-processing with TAL and Distribution Focal Loss (DFL).
  * *Alternative 5B:* DFL removal (streamlined compact regression) to minimize integer operations on edge NPUs.
* **Definitive Ablation Experiment (ABL-05):**
  * Compare: (i) Standard TAL + DFL + NMS, (ii) TAL without DFL + NMS, and (iii) Dual-label one-to-one assignment (NMS-free).
  * *Metrics measured:* Recall, mAP@0.5:0.95, post-processing latency (ms), and end-to-end pipeline FPS.

---

## 3. Decision Summary & Traceability

| Component ID | Current Decision | Classification | Priority for Implementation | Controlling Experiment ID |
|:---|:---|:---|:---:|:---:|
| **CMP-01** | CNN for $P_1\text{--}P_3$ | **Adopted as Baseline Baseline Design** | High | ABL-02A |
| **CMP-02** | Hybrid Visual Mamba for $P_4\text{--}P_5$ | **Design Hypothesis under Investigation** | High | ABL-02B |
| **CMP-03** | Context-Gated Asymmetric Neck | **Design Hypothesis under Investigation** | Medium | ABL-03 |
| **CMP-04** | Decoupled Anchor-Free Head | **Adopted Standard Industry Baseline** | High | ABL-04 |
| **CMP-05** | TAL + Evaluation of NMS-Free | **Conditional Strategy (Train-time only)** | Medium | ABL-05 |
