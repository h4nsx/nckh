# Architecture Decision Log & Component Audit

**Document ID:** `E_research_design/architecture_decision_log.md`  
**Date:** October 2026 (Updated to v1.1 Alignment)  
**Status:** Canonical Architectural Specification & Experimental Contract  
**Associated Project:** *A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices*  
**Core Baseline Reference:** YOLO11s (`yolo11s.pt` / `yolo11.yaml`)

---

## 1. Architectural Philosophy and Design Principles

The design of the proposed hybrid detector is governed by strict, non-negotiable principles:
1. **Hypothesis Separation:** Architectural proposals are hypotheses to be tested, not established truths.
2. **Minimal & Reversible Modification:** Anchor to the standard YOLO11s baseline. Keep the original neck (FPN/PAN) and Detect head completely unchanged in initial phases.
3. **Hardware & Memory Realism:** Flops and parameter counts are secondary to physical Memory Access Cost (MAC), kernel fusion, and cache hierarchy on embedded edge SoCs.
4. **Controlled Modularity:** Every proposed block must have a matched CNN-only control counterpart to enable rigorous ablation isolation.

---

## 2. Component-by-Component Specification (YOLO11s-SS2D Contract)

```
+----------------------------------------------------------------------------------------------------+
|                                    PROPOSED SYSTEM TOPOLOGY (v1)                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Input: 640x640x3 RGB]                                                                            |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 1: Pure CNN High-Resolution Stages P1-P3]                                         |  |
|  | - Baseline: YOLO11s stem, P1 (stride 2), P2 (stride 4), P3 (stride 8: 80x80x128)             |  |
|  | - Focus: Preserves local edges, fine gradients, low MAC at large spatial resolutions         |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 2: Selective Deep-Stage SS2D Integration (P4-P5)]                                  |  |
|  | - Primary Milestone (Case B): Replace deepest C2PSA block at P5 (stride 32: 20x20) with     |  |
|  |   shape-compatible SS2D/Mamba block ([B, C, H, W] -> [B, C, H, W]).                           |  |
|  | - Follow-up Milestone (Case C): SS2D at P4 (stride 16: 40x40) + P5.                          |  |
|  | - Focus: Long-range global context, rack reasoning, linear O(N) sequence scaling              |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 3: Feature Neck]                                                                  |  |
|  | - v1 Baseline & Primary Experiment: Unchanged original YOLO11s FPN/PAN neck.                 |  |
|  | - Optional Follow-up (Case E): Context-Gated Asymmetric Neck (only if justified by evidence)   |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | [Component 4 & 5: Original YOLO11s Detect Head]                                              |  |
|  | - Keep original decoupled anchor-free Detect head unchanged for v1.                          |  |
|  | - Deferred: Dual-assignment / NMS-free head is out of scope for v1.                          |  |
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
| **CMP-01** | Pure CNN for $P_1\text{--}P_3$ | **Adopted Baseline Design (YOLO11s)** | High | Case A / ABL-02A |
| **CMP-02** | Deep-Stage SS2D at $P_5$ (Replacing C2PSA) | **Primary Research Hypothesis (Case B)** | Highest (First Milestone) | Case B (ABL-01) |
| **CMP-02-Ext** | SS2D at $P_4 + P_5$ | **Follow-up Hypothesis (Case C)** | Medium (Pending Case B) | Case C (ABL-02) |
| **CMP-03** | Context-Gated Asymmetric Neck | **Optional Secondary Hypothesis (Case E)** | Low / Conditional | Case E (ABL-03) |
| **CMP-04** | Original YOLO11s Decoupled Head | **Adopted Unchanged (v1 Scope Freeze)** | Baseline | Case A |
| **CMP-05** | Dual-label / NMS-Free Training | **Deferred (Explicitly Out of Scope for v1)** | Deferred | Out of scope |
