# Experiment Record: EXP-01-SANITY-CASE-B

## 1. Identity
- **Experiment ID:** `EXP-01-SANITY-CASE-B`
- **Date:** 2026-10-23
- **Research Question:** RQ1 & RQ2: Feasibility, parameter efficiency, and FLOPs of replacing Layer 10 (C2PSA) with WarehouseSS2DBlock at P5
- **Git Commit:** Working tree on branch `base`
- **Config Path:** `models/yolo11s_ss2d_p5.yaml` + `data/smartspaces_micro/smartspaces_micro.yaml`
- **Operator:** Antigravity AI Assistant & Hans

## 2. Hypothesis and Design
- **Hypothesis:** Replacing the deepest C2PSA self-attention block (Layer 10) with an SS2D (2D Selective State Space) block will reduce parameter count and computational FLOPs while preserving the full multiscale detection interface and achieving stable training convergence.
- **Independent Variable:** Architectural block at Layer 10 (`WarehouseSS2DBlock` vs. baseline `C2PSA`).
- **Controlled Variables:** Input resolution (640x640), batch size (4), optimizer (AdamW, lr=0.00125), data split (`Warehouse_000` train, `Warehouse_015` val).
- **Outcome:** **VERIFIED**. Full training, backward pass, and validation cycle completed with zero NaNs/Infs and lower parameters/FLOPs.

## 3. Data & Splits
- **Dataset:** NVIDIA PhysicalAI-SmartSpaces (Targeted Micro-Subset v1.0)
- **Train Set:** `Warehouse_000` (`Camera_0000`, `Camera_0001`), 300 frames, 12,959 instances.
- **Validation Set:** `Warehouse_015` (`Camera`, `Camera_01`), 100 frames, 3,696 instances.
- **Leakage Safeguard:** Strictly disjoint scenes and camera streams between train and validation.
- **Classes:** `0: worker`, `1: forklift`, `2: pallet_truck`, `3: amr`.

## 4. Comparative Architectural Metrics

| Metric | Baseline YOLO11s (EXP-00) | YOLO11s-SS2D Case B (EXP-01) | Delta / Change |
|:---|:---:|:---:|:---:|
| **Layer 10 Type** | `C2PSA` (Self-Attention) | `WarehouseSS2DBlock` (SSM) | Replacement at P5 |
| **Total Layers (Fused)** | 100 | **98** | -2 layers |
| **Total Parameters (Fused)**| 9,414,348 (~9.41 M) | **9,005,004 (~9.01 M)** | **-409,344 params (-4.35%)** |
| **Model Size (.pt)** | 19.2 MB | **18.3 MB** | **-0.9 MB (-4.69%)** |
| **Computational GFLOPs** | 21.4 GFLOPs | **21.0 GFLOPs** | **-0.4 GFLOPs (-1.87%)** |
| **Train Box Loss (Epoch 1)**| 5.9889 | **5.7863** | -0.2026 (improved) |
| **Train Cls Loss (Epoch 1)**| 4.6421 | **4.5379** | -0.1042 (improved) |
| **Train DFL Loss (Epoch 1)**| 3.9852 | **3.8959** | -0.0893 (improved) |

## 5. Hardware and Measured Runtime
- **Host Device:** Laptop
- **CPU:** Intel Core i7-7600U @ 2.80GHz (2 cores, 4 threads)
- **GPU:** None (pure CPU execution)
- **RAM Utilized (Peak RSS):** 2,845.44 MB (~2.85 GB) (safely within the 7.6 GB RAM ceiling)
- **Total Duration:** 3,453.11 seconds (~57.55 minutes)
- **Measured CPU Inference Latency (fused model):**
  - **Preprocess:** 3.0 ms / image
  - **Inference (CPU):** 329.5 ms / image (~3.0 FPS on dual-core CPU)
  - **Postprocess:** 2.1 ms / image

## 6. Scientific Observations & Comparative Analysis

1. **Parameter & Computational Efficiency**:
   - The SS2D block at P5 achieves a **4.35% parameter reduction** (9.01 M vs 9.41 M) and lowers computational complexity from **21.4 to 21.0 GFLOPs**.
   - Checkpoint storage footprint is reduced from 19.2 MB to 18.3 MB.
2. **Optimization Dynamics (Epoch 1)**:
   - On the same 300-image training set with identical initial learning rate (`0.00125`), the hybrid YOLO11s-SS2D converged to lower training losses across all three components:
     - `box_loss`: 5.786 vs 5.989
     - `cls_loss`: 4.538 vs 4.642
     - `dfl_loss`: 3.896 vs 3.985
3. **CPU Recurrence vs. GPU Parallelism Insight**:
   - While the pure PyTorch sequential loop allows CPU portability and verified correctness without custom compilation, the sequential iteration across sequence length $L=400$ ($20 \times 20$) in Python adds significant CPU overhead during backward backpropagation (57.5 mins vs 7.3 mins for vectorized convs).
   - **Crucial Architectural Takeaway for the Paper**: On CPU, unrolled Python sequential scans suffer from CPU branch overhead. For edge deployment (e.g. Jetson Orin Nano with CUDA/TensorRT), hardware scan kernels or associative parallel scans will be essential to realize the theoretical $\mathcal{O}(N)$ speed advantages.
4. **Artifacts Stored**:
   - Weights: `runs/detect/runs/case_b_prototype/exp_sanity/weights/best.pt` (18.3 MB)
   - Results & plots: `runs/detect/runs/case_b_prototype/exp_sanity/`
