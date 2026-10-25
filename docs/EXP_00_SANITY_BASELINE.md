# Experiment Record: EXP-00-SANITY-BASELINE

## 1. Identity
- **Experiment ID:** `EXP-00-SANITY-BASELINE`
- **Date:** 2026-10-23
- **Research Question:** Baseline pipeline feasibility & latency calibration on CPU edge host
- **Git Commit:** `a02d285` (working branch `base`)
- **Config Path:** `yolo11s.yaml` + `data/smartspaces_micro/smartspaces_micro.yaml`
- **Operator:** Antigravity AI Assistant & Hans

## 2. Hypothesis and Design
- **Hypothesis:** The unmodified YOLO11s baseline can complete a full training epoch and validation cycle on the extracted SmartSpaces micro-benchmark within local laptop CPU memory (< 3.0 GB RSS) without out-of-memory errors or dataset corruption.
- **Independent Variable:** Unmodified standard YOLO11s (control group).
- **Controlled Variables:** Input resolution (640x640), batch size (4), optimizer (AdamW), dataset split (Warehouse_000 train, Warehouse_015 val).
- **Outcome:** **VERIFIED**. Pipeline executed without errors.

## 3. Data & Splits
- **Dataset:** NVIDIA PhysicalAI-SmartSpaces (Targeted Micro-Subset v1.0)
- **Train Set:** `Warehouse_000` (`Camera_0000`, `Camera_0001`), 300 frames, 12,959 instances.
- **Validation Set:** `Warehouse_015` (`Camera`, `Camera_01`), 100 frames, 3,696 instances.
- **Leakage Safeguard:** Strictly independent facilities and disjoint camera streams between train and validation.
- **Classes:** `0: worker`, `1: forklift`, `2: pallet_truck`, `3: amr`.

## 4. Model and Training Hyperparameters
- **Model Architecture:** Official unmodified YOLO11s (`yolo11s.yaml`)
- **Total Layers (Fused):** 100 layers
- **Parameter Count:** 9,414,348 parameters
- **Computational Complexity:** 21.4 GFLOPs (at 640x640 input)
- **Weights File:** Random initialization (`weights/best.pt`, `weights/last.pt`: 19.2 MB)
- **Epochs:** 1 (Sanity pass)
- **Batch Size:** 4
- **Optimizer:** AdamW (`lr=0.00125`, `momentum=0.9`, weight decay=0.0005)
- **Image Size:** 640 x 640

## 5. Hardware and Measured Runtime
- **Host Device:** Laptop
- **CPU:** Intel Core i7-7600U @ 2.80GHz (2 cores, 4 threads)
- **GPU:** None (CPU execution)
- **RAM Utilized (Peak RSS):** 2,601.25 MB (~2.60 GB)
- **Total Epoch Duration:** 436.45 seconds (~7.27 minutes)
- **Measured CPU Inference Latency:**
  - **Preprocess:** 2.8 ms / image
  - **Inference (CPU):** 275.9 ms / image (~3.6 FPS on dual-core CPU)
  - **Postprocess:** 2.0 ms / image
  - **Total Latency:** ~280.7 ms / image

## 6. Scientific Observations & Implications
1. **Memory Feasibility on Laptop:** Peak memory RSS reached 2.60 GB, staying safely within the 7.6 GB RAM ceiling without freezing the system or thrashing swap.
2. **Compute Bottleneck for Multi-Epoch Training:** At ~7.27 minutes per epoch for 300 images on CPU, a 50-epoch training run locally would take ~6.0 hours. This confirms our design decision:
   - Use the laptop for code development, architecture unit testing, and sanity verification (1–2 epochs).
   - Offload full 50–100 epoch convergence training to a GPU instance (e.g. Colab / Cloud GPU).
3. **Control Benchmark Anchor:** We now have the exact reference parameter count (9.41M), GFLOPs (21.4), and CPU latency baseline to compare against our upcoming **Case B (YOLO11s-SS2D)** prototype.
