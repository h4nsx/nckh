# Training, Export, and Edge Deployment Contract
## YOLO11s + SS2D Warehouse Object Detection Research

**Document ID:** `E_research_design/training_export_edge_deployment_contract.md`  
**Purpose:** This document defines the intended training and deployment workflow for this research project. Any coding agent (including Antigravity) must follow it when planning tasks, modifying code, running experiments, or documenting results.

---

## 1. Core Rule

**Train on a suitable GPU machine; deploy the selected trained model to an edge device for inference and benchmarking.**

The edge device is primarily an inference target. Do not assume that full training must run on the Jetson or Raspberry Pi. Do not retrain on the edge device unless a separate experiment explicitly requires it.

### Standard Lifecycle Workflow
1. **Develop and debug code on the developer laptop** (unit tests, shape verifications, small smoke tests).
2. **Train the original YOLO11s baseline on a suitable GPU machine** (cloud GPU or dedicated workstation).
3. **Train the YOLO11s–SS2D (Case B) model under an identical, matched experimental protocol**.
4. **Evaluate both models and select checkpoints using validation results**.
5. **Export each selected checkpoint to a runtime-compatible format** (e.g. TensorRT engine / ONNX), where supported.
6. **Deploy both models to the confirmed edge device**.
7. **Measure edge inference performance and document compatibility limitations honestly**.
8. **Use the held-out test set for final accuracy reporting**, following the research protocol.

---

## 2. Hardware Responsibilities & Boundaries

### 2.1 Developer Laptop
- **Hardware Profile:** Intel Core i7-7600U (2 cores / 4 threads @ 2.80GHz), 8 GB RAM, Intel HD Graphics 620, CPU-only.
- **Process Memory Bound:** Peak RSS strictly constrained below **3.0 GB** for host stability.
- **Allowed Tasks:** Repository inspection, code editing, dataset metadata inspection, small-batch annotation conversion, unit tests, tensor-shape checks, and small CPU smoke tests (1–2 epochs on small subsets).
- **Prohibited Tasks:** **Do not run full convergence training (e.g., 50–100 epochs) on this machine.** The compute and memory limits make full YOLO11s/SS2D training impractical and risk freezing the host.

### 2.2 GPU Training Machine (Remote / Cloud)
- **Role:** Dedicated execution of multi-epoch baseline training, Case B training, and full ablation matrices.
- **Prerequisites to verify prior to training:**
  - GPU model and VRAM (e.g. T4 16GB, A100, or RTX series).
  - OS, Python, PyTorch, CUDA, Ultralytics, and SS2D dependency versions.
  - CUDA availability in PyTorch (`torch.cuda.is_available() == True`).
  - Compatibility of the SS2D scan implementation (pure-PyTorch vs. CUDA parallel scan).
  - Storage space for checkpoints and full dataset partitions.

### 2.3 Edge Device
- **Device Status:** **To be confirmed by developer.** (Do not assume Jetson Nano, Orin Nano, or Raspberry Pi 4/5 until confirmed).
- **Role:** Inference runtime only. Benchmarking latency, throughput (FPS), peak memory (RSS), and export compatibility under batch size 1.
- **Rule:** Never describe edge performance as measured until a physical benchmark has completed on the confirmed device.

---

## 3. Mandatory Model Comparisons

At minimum, preserve and compare:
- **Model A (Control Baseline):** Original pretrained YOLO11s, fine-tuned on the selected warehouse dataset.
- **Model B (Proposed Variant):** YOLO11s with the specified deep C2PSA-to-genuine-SS2D modification at Layer 10 (P5), fine-tuned under matched conditions.

### Strict Control Variables:
- Dataset splits, class mapping, input resolution ($640 \times 640$), augmentation policy, optimizer (AdamW), schedule, checkpoint-selection rule, and evaluation metrics must remain **strictly identical**.
- SS2D must be a genuine selective-scan/state-space implementation. Do not substitute ordinary convolution or attention while calling it SS2D.
- Preserve the original neck and Detect head for the initial Case B experiment.

---

## 4. Dataset and Training Requirements

- Confirm dataset version, license (CC-BY-4.0), annotation format, and class names before training.
- Use only verified classes supported by annotations (`worker`, `forklift`, `pallet_truck`, `amr`). Never invent class IDs.
- **Zero-Leakage Splitting:** Partition by facility scene and disjoint camera streams (`Warehouse_000` train, `Warehouse_015` val).
- Run small smoke tests before full training runs.
- If GPU memory is insufficient, reduce batch size. **Do not silently switch to CPU for a long convergence run.**

---

## 5. Export and Deployment Procedure

1. Preserve the original training checkpoint (`best.pt`) and logs. Never overwrite training weights.
2. Export a separate deployment artifact (e.g., `model.onnx`, `model.engine`).
3. Verify that the deployment runtime supports every required operator, especially custom SS2D selective-scan operations.
4. If export or inference fails, record the exact unsupported operator/error and investigate compatible alternatives. **Do not silently replace SS2D with convolution or attention.**
5. If the proposed model cannot run on the target edge device, report that limitation honestly and benchmark the baseline separately.

---

## 6. Edge Benchmark Protocol

Benchmark baseline and proposed model on the exact same device under matched conditions:
- Device model, RAM, OS, JetPack version, CUDA, TensorRT / ONNX Runtime versions.
- Model artifact and precision (FP32, FP16, INT8).
- Input resolution ($640 \times 640$) and batch size ($=1$).
- Warm-up iteration count ($\ge 10$) and timed iteration count ($\ge 100$).
- Explicit timing boundaries: Model inference vs. End-to-end pipeline (Image read + Preprocess + Forward + NMS).
- Report median, p90/p95 latency, and FPS.
- Peak physical memory (RSS) and power draw (if hardware sensors exist).

---

## 7. Immediate Operational Directions for Antigravity

Before changing code or launching expensive jobs:
1. **Preserve existing artifacts:** Back up `models/warehouse_ss2d.py` as `models/warehouse_ss2d_v1_backup.py`.
2. **Correct directional parameter alignment in SS2D:** Ensure parameter projections $(\mathbf{\Delta}_k, \mathbf{B}_k, \mathbf{C}_k)$ are spatially aligned with each directional input sequence $x_k$ before scanning.
3. **Add unit tests:** Verify non-square ($3\times 5$), $1\times 1$, shape preservation, zero NaNs, and gradient flow in `tests/test_ss2d_directional_alignment.py`.
4. **Audit and update reports:** Update `Reports/SS2D_IMPLEMENTATION_AUDIT.md` to reflect verified mathematical facts, qualified stability claims, and explicit ZOH derivations.
5. **No full training on laptop:** Local CPU runs are strictly limited to unit tests, syntax/shape checks, and verification gates. Multi-epoch training is exported to a GPU machine.
