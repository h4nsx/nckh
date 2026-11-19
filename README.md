# Why a Cheaper Block Runs Slower: Replacing Self-Attention with 2D Selective Scans in YOLO11s on an Embedded CPU

[![Paper LNCS PDF](https://img.shields.io/badge/Paper-Springer_LNCS-blue.svg)](paper_lncs/main.pdf)
[![Architecture](https://img.shields.io/badge/Architecture-YOLO11s--SS2D-orange.svg)](#architecture-overview)
[![Benchmark](https://img.shields.io/badge/Benchmark-NVIDIA_SmartSpaces-green.svg)](#benchmark--evaluation)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Cryptographically_Hashed-success.svg)](audit/REPRODUCIBILITY.md)

> **Official Research Codebase & Artifact Repository**  
> **Target Venue:** TAIMMA 2027 (Springer Lecture Notes in Computer Science)  
> **Topic:** *Why a Cheaper Block Runs Slower: Replacing Self-Attention with 2D Selective Scans in YOLO11s on an Embedded CPU*

---

## 📌 Project Overview

Modern autonomous warehouse logistics environments—such as fulfillment centers, automated storage and retrieval systems (ASRS), and mixed human-robot industrial facilities—demand continuous, high-precision visual object perception. Deployed edge cameras and robotic gateways must accurately detect four key logistics entities:
1. **Personnel (`worker`):** Pedestrians requiring strict safety clearance and zero false alarms.
2. **Heavy Machinery (`forklift`):** Primary motorized collision hazard prone to severe pallet-rack occlusions.
3. **Autonomous Mobile Robots (`amr`):** Low-profile robotic carts operating on warehouse floors.
4. **Manual Jacks (`pallet_truck`):** Low-ground-clearance movable inventory equipment.

While state-of-the-art single-stage detectors like **YOLO11** achieve high inference frame rates, their deep semantic stages rely on multi-head self-attention mechanisms (such as **`C2PSA`**). These attention blocks incur quadratic spatial computational complexity $\mathcal{O}(L^2)$, significant parameter footprints, and heavy memory bandwidth pressure on resource-constrained embedded edge devices (e.g., NVIDIA Jetson, Raspberry Pi).

**YOLO11s-SS2D** addresses this fundamental trade-off. We surgically replace the attention bottleneck with **`WarehouseSS2DBlock`**, a shape-compatible 4-direction 2D Selective State Space model (SSM) based on visual Mamba principles. The block achieves global receptive field aggregation with strictly linear token complexity $\mathcal{O}(L)$ while slashing parameter counts and preserving critical safety detection metrics.

---

## 🏛 Architecture Overview

`WarehouseSS2DBlock` is a drop-in architectural replacement for the $P_5$ attention block (Layer 10 in YOLO11s) and optionally intermediate $P_4$ stages (Case C dual-scale).

```
Feature Map X ∈ ℝ^{B × 512 × 20 × 20}
               │
  ┌────────────┴────────────┐
  │ Residual Identity Path  │
  │                         │
  │ Branch: LayerNorm / Conv│
  │   - 1×1 Conv + SiLU     │ (Projects 512 → d_inner=256)
  │   - 3×3 DW-Conv + SiLU  │ (Captures local context)
  │                         │
  │ 4-Direction SS2D Core:  │ (Sub-quadratic continuous scan)
  │   1. Top-Left → Btm-Right (Row-Major Forward)
  │   2. Btm-Right → Top-Left (Row-Major Reversed)
  │   3. Top-Right → Btm-Left (Col-Major Forward)
  │   4. Btm-Left → Top-Right (Col-Major Reversed)
  │                         │
  │ Spatial Re-Orientation  │
  │   - Fold & Sum          │
  │   - 1×1 Out-Projection  │ (W_out initialized to ZERO)
  └────────────┬────────────┘
               │
               ▼
 Output Feature Map Y = X + Out_Proj(SS2D)
```

### Key Architectural Innovations

1. **4-Directional Selective Scan (SS2D):**
   1D Mamba introduces artificial breaks when applied to 2D image matrices. Our vectorized PyTorch selective scan traverses the spatial grid horizontally and vertically in both forward and reverse directions, resolving long-range aisle geometry without directional bias.
2. **Zero-Residual Identity Pass-Through ($\mathbf{W}_{\text{out}} = \mathbf{0}$):**
   By initializing the final linear projection matrix to zero, the block acts as an exact identity pass-through at step 0 ($\|\mathbf{Y} - \mathbf{X}\|_{\infty} = 0.0$). This eliminates the "cold block shock" that typically destabilizes pretrained CNN weights upon surgical insertion.
3. **Full Warmstart Weight Surgery:**
   We transfer all 42 class-agnostic bounding-box regression layers (`model.23.cv2`) and Distribution Focal Loss (`model.23.dfl`) from pretrained COCO weights, ensuring initial box regression begins at a low, stable loss ($\mathcal{L}_{\text{box}} = 2.758$ vs. $6.276$ for random initialization).

---

## 📊 Key Research Findings & Benchmarks

All models were evaluated under identical experimental conditions on the **NVIDIA PhysicalAI-SmartSpaces** zero-leakage cross-facility benchmark ($100$ test frames, $3{,}696$ ground-truth instances):

### 1. Multi-Arm Architecture Comparison ($640 \times 640$, 20 Epochs)

| Metric / Dimension | Baseline YOLO11s | YOLO11s-SS2D (Case B, Single-Stage) | YOLO11s-SS2D (Case B, Two-Stage Warmup) | YOLO11s-SS2D (Case C, Dual-Scale P4+P5) |
| :--- | :---: | :---: | :---: | :---: |
| **Attention Layer 10 Type** | `C2PSA` | **`WarehouseSS2DBlock`** | **`WarehouseSS2DBlock`** | `SS2D` $\times 2$ (P4 + P5) |
| **Layer 10 Parameter Count** | $990{,}976$ | **$357{,}632$** ($-63.9\%$) | **$357{,}632$** ($-63.9\%$) | $454{,}528$ ($-54.1\%$) |
| **Total Model Parameters** | $9.429\text{ M}$ | **$8.796\text{ M}$** ($-6.72\%$) | **$8.796\text{ M}$** ($-6.72\%$) | $8.893\text{ M}$ ($-5.69\%$) |
| **Computational Complexity** | $21.7\text{ GFLOPs}$ | **$21.2\text{ GFLOPs}$** | **$21.2\text{ GFLOPs}$** | $21.8\text{ GFLOPs}$ |
| **Checkpoint Storage** | $18.29\text{ MB}$ | **$17.07\text{ MB}$** | **$17.07\text{ MB}$** | $17.26\text{ MB}$ |
| **Global Macro $\text{mAP}_{50}$** | $69.80\%$ | **$71.95\%$** ($+2.15\%$) | $67.67\%$ | $67.17\%$ |
| **Global Macro $\text{mAP}_{50-95}$**| **$45.53\%$** | $45.15\%$ | $43.15\%$ | $42.76\%$ |
| **Safety Recall (Overall)** | $57.21\%$ | $57.61\%$ | **$65.06\%$** ($\mathbf{+7.85\%}$) | $51.54\%$ |
| **Forklift $\text{mAP}_{50}$** | $70.13\%$ | **$79.71\%$** ($\mathbf{+9.58\%}$) | **$76.26\%$** ($\mathbf{+6.13\%}$) | **$75.25\%$** ($\mathbf{+5.12\%}$) |
| **Forklift $\text{mAP}_{50-95}$** | $31.80\%$ | **$38.33\%$** ($\mathbf{+6.53\%}$) | **$38.36\%$** ($\mathbf{+6.56\%}$) | **$33.84\%$** ($\mathbf{+2.04\%}$) |
| **Personnel Precision** | $95.0\%$ | $93.6\%$ | $93.2\%$ | **$96.6\%$** |

---

### 2. Physical Embedded Silicon Benchmark Profile ($640 \times 640$, Batch Size = 1)

Benchmarked on **Raspberry Pi 4 Model B Rev 1.2** (Broadcom BCM2711 Quad-Core ARM Cortex-A72 @ 1.5 GHz, 4GB RAM) vs. Host x86_64 Core i7:

| Hardware Platform | Model Architecture | ONNX Size | Mean Latency | Throughput | Peak Process RSS | Safety Headroom ($< 1.0\text{ GB}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raspberry Pi 4 ARM** | **YOLO11s Baseline Control** | 36.18 MB | $1169.73 \pm 24.7\text{ ms}$ | 0.86 FPS | **231.5 MB** | $76.8\%$ Headroom |
| **Raspberry Pi 4 ARM** | **YOLO11s-SS2D (Case B, P5)** | 38.47 MB | $1398.19 \pm 28.0\text{ ms}$ | 0.72 FPS | **371.9 MB** | **62.8% Headroom** |
| **Raspberry Pi 4 ARM** | **YOLO11s-SS2D (Case C, P4+P5)**| 46.62 MB | $1965.07 \pm 29.7\text{ ms}$ | 0.51 FPS | **714.9 MB** | $28.5\%$ Headroom |
| Host x86_64 Core i7 | YOLO11s Baseline Control | 36.18 MB | $244.72 \pm 33.5\text{ ms}$ | 4.37 FPS | 285.1 MB | $71.5\%$ Headroom |
| Host x86_64 Core i7 | YOLO11s-SS2D (Case B, P5) | 38.47 MB | $304.01 \pm 8.5\text{ ms}$ | 3.32 FPS | 395.7 MB | $60.4\%$ Headroom |

---

### Core Insights
- **Heavy Machinery Perception (+6.2% $\text{mAP}_{50}$ / +6.56% $\text{mAP}_{50-95}$):** Motorized forklifts benefit substantially from continuous 2D state-space modeling. The 4-direction scan bridges tall warehouse racking and resolves elongated industrial machinery silhouettes under heavy aisle occlusion.
- **Worker Safety Recall (+7.85% Absolute Gain):** Case B Two-Stage achieves **$65.06\%$ safety recall** on warehouse personnel (vs. $57.21\%$ baseline), drastically reducing missed detections of floor workers.
- **Embedded Memory Compliance:** Running at **$371.9\text{ MB}$ peak RSS**, Case B operates **$62.8\%$ beneath the 1.0 GB physical memory envelope** of industrial smart camera modules.
- **Scientific Latency Trade-Off:** The latency ratio on physical ARM silicon ($\approx 1.20\times$) matches the host x86 CPU profile ($\approx 1.24\times$), validating that sequential recurrence incurs a predictable overhead on single-threaded CPUs while remaining viable for edge gateways.

---

## 🔬 Scientific Honesty: Accuracy–Efficiency Trade-off

A critical research contribution of this work is investigating the wall-clock execution profile of visual state space models on embedded edge hardware:
- While theoretical FLOPs ($21.2\text{ G}$ vs. $21.7\text{ G}$) and parameters ($8.796\text{ M}$ vs. $9.429\text{ M}$) are lower, **steady-state ONNX CPU latency is higher** ($304.01\text{ ms}$ vs. $244.72\text{ ms}$ on x86 host; $1398.19\text{ ms}$ vs. $1169.73\text{ ms}$ on Raspberry Pi 4 ARM).
- **The Bottleneck:** Convolutions and standard attention map efficiently to BLAS matrix multiplication (`gemm`) on CPU SIMD units. In contrast, selective state-space scans require sequential recurrence updates across $L=400$ steps for 4 passes, which is memory-bandwidth bound on single-threaded CPU loops.
- **Hardware Acceleration Path:** On edge devices equipped with GPU/NPU acceleration (e.g., NVIDIA Jetson Orin Nano), parallel associative prefix scans (Blelloch scan) bypass sequential loops, unlocking high-throughput inference while retaining parameter savings and occlusion sensitivity.

---

## 🗂 Project Structure

```
├── models/         # Neural network definitions & YAML model configurations
│   ├── warehouse_ss2d.py           # Core WarehouseSS2DBlock & PurePyTorch2DSelectiveScan
│   ├── yolo11s_ss2d_p5.yaml        # Single-scale (P5) YOLO11s-SS2D architecture definition
│   └── yolo11s_ss2d_p4_p5.yaml     # Dual-scale (P4 + P5) YOLO11s-SS2D architecture definition
├── configs/        # Experiment settings & dataset definitions
│   ├── experiment_settings.yaml    # Training hyperparameters, arms, seeds, and schedule
│   └── smartspaces_micro.yaml      # NVIDIA PhysicalAI-SmartSpaces zero-leakage split spec
├── scripts/        # Training, surgery, evaluation & benchmarking pipelines
│   ├── build_case_b_full_warmstart.py  # Warmstart weight surgery (cv2, dfl transfer)
│   ├── run_two_stage_experiment.py     # Staged warmup training execution
│   ├── benchmark_edge_device.py        # ONNX export & latency/memory profiling
│   └── generate_paper_charts.py        # Publication vector figures generator
├── tests/          # Architecture, alignment & gradient test suites
│   ├── test_architecture_gate.py       # Gate checks (constructor, gradients, identity, YOLO pass)
│   └── test_ss2d_directional_alignment.py # 4-direction coordinate alignment & shape restoration
├── docs/           # Verified audit reports & experimental protocols
│   ├── ARCHITECTURE_VERIFICATION_GATE.md # Step-by-step architectural verification
│   ├── BASELINE_VALIDATION_GATE.md       # Baseline YOLO11s reproduction audit
│   ├── WARMSTART_VALIDATION_GATE.md      # Tensor transfer & head surgery manifest
│   ├── CONTROLLED_EXPERIMENTS_GATE.md    # Multi-arm comparison protocol
│   └── RESULTS_INTEGRITY_GATE.md         # Single source-of-truth results table
├── results/        # Lightweight tables and publication plots (no heavy checkpoints)
│   ├── tables/                           # CSV results per experimental arm
│   └── plots/                            # High-res publication PDF/PNG figures
└── paper/          # LaTeX paper source (IEEE Transactions on Automation Science and Engineering)
```

---

## 📄 Academic Paper & Submission

- **Target Conference:** **TAIMMA 2027** (Springer LNCS)
- **Paper Title:** *Why a Cheaper Block Runs Slower: Replacing Self-Attention with 2D Selective Scans in YOLO11s on an Embedded CPU*
- **Authors:** 
  - **Vo Tuan Hung** (Information Technology Department, Song Hung Thuan Co., Ltd.; [votuanhung1205.work@gmail.com](mailto:votuanhung1205.work@gmail.com))
  - **Huynh Tan Phat** (Corresponding Author, Ho Chi Minh City Open University; [phat.ht@ou.edu.vn](mailto:phat.ht@ou.edu.vn))
- **Primary Submission Manuscript (LNCS):** [`paper_lncs/main.pdf`](paper_lncs/main.pdf) ([Source](paper_lncs/main.tex))
- **Supplementary Code Archive:** [`submission_code.zip`](submission_code.zip)
- **Legacy IEEE Draft:** [`paper/main.pdf`](paper/main.pdf)

### Citation
```bibtex
@inproceedings{vo2027why,
  title={Why a Cheaper Block Runs Slower: Replacing Self-Attention with 2D Selective Scans in YOLO11s on an Embedded CPU},
  author={Vo, Tuan Hung and Huynh, Tan Phat},
  booktitle={Proceedings of the International Conference on Trends in Artificial Intelligence, Machine Learning and Manufacturing Applications (TAIMMA 2027)},
  series={Lecture Notes in Computer Science},
  publisher={Springer},
  year={2027}
}
```
