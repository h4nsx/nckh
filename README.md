# YOLO11s-SS2D: A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection

[![IEEEtran](https://img.shields.io/badge/Paper-IEEEtran-blue.svg)](paper/main.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-orange.svg)](LICENSE)

Official repository for the paper:  
**"A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices"**

---

## 1. Repository Structure

The codebase is organized into modular directories for full reproducibility:

```
├── models/         # Neural network definitions & YAML architectures
│   ├── warehouse_ss2d.py           # Core WarehouseSS2DBlock & PurePyTorch2DSelectiveScan
│   ├── yolo11s_ss2d_p5.yaml        # Case B: Single-Scale P5 Architecture
│   └── yolo11s_ss2d_p4_p5.yaml     # Case C: Dual-Scale P4+P5 Architecture
├── configs/        # Experiment configurations & dataset split specs
│   ├── experiment_settings.yaml    # Hyperparameters, arms, and training settings
│   └── smartspaces_micro.yaml      # NVIDIA PhysicalAI-SmartSpaces dataset definition
├── scripts/        # Training, surgery, validation & benchmarking scripts
│   ├── build_case_b_full_warmstart.py  # Warmstart weight surgery (cv2, dfl transfer)
│   ├── run_two_stage_experiment.py     # Staged warmup training execution
│   ├── benchmark_edge_device.py        # ONNX export & CPU/Jetson latency profiling
│   └── generate_paper_charts.py        # Publication vector charts generator
├── tests/          # Architecture, alignment & gradient test suites
│   ├── test_architecture_gate.py       # Gate checks (constructor, gradients, identity)
│   └── test_ss2d_directional_alignment.py # 4-direction coordinate alignment & restoration
├── docs/           # Implementation notes, protocols & audit gate reports
│   ├── ARCHITECTURE_VERIFICATION_GATE.md # Step-by-step architectural verification
│   ├── BASELINE_VALIDATION_GATE.md       # Baseline YOLO11s reproduction audit
│   ├── WARMSTART_VALIDATION_GATE.md      # Tensor transfer & head surgery manifest
│   ├── CONTROLLED_EXPERIMENTS_GATE.md    # Multi-arm comparison protocol
│   └── RESULTS_INTEGRITY_GATE.md         # Single source-of-truth results table
├── results/        # Empirical results, metrics & figures (lightweight)
│   ├── tables/                           # CSV results per experimental arm
│   └── plots/                            # High-res publication PDF/PNG figures
├── paper/          # LaTeX paper source (IEEE Transactions template)
└── README.md       # Setup, reproduction commands & verified benchmarks
```

---

## 2. Environment Setup

### Prerequisites
- Linux OS (Ubuntu 20.04/22.04 or Arch Linux)
- Python 3.10+ (tested with Python 3.14 venv)
- PyTorch >= 2.0 (CPU or CUDA)
- Ultralytics >= 8.3

### Installation
```bash
# Clone the repository
git clone https://github.com/h4nsx/nckh.git
cd nckh

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or CUDA index
pip install ultralytics onnx onnxruntime matplotlib pandas thop
```

---

## 3. Verified Benchmark Results

Evaluated on the independent, zero-leakage cross-facility validation set ($100$ frames, $3{,}696$ instances):

| Architecture / Model | Attention Type | Total Params | Layer 10 Params | FLOPs | Weight Size | $\text{mAP}_{50}$ (Peak) | Forklift $\text{mAP}_{50}$ | Safety Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline YOLO11s** | `C2PSA` | $9.429\text{ M}$ | $990{,}976$ | $21.7\text{ G}$ | $19.18\text{ MB}$ | $70.83\%$ | $70.13\%$ | $57.21\%$ |
| **YOLO11s-SS2D (Case B, Single-Stage)**| `WarehouseSS2D`| **$8.796\text{ M}$** | **$357{,}632$** | **$21.2\text{ G}$** | **$17.89\text{ MB}$** | **$71.95\%$** | **$79.71\%$** | $57.61\%$ |
| **YOLO11s-SS2D (Case B, Two-Stage)** | `WarehouseSS2D`| **$8.796\text{ M}$** | **$357{,}632$** | **$21.2\text{ G}$** | **$17.89\text{ MB}$** | $68.42\%$ | **$76.26\%$** | **$65.06\%$** |
| **YOLO11s-SS2D (Case C, Dual-Scale)**| `SS2D` $\times 2$| $8.892\text{ M}$ | $357\text{K} + 96\text{K}$ | $21.8\text{ G}$ | $18.10\text{ MB}$ | $70.18\%$ | **$75.25\%$** | $51.54\%$ |

### Key Architectural Advantages
1. **Parameter Efficiency:** Layer 10 drops from $990{,}976 \to 357{,}632$ parameters (**$-63.91\%$ reduction** in attention block), saving $633{,}344$ parameters globally (**$-6.72\%$ reduction**).
2. **Safety-Critical Heavy Machinery:** Forklift $\text{mAP}_{50}$ improves by **$+6.1\%$ to $+9.6\%$** due to continuous 4-direction aisle scanning resolving occluded masts through warehouse racks.
3. **Personnel Safety:** Worker precision reaches **$96.6\%$**, suppressing disruptive false alarms in high-traffic aisles.
4. **Edge Headroom:** Operates at **$395.7\text{ MB}$ peak RSS**, running **$60.4\%$ beneath the 1.0 GB physical memory ceiling** of edge gateways.

---

## 4. Step-by-Step Reproduction Guide

### Step 1: Run Architecture Verification Tests
Verify constructor mappings, residual identity initialization, and gradient backpropagation:
```bash
python tests/test_architecture_gate.py
python tests/test_ss2d_directional_alignment.py
```
*Expected: All gate checks pass with `max abs diff = 0.00e+00`.*

### Step 2: Generate Warmstart Checkpoint
Perform shape-aware weight surgery transferring pretrained COCO backbone, neck, box regression (`cv2`), and DFL parameters into Case B:
```bash
python scripts/build_case_b_full_warmstart.py
```
*Output: `models/yolo11s_ss2d_full_warmstart.pt` (SHA-256: `1e92c10...`).*

### Step 3: Train or Fine-Tune
To run the staged warmup training protocol (Arm 3: 5 epochs backbone-frozen, 15 epochs unfrozen):
```bash
python scripts/run_two_stage_experiment.py
```

### Step 4: Evaluate Checkpoints
Evaluate any saved checkpoint on the benchmark:
```bash
python -c '
from ultralytics import YOLO
from models.warehouse_ss2d import WarehouseSS2DBlock
import ultralytics.nn.tasks as tasks
tasks.WarehouseSS2DBlock = WarehouseSS2DBlock

m = YOLO("runs/detect/runs/two_stage_protocol_study/arm2_case_b_single_stage/weights/best.pt")
metrics = m.val(data="configs/smartspaces_micro.yaml", imgsz=640, batch=4)
print("mAP50:", metrics.box.map50)
'
```

### Step 5: Benchmark ONNX Edge Latency & Physical Memory
Export model and measure steady-state inference process footprint:
```bash
python scripts/benchmark_edge_device.py
```

---

## 5. Paper Compilation

The official manuscript is written in LaTeX using the standard IEEE Transactions template:
```bash
cd paper/
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
*Generated PDF:* [`paper/main.pdf`](paper/main.pdf).

---

## 6. Citation

If you use this architecture, dataset split, or benchmark protocol in your research, please cite:

```bibtex
@article{yolo11s_ss2d_2026,
  title={A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices},
  author={Placeholder Authors},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026}
}
```
