# Remote GPU Execution Guide: 3-Arm Convergence Study

This guide provides step-by-step instructions for running the complete multi-epoch (20–50 epochs) comparative convergence study on an NVIDIA GPU using **Google Colab** (free T4 GPU) or **Kaggle Notebooks**.

---

## 1. Package Contents

The archive `gpu_training_package/warehouse_ss2d_gpu_bundle.tar.gz` (67 MB) contains everything needed:
* **Micro-dataset:** `data/smartspaces_micro/` (300 train images, 100 val images, zero-leakage split).
* **Model Configurations:** `models/yolo11s_ss2d_p5.yaml` and `models/warehouse_ss2d.py`.
* **Verified Warmstart Weights:** `models/yolo11s_ss2d_full_warmstart.pt` (all 42 box regression layers intact).
* **3-Arm Runner:** `scripts/run_two_stage_experiment.py` (with CUDA `--device 0` support).
* **Notebook:** `gpu_training_package/run_convergence_study.ipynb`.

---

## 2. Running on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** > **Change runtime type** > Select **T4 GPU** (or A100 if Colab Pro).
3. Upload `warehouse_ss2d_gpu_bundle.tar.gz` directly into the Colab file explorer (or Google Drive).
4. In a new cell, unpack the bundle:
   ```bash
   !tar -xzvf warehouse_ss2d_gpu_bundle.tar.gz
   ```
5. Install dependencies:
   ```bash
   !pip install -q ultralytics torchvision opencv-python
   ```
6. Open and run [`run_convergence_study.ipynb`](file:///home/hans/Documents/nckh/gpu_training_package/run_convergence_study.ipynb) or execute the 3 arms via terminal:
   ```bash
   # Arm 1: Control Baseline YOLO11s (20 epochs)
   !python scripts/run_two_stage_experiment.py --arm 1 --epochs 20 --device 0 --batch 8

   # Arm 2: Case B Single-Stage Unfrozen (20 epochs)
   !python scripts/run_two_stage_experiment.py --arm 2 --epochs 20 --device 0 --batch 8

   # Arm 3: Case B Two-Stage Protocol (5 warmup + 15 fine-tune)
   !python scripts/run_two_stage_experiment.py --arm 3 --epochs 20 --device 0 --batch 8
   ```
7. Expected GPU Execution Time: ~15 to 20 minutes total for all 3 arms on an NVIDIA T4.

---

## 3. Retrieving and Analyzing the Results

After training completes, download the run artifacts:
* `runs/detect/runs/two_stage_protocol_study/arm1_baseline_control/results.csv`
* `runs/detect/runs/two_stage_protocol_study/arm2_case_b_single_stage/results.csv`
* `runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/results.csv`

Place the CSVs back into the repository under `runs/` or load them into the final notebook cell to generate `gpu_convergence_summary.csv` for LaTeX paper inclusion.
