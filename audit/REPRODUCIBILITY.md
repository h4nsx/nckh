# Artifact Availability and Provenance Ledger

This document establishes the artifact availability, cryptographic hashes (SHA-256), and reproduction procedures required by **Phase 7 of the Action Plan**.

## 1. Artifact Availability Policy

- **Lightweight Artifacts (< 50 MB, Non-Binary / PDF):** Directly tracked in Git under `audit/`, `scripts/`, `models/`, and `paper/`.
- **Heavy Weights & Video Datasets (> 50 MB / Runs):** Kept in local repository storage (and gitignored per standard open-source practice) to prevent repository bloat, fully reproducible via documented execution commands below.

## 2. Cryptographic Manifest

| Artifact | Relative Path | Size | SHA-256 Hash | Availability Status |
|---|---|---|---|:---:|
| **Arm 1 Pretrained Baseline** | `yolo11s.pt` | 18.42 MB | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | Local Reproducible |
| **Arm 1 Checkpoint** | `runs/detect/runs/two_stage_protocol_study/arm1_baseline_control/weights/best.pt` | 18.29 MB | `a213d99a40c4f405dd28f8d07670e34d7e4a000fb8fa51f60549b72a63e52870` | Local Reproducible |
| **Arm 2 Checkpoint** | `runs/detect/runs/two_stage_protocol_study/arm2_case_b_single_stage/weights/best.pt` | 17.07 MB | `16a9ff3914c37e248c881256c768afd67748ea076dccf774eec5cd5d1470f606` | Local Reproducible |
| **Arm 3 Warmup Checkpoint** | `runs/detect/runs/two_stage_protocol_study/arm3_stage1_warmup/weights/best.pt` | 17.06 MB | `f9370283f4912adbfef7bf92f06aeb5b19bab6d4db0e5c6aa0e5f79804fd814c` | Local Reproducible |
| **Arm 3 Fine-tune Checkpoint** | `runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.pt` | 17.07 MB | `5b90cc10afc906ec77d5b9257367032ad11172d013f6be172fdd30a70d007f0e` | Local Reproducible |
| **Arm 4 Checkpoint** | `runs/detect/runs/two_stage_protocol_study/arm4_case_c_dual_scale/weights/best.pt` | 17.26 MB | `22267e09e823cfab58e02b9e15ed976114ad72e1420224509ba0d981d33685c5` | Local Reproducible |
| **Arm 1 ONNX Export** | `yolo11s.onnx` | 36.29 MB | `bc6de8534de80e197a75604e2d475eab182a48f80d458f83d77457e6d8fadf49` | Local Reproducible |
| **Arm 3 ONNX Export** | `runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.onnx` | 38.47 MB | `677274c7d89a350f1ad812ea1c2cef16428b5301b2f04597d04c301319051e0a` | Local Reproducible |
| **Arm 1 Evaluation Record** | `audit/eval_records/arm1_baseline_results.json` | 1361 B | `c6780c99a4ed2084850e5c3d5bf9eac68bbce67a3f15addfba88b93947ba771a` | Tracked in Git |
| **Arm 2 Evaluation Record** | `audit/eval_records/arm2_case_b_single_results.json` | 1383 B | `8f903a74e0ef26a36243a7c5edde24328dd178b7fb4444a632b48c41f8c44abf` | Tracked in Git |
| **Arm 3 Evaluation Record** | `audit/eval_records/arm3_case_b_two_stage_results.json` | 1386 B | `d1ad5d51cd01f34f2ca4421515ac2d1266a52180645c4dc6847ce767b3f0107e` | Tracked in Git |
| **Arm 4 Evaluation Record** | `audit/eval_records/arm4_case_c_dual_results.json` | 1378 B | `db10f5b45d81ce8c65968320f96484d48bba77c99965002cdfb2305f20b735db` | Tracked in Git |
| **Arm 3 Multi-Stage Schedule** | `audit/arm3_two_stage_global_schedule.csv` | 2745 B | `1288b18c17313d87718e4f2f9ec367dc1b97fef7a8a0b76b3be3419e331fd80c` | Tracked in Git |
| **Dataset Micro Sample Spec** | `data/smartspaces_micro/smartspaces_micro.yaml` | 199 B | `2d9685cb3157a06fde0dd489cce106e30b32cbb2d57628084884d26da91650e0` | Local Reproducible |
| **Compiled Manuscript PDF** | `paper/main.pdf` | 8.55 MB | `2289c5e7a868dc0a097361fa853833bf4f6b0278928379ab9e3e3783ea81f3db` | Tracked in Git |

## 3. Step-by-Step Reproduction Procedure

### Step 1: Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Architecture & Unit Invariance Tests
```bash
PYTHONPATH=. python3 tests/test_zero_gamma_identity.py
PYTHONPATH=. python3 tests/test_ss2d_directional_alignment.py
PYTHONPATH=. python3 tests/test_architecture_gate.py
```

### Step 3: Run Staged Multi-Arm Training Protocols
```bash
# Arm 1: Baseline Control
python3 scripts/run_two_stage_experiment.py --arm 1 --epochs 20

# Arm 2: Case B Single-Stage
python3 scripts/run_two_stage_experiment.py --arm 2 --epochs 20

# Arm 3: Case B Two-Stage Warmup (5 warmup + 15 fine-tuning)
python3 scripts/run_two_stage_experiment.py --arm 3 --epochs 20

# Arm 4: Case C Dual-Scale P4+P5
python3 scripts/run_two_stage_experiment.py --arm 4 --epochs 20
```

### Step 4: Regenerate Single-Source LaTeX Tables
```bash
python3 scripts/make_tables.py
```

### Step 5: Edge Silicon ONNX Export & Benchmark
```bash
# Export model to standalone zero-plugin ONNX
python3 -c "from ultralytics import YOLO; YOLO('runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.pt').export(format='onnx', opset=17)"

# Physical/Host Edge Benchmark
python3 scripts/benchmark_edge_device.py --model runs/detect/runs/two_stage_protocol_study/arm3_stage2_finetune/weights/best.onnx --device-name host_cpu
```

### Step 6: Compile Paper
```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
