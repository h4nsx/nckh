#!/usr/bin/env python3
"""
Controlled 3-Arm Experiment Runner for Two-Stage Training Protocol:
- Arm 1: Control Baseline YOLO11s (yolo11s.pt, fine-tuning)
- Arm 2: Case B Single-Stage (models/yolo11s_ss2d_full_warmstart.pt, all layers unfrozen from step 0)
- Arm 3: Case B Two-Stage (models/yolo11s_ss2d_full_warmstart.pt, Stage 1: freeze=10, Stage 2: unfrozen)
All 3 arms run under strictly identical hyperparameters on smartspaces_micro.yaml.
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import WarehouseSS2DBlock

tasks.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.WarehouseSS2DBlock = WarehouseSS2DBlock

from ultralytics import YOLO
import torch

DATASET_YAML = "data/smartspaces_micro/smartspaces_micro.yaml"

def run_arm1_baseline(epochs: int = 2, device: str = "cpu", batch: int = 4):
    print("\n" + "=" * 65)
    print(f"RUNNING ARM 1: Baseline YOLO11s Control (Pretrained COCO, {epochs} Epochs, Device: {device})")
    print("=" * 65)
    model = YOLO("yolo11s.pt")
    model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        workers=2 if device != "cpu" else 1,
        device=device,
        project="runs/two_stage_protocol_study",
        name="arm1_baseline_control",
        exist_ok=True,
        verbose=True
    )

def run_arm2_case_b_single_stage(epochs: int = 2, device: str = "cpu", batch: int = 4):
    print("\n" + "=" * 65)
    print(f"RUNNING ARM 2: Case B Single-Stage Direct Fine-Tuning ({epochs} Epochs, Device: {device}, Unfrozen)")
    print("=" * 65)
    model = YOLO("models/yolo11s_ss2d_full_warmstart.pt")

    model.train(
        data=DATASET_YAML,
        epochs=epochs,
        freeze=0, # No freezing
        imgsz=640,
        batch=batch,
        workers=2 if device != "cpu" else 1,
        device=device,
        project="runs/two_stage_protocol_study",
        name="arm2_case_b_single_stage",
        exist_ok=True,
        verbose=True
    )

def run_arm3_case_b_two_stage(warmup_epochs: int = 1, finetune_epochs: int = 1, device: str = "cpu", batch: int = 4):
    print("\n" + "=" * 65)
    print(f"RUNNING ARM 3: Case B Two-Stage Protocol (Device: {device})")
    print(f"  Stage 1 (Warmup):   {warmup_epochs} Epochs (Backbone Frozen, freeze=10)")
    print(f"  Stage 2 (Fine-Tune): {finetune_epochs} Epochs (Full Model Unfrozen)")
    print("=" * 65)

    # --- Stage 1: Warmup ---
    print("\n>>> STAGE 1: Freezing Backbone Layers 0..9 (Warming up SS2D + Head)...")
    model_stage1 = YOLO("models/yolo11s_ss2d_full_warmstart.pt")

    model_stage1.train(
        data=DATASET_YAML,
        epochs=warmup_epochs,
        freeze=10, # Freeze layers 0..9
        imgsz=640,
        batch=batch,
        workers=2 if device != "cpu" else 1,
        device=device,
        project="runs/two_stage_protocol_study",
        name="arm3_stage1_warmup",
        exist_ok=True,
        verbose=True
    )

    stage1_weights = "runs/detect/runs/two_stage_protocol_study/arm3_stage1_warmup/weights/last.pt"
    print(f"\n>>> Stage 1 complete! Checkpoint saved: {stage1_weights}")

    # --- Stage 2: Full Fine-Tuning ---
    print("\n>>> STAGE 2: Unfreezing All Layers for End-to-End Fine-Tuning...")
    model_stage2 = YOLO(stage1_weights)

    model_stage2.train(
        data=DATASET_YAML,
        epochs=finetune_epochs,
        freeze=0, # Completely unfrozen!
        lr0=0.0005, # Refined learning rate for fine-tuning
        imgsz=640,
        batch=batch,
        workers=2 if device != "cpu" else 1,
        device=device,
        project="runs/two_stage_protocol_study",
        name="arm3_stage2_finetune",
        exist_ok=True,
        verbose=True
    )
    print("\n" + "=" * 65)
    print("ARM 3 (TWO-STAGE PROTOCOL) COMPLETED SUCCESSFULLY!")
    print("=" * 65)

def run_arm4_case_c_single_stage(epochs: int = 2, device: str = "cpu", batch: int = 4):
    print("\n" + "=" * 65)
    print(f"RUNNING ARM 4: Case C Dual-Scale SS2D (P4 + P5, {epochs} Epochs, Device: {device})")
    print("=" * 65)
    model = YOLO("models/yolo11s_ss2d_p4_p5_full_warmstart.pt")

    model.train(
        data=DATASET_YAML,
        epochs=epochs,
        freeze=0,
        imgsz=640,
        batch=batch,
        workers=2 if device != "cpu" else 1,
        device=device,
        project="runs/two_stage_protocol_study",
        name="arm4_case_c_dual_scale",
        exist_ok=True,
        verbose=True
    )

def main():
    parser = argparse.ArgumentParser(description="Multi-Arm Training Protocol Comparative Study")
    parser.add_argument("--arm", choices=["1", "2", "3", "4", "all"], default="3",
                        help="Select experimental arm: 1 (Baseline), 2 (Case B Single-Stage), 3 (Case B Two-Stage), 4 (Case C Dual-Scale), or all")
    parser.add_argument("--epochs", type=int, default=2, help="Total epoch budget (default: 2)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on ('cpu', '0', etc.)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4, can use 8 or 16 on GPU)")
    args = parser.parse_args()

    if args.arm == "1" or args.arm == "all":
        run_arm1_baseline(epochs=args.epochs, device=args.device, batch=args.batch)
    if args.arm == "2" or args.arm == "all":
        run_arm2_case_b_single_stage(epochs=args.epochs, device=args.device, batch=args.batch)
    if args.arm == "3" or args.arm == "all":
        # Split epoch budget: 1/4 warmup + 3/4 fine-tune for longer schedules, or 50/50 for small schedules
        w_ep = max(1, args.epochs // 4) if args.epochs >= 4 else 1
        ft_ep = max(1, args.epochs - w_ep)
        run_arm3_case_b_two_stage(warmup_epochs=w_ep, finetune_epochs=ft_ep, device=args.device, batch=args.batch)
    if args.arm == "4" or args.arm == "all":
        run_arm4_case_c_single_stage(epochs=args.epochs, device=args.device, batch=args.batch)

if __name__ == "__main__":
    main()
