#!/usr/bin/env python3
"""
Diagnostic Runner: EXP-04 Two-Stage Warmup Protocol
Testing whether freezing the backbone and using identity-preserving zero-init
eliminates the initial performance gap between Case B and Baseline.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import WarehouseSS2DBlock

tasks.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.WarehouseSS2DBlock = WarehouseSS2DBlock

from ultralytics import YOLO
import torch

def run_diagnostic():
    print("=" * 65)
    print("EXP-04: CASE B TWO-STAGE WARMUP DIAGNOSTIC RUNNER")
    print("=" * 65)

    model_yaml = "models/yolo11s_ss2d_p5.yaml"
    ckpt_path = "models/yolo11s_ss2d_warmstart_zeroinit.pt"

    print(f"[1/3] Loading architecture: {model_yaml}")
    model = YOLO(model_yaml)

    print(f"[2/3] Loading zero-init warmstart weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model"].state_dict())

    print("[3/3] Freezing Backbone Layers 0..9...")
    model.train(
        data="data/smartspaces_micro/smartspaces_micro.yaml",
        epochs=2,
        freeze=10,           # Freeze layers 0-9!
        imgsz=640,
        batch=4,
        workers=1,
        device="cpu",
        project="runs/diagnostic_runs",
        name="case_b_warmup_freeze10",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("DIAGNOSTIC RUN COMPLETED!")
    print("Check results at: runs/detect/runs/diagnostic_runs/case_b_warmup_freeze10/")
    print("=" * 65)

if __name__ == "__main__":
    run_diagnostic()
