#!/usr/bin/env python3
"""
Pretrained Case B (YOLO11s-SS2D) Fine-Tuning Runner
Loads models/yolo11s_ss2d_warmstart.pt and fine-tunes for 2 epochs on smartspaces_micro
Produces populated mAP metrics, PR curves, confusion matrices, and detection images.
"""

import os
import sys
import time
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import WarehouseSS2DBlock

tasks.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.WarehouseSS2DBlock = WarehouseSS2DBlock

from ultralytics import YOLO

def train_pretrained_case_b():
    print("=" * 60)
    print("STEP 2: PRETRAINED CASE B (YOLO11s-SS2D) (2 Epochs)")
    print("=" * 60)

    # Use warmstarted Case B model
    model = YOLO("models/yolo11s_ss2d_p5.yaml")
    # Load warmstart weights
    import torch
    ckpt = torch.load("models/yolo11s_ss2d_warmstart.pt", map_location="cpu", weights_only=False)
    model.model.load_state_dict(ckpt["model"].state_dict())
    print("Successfully loaded warmstart weights into Case B model.")

    model.train(
        data="data/smartspaces_micro/smartspaces_micro.yaml",
        epochs=2,
        imgsz=640,
        batch=4,
        workers=1,
        device="cpu",
        project="runs/pretrained_runs",
        name="case_b_pretrained",
        exist_ok=True,
        verbose=True
    )

if __name__ == "__main__":
    train_pretrained_case_b()
