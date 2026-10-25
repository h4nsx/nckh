#!/usr/bin/env python3
"""
Pretrained Baseline vs Case B Evaluation Runner
Executes 2 epochs with pretrained weights to produce populated metrics,
non-zero mAP values, real PR curves, and visualized bounding box predictions.
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

def train_pretrained_baseline():
    print("=" * 60)
    print("STEP 1: PRETRAINED BASELINE YOLO11s (2 Epochs)")
    print("=" * 60)
    # yolo11s.pt will fine-tune on smartspaces_micro
    model = YOLO("yolo11s.pt")
    model.train(
        data="data/smartspaces_micro/smartspaces_micro.yaml",
        epochs=2,
        imgsz=640,
        batch=4,
        workers=1,
        device="cpu",
        project="runs/pretrained_runs",
        name="baseline_pretrained",
        exist_ok=True,
        verbose=True
    )

if __name__ == "__main__":
    train_pretrained_baseline()
