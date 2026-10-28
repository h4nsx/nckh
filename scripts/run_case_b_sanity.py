#!/usr/bin/env python3
"""
Experiment Runner: Case B (YOLO11s-SS2D at P5) Sanity Run
Executes training and validation on data/smartspaces_micro/smartspaces_micro.yaml
Records parameter counts, GFLOPs, runtime, memory, and output artifacts.
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

def run_case_b_sanity():
    print("=" * 60)
    print("STARTING EXPERIMENT: EXP-01-SANITY-CASE-B (YOLO11s-SS2D)")
    print("=" * 60)

    start_time = time.time()
    process = psutil.Process(os.getpid())

    model_cfg = "models/yolo11s_ss2d_p5.yaml"
    model = YOLO(model_cfg)
    
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Loaded YOLO11s-SS2D (Case B) successfully. Total parameters: {total_params:,}")

    results = model.train(
        data="data/smartspaces_micro/smartspaces_micro.yaml",
        epochs=1,
        imgsz=640,
        batch=4,
        workers=1,
        device="cpu",
        project="runs/case_b_prototype",
        name="exp_sanity",
        exist_ok=True,
        verbose=True
    )

    elapsed = time.time() - start_time
    rss_mb = process.memory_info().rss / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETED in {elapsed:.2f}s (~{elapsed/60:.2f} min)")
    print(f"Peak Process Memory RSS: {rss_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    run_case_b_sanity()
