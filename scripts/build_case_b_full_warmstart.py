#!/usr/bin/env python3
"""
Full Warmstart Builder for YOLO11s-SS2D (Case B)
-----------------------------------------------
Transfers:
1. All matching backbone CNN layers (0-9)
2. All matching neck layers (11-22)
3. Detect head box-regression branches (model.23.cv2 - 42 tensors)
4. Distribution Focal Loss tensor (model.23.dfl - 1 tensor)
Leaves ONLY:
- Layer 10 (WarehouseSS2DBlock with identity zero-init projection)
- model.23.cv3 (Classification head adapted to 4 warehouse classes)
This ensures a 100% apples-to-apples pretrained foundation with Baseline YOLO11s.
"""

import sys
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

def build_full_warmstart():
    print("=" * 65)
    print("BUILDING FULL WARMSTART CHECKPOINT FOR CASE B (YOLO11s-SS2D)")
    print("=" * 65)

    # 1. Load source COCO pretrained model
    print("[1/4] Loading pretrained YOLO11s weights (yolo11s.pt)...")
    base_model = YOLO("yolo11s.pt")
    base_dict = base_model.model.state_dict()

    # 2. Instantiate Case B architecture
    print("[2/4] Instantiating Case B model (models/yolo11s_ss2d_p5.yaml)...")
    case_b_model = YOLO("models/yolo11s_ss2d_p5.yaml")
    case_b_dict = case_b_model.model.state_dict()

    # 3. Transfer all matching shape tensors
    # This automatically includes cv2 (box regression) and dfl because their shapes
    # depend only on reg_max=16, NOT on the number of classes (nc)!
    matched = []
    skipped = []

    for k, v in base_dict.items():
        if k in case_b_dict and case_b_dict[k].shape == v.shape:
            case_b_dict[k] = v
            matched.append(k)
        else:
            skipped.append(k)

    case_b_model.model.load_state_dict(case_b_dict)

    print(f"[3/4] Transferred {len(matched)} matching tensors with identical shapes.")
    print(f"      - Backbone & Neck transferred: {sum(1 for k in matched if 'model.23' not in k)} tensors")
    print(f"      - Detect Box-Regression (cv2) transferred: {sum(1 for k in matched if 'model.23.cv2' in k)} tensors")
    print(f"      - Detect DFL transferred: {sum(1 for k in matched if 'model.23.dfl' in k)} tensors")
    print(f"      - Skipped for warehouse adaptation: {len(skipped)} tensors")
    print(f"        (Layer 10 WarehouseSS2DBlock + Detect classification cv3)")

    # 4. Save checkpoint
    out_path = Path("models/yolo11s_ss2d_full_warmstart.pt")
    torch.save({"model": case_b_model.model}, str(out_path))
    print(f"[4/4] Saved complete warmstart checkpoint to: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 65)
    return str(out_path)

if __name__ == "__main__":
    build_full_warmstart()
