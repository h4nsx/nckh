#!/usr/bin/env python3
"""
Custom YOLO Model Builder for YOLO11s-SS2D (Case B)
Registers WarehouseSS2DBlock with Ultralytics and enables seamless training/validation.
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import WarehouseSS2DBlock

# Inject custom module into Ultralytics namespace for YAML parsing
tasks.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.block.WarehouseSS2DBlock = WarehouseSS2DBlock

def get_yolo11s_case_b_yaml(output_path: Path = Path("models/yolo11s_ss2d_p5.yaml")) -> str:
    """Generate YOLO11s Case B YAML swapping Layer 10 (C2PSA) with WarehouseSS2DBlock."""
    # Read upstream yolo11.yaml
    upstream_yaml_path = Path(".venv/lib/python3.14/site-packages/ultralytics/cfg/models/11/yolo11.yaml")
    with open(upstream_yaml_path, "r") as f:
        d = yaml.safe_load(f)

    # In parse_model else branch:
    # m(*args) receives args directly from YAML row!
    # For WarehouseSS2DBlock(c1=512, c2=512, d_state=16, d_conv=3, expand=0.5):
    # args: [512, 16, 3, 0.5] where first arg is c2, and c1 defaults to c2 if not passed
    d["backbone"][10] = [-1, 1, "WarehouseSS2DBlock", [16, 3, 0.5]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(d, f, sort_keys=False)

    print(f"[INFO] Generated Case B configuration: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    get_yolo11s_case_b_yaml()
