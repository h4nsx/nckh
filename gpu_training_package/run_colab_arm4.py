import os
import sys
import torch
from pathlib import Path

# 1. Update models/warehouse_ss2d.py on Colab to ensure signature matches
import models.warehouse_ss2d as m_ss2d
import inspect

# Dynamically patch WarehouseSS2DBlock in runtime memory
orig_init = m_ss2d.WarehouseSS2DBlock.__init__

def patched_init(self, c1=512, c2=512, d_state=16, d_conv=3, expand=0.5):
    super(m_ss2d.WarehouseSS2DBlock, self).__init__()
    self.c1 = int(c1)
    self.c2 = int(c2)
    self.d_inner = max(1, int(expand * self.c2))

    self.in_proj = torch.nn.Conv2d(self.c1, self.d_inner, kernel_size=1, bias=False)
    self.norm1 = torch.nn.BatchNorm2d(self.d_inner)
    self.act1 = torch.nn.SiLU()

    self.dw_conv = torch.nn.Conv2d(
        self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=False
    )
    self.norm2 = torch.nn.BatchNorm2d(self.d_inner)
    self.act2 = torch.nn.SiLU()

    self.ss2d = m_ss2d.PurePyTorch2DSelectiveScan(d_model=self.d_inner, d_state=d_state)
    self.out_proj = torch.nn.Conv2d(self.d_inner, self.c2, kernel_size=1, bias=False)
    self.norm3 = torch.nn.BatchNorm2d(self.c2)

    torch.nn.init.zeros_(self.out_proj.weight)
    self.has_residual = (self.c1 == self.c2)

m_ss2d.WarehouseSS2DBlock.__init__ = patched_init

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
tasks.WarehouseSS2DBlock = m_ss2d.WarehouseSS2DBlock
modules.WarehouseSS2DBlock = m_ss2d.WarehouseSS2DBlock
from ultralytics import YOLO

# 2. Write Case C YAML
case_c_yaml = """nc: 80
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, WarehouseSS2DBlock, [256, 256, 16, 3, 0.5]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 1, WarehouseSS2DBlock, [512, 512, 16, 3, 0.5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 7], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[17, 20, 23], 1, Detect, [nc]]
"""
os.makedirs("models", exist_ok=True)
with open("models/yolo11s_ss2d_p4_p5.yaml", "w") as f:
    f.write(case_c_yaml)

# 3. Perform Warmstart Surgery
print(">>> Performing Case C Warmstart Surgery...")
base_model = YOLO("yolo11s.pt")
base_dict = base_model.model.state_dict()

case_c_model = YOLO("models/yolo11s_ss2d_p4_p5.yaml")
case_c_dict = case_c_model.model.state_dict()

layer_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
    7: 8, 8: 9, 9: 10,
    13: 14, 16: 17, 17: 18, 19: 20, 20: 21, 22: 23, 23: 24
}

for base_k, base_v in base_dict.items():
    parts = base_k.split(".")
    if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
        base_idx = int(parts[1])
        if base_idx in layer_map:
            c_idx = layer_map[base_idx]
            target_k = "model." + str(c_idx) + "." + ".".join(parts[2:])
            if target_k in case_c_dict and case_c_dict[target_k].shape == base_v.shape:
                case_c_dict[target_k] = base_v
    else:
        if base_k in case_c_dict and case_c_dict[base_k].shape == base_v.shape:
            case_c_dict[base_k] = base_v

case_c_model.model.load_state_dict(case_c_dict)
warmstart_path = "models/yolo11s_ss2d_p4_p5_full_warmstart.pt"
torch.save({"model": case_c_model.model}, warmstart_path)
print(f">>> Case C Warmstart Checkpoint saved successfully: {warmstart_path}")

# 4. Train Arm 4 (20 Epochs on GPU)
print("\n" + "=" * 65)
print("RUNNING ARM 4: Case C Dual-Scale SS2D (20 Epochs, Tesla T4 GPU)")
print("=" * 65)
model = YOLO(warmstart_path)
model.train(
    data="data/smartspaces_micro/smartspaces_micro.yaml",
    epochs=20,
    freeze=0,
    imgsz=640,
    batch=4,
    workers=2,
    device=0,
    project="runs/two_stage_protocol_study",
    name="arm4_case_c_dual_scale",
    exist_ok=True,
    verbose=True
)
