import os
import sys
import shutil
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# --- 1. Install Dependencies ---
os.system("pip install -q ultralytics torchvision opencv-python")

# --- 2. Unpack bundle or locate dataset ---
bundle_candidates = [
    "warehouse_ss2d_gpu_bundle.tar.gz",
    "/kaggle/input/warehouse_ss2d_gpu_bundle.tar.gz",
    "/kaggle/working/warehouse_ss2d_gpu_bundle.tar.gz"
]

for b in bundle_candidates:
    if os.path.exists(b):
        print(f">>> Unpacking {b}...")
        os.system(f"tar -xzf {b}")
        break

# Verify micro-dataset
if not os.path.exists("data/smartspaces_micro/smartspaces_micro.yaml"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "smartspaces_micro.yaml" in files:
            dataset_dir = Path(root)
            os.makedirs("data/smartspaces_micro", exist_ok=True)
            os.system(f"cp -r {dataset_dir}/* data/smartspaces_micro/")
            break

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# --- 3. Pure PyTorch 2D Selective Scan (Embedded directly - 0 external imports!) ---
class PurePyTorch2DSelectiveScan(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_logs = nn.Parameter(torch.randn(4, d_model, d_state))
        self.D = nn.Parameter(torch.ones(4, d_model))
        self.proj_params = nn.Linear(d_model, 2 * self.d_state + self.d_model, bias=False)

    def _selective_scan_1d(self, u, delta, A, B, C, D):
        batch_size, seq_len, d_model = u.shape
        delta_A = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, d_model, self.d_state))
        delta_B_u = (delta.unsqueeze(-1) * u.unsqueeze(-1)) * B.unsqueeze(2)
        states = []
        h = torch.zeros(batch_size, d_model, self.d_state, device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            states.append(y_t)
        y = torch.stack(states, dim=1)
        y = y + u * D.view(1, 1, d_model)
        return y

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_1 = x.permute(0, 2, 3, 1).reshape(B, L, C)
        x_2 = torch.flip(x_1, dims=[1])
        x_3 = x.permute(0, 3, 2, 1).reshape(B, L, C)
        x_4 = torch.flip(x_3, dims=[1])

        A = -torch.exp(self.A_logs)

        p_1 = self.proj_params(x_1)
        B_1, C_1, d_1 = torch.split(p_1, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_1 = self._selective_scan_1d(x_1, F.softplus(d_1), A[0], B_1, C_1, self.D[0])

        p_2 = self.proj_params(x_2)
        B_2, C_2, d_2 = torch.split(p_2, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_2 = self._selective_scan_1d(x_2, F.softplus(d_2), A[1], B_2, C_2, self.D[1])
        y_2 = torch.flip(y_2, dims=[1])

        p_3 = self.proj_params(x_3)
        B_3, C_3, d_3 = torch.split(p_3, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_3 = self._selective_scan_1d(x_3, F.softplus(d_3), A[2], B_3, C_3, self.D[2])
        y_3 = y_3.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        p_4 = self.proj_params(x_4)
        B_4, C_4, d_4 = torch.split(p_4, [self.d_state, self.d_state, self.d_model], dim=-1)
        y_4 = self._selective_scan_1d(x_4, F.softplus(d_4), A[3], B_4, C_4, self.D[3])
        y_4 = torch.flip(y_4, dims=[1]).reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, L, C)

        y = y_1 + y_2 + y_3 + y_4
        return y.reshape(B, H, W, C).permute(0, 3, 1, 2)


# --- 4. WarehouseSS2DBlock with Full Dynamic Channel Support ---
class WarehouseSS2DBlock(nn.Module):
    def __init__(self, c1: int = 512, c2: int = 512, d_state: int = 16, d_conv: int = 3, expand: float = 0.5):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.d_inner = max(1, int(expand * self.c2))

        self.in_proj = nn.Conv2d(self.c1, self.d_inner, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.d_inner)
        self.act1 = nn.SiLU()

        self.dw_conv = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv // 2, groups=self.d_inner, bias=False
        )
        self.norm2 = nn.BatchNorm2d(self.d_inner)
        self.act2 = nn.SiLU()

        self.ss2d = PurePyTorch2DSelectiveScan(d_model=self.d_inner, d_state=d_state)
        self.out_proj = nn.Conv2d(self.d_inner, self.c2, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(self.c2)

        nn.init.zeros_(self.out_proj.weight)
        self.has_residual = (self.c1 == self.c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act1(self.norm1(self.in_proj(x)))
        out = self.act2(self.norm2(self.dw_conv(out)))
        out = self.norm3(self.out_proj(self.ss2d(out)))
        if self.has_residual:
            out = out + res
        return out


# Register directly into Ultralytics registry
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
tasks.WarehouseSS2DBlock = WarehouseSS2DBlock
modules.WarehouseSS2DBlock = WarehouseSS2DBlock
from ultralytics import YOLO

# Also write to models/warehouse_ss2d.py on disk for safety
with open("models/warehouse_ss2d.py", "w") as f:
    f.write("# Auto-generated WarehouseSS2DBlock registry\n")

# --- 5. Write Case C YAML Architecture ---
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
with open("models/yolo11s_ss2d_p4_p5.yaml", "w") as f:
    f.write(case_c_yaml)

# --- 6. Case C Full Warmstart Weight Surgery ---
print(">>> Performing Case C Warmstart Surgery...")
if not os.path.exists("yolo11s.pt"):
    YOLO("yolo11s.pt")  # Auto-downloads official COCO weights

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
print(f">>> Case C Warmstart Checkpoint created: {warmstart_path}")

# --- 7. Train Arm 4 (20 Epochs on GPU) ---
print("\n" + "=" * 65)
print("RUNNING ARM 4: Case C Dual-Scale SS2D (20 Epochs on Tesla T4 GPU)")
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

# --- 8. Copy Artifacts to /kaggle/working/ for One-Click Download ---
print("\n>>> Copying final artifacts to /kaggle/working/...")
src_res = "runs/two_stage_protocol_study/arm4_case_c_dual_scale/results.csv"
src_pt = "runs/two_stage_protocol_study/arm4_case_c_dual_scale/weights/best.pt"

if os.path.exists(src_res):
    shutil.copy(src_res, "/kaggle/working/arm4_case_c_results.csv")
    print("Saved: /kaggle/working/arm4_case_c_results.csv")

if os.path.exists(src_pt):
    shutil.copy(src_pt, "/kaggle/working/arm4_case_c_best.pt")
    print("Saved: /kaggle/working/arm4_case_c_best.pt")

print("\n" + "=" * 65)
print("ARM 4 EXECUTION COMPLETE! READY FOR DOWNLOAD.")
print("=" * 65)
