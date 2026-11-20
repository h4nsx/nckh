"""Effective memory of the trained selective scan on real validation frames.

For each direction k, Abar = exp(Delta * A) per (token, channel, state). The number of tokens over which the
state decays by 1/e is 1 / (Delta * |A|). We report its distribution after training (best.pt of Arms 2 and 3).
"""
import glob
import sys

import torch
import torch.nn.functional as F

repo, weights = sys.argv[1], sys.argv[2:]
sys.path.insert(0, repo)
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import PurePyTorch2DSelectiveScan, WarehouseSS2DBlock, WarehouseConvBlock, WarehouseSS2DBlock2Dir

for c in (WarehouseSS2DBlock, WarehouseConvBlock, WarehouseSS2DBlock2Dir):
    setattr(tasks, c.__name__, c)
    setattr(modules, c.__name__, c)
import __main__
__main__.PurePyTorch2DSelectiveScan = PurePyTorch2DSelectiveScan
from ultralytics import YOLO

imgs = sorted(glob.glob(f"{repo}/data/smartspaces_micro/images/val/*.jpg"))[::10]  # 10 frames
for w in weights:
    yolo = YOLO(w)
    scans = [m for m in yolo.model.modules() if isinstance(m, PurePyTorch2DSelectiveScan)]
    scan = scans[-1]  # the P5 block
    store = []
    h = scan.proj_params.register_forward_hook(lambda m, i, o: store.append(o.detach()))
    yolo.predict(imgs, verbose=False, device="cpu", imgsz=640)
    h.remove()
    A = -torch.exp(scan.A_logs.detach().float())  # [4, d, n]
    mem_all, abar_all = [], []
    for j, o in enumerate(store):
        k = j % 4
        d = o[..., 2 * scan.d_state:]                         # [b, L, d]
        delta = F.softplus(d.float())
        x = delta.unsqueeze(-1) * A[k].abs().view(1, 1, *A[k].shape)  # [b, L, d, n]
        mem_all.append((1.0 / x).flatten())
        abar_all.append(torch.exp(-x).flatten())
    mem = torch.cat(mem_all)
    abar = torch.cat(abar_all)
    q = torch.quantile(mem[torch.randperm(mem.numel())[:2_000_000]], torch.tensor([0.5, 0.9, 0.99]))
    print(f"{w.split('/')[-3]}: frames={len(imgs)} calls={len(store)} median Abar={abar.median():.3f} "
          f"memory tokens median/p90/p99={q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f} "
          f"frac >=20 tokens={(mem >= 20).float().mean():.4f} frac >=400={(mem >= 400).float().mean():.5f}")
