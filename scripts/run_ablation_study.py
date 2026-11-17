#!/usr/bin/env python3
"""
Multi-Seed Ablation Runner (TAIMMA 2027 revision, Experiments A and C)
----------------------------------------------------------------------
Trains every requested arm for every requested seed under one shared protocol,
then evaluates each run's best.pt and writes one JSON record per (arm, seed) to
results/multiseed/. Aggregate the records with scripts/aggregate_seeds.py.

Arms:
  baseline     YOLO11s with C2PSA (yolo11s.pt)
  identity     layer 10 = nn.Identity           (removes C2PSA, adds nothing)
  convonly     layer 10 = WarehouseConvBlock     (SS2D block without the scan)
  ss2d         layer 10 = WarehouseSS2DBlock     (proposed, 4 directions)
  ss2d_2dir    layer 10 = WarehouseSS2DBlock2Dir (row-major scan only)
  ss2d_2stage  ss2d with stage 1 freeze=10 (layers 0-9) then stage 2 unfrozen
  case_c       SS2D at P4 and P5 (models/yolo11s_ss2d_p4_p5.yaml)

Every non-baseline arm is built from its YAML and receives all shape-compatible
tensors from yolo11s.pt (same procedure as scripts/build_case_b_full_warmstart.py),
so new blocks keep their fresh zero-gamma initialization. The warmstart is rebuilt
per arm under runs/multiseed/warmstarts/; the old models/*_full_warmstart.pt files
are NOT used, because they were saved with the earlier W_out=0 initialization.

Example (3 seeds, GPU 0, held-out test split defined in the data YAML):
  python scripts/run_ablation_study.py --arms baseline identity convonly ss2d \
      --seeds 0 1 2 --epochs 20 --batch 8 --device 0 --eval-split test
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from models.warehouse_ss2d import (
    PurePyTorch2DSelectiveScan,
    WarehouseSS2DBlock,
    WarehouseConvBlock,
    WarehouseSS2DBlock2Dir,
)

for _cls in (WarehouseSS2DBlock, WarehouseConvBlock, WarehouseSS2DBlock2Dir):
    setattr(tasks, _cls.__name__, _cls)
    setattr(modules, _cls.__name__, _cls)

from ultralytics import YOLO
import torch

ARM_YAMLS = {
    "identity": "models/yolo11s_identity_p5.yaml",
    "convonly": "models/yolo11s_convonly_p5.yaml",
    "ss2d": "models/yolo11s_ss2d_p5.yaml",
    "ss2d_2dir": "models/yolo11s_ss2d2dir_p5.yaml",
    "ss2d_2stage": "models/yolo11s_ss2d_p5.yaml",  # shares the "ss2d" architecture
    "case_c": "models/yolo11s_ss2d_p4_p5.yaml",
}
OUT_DIR = ROOT / "results" / "multiseed"


def build_warmstart(arm):
    """Transfer every shape-compatible tensor from yolo11s.pt into the arm's YAML model."""
    out_path = ROOT / "runs" / "multiseed" / "warmstarts" / f"{arm}.pt"
    if out_path.exists():
        return out_path
    base_dict = YOLO("yolo11s.pt").model.state_dict()
    model = YOLO(str(ROOT / ARM_YAMLS[arm]))
    target = model.model.state_dict()
    matched = [k for k, v in base_dict.items() if k in target and target[k].shape == v.shape]
    for k in matched:
        target[k] = base_dict[k]
    model.model.load_state_dict(target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.model}, str(out_path))
    print(f"[WARMSTART] {arm}: transferred {len(matched)}/{len(target)} tensors -> {out_path}")
    return out_path


def build_model(arm):
    if arm == "baseline":
        return YOLO("yolo11s.pt")
    return YOLO(str(build_warmstart(arm)))


def train(model, args, name, epochs, seed, freeze=0):
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=seed,
        deterministic=True,
        freeze=freeze,
        project=str(ROOT / "runs" / "multiseed"),
        name=name,
        exist_ok=True,
        plots=False,
    )
    return Path(model.trainer.save_dir)


def final_epoch_row(save_dir):
    with open(save_dir / "results.csv", newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    last = rows[-1]
    return {
        "epoch": int(float(last["epoch"])),
        "mAP50": float(last["metrics/mAP50(B)"]),
        "mAP50_95": float(last["metrics/mAP50-95(B)"]),
        "precision": float(last["metrics/precision(B)"]),
        "recall": float(last["metrics/recall(B)"]),
    }


def evaluate(best_pt, args):
    m = YOLO(str(best_pt)).val(
        data=args.data, split=args.eval_split, imgsz=640, batch=args.batch,
        device=args.device, plots=False, verbose=False,
    )
    box = m.box
    per_class = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        per_class[m.names[int(cls_idx)]] = {
            "AP50": float(box.ap50[i]),
            "AP50_95": float(box.ap[i]),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
        }
    p, r = float(box.mp), float(box.mr)
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0,
        "per_class": per_class,
    }


def run_one(arm, seed, args):
    name = f"{arm}_seed{seed}"
    model = build_model(arm)
    if arm == "ss2d_2stage":
        warm = max(1, round(args.epochs * args.warmup_frac))
        stage1_dir = train(model, args, name + "_stage1", warm, seed, freeze=10)
        model = YOLO(str(stage1_dir / "weights" / "last.pt"))
        save_dir = train(model, args, name + "_stage2", args.epochs - warm, seed)
    else:
        save_dir = train(model, args, name, args.epochs, seed)

    best_pt = save_dir / "weights" / "best.pt"
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    record = {
        "arm": arm,
        "seed": seed,
        "eval_split": args.eval_split,
        "scan_chunk_size": PurePyTorch2DSelectiveScan.default_chunk_size,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": str(args.device),
        "best_epoch": int(ckpt.get("epoch", -1)) + 1,  # Ultralytics stores a 0-based epoch
        "final_epoch_val": final_epoch_row(save_dir),
        "best_ckpt_eval": evaluate(best_pt, args),
        "save_dir": str(save_dir),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"[SAVED] {out}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed ablation study")
    parser.add_argument("--arms", nargs="+", default=["baseline", "identity", "convonly", "ss2d"],
                        choices=["baseline", *ARM_YAMLS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-frac", type=float, default=0.25, help="ss2d_2stage: share of epochs in stage 1")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data", default="data/smartspaces_micro/smartspaces_micro.yaml")
    parser.add_argument("--eval-split", default="val", choices=["val", "test"],
                        help="Use 'test' once a held-out test facility is defined in the data YAML")
    parser.add_argument("--scan-chunk", type=int, default=0,
                        help="0 = sequential reference scan; e.g. 20 = chunked scan (same math, fewer ops)")
    args = parser.parse_args()

    PurePyTorch2DSelectiveScan.default_chunk_size = args.scan_chunk
    for seed in args.seeds:
        for arm in args.arms:
            run_one(arm, seed, args)


if __name__ == "__main__":
    main()
