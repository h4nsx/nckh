#!/usr/bin/env python3
"""
ONNX Export for Edge Benchmarking (TAIMMA 2027 revision, Experiments G1/G2)
--------------------------------------------------------------------------
Exports one model to ONNX (opset 17, onnxslim-simplified, batch 1, 640x640, no NMS)
so that scripts/benchmark_edge_device.py and scripts/profile_onnx_operators.py can run
it on the Raspberry Pi. Run this on a PC with torch + ultralytics, then copy the .onnx
files to the Pi (they are git-ignored).

Two modes:
  * trained:   --weights path/to/best.pt    (G2: re-measure the models in the paper)
  * untrained: --untrained                  (G1: runtime-only ablation; latency, node
                                             count and memory do not depend on weights)

The selective scan can be exported as the sequential reference (--chunk 0, the form used
for the paper's current numbers) or as the mathematically equivalent chunked scan
(--chunk 20), which is the controlled intervention for the graph-unrolling hypothesis.

Examples:
  python scripts/export_onnx.py --arm baseline --weights runs/.../arm1/weights/best.pt --out onnx/arm1_baseline.onnx
  python scripts/export_onnx.py --arm ss2d --weights runs/.../arm2/weights/best.pt --out onnx/arm2_ss2d_seq.onnx
  python scripts/export_onnx.py --arm ss2d --weights runs/.../arm2/weights/best.pt --chunk 20 --out onnx/arm2_ss2d_chunk20.onnx
  python scripts/export_onnx.py --arm identity --untrained --out onnx/g1_identity.onnx
"""

import argparse
import hashlib
import shutil
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
    set_scan_chunk_size,
)

for _cls in (WarehouseSS2DBlock, WarehouseConvBlock, WarehouseSS2DBlock2Dir):
    setattr(tasks, _cls.__name__, _cls)
    setattr(modules, _cls.__name__, _cls)

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

ARM_CFGS = {
    "baseline": "yolo11s.yaml",  # resolved from the ultralytics package
    "identity": str(ROOT / "models/yolo11s_identity_p5.yaml"),
    "convonly": str(ROOT / "models/yolo11s_convonly_p5.yaml"),
    "ss2d": str(ROOT / "models/yolo11s_ss2d_p5.yaml"),
    "ss2d_2dir": str(ROOT / "models/yolo11s_ss2d2dir_p5.yaml"),
    "case_c": str(ROOT / "models/yolo11s_ss2d_p4_p5.yaml"),
}
CLASS_NAMES = ["worker", "forklift", "pallet_truck", "amr"]


def build_untrained(arm: str) -> YOLO:
    """Model from YAML with 4 classes and random weights (same head size as the trained models)."""
    yolo = YOLO(ARM_CFGS[arm])
    model = DetectionModel(ARM_CFGS[arm], nc=len(CLASS_NAMES), verbose=False)
    model.names = dict(enumerate(CLASS_NAMES))
    model.args = getattr(yolo.model, "args", {})
    yolo.model = model
    return yolo


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export one arm to ONNX for edge benchmarking")
    parser.add_argument("--arm", required=True, choices=list(ARM_CFGS))
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", type=str, help="Trained checkpoint (best.pt)")
    src.add_argument("--untrained", action="store_true", help="Random weights from the arm's YAML")
    parser.add_argument("--chunk", type=int, default=0, help="0 = sequential scan; e.g. 20 = chunked scan")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", type=str, required=True, help="Destination .onnx path")
    args = parser.parse_args()

    # Instances created from now on use this scan mode; loaded checkpoints are switched below.
    PurePyTorch2DSelectiveScan.default_chunk_size = args.chunk
    yolo = YOLO(args.weights) if args.weights else build_untrained(args.arm)
    n_scans = set_scan_chunk_size(yolo.model, args.chunk)
    if args.arm in ("ss2d", "ss2d_2dir", "case_c") and n_scans == 0:
        sys.exit(f"[ERROR] arm '{args.arm}' should contain a selective scan, but none was found")
    print(f"[INFO] arm={args.arm} source={'weights' if args.weights else 'untrained'} "
          f"chunk={args.chunk} selective_scans={n_scans} nc={len(yolo.model.names)}")

    exported = yolo.export(format="onnx", opset=args.opset, simplify=True, imgsz=args.imgsz,
                           batch=1, dynamic=False, half=False, nms=False, device="cpu")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(out))

    try:
        import onnx
        n_nodes = len(onnx.load(str(out), load_external_data=False).graph.node)
    except ImportError:
        n_nodes = "onnx not installed"
    print(f"[SAVED] {out}  ({out.stat().st_size / 2**20:.1f} MiB, {n_nodes} nodes)")
    print(f"[SHA256] {sha256(out)}")


if __name__ == "__main__":
    main()
