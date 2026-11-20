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


def build_untrained(arm: str, seed: int = 0) -> YOLO:
    """Model from YAML with 4 classes and random weights (same head size as the trained models).

    The weights depend only on `seed`, so the sequential and chunked exports of an arm share
    identical parameters and their outputs can be compared. The zero-gamma initialization of
    the SS2D/conv blocks is replaced by gamma = 1; otherwise the branch would output zeros,
    the output comparison would not exercise the scan, and a graph optimizer could in
    principle prune work that a trained model must do.
    """
    import torch
    yolo = YOLO(ARM_CFGS[arm])
    torch.manual_seed(seed)
    model = DetectionModel(ARM_CFGS[arm], nc=len(CLASS_NAMES), verbose=False)
    for m in model.modules():
        if isinstance(m, WarehouseSS2DBlock):  # includes the conv-only and 2-direction variants
            torch.nn.init.ones_(m.norm3.weight)
    # With default running statistics (mean 0, var 1), activations of a randomly initialized
    # network shrink layer by layer in eval mode (the SS2D branch output was ~1e-9), so the
    # exported outputs would barely depend on the scan. Calibrate BatchNorm statistics on
    # seeded random inputs so that every layer works at unit scale.
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over the calibration batches
    model.train()
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for _ in range(2):
            model(torch.rand(2, 3, 640, 640, generator=gen))
    for bn in bns:
        bn.momentum = 0.03
    model.eval()
    model.names = dict(enumerate(CLASS_NAMES))
    model.args = getattr(yolo.model, "args", {})
    model.task = getattr(yolo.model, "task", "detect")
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
    parser.add_argument("--chunk", type=int, default=0, help="0 = unrolled sequential scan; -1 = same scan as one ONNX Loop; "
                             "-2 = parallel associative scan (log2 L steps); e.g. 20 = chunked scan")
    parser.add_argument("--hoist-proj", action="store_true",
                        help="Compute the shared W_proj once and permute it (identical outputs, 1/4 of the MACs)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out", type=str, required=True, help="Destination .onnx path")
    args = parser.parse_args()

    # Instances created from now on use this scan mode; loaded checkpoints are switched below.
    PurePyTorch2DSelectiveScan.default_chunk_size = args.chunk
    yolo = YOLO(args.weights) if args.weights else build_untrained(args.arm)
    n_scans = set_scan_chunk_size(yolo.model, args.chunk)
    for m in yolo.model.modules():
        if isinstance(m, PurePyTorch2DSelectiveScan):
            m.hoist_proj = args.hoist_proj
    if args.arm in ("ss2d", "ss2d_2dir", "case_c") and n_scans == 0:
        sys.exit(f"[ERROR] arm '{args.arm}' should contain a selective scan, but none was found")
    print(f"[INFO] arm={args.arm} source={'weights' if args.weights else 'untrained'} "
          f"chunk={args.chunk} hoist_proj={args.hoist_proj} selective_scans={n_scans} nc={len(yolo.model.names)}")

    # Ultralytics writes the export next to the checkpoint (e.g. weights/best.onnx). Keep an
    # existing file there (such as the measured export) from being overwritten and moved away.
    sibling = Path(args.weights).with_suffix(".onnx") if args.weights else None
    backup = None
    if sibling is not None and sibling.exists():
        backup = sibling.with_name(sibling.name + ".keep")
        sibling.replace(backup)
    try:
        exported = yolo.export(format="onnx", opset=args.opset, simplify=True, imgsz=args.imgsz,
                               batch=1, dynamic=False, half=False, nms=False, device="cpu")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(out))
    finally:
        if backup is not None:
            backup.replace(sibling)

    try:
        import onnx
        n_nodes = len(onnx.load(str(out), load_external_data=False).graph.node)
    except ImportError:
        n_nodes = "onnx not installed"
    print(f"[SAVED] {out}  ({out.stat().st_size / 2**20:.1f} MiB, {n_nodes} nodes)")
    print(f"[SHA256] {sha256(out)}")


if __name__ == "__main__":
    main()
