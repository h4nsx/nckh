#!/usr/bin/env python3
"""
Checks that two ONNX exports of the same trained weights compute the same function,
e.g. the sequential-scan and chunked-scan exports of Arm 2. Run before reporting the
chunked model's latency as an intervention on the same model.

Example:
  python scripts/compare_onnx_outputs.py onnx/arm2_ss2d_seq.onnx onnx/arm2_ss2d_chunk20.onnx
"""

import argparse
import sys

import numpy as np


def run(path, x):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def main():
    parser = argparse.ArgumentParser(description="Compare the outputs of two ONNX models")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-4,
                        help="Max allowed max|diff| / max|reference| (outputs include pixel box coordinates)")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    worst = 0.0  # worst relative difference
    for i in range(args.trials):
        # Image-like inputs in [0, 1], as produced by the letterbox preprocessing
        x = rng.random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
        a, b = run(args.reference, x), run(args.candidate, x)
        if a.shape != b.shape:
            sys.exit(f"[FAIL] output shapes differ: {a.shape} vs {b.shape}")
        diff = float(np.max(np.abs(a - b)))
        rel = diff / (float(np.max(np.abs(a))) + 1e-12)
        worst = max(worst, rel)
        print(f"trial {i}: max |diff| = {diff:.3e} (relative {rel:.3e})")
    print("[PASS]" if worst <= args.rtol else "[FAIL]", f"worst relative diff = {worst:.3e}, rtol = {args.rtol}")
    sys.exit(0 if worst <= args.rtol else 1)


if __name__ == "__main__":
    main()
