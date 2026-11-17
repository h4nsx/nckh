#!/usr/bin/env python3
"""
Standard Edge Device Benchmarking Script:
Evaluates and benchmarks YOLO11s (Baseline) vs. YOLO11s-SS2D on CPU edge
hardware (Raspberry Pi 4 / x86 host) with ONNX Runtime.

Measures:
1. Warmup (10 iterations by default, not timed)
2. Steady-State Latency (Median, Mean, P90, P95, StdDev over N iterations)
   - forward only (session.run), and
   - end-to-end on real images (--images): letterbox preprocessing,
     forward pass and NMS post-processing, each timed separately
3. Throughput (FPS = 1000 / median latency)
4. Memory, all in MiB (1 MiB = 1024 * 1024 bytes):
   - rss_idle_mib:  RSS after importing ONNX Runtime, before loading the model
   - rss_end_mib:   RSS after the timed loop
   - peak_rss_mib:  process high-water mark (VmHWM / ru_maxrss), i.e. true peak,
                    which usually occurs while the session is being created
5. Execution setup: provider, intra-op threads, ONNX node count

Revision note (TAIMMA 2027, round 3): the earlier version reported the RSS
*after* the timed loop as "peak RSS", let ONNX Runtime pick TensorRT/CUDA when
available, used the default thread count, and timed only session.run on random
noise. Values in the paper measured with that version must be re-measured.

Example (Raspberry Pi 4, 4 threads, 100 real validation frames):
  python scripts/benchmark_edge_device.py --model baseline.onnx --threads 4 \
      --images data/smartspaces_micro/images/val --out results/edge/pi4_baseline_t4.json
"""

import os
import sys
import json
import time
import argparse
import platform
from pathlib import Path

import numpy as np

MIB = 1024 * 1024


def current_rss_mib():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / MIB
    except ImportError:
        return float("nan")


def peak_rss_mib():
    """Process high-water mark of resident memory, in MiB."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / MIB  # value is in kB
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes
        return maxrss / MIB if sys.platform == "darwin" else maxrss * 1024 / MIB
    except ImportError:  # Windows
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().peak_wset / MIB
        except (ImportError, AttributeError):
            return float("nan")


def summarize(latencies_ms):
    a = np.asarray(latencies_ms)
    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(np.mean(a)),
        "std_ms": float(np.std(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "fps": float(1000.0 / np.median(a)),
    }


def letterbox(img, imgsz):
    """Resize with unchanged aspect ratio and pad to imgsz x imgsz (Ultralytics convention)."""
    import cv2
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0


def nms_postprocess(pred, conf_thres=0.25, iou_thres=0.7):
    """Class-aware NMS on a YOLO11 ONNX output of shape [1, 4 + nc, A]."""
    p = pred[0].T  # [A, 4 + nc]
    scores_all = p[:, 4:]
    cls = scores_all.argmax(1)
    scores = scores_all[np.arange(len(cls)), cls]
    keep = scores > conf_thres
    boxes, scores, cls = p[keep, :4], scores[keep], cls[keep]
    if len(scores) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)
    offset = xyxy + cls[:, None] * 4096.0  # separate classes
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        i = order[0]
        kept.append(i)
        rest = order[1:]
        xx1 = np.maximum(offset[i, 0], offset[rest, 0])
        yy1 = np.maximum(offset[i, 1], offset[rest, 1])
        xx2 = np.minimum(offset[i, 2], offset[rest, 2])
        yy2 = np.minimum(offset[i, 3], offset[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area(offset[[i]]) + area(offset[rest]) - inter + 1e-9)
        order = rest[iou <= iou_thres]
    kept = np.array(kept)
    return np.concatenate([xyxy[kept], scores[kept, None], cls[kept, None]], 1)


def count_onnx_nodes(model_path):
    try:
        import onnx
        return len(onnx.load(model_path, load_external_data=False).graph.node)
    except ImportError:
        return None


def benchmark_onnx(model_path: str, num_runs: int = 100, warmup: int = 10, imgsz: int = 640,
                   threads: int = 0, images: str = None):
    print("\n" + "=" * 65)
    print(f"BENCHMARKING ONNX RUNTIME: {os.path.basename(model_path)}")
    print("=" * 65)

    import onnxruntime as ort
    rss_idle = current_rss_mib()

    # CPU execution provider only, so that results never silently switch to a GPU EP
    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        session_opts.intra_op_num_threads = threads
        session_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, session_opts, providers=["CPUExecutionProvider"])
    rss_after_load = current_rss_mib()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = [1, 3, imgsz, imgsz]

    # --- Forward-only timing on a fixed random tensor ---
    dummy_input = np.random.default_rng(0).standard_normal(input_shape).astype(np.float32)
    print(f"\n1. Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: dummy_input})

    print(f"2. Timing forward pass over {num_runs} iterations (Batch Size = 1)...")
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    forward = summarize(latencies)
    # Memory of the inference session alone: read before OpenCV, real images or onnx.load
    # (used for the node count below) can raise the high-water mark.
    rss_end_forward = current_rss_mib()
    peak_rss_forward = peak_rss_mib()

    # --- End-to-end timing on real images ---
    end_to_end = None
    if images:
        import cv2
        paths = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        paths = paths[:num_runs]
        print(f"3. Timing end-to-end on {len(paths)} images (preprocess + forward + NMS)...")
        pre, fwd, post, total, n_det = [], [], [], [], []
        for path in paths:
            img = cv2.imread(str(path))  # disk read is excluded from timing
            t0 = time.perf_counter()
            x = letterbox(img, imgsz)
            t1 = time.perf_counter()
            out = session.run([output_name], {input_name: x})[0]
            t2 = time.perf_counter()
            det = nms_postprocess(out)
            t3 = time.perf_counter()
            pre.append((t1 - t0) * 1000.0)
            fwd.append((t2 - t1) * 1000.0)
            post.append((t3 - t2) * 1000.0)
            total.append((t3 - t0) * 1000.0)
            n_det.append(len(det))
        end_to_end = {
            "n_images": len(paths),
            "preprocess": summarize(pre),
            "forward": summarize(fwd),
            "postprocess_nms": summarize(post),
            "total": summarize(total),
            "mean_detections": float(np.mean(n_det)) if n_det else 0.0,
        }

    rss_end_all = current_rss_mib()
    peak_rss_all = peak_rss_mib()
    record = {
        "model": os.path.basename(model_path),
        "onnx_size_mib": os.path.getsize(model_path) / MIB,
        "onnx_nodes": count_onnx_nodes(model_path),  # after all memory readings
        "provider": session.get_providers()[0],
        "intra_op_threads": threads if threads > 0 else "ort_default",
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "onnxruntime": ort.__version__,
        "input_shape": input_shape,
        "warmup": warmup,
        "runs": num_runs,
        "forward_only": forward,
        "end_to_end": end_to_end,
        "rss_idle_mib": rss_idle,
        "rss_after_load_mib": rss_after_load,
        "rss_end_mib": rss_end_forward,         # comparable to the paper's current "RSS" column
        "peak_rss_mib": peak_rss_forward,       # session creation + forward loop (report this)
        "rss_end_with_e2e_mib": rss_end_all,
        "peak_rss_with_e2e_mib": peak_rss_all,  # also includes OpenCV and image decoding
    }

    print("\n" + "-" * 40)
    print("BENCHMARK METRICS SUMMARY")
    print("-" * 40)
    print(f"Model:            {record['model']}  ({record['onnx_nodes']} ONNX nodes)")
    print(f"Provider/threads: {record['provider']} / {record['intra_op_threads']}")
    print(f"Forward median:   {forward['median_ms']:.2f} ms  (P95 {forward['p95_ms']:.2f} ms, {forward['fps']:.2f} FPS)")
    if end_to_end:
        print(f"End-to-end median:{end_to_end['total']['median_ms']:.2f} ms  "
              f"(pre {end_to_end['preprocess']['median_ms']:.1f} / fwd {end_to_end['forward']['median_ms']:.1f} / "
              f"nms {end_to_end['postprocess_nms']['median_ms']:.1f})")
    print(f"RSS idle / end:   {record['rss_idle_mib']:.1f} / {record['rss_end_mib']:.1f} MiB")
    print(f"Peak RSS (HWM):   {record['peak_rss_mib']:.1f} MiB")
    print("-" * 40)
    return record


def main():
    parser = argparse.ArgumentParser(description="Edge Device Latency, FPS and Memory Benchmark (CPU EP)")
    parser.add_argument("--model", type=str, default="models/yolo11s_ss2d_full_warmstart.onnx",
                        help="Path to ONNX model artifact")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--threads", type=int, default=0,
                        help="ONNX Runtime intra-op threads (0 = ORT default); report 1 and 4 on the Pi")
    parser.add_argument("--images", type=str, default=None,
                        help="Folder of real frames for end-to-end timing (preprocess + forward + NMS)")
    parser.add_argument("--out", type=str, default=None, help="Write the metrics record as JSON")
    args = parser.parse_args()

    record = benchmark_onnx(model_path=args.model, num_runs=args.runs, warmup=args.warmup,
                            imgsz=args.imgsz, threads=args.threads, images=args.images)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
