#!/usr/bin/env python3
"""
ONNX Operator-Level Graph Profiler for TAIMMA 2027
--------------------------------------------------
Resolves Reviewer Issue D8:
Empirically quantifies the exact architectural and execution profile:
1. Baseline YOLO11s (C2PSA): 320 ONNX nodes (dense parallel GEMM/Conv)
2. YOLO11s-SS2D (Case B): 14,767 ONNX nodes (due to sequential recurrence unrolling over L=400 tokens x 4 directions)
3. Profiling breakdown: Gather, Mul, Unsqueeze, Add, ReduceSum vs Conv
Proves that CPU latency is strictly governed by operator graph unrolling,
not by theoretical FLOPs.
"""

import os
import sys
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np

def profile_graph(onnx_path: str, threads: int = 0, out_dir: str = ".", warmup: int = 2, runs: int = 5):
    import onnx
    import onnxruntime as ort
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(onnx_path).stem

    print("=" * 65)
    print(f"PROFILING ONNX GRAPH: {os.path.basename(onnx_path)}")
    print("=" * 65)

    model = onnx.load(onnx_path)
    nodes = model.graph.node
    total_nodes = len(nodes)
    op_counts = Counter(n.op_type for n in nodes)

    print(f"Total Graph Nodes: {total_nodes}")
    print("Operator Distribution:")
    for op, cnt in op_counts.most_common(10):
        print(f"  - {op:<15}: {cnt:>6} nodes ({cnt/total_nodes*100:>5.1f}%)")

    # Run profiling session
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.profile_file_prefix = str(Path(out_dir) / f"ort_profile_{stem}")
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Save the graph ORT actually executes, to report node counts after its fusions
    optimized_path = Path(out_dir) / f"{stem}_ort_optimized.onnx"
    opts.optimized_model_filepath = str(optimized_path)
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    optimized_nodes = len(onnx.load(str(optimized_path), load_external_data=False).graph.node)
    print(f"Nodes after ORT graph optimization: {optimized_nodes}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup and profiled runs. Profiling is on for the whole session, so warmup runs are
    # recorded too; they are excluded below by timestamp. Keep the counts small: an unrolled
    # scan graph emits tens of thousands of events per run, and ORT stops recording at its
    # event cap (about 1M), which would silently truncate the later runs.
    for _ in range(warmup + runs):
        session.run([output_name], {input_name: dummy})

    prof_file = session.end_profiling()
    print(f"\nSaved detailed ORT profile log to: {prof_file}")

    # ORT writes one event per line; parse line by line so a multi-hundred-MB profile does
    # not have to fit in memory as one JSON document (json.load was OOM-killed on a Pi 4).
    run_spans, node_events = [], []
    with open(prof_file, "r") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("cat") == "Session" and ev.get("name") == "model_run":
                run_spans.append((ev.get("ts", 0), ev.get("dur", 0)))
            elif ev.get("cat") == "Node" and str(ev.get("name", "")).endswith("_kernel_time"):
                node_events.append((ev.get("ts", 0), ev.get("args", {}).get("op_name", "Unknown"),
                                    ev.get("dur", 0)))
    run_spans.sort()
    if len(run_spans) <= warmup:
        raise RuntimeError(f"only {len(run_spans)} model_run events recorded; profile truncated")
    cutoff = run_spans[warmup - 1][0] + run_spans[warmup - 1][1] if warmup > 0 else 0
    captured_runs = len(run_spans) - warmup
    if captured_runs < runs:
        print(f"[WARN] only {captured_runs}/{runs} profiled runs were recorded (event cap)")

    op_durations = Counter()
    for ts, op_name, dur in node_events:  # microseconds
        if ts >= cutoff:
            op_durations[op_name] += dur
    del node_events

    total_dur = sum(op_durations.values())
    print("\nOperator Time Breakdown (Top 8):")
    for op, dur in op_durations.most_common(8):
        print(f"  - {op:<15}: {dur/1000.0:>8.2f} ms ({dur/total_dur*100:>5.1f}%)")

    return {
        "model": os.path.basename(onnx_path),
        "total_nodes": total_nodes,
        "nodes_after_ort_optimization": optimized_nodes,
        "intra_op_threads": threads if threads > 0 else "ort_default",
        "warmup_runs": warmup,
        "profiled_runs_requested": runs,
        "profiled_runs": captured_runs,
        "mean_model_run_ms": sum(d for _, d in run_spans[warmup:]) / captured_runs / 1000.0,
        "top_ops": dict(op_counts.most_common(10)),
        # Kernel time summed over the profiled runs; divide by profiled_runs for per-inference time
        "durations_ms": {k: v/1000.0 for k, v in op_durations.most_common(10)},
        "kernel_total_ms": total_dur / 1000.0,
        "ort_profile_file": prof_file,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ONNX node counts and per-operator CPU time")
    parser.add_argument("--model", nargs="+", required=True, help="One or more .onnx files")
    parser.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = default)")
    parser.add_argument("--out-dir", default="results/edge/profiles", help="Where JSON and ORT logs go")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs, excluded from the timings")
    parser.add_argument("--runs", type=int, default=5, help="Profiled runs (keep small for unrolled graphs)")
    args = parser.parse_args()

    for path in args.model:
        result = profile_graph(path, threads=args.threads, out_dir=args.out_dir,
                               warmup=args.warmup, runs=args.runs)
        out = Path(args.out_dir) / f"{Path(path).stem}_t{args.threads or 'default'}_profile.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"[SAVED] {out}")
