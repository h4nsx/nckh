#!/usr/bin/env python3
"""
Hardware performance counters for ONNX Runtime inference on the Raspberry Pi 4 (Cortex-A72).

Creates the session and warms up first, then attaches `perf stat -p <own pid>` only around the
timed session.run calls, so the (up to ~47 s) session creation of the unrolled graph is excluded.
The Cortex-A72 has 6 programmable counters, so the events are measured in two groups of 6, each
over its own block of calls (no multiplexing). Raw ARMv8 PMU event numbers are used because the
symbolic names differ between kernels.

Derived quantities (per inference call):
  IPC                      = INST_RETIRED / CPU_CYCLES
  L2 refills               = L2D_CACHE_REFILL (the L2 is the last-level cache on the Pi 4)
  DRAM traffic (estimate)  = (L2D_CACHE_REFILL + L2D_CACHE_WB) * 64 B
  DRAM bandwidth (estimate)= DRAM traffic / wall time of the calls

Usage (inside the Pi venv; needs kernel.perf_event_paranoid <= 1 or root):
  python3 scripts/perf_counters_pi.py --model ~/onnx_caseb/yolo11s_ss2d_arm3_par.onnx --threads 4 \
      --runs 10 --warmup 3 --perf perf --out results/edge/perf_x/par_t4.json
"""

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort

# ARMv8 common PMU events (Arm ARM D7.10; Cortex-A72 TRM, PMU events)
GROUPS = {
    "core": {
        "CPU_CYCLES": "r11",
        "INST_RETIRED": "r08",
        "STALL_FRONTEND": "r23",
        "STALL_BACKEND": "r24",
        "L1D_CACHE": "r04",
        "L1D_CACHE_REFILL": "r03",
    },
    "memory": {
        "L2D_CACHE": "r16",
        "L2D_CACHE_REFILL": "r17",
        "L2D_CACHE_WB": "r18",
        "BUS_ACCESS": "r19",
        "MEM_ACCESS": "r13",
        "L1I_CACHE_REFILL": "r01",
    },
}
LINE_BYTES = 64


def run_calls(session, feed, n):
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def perf_block(perf, events, session, feed, n):
    """Count `events` for this process (all threads) over n calls; return counts and timings."""
    with tempfile.NamedTemporaryFile("r", suffix=".csv", delete=False) as tmp:
        out_path = tmp.name
    names = list(events)
    cmd = [perf, "stat", "-x", ",", "-o", out_path, "-e", ",".join(events[k] for k in names),
           "-p", str(os.getpid())]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(1.0)  # let perf attach to every thread before the first call
    if proc.poll() is not None:
        sys.exit(f"[ERROR] perf exited early: {proc.stderr.read().decode(errors='replace')}")
    t0 = time.perf_counter()
    lat = run_calls(session, feed, n)
    wall = time.perf_counter() - t0
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=30)
    counts, raw = {}, []
    with open(out_path) as f:
        for line in f:
            raw.append(line.rstrip("\n"))
            parts = line.strip().split(",")
            if len(parts) < 3 or line.startswith("#"):
                continue
            value, event = parts[0], parts[2]
            for k in names:
                if events[k] == event or event.endswith("/" + events[k] + "/"):
                    counts[k] = None if value.startswith("<") else int(float(value))
            if len(parts) > 4 and parts[4] not in ("", "100.00"):
                print(f"[WARN] {event} counted {parts[4]}% of the time (multiplexed)")
    os.unlink(out_path)
    missing = [k for k in names if counts.get(k) is None]
    if missing:
        print(f"[WARN] events not counted: {missing}")
    return counts, lat, wall, raw


def main():
    ap = argparse.ArgumentParser(description="perf stat counters around ONNX Runtime inference")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=10, help="timed calls per counter group")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--perf", default="perf", help="perf binary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    t0 = time.perf_counter()
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    create_s = time.perf_counter() - t0
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    feed = {inp.name: np.random.default_rng(0).standard_normal(shape).astype(np.float32)}
    run_calls(session, feed, args.warmup)

    rec = {"model": os.path.basename(args.model), "threads": args.threads, "runs_per_group": args.runs,
           "warmup": args.warmup, "session_create_s": create_s, "onnxruntime": ort.__version__,
           "platform": platform.platform(), "groups": {}}
    total = {}
    for g, events in GROUPS.items():
        counts, lat, wall, raw = perf_block(args.perf, events, session, feed, args.runs)
        rec["groups"][g] = {"counts": counts, "latency_ms": lat, "median_ms": float(np.median(lat)),
                            "wall_s": wall, "perf_raw": raw}
        total.update({k: (v / args.runs if v is not None else None) for k, v in counts.items()})
        print(f"[{g}] median {np.median(lat):.1f} ms over {args.runs} calls")

    per_call = total
    d = {}
    if per_call.get("CPU_CYCLES") and per_call.get("INST_RETIRED"):
        d["ipc"] = per_call["INST_RETIRED"] / per_call["CPU_CYCLES"]
        if per_call.get("STALL_BACKEND") is not None:
            d["backend_stall_frac"] = per_call["STALL_BACKEND"] / per_call["CPU_CYCLES"]
        if per_call.get("STALL_FRONTEND") is not None:
            d["frontend_stall_frac"] = per_call["STALL_FRONTEND"] / per_call["CPU_CYCLES"]
    if per_call.get("L1D_CACHE") and per_call.get("L1D_CACHE_REFILL") is not None:
        d["l1d_miss_rate"] = per_call["L1D_CACHE_REFILL"] / per_call["L1D_CACHE"]
    if per_call.get("L2D_CACHE") and per_call.get("L2D_CACHE_REFILL") is not None:
        d["l2_miss_rate"] = per_call["L2D_CACHE_REFILL"] / per_call["L2D_CACHE"]
    if per_call.get("L2D_CACHE_REFILL") is not None and per_call.get("L2D_CACHE_WB") is not None:
        mem_ms = rec["groups"]["memory"]["median_ms"]
        traffic = (per_call["L2D_CACHE_REFILL"] + per_call["L2D_CACHE_WB"]) * LINE_BYTES
        d["dram_traffic_mib_per_call"] = traffic / 2**20
        d["dram_bw_gib_s"] = traffic / (mem_ms / 1000.0) / 2**30
    rec["per_call"] = per_call
    rec["derived"] = d

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"Session creation: {create_s:.2f} s")
    for k, v in per_call.items():
        print(f"  {k:18s} {v:,.0f} per call" if v is not None else f"  {k:18s} n/a")
    for k, v in d.items():
        print(f"  {k:26s} {v:.4f}")
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
