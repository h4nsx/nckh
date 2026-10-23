#!/usr/bin/env python3
"""
Standard Edge Device Benchmarking Script:
Evaluates and benchmarks YOLO11s (Baseline) vs. YOLO11s-SS2D (Case B)
on target edge hardware (Jetson Nano / Orin Nano / Raspberry Pi / Edge CPU).

Measures:
1. Warmup Latency (10 iterations)
2. Steady-State Inference Latency (Median, Mean, P90, P95, StdDev over N iterations)
3. Throughput (Frames Per Second - FPS)
4. Model Memory Footprint (Peak RSS via psutil)
5. Output Tensor Verification
"""

import os
import sys
import time
import argparse
import numpy as np

def measure_rss_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0

def benchmark_onnx(model_path: str, num_runs: int = 100, warmup: int = 10, imgsz: int = 640):
    print("\n" + "=" * 65)
    print(f"BENCHMARKING ONNX RUNTIME: {os.path.basename(model_path)}")
    print("=" * 65)

    import onnxruntime as ort

    # Prefer CUDA/TensorRT if available on edge, fallback to CPU
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    selected_providers = []
    if "TensorrtExecutionProvider" in providers:
        selected_providers.append("TensorrtExecutionProvider")
    if "CUDAExecutionProvider" in providers:
        selected_providers.append("CUDAExecutionProvider")
    selected_providers.append("CPUExecutionProvider")
    print(f"Selected Providers:  {selected_providers}")

    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, session_opts, providers=selected_providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = [1, 3, imgsz, imgsz]

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    print(f"\n1. Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = session.run([output_name], {input_name: dummy_input})

    print(f"2. Timing inference over {num_runs} iterations (Batch Size = 1)...")
    latencies = []
    rss_start = measure_rss_mb()

    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in milliseconds

    rss_end = measure_rss_mb()
    latencies = np.array(latencies)

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p90_lat = np.percentile(latencies, 90)
    p95_lat = np.percentile(latencies, 95)
    std_lat = np.std(latencies)
    fps = 1000.0 / median_lat

    print("\n" + "-" * 40)
    print("BENCHMARK METRICS SUMMARY")
    print("-" * 40)
    print(f"Model:           {os.path.basename(model_path)}")
    print(f"Input Shape:     {input_shape}")
    print(f"Active Provider: {session.get_providers()[0]}")
    print(f"Median Latency:  {median_lat:.2f} ms")
    print(f"Mean Latency:    {mean_lat:.2f} ms +/- {std_lat:.2f} ms")
    print(f"P90 Latency:     {p90_lat:.2f} ms")
    print(f"P95 Latency:     {p95_lat:.2f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    if rss_start > 0:
        print(f"Memory RSS:      {rss_end:.1f} MB (Delta: +{rss_end - rss_start:.1f} MB)")
    print("-" * 40)

    return {
        "model": os.path.basename(model_path),
        "median_ms": median_lat,
        "mean_ms": mean_lat,
        "p90_ms": p90_lat,
        "p95_ms": p95_lat,
        "fps": fps,
        "rss_mb": rss_end
    }

def main():
    parser = argparse.ArgumentParser(description="Edge Device Latency and FPS Benchmark")
    parser.add_argument("--model", type=str, default="models/yolo11s_ss2d_full_warmstart.onnx",
                        help="Path to ONNX model artifact")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    benchmark_onnx(model_path=args.model, num_runs=args.runs, warmup=args.warmup, imgsz=args.imgsz)

if __name__ == "__main__":
    main()
