#!/usr/bin/env bash
# Runs every ONNX file in a folder through the edge benchmark (1 and 4 threads) and the
# operator profiler on a Raspberry Pi, and records the system state alongside the results.
#
# Usage (from the repository root, inside the Pi virtual environment):
#   bash scripts/run_pi_benchmarks.sh <onnx_dir> [images_dir]
# Example:
#   bash scripts/run_pi_benchmarks.sh ~/onnx ~/val_images
#
# Each benchmark runs in its own Python process, so the peak RSS of one model is never
# inflated by another. Results go to results/edge/<hostname>_<date>/.
set -euo pipefail

ONNX_DIR="${1:?usage: run_pi_benchmarks.sh <onnx_dir> [images_dir]}"
IMAGES="${2:-}"
COOLDOWN="${COOLDOWN:-30}"   # seconds between runs, to limit thermal throttling
OUT="${OUT:-results/edge/$(hostname)_$(date +%Y%m%d_%H%M)}"   # set OUT to resume into an existing folder
mkdir -p "$OUT/profiles"

shopt -s nullglob
models=("$ONNX_DIR"/*.onnx)
if [ ${#models[@]} -eq 0 ]; then
    echo "[ERROR] no .onnx files in $ONNX_DIR" >&2
    exit 1
fi
if [ -n "$IMAGES" ] && [ ! -d "$IMAGES" ]; then
    echo "[ERROR] images folder not found: $IMAGES" >&2
    exit 1
fi

{
    echo "date: $(date -Iseconds)"
    echo "--- uname";        uname -a
    echo "--- os";           cat /etc/os-release 2>/dev/null || true
    echo "--- model";        cat /proc/device-tree/model 2>/dev/null; echo
    echo "--- memory";       free -m
    echo "--- cpus";         nproc
    echo "--- governor";     cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a
    echo "--- max freq kHz"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo n/a
    echo "--- temperature";  vcgencmd measure_temp 2>/dev/null || echo n/a
    echo "--- throttled";    vcgencmd get_throttled 2>/dev/null || echo n/a
    echo "--- python";       python3 --version
    echo "--- packages";     python3 -c "import onnxruntime, onnx, numpy, cv2, psutil; print('onnxruntime', onnxruntime.__version__); print('onnx', onnx.__version__); print('numpy', numpy.__version__); print('opencv', cv2.__version__); print('psutil', psutil.__version__)"
    echo "--- onnx files";   for m in "${models[@]}"; do sha256sum "$m"; done
} > "$OUT/system_info.txt" 2>&1
echo "[INFO] system info -> $OUT/system_info.txt"

for model in "${models[@]}"; do
    name="$(basename "$model" .onnx)"
    for threads in 1 4; do
        echo "[RUN] $name  threads=$threads"
        cmd=(python3 -u scripts/benchmark_edge_device.py --model "$model" --threads "$threads"
             --runs 100 --warmup 10 --out "$OUT/${name}_t${threads}.json")
        if [ -n "$IMAGES" ]; then cmd+=(--images "$IMAGES"); fi
        "${cmd[@]}" | tee "$OUT/${name}_t${threads}.log"
        echo "$name t$threads $(vcgencmd measure_temp 2>/dev/null) $(vcgencmd get_throttled 2>/dev/null)" >> "$OUT/thermal_log.txt"
        sleep "$COOLDOWN"
    done
    echo "[PROFILE] $name"
    # A failed profile (e.g. out of memory) must not abort the remaining models
    python3 -u scripts/profile_onnx_operators.py --model "$model" --threads 4 --out-dir "$OUT/profiles" \
        | tee "$OUT/profiles/${name}.log" || echo "[WARN] profiling failed for $name; see $OUT/profiles/${name}.log"
    rm -f "$OUT/profiles/${name}_ort_optimized.onnx"   # large and reproducible; keep only the counts
    sleep "$COOLDOWN"
done

echo "[DONE] results in $OUT"
echo "       commit them with: git add -f $OUT && git commit -m 'Pi 4 edge measurements'"
