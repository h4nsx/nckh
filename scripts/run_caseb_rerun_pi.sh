#!/usr/bin/env bash
# Re-measure Case B on the Raspberry Pi with the trained four-class Arm 3 export, plus the Arm 1
# baseline in the same session as a control. Same protocol as the paper: 1 thread (10 warmup + 100
# timed) and 4 threads (5 warmup + 30 timed), each run started below 55 C, ARM clock logged every 5 s.
#
# Usage (inside the Pi venv, from ~/edge_benchmark):
#   bash scripts/run_caseb_rerun_pi.sh ~/onnx_caseb
# Results: results/edge/caseb_<host>_<date>/
set -euo pipefail

ONNX_DIR="${1:?usage: run_caseb_rerun_pi.sh <onnx_dir>}"
OUT="results/edge/caseb_$(hostname)_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT/profiles"
MODELS=(yolo11s_baseline yolo11s_ss2d_arm3)

{
    echo "date: $(date -Iseconds)"; uname -a; cat /proc/device-tree/model 2>/dev/null; echo
    vcgencmd get_throttled 2>/dev/null || true
    python3 --version
    python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
    for m in "${MODELS[@]}"; do sha256sum "$ONNX_DIR/$m.onnx"; done
} > "$OUT/system_info.txt" 2>&1

( while true; do echo "$(date +%T) $(vcgencmd measure_clock arm) $(vcgencmd measure_temp)"; sleep 5; done ) \
    > "$OUT/clock.log" 2>&1 &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT

wait_cool() {
    while [ "$(vcgencmd measure_temp | grep -oE '[0-9]+' | head -1)" -ge 55 ]; do sleep 10; done
}

run() {  # model threads runs warmup
    wait_cool
    echo "[RUN] $1 threads=$2 runs=$3  $(date +%T)"
    python3 -u scripts/benchmark_edge_device.py --model "$ONNX_DIR/$1.onnx" --threads "$2" \
        --runs "$3" --warmup "$4" --out "$OUT/$1_t$2.json" | tee "$OUT/$1_t$2.log"
}

for m in "${MODELS[@]}"; do run "$m" 1 100 10; done
for m in "${MODELS[@]}"; do run "$m" 4 30 5; done

wait_cool
echo "[PROFILE] yolo11s_ss2d_arm3  $(date +%T)"
python3 -u scripts/profile_onnx_operators.py --model "$ONNX_DIR/yolo11s_ss2d_arm3.onnx" --threads 4 --out-dir "$OUT/profiles" \
    | tee "$OUT/profiles/yolo11s_ss2d_arm3.log"

echo "[DONE] results in $OUT"
