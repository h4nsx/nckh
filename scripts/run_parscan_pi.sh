#!/usr/bin/env bash
# Parallel associative scan on the Raspberry Pi: the trained Arm 3 (Case B) weights exported with the
# scan as a Hillis-Steele parallel scan (export_onnx.py --chunk -2, ~619 nodes) instead of the unrolled
# sequential scan (14,771 nodes). Same function: max relative output difference 1.9e-7 versus the
# measured Arm 3 export (scripts/compare_onnx_outputs.py). The Arm 1 baseline and the sequential Arm 3
# export run again in the same session as controls. Same protocol as the paper: 1 thread
# (10 warmup + 100 timed) and 4 threads (5 warmup + 30 timed), each run started below 55 C, ARM clock
# and temperature logged every 5 s, then an operator profile of the parallel-scan model.
#
# Usage (inside the Pi venv, from ~/edge_benchmark; the three .onnx files in one folder):
#   nohup bash scripts/run_parscan_pi.sh ~/onnx_caseb > results/edge/parscan_run.log 2>&1 &
# A lock file prevents a second copy from starting by accident.
# Results: results/edge/parscan_<host>_<date>/
set -euo pipefail

exec 9> /tmp/run_parscan_pi.lock
flock -n 9 || { echo "[ERROR] another run_parscan_pi.sh is already running"; exit 1; }

ONNX_DIR="${1:?usage: run_parscan_pi.sh <onnx_dir>}"
OUT="results/edge/parscan_$(hostname)_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT/profiles"
MODELS=(yolo11s_baseline yolo11s_ss2d_arm3 yolo11s_ss2d_arm3_par)

for m in "${MODELS[@]}"; do
    [ -f "$ONNX_DIR/$m.onnx" ] || { echo "[ERROR] missing $ONNX_DIR/$m.onnx"; exit 1; }
done

{
    echo "date: $(date -Iseconds)"; uname -a; cat /proc/device-tree/model 2>/dev/null; echo
    vcgencmd get_throttled 2>/dev/null || true
    python3 --version
    python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
    python3 -c "import onnx; print('onnx', onnx.__version__)" 2>/dev/null || true
    for m in "${MODELS[@]}"; do sha256sum "$ONNX_DIR/$m.onnx"; done
    for m in "${MODELS[@]}"; do
        python3 -c "import onnx,sys; print(sys.argv[1], len(onnx.load(sys.argv[1]).graph.node), 'nodes')" \
            "$ONNX_DIR/$m.onnx" 2>/dev/null || true
    done
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
echo "[PROFILE] yolo11s_ss2d_arm3_par  $(date +%T)"
python3 -u scripts/profile_onnx_operators.py --model "$ONNX_DIR/yolo11s_ss2d_arm3_par.onnx" --threads 4 \
    --out-dir "$OUT/profiles" | tee "$OUT/profiles/yolo11s_ss2d_arm3_par.log"

echo "[DONE] results in $OUT"
echo "ALL_DONE"
