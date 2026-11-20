#!/usr/bin/env bash
# Final single-session Raspberry Pi run (session 4): every SS2D-P5 graph form and the baseline in one
# session with the CPU governor locked to "performance" (no ondemand down-clocking), plus ONNX Runtime
# session-creation time for each model.
#
# Models (trained weights): Arm 1 baseline, Arm 3 unrolled scan (as in the paper), Arm 3 with the shared
# W_proj computed once (export_onnx.py --hoist-proj; outputs bit-identical), Arm 3 parallel scan.
# Protocol as in the paper: 1 thread 10 warmup + 100 timed calls, 4 threads 5 warmup + 30 timed calls, each
# run started below 55 C, clock and temperature logged every 5 s.
#
# Before running (once per boot; needs the sudo password):
#   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Usage (inside the Pi venv, from ~/edge_benchmark; all .onnx files in one folder):
#   nohup bash scripts/run_final_pi.sh ~/onnx_caseb > results/edge/final_run.log 2>&1 &
# Optional: CASEC=~/onnx_rest/yolo11s_case_c.onnx also times one session creation of SS2D-P4P5 (slow).
# Results: results/edge/final_<host>_<date>/
set -euo pipefail

exec 9> /tmp/run_final_pi.lock
flock -n 9 || { echo "[ERROR] another run_final_pi.sh is already running"; exit 1; }

ONNX_DIR="${1:?usage: run_final_pi.sh <onnx_dir>}"
MODELS=(yolo11s_baseline yolo11s_ss2d_arm3 yolo11s_ss2d_arm3_hoist yolo11s_ss2d_arm3_par)

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ "$(cat "$g")" != "performance" ]; then
        echo "[ERROR] $g is '$(cat "$g")'; run: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
        exit 1
    fi
done
for m in "${MODELS[@]}"; do
    [ -f "$ONNX_DIR/$m.onnx" ] || { echo "[ERROR] missing $ONNX_DIR/$m.onnx"; exit 1; }
done

OUT="results/edge/final_$(hostname)_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
{
    echo "date: $(date -Iseconds)"; uname -a; cat /proc/device-tree/model 2>/dev/null; echo
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    vcgencmd get_throttled 2>/dev/null || true
    python3 --version
    python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
    for m in "${MODELS[@]}"; do sha256sum "$ONNX_DIR/$m.onnx"; done
    [ -n "${CASEC:-}" ] && sha256sum "$CASEC"
} > "$OUT/system_info.txt" 2>&1

( while true; do echo "$(date +%T) $(vcgencmd measure_clock arm) $(vcgencmd measure_temp)"; sleep 5; done ) \
    > "$OUT/clock.log" 2>&1 &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT

wait_cool() {
    while [ "$(vcgencmd measure_temp | grep -oE '[0-9]+' | head -1)" -ge 55 ]; do sleep 10; done
}

run() {  # model_path tag threads runs warmup
    wait_cool
    echo "[RUN] $2 threads=$3 runs=$4  $(date +%T)"
    python3 -u scripts/benchmark_edge_device.py --model "$1" --threads "$3" \
        --runs "$4" --warmup "$5" --out "$OUT/$2_t$3.json" | tee "$OUT/$2_t$3.log"
}

for m in "${MODELS[@]}"; do run "$ONNX_DIR/$m.onnx" "$m" 1 100 10; done
for m in "${MODELS[@]}"; do run "$ONNX_DIR/$m.onnx" "$m" 4 30 5; done

if [ -n "${CASEC:-}" ]; then
    run "$CASEC" yolo11s_case_c_session 4 2 0
fi

echo "[DONE] results in $OUT"
echo "ALL_DONE"
