#!/usr/bin/env bash
# Hardware counters (perf stat) for the baseline, the unrolled SS2D-P5 scan, and the parallel scan
# on the Raspberry Pi 4, 4 and 1 threads; see scripts/perf_counters_pi.py.
#
# Once per boot (needs the sudo password):
#   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#   sudo sysctl kernel.perf_event_paranoid=1
# Usage (inside the Pi venv, from ~/edge_benchmark):
#   nohup bash scripts/run_perf_pi.sh ~/onnx_caseb > results/edge/perf_run.log 2>&1 &
set -euo pipefail

exec 9> /tmp/run_perf_pi.lock
flock -n 9 || { echo "[ERROR] another run_perf_pi.sh is already running"; exit 1; }

ONNX_DIR="${1:?usage: run_perf_pi.sh <onnx_dir>}"
MODELS=(yolo11s_baseline yolo11s_ss2d_arm3 yolo11s_ss2d_arm3_par)

# perf binary: Debian's /usr/bin/perf wrapper refuses kernels it was not built for; fall back to perf_X.Y
PERF=""
if perf --version >/dev/null 2>&1; then PERF=perf; else
    for p in /usr/bin/perf_*; do [ -x "$p" ] && "$p" --version >/dev/null 2>&1 && PERF="$p"; done
fi
[ -n "$PERF" ] || { echo "[ERROR] no working perf binary (sudo apt install linux-perf)"; exit 1; }
echo "perf: $PERF ($("$PERF" --version))"

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ "$(cat "$g")" = "performance" ] || { echo "[ERROR] $g is not 'performance'"; exit 1; }
done
[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 1 ] || {
    echo "[ERROR] perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid); run: sudo sysctl kernel.perf_event_paranoid=1"; exit 1; }
for m in "${MODELS[@]}"; do [ -f "$ONNX_DIR/$m.onnx" ] || { echo "[ERROR] missing $ONNX_DIR/$m.onnx"; exit 1; }; done

# Smoke test: can perf count the raw PMU events at all?
"$PERF" stat -x , -e r11,r08,r17 -- sleep 0.2 2>&1 | head -5

OUT="results/edge/perf_$(hostname)_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
{
    echo "date: $(date -Iseconds)"; uname -a; cat /proc/device-tree/model 2>/dev/null; echo
    echo "perf: $("$PERF" --version)"; echo "paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    vcgencmd get_throttled 2>/dev/null || true
    python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
    for m in "${MODELS[@]}"; do sha256sum "$ONNX_DIR/$m.onnx"; done
} > "$OUT/system_info.txt" 2>&1

( while true; do echo "$(date +%T) $(vcgencmd measure_clock arm) $(vcgencmd measure_temp)"; sleep 5; done ) \
    > "$OUT/clock.log" 2>&1 &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT

wait_cool() { while [ "$(vcgencmd measure_temp | grep -oE '[0-9]+' | head -1)" -ge 55 ]; do sleep 10; done; }

for t in 4 1; do
    for m in "${MODELS[@]}"; do
        wait_cool
        echo "[RUN] $m threads=$t  $(date +%T)"
        python3 -u scripts/perf_counters_pi.py --model "$ONNX_DIR/$m.onnx" --threads "$t" \
            --runs 10 --warmup 3 --perf "$PERF" --out "$OUT/${m}_t${t}.json" | tee "$OUT/${m}_t${t}.log"
    done
done
echo "[DONE] results in $OUT"
echo "ALL_DONE"
