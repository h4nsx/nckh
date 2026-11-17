#!/usr/bin/env bash
# G1 runtime intervention on the Raspberry Pi: same SS2D block exported in different graph
# forms (unrolled sequential, one ONNX Loop, chunked), plus identity / conv-only / baseline.
# Untrained weights are fine: latency, node count, and memory do not depend on weight values.
#
# Usage (inside the Pi venv, from ~/edge_benchmark):
#   bash scripts/run_g1_pi.sh ~/onnx_g1
# Runs 1 and 4 threads, 3 warmup + N timed calls (default 20), each run started below 55 C,
# while logging the ARM clock and temperature every 5 s. Results: results/edge/g1_<host>_<date>/
set -euo pipefail

ONNX_DIR="${1:?usage: run_g1_pi.sh <onnx_dir>}"
N="${N:-20}"
OUT="results/edge/g1_$(hostname)_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
MODELS=(g1_baseline g1_identity g1_convonly g1_ss2d_seq g1_ss2d_loop g1_ss2d_chunk4)

{
    echo "date: $(date -Iseconds)"; uname -a; cat /proc/device-tree/model 2>/dev/null; echo
    vcgencmd get_throttled 2>/dev/null || true
    python3 --version
    python3 -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
    for m in "${MODELS[@]}"; do sha256sum "$ONNX_DIR/$m.onnx"; done
} > "$OUT/system_info.txt" 2>&1

# Clock/temperature log for the whole session (stopped on exit)
( while true; do echo "$(date +%T) $(vcgencmd measure_clock arm) $(vcgencmd measure_temp)"; sleep 5; done ) \
    > "$OUT/clock.log" 2>&1 &
CLOCK_PID=$!
trap 'kill $CLOCK_PID 2>/dev/null || true' EXIT

wait_cool() {
    while [ "$(vcgencmd measure_temp | grep -oE '[0-9]+' | head -1)" -ge 55 ]; do sleep 10; done
}

for threads in 1 4; do
    for m in "${MODELS[@]}"; do
        wait_cool
        echo "[RUN] $m threads=$threads  $(date +%T)"
        python3 -u scripts/benchmark_edge_device.py --model "$ONNX_DIR/$m.onnx" --threads "$threads" \
            --runs "$N" --warmup 3 --out "$OUT/${m}_t${threads}.json" | tee "$OUT/${m}_t${threads}.log"
    done
done

echo "[DONE] results in $OUT"
