# Raspberry Pi 4 measurements (host `saurieng`)

Raspberry Pi 4 Model B Rev 1.2, 4 GB, Debian 13, kernel 6.18.50, Python 3.13.5, ONNX Runtime 1.31.0
(see `system_info_run1.txt`; `system_info.txt` was rewritten by the second session with the same environment).
Scripts: `scripts/benchmark_edge_device.py`, `scripts/profile_onnx_operators.py`, `scripts/run_pi_benchmarks.sh`.

| Files | Used in the paper | Notes |
|---|---|---|
| `yolo11s_*_t1.json` | Yes (Table VIII, 1 thread) | 10 warmup + 100 timed calls; SoC 64-67 °C at the end of each run (`thermal_log.txt`); clock not logged |
| `rerun_t4_n30/yolo11s_*_t4.json` | Yes (Table VIII, 4 threads) | 5 warmup + 30 timed calls, each started below 55 °C; ARM clock 1.5 GHz throughout, max 64.7 °C (`logs/clock_t4b.log`) |
| `yolo11s_*_t4.json` | No | 100 timed calls, 4 threads; SoC reached 79-80 °C and the soft-temperature-limit flag was set (`thermal_log.txt`, `throttled=0x80000`) |
| `rerun_t4_cooled/*.json` | No | Baseline and Case B only, 100 calls (despite the folder name, no fan); clock dropped to 1.23-1.48 GHz during the Case B run at 81-84 °C (`logs/clock_t4.log`) |
| `profiles/*_t4_profile.json` | Yes (Table IX) | 2 warmup + 5 profiled calls, 4 threads; clock not logged. Raw ORT traces stay on the Pi (up to ~1 GB) |

The first profiler version loaded the whole ORT trace with `json.load` and was killed by the OOM killer on
Case C (3.06 GB anonymous RSS); the profiler now parses the trace line by line. The baseline profile was
re-run with the new profiler so all three profiles use the same method.

`saurieng_20261107_2109/` (only on the Pi) is an aborted start of the first session and contains no results.
