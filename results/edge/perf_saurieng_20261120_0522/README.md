# Hardware counters, Raspberry Pi 4 (20 Nov 2026)

`scripts/run_perf_pi.sh` + `scripts/perf_counters_pi.py`; perf 6.18.60 on kernel 6.18.50, ORT 1.31,
governor `performance`, `throttled=0x0` at start. All 2,276 clock samples at 1.5 GHz, max 71.5 °C.
Per call; 10 timed calls per counter group after 3 warm-up calls; session creation excluded.

| Model | Thr | Median ms | Cycles | Instr. | IPC | L2 refills | Est. DRAM traffic | Est. BW |
|---|---|---|---|---|---|---|---|---|
| Baseline | 1 | 3165 | 4.75 G | 4.18 G | 0.88 | 17.5 M | 1383 MiB | 0.43 GiB/s |
| Unrolled (Arm 3) | 1 | 3440 | 5.16 G | 4.59 G | 0.89 | 21.8 M | 1707 MiB | 0.48 GiB/s |
| Parallel scan | 1 | 4288 | 6.43 G | 4.90 G | 0.76 | 54.0 M | 3695 MiB | 0.84 GiB/s |
| Baseline | 4 | 1240 | 7.31 G | 5.96 G | 0.82 | 22.7 M | 1918 MiB | 1.51 GiB/s |
| Unrolled (Arm 3) | 4 | 1376 | 8.12 G | 7.26 G | 0.89 | 25.3 M | 2112 MiB | 1.49 GiB/s |
| Parallel scan | 4 | 2176 | 12.73 G | 9.20 G | 0.72 | 57.8 M | 4123 MiB | 1.83 GiB/s |

One thread, parallel vs unrolled: instructions +7%, cycles +25%, L2 refills 2.5x, DRAM traffic 2.2x.
Unrolled vs baseline: instructions +10%, cycles +9%, IPC unchanged.

Caveats
- STALL_FRONTEND (0x23) and STALL_BACKEND (0x24) read 0: not implemented on the Cortex-A72 (ARMv8.0). Stall fractions are invalid and not used.
- Four-thread counts sum all threads, including ONNX Runtime thread-pool spinning; not compared in the paper.
- DRAM traffic = (L2D_CACHE_REFILL + L2D_CACHE_WB) x 64 B, an estimate.
- The parallel-scan 1-thread run was started manually at 56.4 °C (the <55 °C wait was skipped because the idle Pi stayed at 57-58 °C); clock stayed at 1.5 GHz.
