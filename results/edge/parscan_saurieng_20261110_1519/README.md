# Raspberry Pi 4, session 3 (2026-11-10, 15:19-16:11): parallel associative scan

Question: does removing the 400-step dependency chain of the scan (same weights, same function) lower latency?

Models (SHA-256 and node counts in `system_info.txt`), all four-class `[1, 8, 8400]`:

| File | What | Nodes |
|---|---|---|
| `yolo11s_baseline.onnx` (`ca47c3d8...`) | Arm 1 `best.onnx` | 320 |
| `yolo11s_ss2d_arm3.onnx` (`677274c7...`) | Arm 3 `best.onnx`, unrolled sequential scan (as in the paper) | 14,771 |
| `yolo11s_ss2d_arm3_par.onnx` (`8c2a2fa4...`) | Arm 3 `best.pt` exported with `scripts/export_onnx.py --chunk -2` (Hillis-Steele parallel scan, 9 steps for L = 400) | 619 (541 after ORT optimization) |

Equivalence: `scripts/compare_onnx_outputs.py` (x86 host, 3 random inputs) gives a worst relative output difference
of 1.9e-7 between the sequential and parallel Arm 3 exports; `tests/test_chunked_scan_equivalence.py` checks the
parallel scan against the sequential recurrence in float64 (forward 2.8e-14, gradients allclose 1e-7).

Script: `scripts/run_parscan_pi.sh`. Protocol as in the paper: 1 thread 10 warmup + 100 timed calls, 4 threads
5 warmup + 30 timed calls, each run started below 55 C, clock and temperature every 5 s (`clock.log`, max 75.9 C).
Console log: `../pi_parscan_20261110.log`. Used in the paper: lower block of Table 7 (LNCS version).

| Model | Threads | Median (ms) | Mean +- SD (ms) | Peak RSS (MiB) | Rel. to baseline | Rel. to sequential |
|---|---|---|---|---|---|---|
| Baseline | 1 | 3347.87 | 3347.46 +- 6.1 | 233.2 | 1.00 | |
| Arm 3 sequential | 1 | 3449.01 | 3565.69 +- 504.7 * | 373.1 | 1.03 | 1.00 |
| Arm 3 parallel | 1 | 4072.93 | 4073.62 +- 2.4 | 389.0 | 1.22 | 1.18 |
| Baseline | 4 | 1111.82 | 1114.71 +- 8.9 | 233.3 | 1.00 | |
| Arm 3 sequential | 4 | 1327.94 | 1329.63 +- 4.6 | 373.6 | 1.19 | 1.00 |
| Arm 3 parallel | 4 | 2103.98 | 2105.57 +- 5.5 | 389.4 | 1.89 | 1.58 |

One-to-four-thread speed-up: baseline 3.01x, sequential 2.60x, parallel 1.94x.

\* The sequential single-thread run is noisy (P95 4232 ms): one clock sample inside the run (15:36:34) shows
600 MHz at 62.8 C, i.e. not thermal; all other samples inside timed runs are 1.5 GHz. Its median (3449 ms) is 2.2%
above the session-2 value (3374.10 ms), and the single-thread baseline is 4.9% above session 2 (3192.88 ms), so the
paper keeps the session-2 numbers for these two models; this session is used only for the parallel-scan comparison.
The other five runs have no low-clock sample.

Operator profile of the parallel model (4 threads, 5 profiled calls; `profiles/yolo11s_ss2d_arm3_par.log`):
Conv 4522 ms (41.1%, ~904 ms per call, as for the other models), Mul 2270 ms, Add 1117 ms, Pad 885 ms, Slice 820 ms
(together ~1018 ms per call). The raw ORT trace and the per-operator JSON were not copied from the Pi.

Interpretation (not tested with hardware counters): the parallel scan removes the dependency chain but performs
O(L log L) element-wise work on [1, 400, 256, 16] state tensors (~6.5 MB each) per direction and step, so it moves far
more data than the sequential scan; the lower thread speed-up is consistent with a memory-bandwidth limit.
