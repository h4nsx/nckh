# Session 4: final single-session Pi run (19-20 Nov 2026)

Raspberry Pi 4 Model B (Cortex-A72, 1.5 GHz, no heatsink or fan, official 5 V / 3 A PSU), ONNX Runtime 1.31.0,
Python 3.13.5, CPU governor `performance`. Script: `scripts/run_final_pi.sh ~/onnx_caseb`.
Protocol: 1 thread 10 warmup + 100 timed calls; 4 threads 5 warmup + 30 timed calls; each run started below 55 C.
Clock: 1746 samples every 5 s, all at 1.500 GHz; maximum 75 C. `vcgencmd get_throttled` = 0x0 at the start.
ONNX SHA-256 in `system_info.txt` (baseline ca47c3d8, Arm 3 unrolled 677274c7, Arm 3 W_proj once 7dffd804,
Arm 3 parallel scan 8c2a2fa4).

| Model | Nodes | 1 thread median (SD) ms | Rel. | 4 threads median (SD) ms | Rel. | 1->4 speed-up | Session creation 1t / 4t (s) | Peak RSS 4t (MiB) |
|---|---|---|---|---|---|---|---|---|
| Baseline (C2PSA) | 320 | 3204.13 (4.2) | 1.00 | 1171.80 (22.2) | 1.00 | 2.73x | 0.53 / 0.26 | 233.4 |
| SS2D-P5 unrolled (Arm 3) | 14,771 | 3396.60 (3.4) | 1.060 | 1370.85 (25.5) | 1.170 | 2.48x | 46.31 / 47.29 | 373.6 |
| SS2D-P5, W_proj once | 14,773 | 3406.96 (3.0) | 1.063 | 1407.45 (25.9) | 1.201 | 2.42x | 43.48 / 44.72 | 374.1 |
| SS2D-P5, parallel scan | 619 | 4089.56 (5.5) | 1.276 | 2253.62 (17.1) | 1.923 | 1.81x | 0.34 / 0.34 | 389.6 |

Within session: parallel scan vs unrolled +20.4% (1t), +64.4% (4t); W_proj once vs unrolled +0.3% (1t), +2.7% (4t).
Against session 2 (same baseline and Arm 3 files): baseline +0.35% (1t), +4.9% (4t); Arm 3 +0.7% (1t), +4.3% (4t);
the ratios (1.06 / 1.17 here, 1.06 / 1.18 in session 2) agree.
