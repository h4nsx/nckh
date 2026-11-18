# Raspberry Pi 4, session 2 (2026-11-10): G1 runtime intervention

Untrained exports (seed 0, BN statistics calibrated on random inputs, gamma = 1), produced by
`scripts/export_onnx.py --untrained` with `--chunk 0` (unrolled), `-1` (one ONNX Loop per direction) and `4`
(chunked). The three SS2D forms were checked for equal outputs with `scripts/compare_onnx_outputs.py`
(max relative difference 2.3e-6; SS2D vs conv-only negative control 0.154). SHA-256 in `system_info.txt`.
Script: `scripts/run_g1_pi.sh`, 1 and 4 threads, 3 warmup + 20 timed calls, each run started below 55 C;
`clock.log`: 1.5 GHz whenever a model was running. Used in the paper: Table X.
