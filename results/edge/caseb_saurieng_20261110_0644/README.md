# Raspberry Pi 4, session 2 (2026-11-10): baseline and trained Case B

Models (SHA-256 in `system_info.txt`): `yolo11s_baseline.onnx` = Arm 1 `best.onnx` (`ca47c3d8...`),
`yolo11s_ss2d_arm3.onnx` = Arm 3 `best.onnx` (`677274c7...`), both four-class `[1, 8, 8400]`.
Script: `scripts/run_caseb_rerun_pi.sh`. 1 thread: 10 warmup + 100 timed; 4 threads: 5 warmup + 30 timed; each run
started below 55 C; `clock.log` samples clock and temperature every 5 s: 1.5 GHz whenever a model was running,
600-700 MHz only while idle (cooling waits), maximum 76.9 C over the session.
`throttled=0xe0000` in `system_info.txt` is the since-boot history (no under-voltage, no active limit at start).
The raw ORT trace (~104 MB) and the ORT-optimized graph (~40 MB) were not committed; they are in the
archive `pi_rerun_results.tgz` kept by the authors. Combined console log: `../pi_rerun_20261110.log`.
An earlier start of this session (two copies launched at once by mistake) was aborted and discarded on the Pi.
Used in the paper: Table VIII (session 2 rows) and the Case B row of Table IX.
