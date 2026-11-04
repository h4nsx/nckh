# BASELINE MANIFEST (Stage 0)

Created: 2026-11-04
Git Commit: 8d5e699

## Critical File Hashes (SHA-256)

| File Path | SHA-256 Checksum | Description |
|---|---|---|
| `models/yolo11s_ss2d_p5.yaml` | `834c9dfd58c4e97658a4df542847e90f82433e8f4a2ee57de1a04056a7a6bc6d` | Model configuration replacing P5 C2PSA |
| `models/warehouse_ss2d.py` | `9bceeacc864a19835033fd772921bc1a52cf10d26afeebd5942eb1dd2bea6440` | WarehouseSS2DBlock & PurePyTorch2DSelectiveScan |
| `configs/smartspaces_micro.yaml` | `2d9685cb3157a06fde0dd489cce106e30b32cbb2d57628084884d26da91650e0` | Dataset split & class taxonomy configuration |
| `configs/experiment_settings.yaml` | `8900c60fc3177e4d5a0f26de0d9c934f0dd5e4269a439e3bbadd0e3a0ed093b6` | Hyperparameter & training configuration |
| `yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | Official baseline checkpoint |
| `results/tables/arm1_baseline_results.csv` | `b552da5f1bed45dd2374e268e8bf54383c6c4f2d9f6dca30079bb906cba6346f` | Arm 1 baseline convergence log |
| `results/tables/arm2_case_b_single_results.csv` | `341023fa3491119247028cd6dc2810f3b1e09a0d2cd6c0dc2f35a65a3f2a4879` | Arm 2 Case B single-stage log |
| `results/tables/arm3_case_b_two_stage_results.csv` | `90d7435a61408198eabd1c1ea354cd0caf22566cce5b8456a35300dd22d8c482` | Arm 3 Case B two-stage log |
| `results/tables/arm4_case_c_dual_results.csv` | `12454a1280050138f0e3587ebd8be7c52b2248e96ba780951d337db5dde6829a` | Arm 4 Case C dual-stage log |
| `results/tables/gpu_convergence_summary.csv` | `6472f19d6eee7b86fd2e38a55fabc12bb7bbfc55e7227826ed27c80497ac19b4` | Final GPU convergence summary |

