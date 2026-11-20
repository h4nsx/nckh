# Effective memory of the trained P5 selective scan (2026-11-19)

Script: `scripts/measure_scan_memory.py <repo> <best.pt> ...` (CPU, PyTorch 2.14.1, Ultralytics 8.4.185),
on every 10th validation frame of `data/smartspaces_micro/images/val` (10 frames, one batched forward pass).
For each direction k, token t, channel and state, the per-step decay is Abar = exp(-Delta * |A|) and the number of
tokens until a state entry falls to 1/e is 1 / (Delta * |A|).

| Checkpoint | median Abar | memory (tokens) median / p90 / p99 | fraction >= 20 tokens | fraction >= 400 tokens |
|---|---|---|---|---|
| Arm 2 `best.pt` | 0.521 | 1.53 / 5.64 / 16.05 | 0.0052 | 0 |
| Arm 3 `best.pt` | 0.523 | 1.55 / 5.71 / 16.32 | 0.0053 | 0 |

Quantiles are computed on a random subsample of 2,000,000 entries. Used in the LNCS paper, Section 3.2.

Also checked (same day): projecting each of the four permuted sequences with the shared W_proj equals projecting
once and permuting the result (max difference 3.1e-15 in float64); per-direction cost 117.96 M MACs, once 29.49 M.
