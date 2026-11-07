# Experiment Runbook – Priority 1 (TAIMMA 2027 revision)

The text of the paper is now aligned with what has actually been run. Moving the
recommendation above *Borderline* requires the experiments below; no amount of
rewriting can substitute for them. All new numbers must come from these scripts,
not from manual entry.

## 0. Sanity checks (minutes, CPU)

```bash
python -m pytest tests/test_zero_gamma_identity.py tests/test_chunked_scan_equivalence.py -q
```

`test_chunked_scan_equivalence.py` checks that the chunked scan reproduces the
sequential reference scan (forward and gradients, double precision). If it fails,
do **not** use `--scan-chunk`; train with the sequential scan instead.

## 1. Held-out test facility (data)

Add a third SmartSpaces warehouse as `test:` in the data YAML (same 0.5 FPS
sampling, disjoint cameras). Commit the frame list (`splits/test.txt`) and its
SHA-256. Do not look at test results until all arms and seeds are finished.

## 2. Multi-seed ablation (Experiments A + C)

```bash
python scripts/run_ablation_study.py --arms baseline identity convonly ss2d ss2d_2dir ss2d_2stage \
    --seeds 0 1 2 3 4 5 --epochs 20 --batch 8 --device 0 --eval-split test --scan-chunk 20
python scripts/aggregate_seeds.py --source best
python scripts/aggregate_seeds.py --source final
```

* One JSON per (arm, seed) lands in `results/multiseed/`; commit them.
* `aggregate_seeds.py` writes `paper/tables/tab_multiseed.tex` and prints paired
  differences vs. the baseline (t-interval, paired t-test, exact sign-flip test,
  Holm-corrected). With 6 seeds the smallest sign-flip p-value is 2/64 = 0.031.
* Record the GPU model, batch size and wall-clock time of every run in the paper.

What each comparison answers:

| Comparison | Question |
|---|---|
| ss2d vs. baseline | Does the replacement change accuracy beyond seed variance? |
| identity vs. baseline | Does simply removing C2PSA change accuracy? |
| ss2d vs. identity / convonly | Does the selective scan itself contribute? |
| ss2d vs. ss2d_2dir | Does column-wise scanning help on rack-structured scenes? |
| ss2d_2stage vs. ss2d | Is two-stage training still useful with zero-gamma? |

## 3. Edge re-measurement (Experiment D)

1. Export each trained `best.pt` to ONNX twice: sequential scan and chunked scan
   (`PurePyTorch2DSelectiveScan.default_chunk_size = 20` before loading).
2. Run `scripts/profile_onnx_operators.py` on each file **on the Raspberry Pi 4**
   and commit the printed node counts plus the ORT profile JSON.
3. Re-run `scripts/benchmark_edge_device.py` on the Pi for baseline, identity,
   ss2d-sequential and ss2d-chunked, once with `--threads 1` and once with
   `--threads 4`, always with `--images <val or test frames> --out results/edge/<name>.json`.
   The revised script forces the CPU execution provider and records forward-only and
   end-to-end latency (preprocess / forward / NMS), idle RSS, RSS after the loop, and the
   true peak RSS (VmHWM), all in MiB. The paper's current Table VII predates this script
   (its "RSS" is the value after the timed loop, not the peak), so replace it entirely.
   Also record the Pi RAM size, OS image, CPU governor, cooling, and `vcgencmd get_throttled`.

The chunked run is the controlled intervention that tests the claim that graph
unrolling, not SSM arithmetic, causes the latency and memory overhead.

## 4. Paper updates after the runs

* Replace Table IV/V numbers with `tab_multiseed.tex` (keep the single-seed tables
  only as historical context, or drop them).
* Rewrite Limitations items (i), (iv), (v), (vi) to reflect what was done.
* Redraw Fig. 1 (BN after out_proj, zero-gamma, no "vectorized").
