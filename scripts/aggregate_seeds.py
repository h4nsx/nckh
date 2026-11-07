#!/usr/bin/env python3
"""
Seed-Level Aggregation and Paired Comparison (TAIMMA 2027 revision, Experiment A)
---------------------------------------------------------------------------------
Reads results/multiseed/*.json written by scripts/run_ablation_study.py and reports,
for each arm, mean +- sample std and a 95% t-interval over seeds, plus a paired
comparison against the baseline on matching seeds:
  - mean difference with a 95% t-interval
  - paired t-test p-value (needs scipy; omitted otherwise)
  - exact sign-flip permutation p-value (distribution-free)
Paired p-values are Holm-corrected across the compared arms.

Note on power: with n seeds the smallest attainable sign-flip p-value is 2 / 2**n,
so n >= 6 seeds are needed for it to reach 0.05.

Writes a Markdown summary to stdout and a LaTeX table to paper/tables/tab_multiseed.tex.
"""

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "results" / "multiseed"
OUT_TEX = ROOT / "paper" / "tables" / "tab_multiseed.tex"

# Two-sided 95% Student-t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
       16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}

METRICS = [("mAP50", "mAP$_{50}$"), ("mAP50_95", "mAP$_{50-95}$"), ("recall", "Recall"), ("f1", "F1")]


def t_crit(df):
    if df in T95:
        return T95[df]
    keys = [k for k in T95 if k <= df]
    return T95[max(keys)] if keys else float("nan")


def mean_ci(values):
    n = len(values)
    m = statistics.fmean(values)
    if n < 2:
        return m, float("nan"), float("nan")
    s = statistics.stdev(values)
    return m, s, t_crit(n - 1) * s / math.sqrt(n)


def sign_flip_p(diffs):
    observed = abs(statistics.fmean(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if abs(statistics.fmean(d * s for d, s in zip(diffs, signs))) >= observed - 1e-12:
            hits += 1
    return hits / total


def paired_t_p(diffs):
    try:
        from scipy import stats
    except ImportError:
        return None
    if len(diffs) < 2 or statistics.stdev(diffs) == 0:
        return None
    return float(stats.ttest_1samp(diffs, 0.0).pvalue)


def holm(pvals):
    """Holm step-down correction; keeps None entries as None."""
    idx = sorted((i for i, p in enumerate(pvals) if p is not None), key=lambda i: pvals[i])
    adjusted = list(pvals)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (len(idx) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def load(source):
    by_arm = defaultdict(dict)
    for path in sorted(IN_DIR.glob("*.json")):
        rec = json.loads(path.read_text())
        block = rec["best_ckpt_eval"] if source == "best" else rec["final_epoch_val"]
        if "f1" not in block:
            p, r = block["precision"], block["recall"]
            block["f1"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
        by_arm[rec["arm"]][rec["seed"]] = block
    return by_arm


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument("--source", choices=["best", "final"], default="best",
                        help="best = best.pt evaluated on --eval-split; final = last-epoch validation")
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()

    by_arm = load(args.source)
    if args.baseline not in by_arm:
        raise SystemExit(f"No records for baseline arm '{args.baseline}' in {IN_DIR}")

    print(f"# Multi-seed summary (source: {args.source})\n")
    print("| Arm | n | " + " | ".join(label for _, label in METRICS) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    tex_rows = []
    for arm, seeds in by_arm.items():
        cells, tex_cells = [], []
        for key, _ in METRICS:
            m, s, h = mean_ci([100 * seeds[k][key] for k in sorted(seeds)])
            cells.append(f"{m:.2f} ± {s:.2f} (±{h:.2f})")
            tex_cells.append(f"${m:.2f} \\pm {s:.2f}$")
        print(f"| {arm} | {len(seeds)} | " + " | ".join(cells) + " |")
        tex_rows.append(f"{arm.replace('_', chr(92) + '_')} & {len(seeds)} & " + " & ".join(tex_cells) + " \\\\")

    print("\nCells: mean ± sample std (± half-width of the 95% t-interval), in %.\n")
    print(f"## Paired difference vs. {args.baseline} (matching seeds, mAP50-95)\n")
    print("| Arm | n | Δ mean (95% CI) | paired t p (Holm) | sign-flip p (Holm) |")
    print("|---|---|---|---|---|")
    base = by_arm[args.baseline]
    arms, diffs_all = [], []
    for arm, seeds in by_arm.items():
        if arm == args.baseline:
            continue
        common = sorted(set(seeds) & set(base))
        if not common:
            continue
        arms.append((arm, common))
        diffs_all.append([100 * (seeds[k]["mAP50_95"] - base[k]["mAP50_95"]) for k in common])
    t_adj = holm([paired_t_p(d) for d in diffs_all])
    f_adj = holm([sign_flip_p(d) for d in diffs_all])
    for (arm, common), d, tp, fp in zip(arms, diffs_all, t_adj, f_adj):
        m, _, h = mean_ci(d)
        tp_s = "n/a" if tp is None else f"{tp:.3f}"
        print(f"| {arm} | {len(common)} | {m:+.2f} (±{h:.2f}) | {tp_s} | {fp:.3f} |")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    header = " & ".join(f"\\textbf{{{label}}}" for _, label in METRICS)
    OUT_TEX.write_text("\n".join([
        "% Auto-generated by scripts/aggregate_seeds.py - DO NOT EDIT MANUALLY",
        "\\begin{table}[!t]",
        f"\\caption{{Multi-Seed Results (mean $\\pm$ sample std over seeds, \\%, source: {args.source} checkpoints)}}",
        "\\label{tab:multiseed}",
        "\\centering",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}",
        "\\toprule",
        f"\\textbf{{Arm}} & \\textbf{{Seeds}} & {header} \\\\",
        "\\midrule",
        *tex_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ]))
    print(f"\n[SAVED] {OUT_TEX}")


if __name__ == "__main__":
    main()
