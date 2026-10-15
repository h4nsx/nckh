# Ablation Study Plan

## Purpose

Ablation experiments estimate the contribution of individual design choices. They must be controlled, interpretable, and tied to research questions.

## 1. Core ablations

| ID | Ablation | Comparison | What it tests |
|---|---|---|---|
| ABL-01 | Remove Mamba | Hybrid vs otherwise matched CNN-only model | Whether Mamba contributes |
| ABL-02 | Change Mamba placement | Backbone vs neck vs low-resolution stage, if feasible | Where contextual modeling is useful |
| ABL-03 | Change fusion | Selected fusion alternatives | Whether the fusion mechanism matters |
| ABL-04 | Remove auxiliary data | SmartSpaces-only vs auxiliary strategy | Whether auxiliary data contributes |
| ABL-05 | Remove compression | Original vs compressed model | Compression impact |
| ABL-06 | Input resolution | Two or more justified resolutions | Accuracy/latency trade-off |
| ABL-07 | Model scale | Small set of sizes if architecture supports it | Scaling behavior |

Only run ablations relevant to the final architecture. If a component is not included in the final method, its ablation is not needed.

## 2. Keep comparisons fair

For each ablation, specify:
- one variable that changes;
- all factors that remain constant;
- training budget and checkpoint-selection rule;
- seed policy;
- accuracy and efficiency metrics;
- expected interpretation;
- limitations.

If several variables change at once, the result cannot be attributed cleanly to one component.

## 3. Example ablation table

| Variant | CNN backbone | Mamba block | Fusion | Auxiliary data | Compression | mAP@0.5:0.95 | Params | FLOPs/MACs | Latency |
|---|---|---|---|---|---|---|---|---|---|
| V0 | Yes | No | Baseline | No | No | TBD | TBD | TBD | TBD |
| V1 | Yes | Yes | Candidate | No | No | TBD | TBD | TBD | TBD |
| V2 | Yes | Yes | Alternative | No | No | TBD | TBD | TBD | TBD |
| V3 | Yes | Yes | Candidate | Yes | No | TBD | TBD | TBD | TBD |
| V4 | Yes | Yes | Candidate | No | Yes | TBD | TBD | TBD | TBD |

## 4. Reporting guidance

- Report absolute values and differences, not only percentage improvements.
- Use consistent units and explain the denominator for relative changes.
- Do not hide ablations that reduce accuracy.
- Discuss whether gains justify added complexity.
- If multiple seeds are feasible, report mean and standard deviation.
- Avoid strong causal claims when the setup changes multiple factors or has high variance.

## 5. Interpretation template

> Compared with [control variant], [ablation variant] changed [metric] from [measured value] to [measured value] under [controlled conditions]. This suggests [limited interpretation]. The result does not establish [important limitation].

Use only after measured results are available.
