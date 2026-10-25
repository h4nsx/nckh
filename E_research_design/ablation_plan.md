# Ablation Study Plan

## Purpose

Ablation experiments estimate the contribution of individual design choices. They must be controlled, interpretable, and tied to research questions.

## 1. Core ablations

| ID | Matching Case | Ablation Question | Comparison | What It Isolates |
|:---|:---|:---|:---|:---|
| **ABL-01** | **Case B vs Case A** | Effect of replacing C2PSA with SS2D at $P_5$ | Unmodified YOLO11s (C2PSA) vs YOLO11s-SS2D ($P_5$) | True contribution of $P_5$ selective scan context |
| **ABL-02** | **Case C vs Case B** | Single vs dual SSM feature stages | SS2D at $P_5$ only vs SS2D at $P_4 + P_5$ | Trade-off of adding Mamba at stride 16 |
| **ABL-03** | **Case D vs Case B** | Replace vs supplement attention | SS2D alone at $P_5$ vs C2PSA + SS2D combined | Whether attention and SSM are complementary |
| **ABL-04** | **Case E vs Case B/C** | Feature neck modulation | Standard FPN/PAN neck vs Context-Gated Neck | Impact of $P_5 \to P_3$ context gating on small objects |
| **ABL-05** | **Edge Quant** | Precision and engine optimization | FP32 PyTorch vs FP16/INT8 TensorRT | Edge acceleration and quantization degradation |

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
