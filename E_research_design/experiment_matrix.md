# Experiment Matrix

## Purpose

Plan a compact, interpretable experiment set. Add experiments only when they answer a research question or validate a deployment claim.

## 1. Core experiment plan

| ID | Case | Question / Objective | Comparison | Controlled Factors | Primary Measurements | Priority / Status |
|:---|:---|:---|:---|:---|:---|:---|
| **E01** | **Case A** | Establish reference baseline | Unmodified YOLO11s | Input 640x640, data split, hyperparams | mAP50, mAP50-95, Params, GFLOPs, Edge Latency, FPS | **P0 (Prerequisite)** |
| **E02** | **Case B** | Deep-stage SSM integration | Replace C2PSA at $P_5$ with shape-matched SS2D | Same data, neck, head, and training budget | $\Delta$ mAP, $\Delta$ GFLOPs, $\Delta$ Latency, peak VRAM | **P1 (First Milestone)** |
| **E03** | **Case C** | Two-stage SSM integration | SS2D at both $P_4$ and $P_5$ vs Case B | Matched protocol, evaluate cost vs accuracy | AP gains on occluded targets vs edge latency budget | **P2 (Conditional on B)** |
| **E04** | **Case D** | Supplementary vs replacement | Retain C2PSA + Add SS2D at $P_5$ vs Case B | Same backbone topology | Trade-off of dual attention vs single SSM | **P3 (Lower Priority)** |
| **E05** | **Case E** | Multi-scale context gating | Original Neck vs Context-Gated Neck (Best backbone) | Same backbone (from B/C), same head | AP_S (small objects), neck latency overhead | **P3 (Secondary)** |
| **E06** | **Edge** | Physical edge deployment | FP32 vs FP16 vs INT8 (TensorRT/ONNX) on Jetson | Batch=1, 100 warmups, 500 timed runs | p50/p95 latency, FPS, memory footprint | **P1 (Core Goal)** |

## 2. Experiment record fields

For every run, record:
- Experiment ID.
- Date/time.
- Git commit.
- Config file and hash.
- Dataset version and split manifest.
- Seed.
- Model and checkpoint.
- Hardware/runtime.
- Training and inference settings.
- Results and units.
- Deviations, failures, and notes.
- Artifact locations.

Use `templates/experiment_record.md` for each important run.

## 3. Pre-registration fields

Before running a key comparison, write down:
- Research question.
- Independent variable.
- Dependent variables.
- Controlled variables.
- Expected outcome [HYPOTHESIS].
- Outcome that would contradict the hypothesis.
- Checkpoint selection method.
- Number of seeds.
- Statistical summary, if multiple runs are available.

## 4. Result table template

| Model | Input | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params | FLOPs/MACs | Size | Latency | FPS | Hardware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Proposed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Do not fill the table with estimates. Use `N/A` if a metric is not measured, and explain why.

## 5. Evaluation protocol checklist

- [ ] Test set was not used for model selection.
- [ ] Data splits avoid leakage between related video frames/scenes/cameras where applicable.
- [ ] Class mapping is identical for comparisons.
- [ ] Confidence and NMS settings are documented.
- [ ] Input resolution and batch size are reported.
- [ ] Warm-up and synchronization behavior are documented.
- [ ] Preprocessing/postprocessing inclusion is explicit.
- [ ] Latency is measured over enough repetitions and summarized consistently.
- [ ] Hardware and software versions are recorded.
- [ ] Failed and negative results are retained.

## 6. Experiment prioritization

### Minimum viable paper
1. Reproducible baseline.
2. Proposed method vs baseline.
3. Core ablation isolating the proposed component.
4. Accuracy and complexity comparison.
5. Error visualization and limitations.

### Stronger paper, if resources allow
- Multiple seeds.
- More than one relevant baseline.
- Auxiliary-data study.
- Edge hardware benchmark.
- Compression study.
- Stratified robustness analysis.

## 7. Stop rules

Pause or revise the plan if:
- the dataset license or access conditions prohibit intended use;
- annotation conversion cannot be validated;
- the proposed block cannot be trained/exported reliably;
- the compute cost is beyond available resources;
- the supposed contribution duplicates a close prior work;
- the evaluation setup cannot support the claims intended for the paper.
