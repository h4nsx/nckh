# Experiment Matrix

## Purpose

Plan a compact, interpretable experiment set. Add experiments only when they answer a research question or validate a deployment claim.

## 1. Core experiment plan

| ID | Question | Comparison | Controlled factors | Primary measurements | Status |
|---|---|---|---|---|---|
| E01 | Can the baseline pipeline be reproduced? | Baseline training/evaluation sanity run | Dataset split, metric code, config | mAP, precision/recall, logs | Planned |
| E02 | Does the hybrid help? | CNN-only baseline vs hybrid | Data, input size, training budget, evaluation | mAP, Params, FLOPs/MACs, latency | Planned |
| E03 | Where should Mamba be placed? | Selected placement candidates | Same data and training/evaluation protocol | Accuracy and efficiency | Planned |
| E04 | Which fusion is useful? | Addition vs concatenation/projection or selected alternatives | Same block placement and data | Accuracy, cost, latency | Conditional |
| E05 | Does auxiliary data help? | SmartSpaces-only vs auxiliary-data strategy | Target test set fixed; avoid leakage | mAP, per-class AP, robustness | Conditional |
| E06 | Does compression help? | Original vs compressed model | Same device/runtime/input protocol | Accuracy, size, latency, memory | Conditional |
| E07 | Does it deploy in real time? | Final model on named device | Fixed protocol and warm-up | p50/p95 latency, FPS, peak memory | Conditional |
| E08 | Is performance robust? | Stratified evaluation subsets | Same checkpoint and metric protocol | Per-condition/per-size AP and recall | Conditional |

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
