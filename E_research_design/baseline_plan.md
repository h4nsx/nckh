# Baseline Plan

## Purpose

Establish trustworthy reference results before evaluating the proposed model. Baselines should be selected based on relevance, reproducibility, implementation availability, and fair comparison.

## 1. Baseline categories

| Category | Role | Selection rule |
|---|---|---|
| Standard YOLO baseline | Main detector reference | Choose a stable, documented version and a model size that fits the task and compute budget. |
| Lightweight YOLO variant | Efficiency reference | Use a small/nano variant with documented training and inference behavior. |
| CNN-only ablation | Isolate the contribution of Mamba | Keep the proposed model's surrounding design as similar as possible while removing/replacing the Mamba component. |
| Closest published hybrid | Compare to the most relevant prior work | Include only if implementation, data, and evaluation can be reproduced or fairly compared. |
| Optional classical/lightweight detector | Broader context | Include only if it adds useful context and is feasible to run fairly. |

Do not select a baseline solely because it produces a weak result. Explain why each baseline is relevant.

## 2. Lock the baseline specification

Record before training:

- Framework and exact version.
- Model name, repository URL, release/tag, and commit hash.
- Pretrained weights and their provenance/license.
- Input resolution.
- Dataset version and split manifest.
- Class mapping.
- Augmentation configuration.
- Optimizer, learning-rate schedule, epochs, batch size, and early-stopping policy.
- Random seed(s).
- Hardware, driver, CUDA/runtime versions where relevant.
- Evaluation script and metric implementation.
- Checkpoint-selection rule.

## 3. Fair-comparison rules

- Use the same target dataset split and class mapping.
- Keep input resolution consistent for the primary comparison, unless resolution is the explicit variable.
- Report both the original training setup and any adaptations.
- Give each model an appropriate and documented training budget.
- Avoid comparing a pretrained model against a randomly initialized model without explaining the distinction.
- Use identical evaluation code and confidence/NMS settings where compatible.
- If a model requires a different input size or runtime, report it explicitly.
- Report multiple seeds when feasible; otherwise disclose the limitation.
- Do not compare published numbers as if they were reproduced results.

## 4. Minimum metrics

### Detection
- Precision.
- Recall.
- mAP@0.5.
- mAP@0.5:0.95.
- Per-class AP and, where useful, confusion matrix.

### Efficiency
- Parameter count.
- FLOPs or MACs, stating the tool and input size.
- Weight-file size.
- Peak memory where measurable.
- Batch-1 latency and/or throughput under a specified protocol.
- FPS only when its definition and timing scope are clear.

## 5. Baseline record template

| Field | Value |
|---|---|
| Baseline ID | B01 |
| Model and size | TBD |
| Source/repository | TBD |
| Commit/version | TBD |
| Pretrained weights | TBD |
| Dataset and split | TBD |
| Input size | TBD |
| Training budget | TBD |
| Seed(s) | TBD |
| Hardware | TBD |
| Evaluation script | TBD |
| Detection metrics | TBD |
| Efficiency metrics | TBD |
| Deviations from original method | TBD |
| Reproducible? | TBD |

## 6. Baseline acceptance criteria

A baseline is ready for use when:
- The training and evaluation pipeline completes successfully.
- Dataset labels and class IDs have been validated.
- The metric script passes a small sanity test.
- The selected checkpoint is determined by a pre-declared rule.
- The configuration and logs are saved.
- At least a small batch of predictions has been visually inspected.
- The result can be rerun from the recorded configuration.

## 7. Current status

No baseline has been selected or trained in this document. Fill it after checking available compute, framework support, dataset format, and the most relevant literature.
