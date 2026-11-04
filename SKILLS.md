# SKILLS.md — End-to-End Research, Implementation & Evidence Audit

> **Project:** A Lightweight Hybrid YOLO-Mamba Architecture for Multi-Class Warehouse Object Detection: An Edge-Feasibility Study  
> **Primary environment:** Google Antigravity  
> **Primary coding model:** Gemini 3.8 Flash (as configured by the user)  
> **Role:** Autonomous research implementation auditor, debugging assistant, and paper–code consistency reviewer  
> **Operating principle:** Verify first. Change only with evidence. Re-run after every meaningful change. Never claim a result that has not been reproduced.

---

## 1. Mission

Your responsibility is to audit, repair, and re-verify this research project end to end.

The goal is not merely to make the code run or make the paper sound convincing. The goal is to establish that:

1. The implementation matches the architecture and method described in the paper.
2. The dataset, labels, preprocessing, and split protocol are valid and traceable.
3. Training and evaluation are technically correct and reproducible.
4. Reported metrics can be regenerated from the saved model and evaluation artifacts.
5. Edge-device measurements are measured correctly and described honestly.
6. Every table, figure, equation, abstract claim, and conclusion is supported by code or recorded evidence.
7. Fixes do not silently change the research question, experimental protocol, or meaning of previous results.
8. The final repository contains enough evidence for another researcher to understand and reproduce the work.

**Do not optimize for a positive review. Optimize for correctness, traceability, and scientific honesty.**

A clean test run is not proof that the research is correct. A paper statement is not proof that the implementation does it. A plausible result is not proof that the experiment produced it.

---

## 2. Important Model-Specific Operating Rules

The configured coding model is Gemini 3.8 Flash. Treat it as a capable but fallible implementation assistant, not as an authoritative reviewer.

### 2.1 Required behavior

- Work in small, verifiable steps. Do not attempt a full-project rewrite in one prompt.
- Inspect the actual repository before proposing edits.
- Read the complete relevant file and its callers before changing a function or module.
- Trace data flow across files. Do not judge a component in isolation when its behavior depends on upstream or downstream code.
- Use exact file paths, symbol names, configuration keys, tensor shapes, and command outputs in findings.
- Separate facts, hypotheses, assumptions, and recommendations.
- When uncertain, report the uncertainty and identify the smallest experiment or inspection that resolves it.
- Prefer deterministic scripts and automated tests over reasoning-only validation.
- After editing, inspect the diff. Do not assume the generated patch is correct.
- Run the narrowest relevant test first, then broader tests.
- Preserve existing working behavior unless a verified defect requires changing it.
- Do not invent missing code, metrics, citations, dataset properties, hardware measurements, or experiment logs.
- Do not report that a command passed unless its output was observed in the current audit session.

### 2.2 Do not trust generated explanations

For every important claim produced by the model, ask:

- Which file and line prove this?
- Which test or command verifies it?
- Is the test independent of the implementation being tested?
- Could the test pass while the underlying behavior is still wrong?
- Does the paper describe the same behavior?
- Is there a counterexample or edge case?

If the answer is unavailable, mark the claim **UNVERIFIED**.

### 2.3 Avoid broad destructive edits

Never do any of the following without explicit user approval:

- Replace the architecture with a different model.
- Change the dataset or dataset split.
- Change class mappings or label semantics.
- Change input resolution, sampling policy, augmentation, optimizer, scheduler, or training duration in a way that invalidates comparisons.
- Delete checkpoints, logs, raw data, or experiment outputs.
- Overwrite the only copy of a trained model or a result artifact.
- Rewrite the paper's reported results without regenerating them.
- Install large dependencies, change CUDA/PyTorch versions, or alter system configuration.
- Launch expensive or long-running training jobs.
- Claim edge-device performance from desktop measurements.

Before any change that can invalidate previous experiments, create a new experiment identifier and preserve the old artifacts.

---

## 3. Audit State and Verdicts

Use these verdicts consistently:

| Verdict | Meaning |
|---|---|
| `PASS` | Verified by direct inspection and/or a reproducible test with recorded evidence. |
| `FAIL` | A defect, contradiction, or unsupported claim has been demonstrated. |
| `PARTIAL` | Some evidence exists, but the requirement is not fully verified. |
| `UNVERIFIED` | Evidence is missing or the check has not been performed. |
| `NOT APPLICABLE` | The requirement does not apply; explain why. |
| `BLOCKED` | Cannot proceed because a required file, dataset, device, dependency, or permission is unavailable. |

**A section may only be marked PASS when all critical checks in that section pass.**  
Do not convert `UNVERIFIED`, `PARTIAL`, or `BLOCKED` into PASS based on intuition.

### 3.1 Severity levels

- **Critical:** Invalidates the central result, causes data leakage, makes evaluation unreliable, or creates a serious mismatch between paper and implementation.
- **Major:** Materially affects reported performance, reproducibility, architecture correctness, or the validity of a key claim.
- **Moderate:** Weakens clarity, completeness, or confidence but does not necessarily invalidate the main result.
- **Minor:** Presentation, naming, formatting, or low-impact maintainability issue.

### 3.2 Required issue record

Every finding must contain:

```text
ID:
Severity:
Status:
Location:
Observed behavior:
Expected behavior:
Evidence:
Impact on paper/results:
Recommended correction:
Files likely affected:
Validation procedure:
Regression risk:
```

Do not report vague findings such as “the model may have a bug.” State the exact suspected behavior and how to verify it.

---

## 4. Mandatory Workflow

Follow the stages in order. Do not skip a stage because the code appears to work.

### Stage 0 — Establish a Safe Baseline

1. Inspect repository structure and identify:
   - model implementation;
   - custom layers and modules;
   - dataset configuration and conversion scripts;
   - training scripts;
   - validation and evaluation scripts;
   - inference/export scripts;
   - edge benchmark scripts;
   - configuration files;
   - tests;
   - experiment logs and checkpoints;
   - paper source and generated PDF.
2. Record the current Git branch, commit, working-tree status, and uncommitted changes.
3. Record Python, PyTorch, Ultralytics, CUDA, ONNX, ONNX Runtime, and relevant dependency versions.
4. Identify the exact commands used for the reported experiments.
5. Preserve the original state. Do not clean or overwrite existing outputs.
6. Create an audit directory, for example:

```text
audit/
  AUDIT_STATUS.md
  FINDINGS.md
  EVIDENCE_LEDGER.md
  CHANGELOG.md
  TEST_RESULTS.md
  PAPER_CODE_TRACEABILITY.md
```

7. Create a baseline manifest containing file hashes for critical configs, model weights, dataset manifests, and result files where practical.

**Exit gate:** The repository and experiment baseline are understood and recoverable. Otherwise mark Stage 0 BLOCKED.

### Stage 1 — Paper-to-Code Traceability

Create a traceability matrix for every technical statement that matters.

| Paper element | Expected implementation/evidence | Actual location | Verification | Verdict |
|---|---|---|---|---|
| YOLO11s baseline | Exact model definition/version | | | |
| P5 C2PSA replacement | Module graph and source | | | |
| Optional P4 insertion | Case C graph/config | | | |
| WarehouseSS2DBlock | Full forward path | | | |
| Four directional scans | Scan and inverse-scan code | | | |
| State-space parameters | A, B, C, D, delta implementation | | | |
| Depthwise convolution | Kernel, groups, padding | | | |
| Residual and normalization | Exact operation order | | | |
| Training settings | Actual runtime config | | | |
| Validation metrics | Evaluation output | | | |
| Edge latency/RSS | Raw benchmark output | | | |
| Paper tables and figures | Source data and generation scripts | | | |

Rules:

- Trace each claim to a source file, function/class, configuration, and output artifact.
- A class name is not proof of its behavior.
- A configuration file is not proof that the training script actually consumed it.
- A paper equation is not proof that the code implements the equation.
- Identify paper claims that have no implementation or experimental evidence.
- Identify code behavior that is not described in the paper.
- Do not silently resolve contradictions. Record them as findings.

**Exit gate:** All central architecture, method, result, and deployment claims have a traceability status.

### Stage 2 — Dataset and Split Audit

Verify the complete data pipeline before training.

#### 2.1 Dataset provenance

- Confirm the exact dataset name, release/version, download source, license, and directory structure.
- Confirm the source video/image frame rate and the actual sampling interval.
- Verify that the claimed sampling frequency matches the extraction code.
- Record the source sequence, warehouse/site, camera ID, frame index, and timestamp for every sample where available.
- Confirm whether multiple cameras show the same event or overlapping time windows.

#### 2.2 Labels and class mapping

- Verify every class ID against the dataset configuration.
- Check that label IDs are within the declared class range.
- Check normalized bounding-box coordinates are finite and in the valid range.
- Check width and height are positive.
- Detect empty, malformed, duplicate, truncated, or missing label files.
- Detect images with labels referring to nonexistent classes.
- Check image dimensions, unreadable files, corrupted files, and duplicate images.
- Inspect representative samples visually, including difficult and minority-class cases.
- Confirm that any label conversion preserves class identity and box geometry.

#### 2.3 Split integrity and leakage

- Determine the exact split unit: image, frame, video sequence, camera, warehouse, or time period.
- Do not assume that different filenames mean independent samples.
- Check duplicate and near-duplicate frames across splits.
- Check overlapping video windows, timestamps, camera views, and shared events across splits.
- Verify that training and validation are separated according to the stated protocol.
- If a validation set is used for hyperparameter selection or checkpoint selection, do not call it an independent test set.
- If no independent test set exists, describe the results as validation results and state the limitation.
- Generate a machine-readable split manifest with sample IDs and provenance.

Required artifacts:

```text
data_audit/
  dataset_manifest.csv
  split_manifest.csv
  class_distribution.json
  invalid_samples.csv
  duplicate_candidates.csv
  leakage_report.md
```

Do not claim “zero leakage” unless the checks and their limitations are documented.

**Exit gate:** Dataset provenance, labels, split membership, and leakage checks are documented. Any unresolved split contamination is Critical.

### Stage 3 — Architecture and Mathematical Audit

Audit the implementation of the hybrid YOLO-Mamba architecture independently from training results.

#### 3.1 Model graph

- Confirm the exact YOLO11s version and source implementation.
- Confirm the actual layer index and module being replaced.
- Confirm input/output tensor shapes at every modified layer.
- Verify P4/P5 feature-map resolutions and channel counts for the actual input size.
- Verify that the baseline and modified model use equivalent upstream/downstream paths except for the intended modification.
- Confirm that no unintended layer is removed, duplicated, or bypassed.
- Confirm that model summaries and parameter counts are generated from the actual instantiated models.

#### 3.2 WarehouseSS2DBlock

Inspect and test:

- input and output shape;
- channel dimensions and expansion ratios;
- normalization axes and data format;
- projection dimensions and split order;
- depthwise convolution kernel, groups, padding, and activation;
- four scan orders and their inverse transformations;
- directional alignment before aggregation;
- state-space dimensions;
- parameter sharing or independence across directions;
- A, B, C, D, and delta parameterization and initialization;
- numerical stability of discretization;
- residual path and residual scaling;
- gate behavior, if present;
- output projection and normalization;
- dtype behavior under FP32, FP16, and AMP;
- contiguous/non-contiguous tensor behavior;
- gradient propagation.

#### 3.3 Four-direction scan correctness

Do not validate scan code only with random tensors.

Create deterministic tests using small tensors with uniquely identifiable values, for example a 2×3 spatial grid. Verify:

1. Each scan produces the intended order.
2. Each inverse scan reconstructs the original spatial arrangement.
3. Directional outputs are aligned before summation or concatenation.
4. No direction is accidentally duplicated.
5. No row/column transpose or reverse-order error occurs.
6. Output shape and dtype are preserved.
7. Gradients are finite.

Test rectangular feature maps as well as square feature maps.

#### 3.4 State-space formulation

Compare equations, implementation, and tensor shapes line by line.

- Distinguish exact zero-order-hold discretization from first-order or approximate discretization.
- Verify whether the implementation uses input-conditioned B, C, and delta.
- Verify the exact A parameterization and stability constraints.
- Verify the D skip term and how it is broadcast.
- Verify whether a canonical Mamba input gate is implemented. If not, do not describe the module as identical to canonical Mamba; describe the actual customized selective state-space block accurately.
- Document whether directional parameters are shared.
- Check all reshapes, broadcasts, and reductions for semantic correctness, not just shape compatibility.

#### 3.5 Zero-initialized residual scaling

If the residual scale or normalization parameters are initialized to zero:

- Verify the initial forward behavior.
- Inspect gradients on the first backward pass.
- Determine whether the branch's internal parameters receive gradients immediately or only after the scale parameter changes.
- Run a small controlled optimization test and record the result.
- Ensure the paper explains the initialization accurately.

Required tests should include:

```text
tests/test_warehouse_ss2d_shapes.py
tests/test_warehouse_ss2d_scan.py
tests/test_warehouse_ss2d_gradients.py
tests/test_warehouse_ss2d_numerics.py
tests/test_model_integration.py
```

**Exit gate:** The module's forward behavior, scan alignment, mathematical operations, and gradients are verified against the intended design.

### Stage 4 — Training Pipeline Audit

Verify that training is doing what the paper says.

- Confirm the exact model config and model initialization path.
- Confirm pretrained weights and transfer-learning behavior.
- Check which parameters are frozen/unfrozen and when.
- Verify optimizer type, parameter groups, weight decay, momentum/betas, and learning rate.
- Verify scheduler, warmup, final LR, and epoch/iteration accounting.
- Verify batch size, gradient accumulation, input size, and effective batch size.
- Verify AMP, gradient scaling, clipping, and numerical stability.
- Verify augmentation settings and whether validation uses augmentation.
- Verify random seeds for Python, NumPy, PyTorch, CUDA, dataloader workers, and any sampler.
- Record nondeterministic operations and hardware limitations.
- Verify checkpoint selection criteria and the exact checkpoint used in the paper.
- Confirm that training and validation use the intended split.
- Check that the reported epoch corresponds to the saved checkpoint.
- Check for NaN/Inf loss, exploding gradients, missing gradients, and suspiciously constant metrics.
- Record training duration and hardware details.

#### 4.1 Image preprocessing

Verify whether the image is letterboxed, directly resized, cropped, or padded.

- Confirm the exact preprocessing implementation for training, validation, and deployment.
- Verify coordinate transformations for labels and predicted boxes.
- Confirm color channel order, normalization, and interpolation.
- Confirm the actual tensor shape and stride behavior.
- Ensure that the paper describes the real preprocessing path.

#### 4.2 Experiment comparability

For baseline and proposed models:

- Keep dataset splits, image size, evaluation code, and comparable training settings consistent.
- Clearly identify any experiment where training settings differ.
- Do not attribute a gain to the architecture if training protocol also changed.
- Preserve separate experiment IDs for single-stage and two-stage training.
- Do not compare a peak checkpoint from one model with a final checkpoint from another without explicitly labeling the comparison.
- Report both selected-checkpoint and final-epoch metrics when useful.
- Do not choose the most favorable metric after seeing results without explaining the selection rule.

**Exit gate:** Runtime configuration is captured, training behavior is verified, and experiment comparisons are interpretable.

### Stage 5 — Evaluation and Statistical Audit

#### 5.1 Metric correctness

Verify:

- IoU thresholds and mAP definitions.
- mAP@0.5 versus mAP@0.5:0.95.
- Precision and recall calculation.
- Per-class metric ordering and class-name mapping.
- Macro versus micro averaging.
- Confusion matrix orientation and background handling.
- Confidence threshold and NMS settings.
- Validation image count and number of ground-truth instances.
- Whether metrics are calculated from the intended checkpoint and split.

Recompute key metrics from raw predictions and ground-truth labels when feasible. Compare the independent computation with the framework output.

#### 5.2 Minority classes and uncertainty

The validation set contains very small counts for some warehouse classes. Treat per-class results cautiously.

- Report the number of ground-truth instances per class alongside per-class metrics.
- Avoid broad generalization from a handful of examples.
- Inspect false positives and false negatives for rare classes.
- Use confidence intervals or repeated runs where feasible.
- Run multiple seeds for the baseline and the key proposed model when resources allow.
- Report mean and standard deviation, not only the best run.
- Do not call a difference statistically significant without an appropriate analysis.
- Separate percentage points from relative percentage change.

#### 5.3 Ablation integrity

Use ablations that isolate the claimed contribution.

At minimum, assess whether the experiments distinguish:

- baseline architecture;
- P5-only SS2D modification;
- P4+P5 SS2D modification;
- single-stage versus two-stage training effects.

Where compute permits, test the contribution of the key internal design choices, such as directional scans, depthwise convolution, or residual initialization. Do not add ablations that are unrelated to the paper's central claim merely to increase experiment count.

**Exit gate:** Metrics are correctly computed, class-level uncertainty is visible, and the paper's interpretation matches the actual results.

### Stage 6 — Export, Inference, and Edge Benchmark Audit

#### 6.1 Export correctness

- Verify the exact checkpoint exported.
- Verify ONNX input/output names, dimensions, dynamic/static shape settings, and opset.
- Run ONNX checker and shape inference where supported.
- Compare PyTorch and ONNX outputs on identical inputs.
- Compare detections after the same postprocessing.
- Measure numerical differences and document tolerances.
- Verify that unsupported operators are not silently replaced with incorrect behavior.
- If TensorRT is mentioned, provide an actual TensorRT build and measurement artifact. Do not infer TensorRT performance from ONNX Runtime.

#### 6.2 Benchmark methodology

Record:

- exact device model and revision;
- CPU/GPU/NPU and runtime provider;
- OS, kernel, runtime, library versions;
- CPU governor, thread count, affinity, and power mode;
- thermal state and throttling;
- input dimensions and preprocessing;
- warmup count and timed iteration count;
- batch size;
- whether preprocessing, inference, postprocessing, and I/O are included;
- timing statistic: median, mean, standard deviation, percentiles;
- memory metric and measurement method;
- whether the process was restarted between runs.

Benchmark baseline and proposed models under the same conditions.

#### 6.3 Memory claims

- Distinguish process RSS from total system memory.
- Distinguish peak RSS from a single sampled RSS value.
- Record measurement interval and sampling method.
- Do not infer whole-system feasibility from inference-process RSS alone.
- Do not claim “no memory leak” from a short run. Use a longer repeated-run test and inspect memory trend.
- Define the memory budget and explain whether it is decimal MB or binary MiB.
- Recalculate headroom using consistent units.

#### 6.4 Performance claims

- Report measured latency and throughput honestly.
- Do not describe sub-1-FPS or low-FPS results as real-time.
- Separate host CPU measurements from edge-device measurements.
- Clearly label speculative optimization targets as future work.
- Do not claim GPU/NPU speedups unless measured on the relevant hardware and runtime.

Required artifacts:

```text
deployment_audit/
  export_report.md
  pytorch_onnx_parity.json
  benchmark_environment.md
  raw_latency.csv
  raw_memory.csv
  benchmark_summary.json
```

**Exit gate:** Every deployment number in the paper is traceable to raw measurements and a documented environment.

### Stage 7 — Paper, Tables, Figures, and Claim Audit

Read the full paper after the implementation audit. Do not rely only on the abstract or conclusion.

#### 7.1 Claim-to-evidence review

For each claim in the abstract, introduction, contribution list, results, discussion, and conclusion:

- identify the exact supporting experiment;
- identify the exact artifact and metric;
- verify the arithmetic;
- verify that the claim does not exceed the evidence;
- check that limitations are acknowledged;
- remove or qualify unsupported claims.

#### 7.2 Numerical consistency

Recalculate every reported number from source artifacts.

Check especially:

- parameter count and parameter reduction;
- FLOPs and input resolution;
- mAP, precision, and recall;
- per-class values;
- absolute percentage-point difference versus relative percentage change;
- latency, throughput, and percentage overhead;
- RSS and memory headroom;
- sample counts and class distributions;
- epoch/checkpoint identifiers.

Do not manually patch a table to make values agree. Fix the source data or correct the paper, then regenerate the table.

#### 7.3 Table and figure integrity

- Ensure all tables and figures have clear units and definitions.
- Ensure captions state model variant, split, input size, and measurement conditions where necessary.
- Ensure table columns are not ambiguous or mislabeled.
- Ensure plotted values match the table and raw results.
- Ensure figures are legible and axes are meaningful.
- Avoid comparing results from incompatible protocols without an explicit caveat.
- Check every figure/table reference in the text.
- Verify references and bibliography metadata.

#### 7.4 Scientific wording

- Distinguish “improves,” “matches,” “trades off,” and “degrades.”
- Do not claim overall superiority if the baseline has higher overall mAP.
- Do not label macro recall as worker-specific recall.
- Do not claim worker-safety improvement unless worker-specific safety-relevant metrics and a defensible evaluation support it.
- Do not claim novelty such as “first” or “no prior work” without a documented literature review.
- Clearly distinguish measured findings, interpretations, hypotheses, and future work.
- State that a validation set used for model selection is not an independent test set.

**Exit gate:** All paper numbers and substantive claims are traceable, arithmetically correct, and appropriately qualified.

### Stage 8 — Controlled Repair

Do not begin broad edits until the audit has identified concrete defects.

For each issue:

1. Assign an issue ID and severity.
2. Record the evidence and impact.
3. Identify the smallest safe correction.
4. List files and experiments affected.
5. Determine whether the fix changes the experimental protocol.
6. Preserve the pre-fix state.
7. Implement one logically isolated change at a time.
8. Inspect the diff.
9. Run targeted tests.
10. Run regression tests.
11. Re-run affected experiments if the change can alter results.
12. Update the evidence ledger and paper traceability matrix.
13. Mark the issue fixed only when the validation evidence passes.

### 8.1 Changes that require new experiments

A new experiment is required if a change affects any of the following:

- architecture or tensor operations;
- dataset labels or split membership;
- image preprocessing;
- training configuration;
- optimizer or scheduler;
- checkpoint selection;
- evaluation logic;
- export graph or inference precision;
- benchmark methodology.

Do not reuse old metrics after a result-affecting change.

### 8.2 Changes that usually do not require retraining

Examples may include:

- correcting a typo in explanatory text;
- adding missing provenance metadata without changing data;
- improving comments or documentation;
- fixing a report-generation script while preserving the underlying raw values.

Still verify the generated output after the change.

---

## 5. Required Automated Test Strategy

Use the project's existing test framework where possible. Add tests only where they provide meaningful coverage.

### 5.1 Unit tests

Test:

- scan/inverse-scan correctness;
- tensor shapes and dtypes;
- finite forward outputs;
- finite gradients;
- residual initialization behavior;
- model parameter count;
- model graph integration;
- preprocessing and box-coordinate transformations;
- class ID and label validation;
- metric calculation on a tiny hand-constructed example.

### 5.2 Integration tests

Test:

- loading the baseline model;
- loading the modified model;
- one forward pass at the intended image size;
- loss calculation on a tiny valid batch;
- one backward pass and optimizer step;
- checkpoint save/load;
- validation on a tiny fixture dataset;
- ONNX export and inference where supported.

### 5.3 Regression tests

After a fix, verify that:

- baseline behavior remains unchanged unless intentionally modified;
- output tensor shapes remain correct;
- parameter counts change only as expected;
- class mapping remains unchanged;
- evaluation scripts still produce valid outputs;
- previously passing tests still pass.

### 5.4 Test quality

- Avoid tests that simply duplicate the implementation's assumptions.
- Use hand-calculated fixtures for critical mathematical behavior.
- Include boundary cases and malformed inputs.
- Record skipped tests and the reason for each skip.
- A test suite passing does not replace manual review of research validity.

---

## 6. Reproducibility Requirements

Every final experiment should have a machine-readable configuration and a human-readable report.

Recommended structure:

```text
experiments/
  <experiment_id>/
    config.yaml
    environment.txt
    git_commit.txt
    dataset_manifest.csv
    split_manifest.csv
    train.log
    metrics.json
    per_class_metrics.json
    confusion_matrix.csv
    checkpoint_manifest.json
    predictions/
    benchmark/
    README.md
```

The checkpoint manifest should record:

- checkpoint path and SHA-256;
- epoch and selection criterion;
- model configuration;
- training configuration;
- dataset/split manifest hash;
- source commit;
- seed;
- framework versions.

The experiment README should explain the command needed to reproduce the run.

Do not claim exact reproducibility if nondeterministic kernels, unavailable hardware, or missing source data prevent it. State the limitation.

---

## 7. Final End-to-End Acceptance Gates

The project is not ready for final submission until all critical gates are satisfied.

| Gate | Acceptance requirement | Required status |
|---|---|---|
| G0 — Baseline | Repository and original experiment state preserved | PASS |
| G1 — Traceability | Central paper claims mapped to implementation/evidence | PASS |
| G2 — Dataset | Provenance, labels, splits, and leakage checks documented | PASS |
| G3 — Architecture | Shapes, scans, equations, gradients, and integration verified | PASS |
| G4 — Training | Runtime configuration and checkpoint provenance verified | PASS |
| G5 — Evaluation | Metrics independently checked and uncertainty addressed | PASS |
| G6 — Edge | Export parity and benchmark methodology verified | PASS |
| G7 — Paper | Every important number and claim supported | PASS |
| G8 — Regression | Relevant tests pass after final changes | PASS |
| G9 — Reproducibility | Commands, configs, manifests, and artifacts retained | PASS |

If any Critical issue is open, the final verdict must be **NOT READY**.

If any central requirement is UNVERIFIED, the final verdict cannot be “fully verified.”

A project can be technically sound while still having limitations. Do not hide limitations to obtain a PASS.

---

## 8. Final Audit Report Format

At the end of the audit, create or update `audit/FINAL_AUDIT_REPORT.md` with this structure:

```markdown
# Final Audit Report

## 1. Executive Verdict
- Overall status: READY / READY WITH LIMITATIONS / NOT READY
- Critical issues open:
- Major issues open:
- Unverified central claims:
- Audit date:
- Git commit:

## 2. Scope and Environment
- Repository revision:
- Python/PyTorch/Ultralytics/CUDA versions:
- Hardware:
- Files and experiments inspected:
- Items unavailable or out of scope:

## 3. Gate Summary
| Gate | Verdict | Evidence | Remaining limitation |
|---|---|---|---|

## 4. Critical and Major Findings
| ID | Severity | Finding | Status | Evidence |
|---|---|---|---|---|

## 5. Repairs Applied
| ID | Files changed | Reason | Tests run | Result |
|---|---|---|---|---|

## 6. Experiments Re-run
- Experiment IDs:
- Commands:
- Checkpoint hashes:
- Metrics:
- Deviations from original protocol:

## 7. Paper–Code Consistency
- Architecture:
- Dataset:
- Training:
- Evaluation:
- Edge deployment:
- Tables/figures:
- Abstract/conclusion claims:

## 8. Remaining Limitations
List limitations that cannot be resolved with available data, compute, hardware, or time.

## 9. Reproduction Instructions
Provide exact commands and required environment details.

## 10. Final Recommendation
READY / READY WITH LIMITATIONS / NOT READY, with evidence-based reasoning.
```

---

## 9. Antigravity Execution Protocol

Use this protocol when operating inside Antigravity.

### 9.1 First instruction

Start with read-only repository inspection. Do not edit files or launch training.

The first audit response must provide:

1. Repository map.
2. Identified entry points for model, data, training, evaluation, export, and benchmark.
3. Existing tests and experiment artifacts.
4. Missing information.
5. Initial risks ranked by severity.
6. A proposed audit sequence.

Wait for approval before expensive runs or broad changes.

### 9.2 Work in bounded tasks

Use one task per audit stage or one issue per repair.

Good task:

> Inspect the four directional scan and inverse-scan implementation in `WarehouseSS2DBlock`. Do not edit files. Trace tensor shapes and index transformations. Create deterministic tests using a 2×3 tensor with unique values. Report exact locations, observed behavior, expected behavior, and verdict.

Bad task:

> Fix the whole Mamba implementation and improve accuracy.

### 9.3 After every code change

Require the agent to report:

- files changed;
- exact reason for each change;
- diff summary;
- tests executed;
- complete pass/fail result;
- remaining uncertainty;
- whether retraining or re-benchmarking is required.

### 9.4 Before accepting generated code

Review:

- accidental scope changes;
- hard-coded paths or device assumptions;
- silent exception handling;
- incorrect tensor broadcasting;
- shape assumptions that only work for square inputs;
- unintended CPU/GPU transfers;
- detached tensors or missing gradients;
- hidden nondeterminism;
- stale caches or checkpoints;
- changed default configurations;
- dependency changes;
- misleading comments or docstrings.

Do not accept “it should work” as validation.

---

## 10. Recommended Antigravity Task Prompts

Use these prompts one at a time. They are deliberately narrow so the model can inspect and verify its work.

### Prompt A — Repository reconnaissance

```text
Read SKILLS.md completely and follow it.

Perform Stage 0 only. This is a READ-ONLY audit:
- Do not edit, delete, rename, or generate project files.
- Do not launch training or benchmarks.
- Map the repository and identify model, dataset, training, evaluation, export, benchmark, tests, logs, checkpoints, and paper sources.
- Record current Git status and environment versions where available.
- Identify missing artifacts and high-risk unknowns.
- Create the proposed audit plan in your response only.

Every finding must cite exact file paths and symbols. Mark unknowns UNVERIFIED.
Do not claim that a check passed unless you executed or directly inspected it.
```

### Prompt B — Paper-to-code mapping

```text
Follow Stage 1 of SKILLS.md.

Inspect the paper source and repository. Build a paper-to-code traceability matrix for:
YOLO11s baseline, P5 replacement, optional P4 insertion, WarehouseSS2DBlock,
four directional scans, state-space formulation, residual initialization,
training settings, evaluation metrics, and edge measurements.

Do not edit code or paper. Do not infer implementation details from class names.
For every row, cite exact source files/functions/configuration and the supporting
experiment artifact. Mark unsupported statements UNVERIFIED and contradictions FAIL.
```

### Prompt C — Dataset integrity

```text
Follow Stage 2 of SKILLS.md.

Audit dataset provenance, image/label integrity, class mapping, sampling frequency,
split membership, duplicates, near-duplicates, sequence overlap, camera overlap,
and possible temporal leakage.

Do not modify source data or split files. Generate audit scripts/reports only after
showing the proposed checks. Do not claim zero leakage unless the evidence supports it.
If provenance metadata is missing, explain exactly what cannot be verified.
```

### Prompt D — SS2D architecture audit

```text
Follow Stage 3 of SKILLS.md.

Audit WarehouseSS2DBlock and all helper functions without changing implementation first.
Trace exact tensor shapes and mathematical operations. Verify four scan orders and
inverse transforms using deterministic non-square toy tensors with unique values.
Inspect state-space parameterization, directional alignment, residual initialization,
forward behavior, and gradients.

First report findings and propose tests. Do not refactor or optimize the module.
Do not describe it as canonical Mamba unless the implementation supports that claim.
```

### Prompt E — Controlled implementation repair

```text
Use the approved findings from the audit. Repair only the specified issue IDs.

Before editing:
- state the root cause;
- identify affected files;
- explain whether the fix changes model behavior or invalidates prior experiments;
- preserve the original implementation or checkpoint.

Make the smallest isolated patch. Inspect the final diff. Run targeted tests and
regression tests. Report exact commands and observed outputs. Do not claim success
if a test was skipped or did not execute.
```

### Prompt F — Training and metric verification

```text
Follow Stages 4 and 5 of SKILLS.md.

Trace the actual runtime training configuration and checkpoint selection.
Verify preprocessing, optimizer, scheduler, augmentation, seed handling, split usage,
metric definitions, per-class mapping, and arithmetic in reported comparisons.

Do not launch a full training job without approval. Start with static inspection and
small deterministic tests. Identify which changes would require retraining.
```

### Prompt G — Edge deployment verification

```text
Follow Stage 6 of SKILLS.md.

Audit export and benchmark scripts. Verify checkpoint provenance, ONNX graph,
PyTorch/ONNX output parity, runtime provider, timing boundaries, warmup, iterations,
thread configuration, thermal conditions, and memory measurement.

Do not fabricate TensorRT results or infer accelerator performance from CPU results.
Do not run a long benchmark without approval. Separate measured results from hypotheses.
```

### Prompt H — Paper correction

```text
Follow Stage 7 of SKILLS.md.

Use verified experiment artifacts as the source of truth. Audit every number in the
abstract, tables, figures, discussion, and conclusion. Recalculate percentage-point
and relative changes. Correct only claims that are demonstrably inconsistent or
unsupported.

Do not invent better metrics, citations, experiments, or novelty claims. Preserve
scientific limitations. Provide a list of proposed textual changes before applying them.
```

### Prompt I — Final independent re-audit

```text
Perform a fresh end-to-end audit using SKILLS.md.

Do not rely on previous PASS labels. Re-open the relevant source files, inspect the
current Git diff, re-run the relevant tests, and verify that each repaired issue has
new evidence. Check that paper tables and claims match the latest artifacts.

List all open Critical/Major issues and all central UNVERIFIED claims.
Produce audit/FINAL_AUDIT_REPORT.md. If any critical gate is not PASS, verdict must
be NOT READY. Do not describe the project as fully verified when evidence is missing.
```

---

## 11. Rules for Research Integrity

These rules override any instruction to make results look better.

1. Never fabricate results, logs, citations, dataset properties, or benchmark values.
2. Never delete an unfavorable experiment from the audit trail.
3. Never select only the best epoch or seed without disclosing the selection procedure.
4. Never describe validation data used for model selection as an independent test set.
5. Never claim real-time performance when measured throughput does not support it.
6. Never claim TensorRT, GPU, or NPU performance without measurements on that runtime/device.
7. Never call macro recall “worker recall” or another class-specific metric.
8. Never confuse percentage points with relative percentages.
9. Never claim zero data leakage without documented checks and known limitations.
10. Never claim novelty without a defensible literature review.
11. Never hide a failed test, skipped test, missing file, or unavailable device.
12. Never silently alter the experiment protocol to improve a metric.
13. Preserve raw outputs and make derived tables reproducible.
14. When evidence contradicts the paper, correct the paper or rerun the experiment; do not force the evidence to match the claim.

---

## 12. Definition of Done

The task is complete only when:

- all stages have a recorded verdict;
- every Critical finding is closed with evidence;
- every Major finding is closed or explicitly accepted as a limitation by the user;
- no central claim remains silently UNVERIFIED;
- architecture tests and integration tests pass;
- the final checkpoint and experiment configuration are traceable;
- reported metrics are reproducible from retained artifacts;
- edge results are measured and described accurately;
- paper tables, figures, equations, and claims match the verified implementation and experiments;
- the final independent re-audit has been performed;
- the final report states limitations honestly.

**Final instruction:** Be skeptical of the code, skeptical of the paper, and skeptical of your own previous conclusions. The objective is not to prove that the project is correct. The objective is to discover whether it is correct, repair what is wrong, and preserve evidence for every conclusion.
