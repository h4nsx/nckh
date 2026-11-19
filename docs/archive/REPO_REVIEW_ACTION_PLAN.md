# Repository Review Action Plan

**Repository:** `h4nsx/nckh`\
**Reviewed branch:** `main`\
**Review baseline:** `b6bb157`\
**Purpose:** A step-by-step remediation plan based on the repository
review. Re-check the current files before editing because the repository
may have changed since that review.

## How to execute

Work through the phases in order. Do not start the next phase until the
current phase's acceptance criteria are met.

For each task: 1. Inspect the current implementation before changing it.
2. Make one focused change at a time. 3. Run the listed tests or checks.
4. Record the evidence, command output, and commit SHA. 5. Update the
manuscript only after the underlying implementation or evidence is
settled.

**Evidence limitation:** The review covered source code and manuscript
content. It did not independently rerun training, checkpoint evaluation,
ONNX conversion/parity, or Raspberry Pi benchmarks. A passing code test
does not validate experimental metrics.

### Priority

-   **P0:** Results integrity; resolve before treating results as
    publication-ready.
-   **P1:** Technical correctness and reproducibility; resolve before
    final submission.
-   **P2:** Manuscript clarity and polish.

## Execution overview

  Phase   Focus                                                 Priority
  ------- ----------------------------------------------------- ----------
  1       Freeze the evidence baseline                          P0
  2       Make evaluation outputs the source of truth           P0
  3       Standardize edge benchmark methodology                P0
  4       Align manuscript claims with evidence                 P0
  5       Validate SS2D implementation and tests                P1
  6       Clarify Arm 3 training schedule and epoch numbering   P1
  7       Package reproducibility artifacts                     P1
  8       Final manuscript cleanup and release audit            P1/P2

------------------------------------------------------------------------

# Phase 1 --- Freeze the evidence baseline

## Task 1.1 --- Record the starting state

**Priority:** P0\
**Deliverable:** `audit/REMEDIATION_BASELINE.md`

### Actions

-   Confirm the current branch and commit.
-   Check whether the working tree is clean.
-   Create a dedicated remediation branch.
-   Record known software and hardware versions used in the experiments.
-   Do not delete, overwrite, or regenerate existing experiment outputs.

Suggested commands:

``` bash
git checkout main
git pull --ff-only
git status --short
git rev-parse HEAD
git checkout -b review/remediation
```

Record Python, PyTorch, Ultralytics, ONNX Runtime, OS/kernel, and CUDA
versions where applicable. Mark unknown values as `UNKNOWN`; do not
guess.

### Acceptance criteria

-   [ ] Starting commit SHA recorded.
-   [ ] Working-tree status recorded.
-   [ ] Remediation branch created.
-   [ ] Known environment versions recorded; unknowns explicitly marked.
-   [ ] Existing experiment artifacts preserved.

------------------------------------------------------------------------

# Phase 2 --- Make evaluation outputs the source of truth

## Task 2.1 --- Remove hardcoded metrics from table generation

**Priority:** P0\
**Main file:** `scripts/make_tables.py`

### Problem

The reviewed version contains a `CHECKPOINT_RESULTS` dictionary with
manually embedded metrics. The script also reads CSV files for
training/convergence information. This creates multiple sources of
truth. The previously observed differences between CSV peak mAP50 and
checkpoint metrics are not proof that either value is wrong; they
require provenance.

### Actions

1.  Inspect the current script and list every hardcoded experimental
    value: mAP50, mAP50-95, precision, recall, per-class metrics,
    latency, FPS, model size, and parameter counts.
2.  Define a canonical machine-readable evaluation record format (JSON
    is suitable).
3.  For every evaluated checkpoint, record at least:
    -   experiment/arm ID;
    -   checkpoint path and SHA-256;
    -   dataset ID and split;
    -   evaluation timestamp;
    -   exact evaluation command or serialized arguments;
    -   software versions;
    -   evaluation thresholds/settings;
    -   overall metrics;
    -   per-class metrics.
4.  Update `make_tables.py` to read those records instead of a Python
    dictionary of manually entered values.
5.  Fail clearly when an expected experiment or metric is missing. Do
    not silently substitute zero, `None`, or a fallback constant.
6.  Keep training-curve peak metrics distinct from final/best-checkpoint
    re-evaluation metrics. Label them accurately.
7.  Add schema and value validation: required IDs, finite numeric
    values, valid metric ranges, checkpoint hash, dataset split, and
    evaluation configuration.

Example record shape (adapt to the actual evaluator output):

``` json
{
  "experiment_id": "arm1_baseline",
  "checkpoint_path": "artifacts/arm1/best.pt",
  "checkpoint_sha256": "REQUIRED",
  "dataset_id": "PhysicalAI-SmartSpaces",
  "dataset_split": "validation",
  "evaluation_timestamp": "ISO-8601",
  "evaluation_config": {},
  "software_versions": {},
  "metrics": {
    "map50": 0.0,
    "map50_95": 0.0,
    "precision": 0.0,
    "recall": 0.0
  },
  "per_class_metrics": {}
}
```

### Acceptance criteria

-   [ ] No reported experimental metric is hardcoded in
    `make_tables.py`.
-   [ ] Every generated table value traces to a canonical evaluation
    record.
-   [ ] Each record identifies its checkpoint by SHA-256.
-   [ ] Dataset split and evaluation settings are recorded.
-   [ ] Missing or malformed records cause explicit errors.
-   [ ] Re-running table generation with unchanged records produces
    unchanged tables.

### Tests

-   [ ] Valid records generate expected table values.
-   [ ] Missing experiment record raises an error.
-   [ ] Missing metric raises an error.
-   [ ] NaN/Inf or out-of-range metrics are rejected.
-   [ ] Missing checkpoint hash is rejected.

### Stop condition

If the original checkpoints or evaluation outputs cannot be recovered,
do not recreate evaluation records by copying values from the paper.
Mark provenance as unavailable and decide whether to rerun evaluation or
remove the affected claim.

------------------------------------------------------------------------

# Phase 3 --- Standardize edge benchmark methodology

## Task 3.1 --- Unify latency, FPS, and iteration reporting

**Priority:** P0\
**Main files:** `scripts/benchmark_edge_device.py`,
`paper/sections/04_experimental_setup.tex`,
`paper/sections/05_results_discussion.tex`, `report.md`

### Problems identified

-   The reviewed script calculates FPS as `1000 / median_latency_ms`.
-   The manuscript/report describe mean latency and FPS, and the values
    do not consistently establish which statistic was used.
-   The manuscript describes 100 timed iterations, while the report
    records 50 for some arms and 30 for another.
-   The script measures `session.run()` rather than separately measuring
    preprocessing and NMS, despite methodology wording that may imply a
    broader pipeline.
-   The setup mentions CPU and TensorRT providers, while the reviewed
    evidence centers on Raspberry Pi CPU execution.

### Actions

**A. Define benchmark scope.** Choose and clearly name: - model-only
inference (`session.run()`); or - end-to-end detection (preprocessing +
inference + postprocessing/NMS); or - both, reported separately.

Do not describe model-only timing as end-to-end latency.

**B. Define statistics.** Record: - warmup count; - timed iteration
count; - mean and median latency; - standard deviation; - p90/p95 if
useful; - exact FPS formula.

Choose one primary FPS convention and use it everywhere. For example,
`FPS = 1000 / mean_latency_ms`. If median-based FPS is chosen, label it
accordingly. Do not compare FPS derived from median latency with mean
latency as though they are equivalent.

**C. Reconcile iteration counts.** - Choose a common warmup and
timed-iteration policy where feasible. - If counts differ for a valid
reason, report the actual count for each arm and justify it. - Correct
the paper's 100-iteration statement using raw logs, not assumptions.

**D. Record environment and artifact provenance.** Each run should
record: - device model/revision, OS/kernel, CPU, memory; - ONNX Runtime
version and actual execution provider; - ONNX file SHA-256; - input
size, batch size, precision; - warmup/timed counts; - raw per-iteration
latency values or detailed raw logs.

**E. Verify TensorRT claims.** - If TensorRT was actually tested,
provide its device, versions, configuration, and raw evidence. - If not,
remove it from completed-experiment descriptions or label it as future
work. - Never imply TensorRT results from CPU-only Raspberry Pi
measurements.

**F. Update the script.** - Expose warmup and timed iteration counts as
explicit arguments. - Save raw measurements and machine-readable
summaries. - Record the provider actually used; do not silently switch
providers. - Distinguish model-only and end-to-end measurements.

### Acceptance criteria

-   [ ] Script and paper define the same benchmark scope.
-   [ ] One documented FPS convention is used for all arms.
-   [ ] Mean/median/percentile labels are correct.
-   [ ] Actual iteration counts are recorded.
-   [ ] Execution-provider statements match runtime evidence.
-   [ ] Raw benchmark measurements/logs are retained.
-   [ ] Benchmark tables are generated from benchmark records.

### Tests

-   [ ] FPS calculation tested with known latency samples.
-   [ ] Summary statistics tested.
-   [ ] Required device/runtime/provider fields validated.
-   [ ] Output JSON can be consumed by the table-generation pipeline.

------------------------------------------------------------------------

# Phase 4 --- Align manuscript claims with evidence

## Task 4.1 --- Verify or remove Arm 4 precision/recall claims

**Priority:** P0\
**Main file:** `paper/sections/05_results_discussion.tex`

The reviewed text states Arm 4 worker precision of 96.6% and AMR recall
of 80.2%, while the reviewed main results table reports mAP50 and
mAP50-95 rather than those precision/recall values.

### Actions

-   Locate the original evaluation output for both values.
-   Verify checkpoint, dataset split, class mapping, confidence
    threshold, IoU threshold, and evaluator.
-   Add a precision/recall table with metric definitions and provenance
    if the evidence exists.
-   Otherwise remove the unsupported values and conclusions that depend
    on them.
-   Never derive class precision/recall from mAP values.

### Acceptance criteria

-   [ ] Each value has a traceable evaluation record.
-   [ ] Metric definition and thresholds are documented.
-   [ ] Table and discussion agree.
-   [ ] Unsupported values are removed, not estimated.

## Task 4.2 --- Remove unsupported statistical-reliability language

**Priority:** P0\
**Files:** Results and Conclusion sections.

The reviewed experiment uses one training seed (seed 0). Search for
`statistically`, `reliable`, `significant`, `consistent`, `robust`,
`variance`, and `stable`.

### Actions

-   For each claim, identify whether it is supported by multiple
    independent training seeds, a valid confidence interval/statistical
    test, repeated benchmark runs, or a defined descriptive statistic.
-   Replace unsupported claims with appropriately limited wording.
-   If additional seeds are feasible, run them and report the number of
    independent seeds and observed variation.
-   Do not confuse repeated inference iterations with independent
    training runs.

### Acceptance criteria

-   [ ] No statistical-significance claim without appropriate analysis.
-   [ ] Single-seed limitation is explicit.
-   [ ] Benchmark variability is not presented as training-seed
    variability.

## Task 4.3 --- Correct validation/test terminology

**Priority:** P0\
**Files:** `paper/tables/tab_accuracy_results.tex`, Experimental Setup,
Results.

### Actions

-   Document whether the validation partition was used for checkpoint
    selection, hyperparameter tuning, model selection, or final
    evaluation only.
-   If used during model selection, call it a validation partition.
-   Use "independent test" only for data not used in model or
    hyperparameter selection.
-   Replace the reviewed title "Independent Validation Partition" if it
    overstates independence.

### Acceptance criteria

-   [ ] Partition usage is documented.
-   [ ] Table and manuscript terminology match actual usage.
-   [ ] Validation data are not mislabeled as an independent test set.

## Task 4.4 --- Reconcile pallet-truck uncertainty values

**Priority:** P0\
**Files:** Results and Conclusion.

The reviewed Results section reports approximately ±10--15%, while the
Conclusion reports approximately ±5.0%.

### Actions

-   Find the source and calculation for both numbers.
-   Determine whether they represent training-seed variation, repeated
    benchmark variation, confidence intervals, or another measure.
-   If they describe the same quantity, use one consistent calculation.
-   If no valid calculation exists, remove the numerical ranges and
    describe the rare-class limitation qualitatively.
-   Do not call a range a confidence interval unless it was calculated
    as one.

### Acceptance criteria

-   [ ] Every uncertainty value has a defined calculation and source.
-   [ ] Results and Conclusion do not conflict.
-   [ ] Unsupported numbers are removed.

------------------------------------------------------------------------

# Phase 5 --- Validate SS2D implementation and tests

## Task 5.1 --- Correct discretization terminology

**Priority:** P1\
**Main file:** `models/warehouse_ss2d.py`

### Problem

The reviewed implementation calculates `A_bar = exp(delta * A)` but
approximates the input contribution as `delta * B * u`. This is not
exact zero-order-hold discretization for both terms. The "Simplified
ZOH" wording can therefore mislead readers.

### Actions

-   Confirm the equations actually implemented.
-   Update code comments and documentation to describe the approximation
    accurately.
-   Align the paper's equations and terminology with the code.
-   Do not change the mathematical implementation merely to preserve the
    existing label.
-   Treat any move to exact ZOH as a separate architecture change
    requiring new tests and experiments.

### Acceptance criteria

-   [ ] Code comments match the implemented equations.
-   [ ] Paper equations match code.
-   [ ] No exact-ZOH claim is made for an approximation.

## Task 5.2 --- Strengthen zero-gamma identity and gradient tests

**Priority:** P1\
**Main file:** `tests/test_zero_gamma_identity.py`

### Required test cases

1.  **Initialization identity:** verify output matches input within a
    documented tolerance at initialization.
2.  **Forward validity:** verify expected shape and finite output.
3.  **Initial backward:** verify expected gradients exist. Do not
    require upstream branch gradients to be non-zero at exact zero-gamma
    initialization if the architecture gates them to zero.
4.  **Gamma gradient:** verify the zero-initialized residual scaling
    parameter receives a finite, meaningful gradient for a
    non-degenerate input/loss.
5.  **After gamma update:** move gamma away from zero with an optimizer
    step, then verify relevant upstream parameters receive finite
    gradients on a subsequent backward pass.
6.  **Numerical stability:** verify no NaN/Inf in outputs or gradients.

Use deterministic seeds and non-degenerate inputs. Avoid brittle
non-zero assertions that depend on uncontrolled random values.

## Task 5.3 --- Strengthen directional alignment tests

**Priority:** P1\
**Main file:** `tests/test_ss2d_directional_alignment.py`

### Actions

-   Test all four scan directions using a small hand-checkable tensor.
-   Verify inverse transforms restore the original spatial alignment.
-   Verify directional outputs are aligned before shared
    projection/fusion.
-   Include non-square feature maps to expose height/width swaps.
-   Include batch size greater than one.
-   Check shape and finite values.
-   Keep expected outputs independent of the helper under test, so the
    same bug cannot exist in both implementation and expectation.

### Acceptance criteria for Phase 5

-   [ ] Identity behavior is tested at initialization.
-   [ ] Gradient behavior is tested at initialization and after gamma
    updates.
-   [ ] Four directional transformations are independently validated.
-   [ ] Non-square inputs and batch size greater than one are covered.
-   [ ] Outputs and gradients are checked for finite values.
-   [ ] `audit/ARCHITECTURE_AUDIT_REPORT.md` distinguishes tested
    properties, code-inspection findings, assumptions, and
    experiment-dependent claims.
-   [ ] Audit claims do not exceed test coverage.

------------------------------------------------------------------------

# Phase 6 --- Clarify Arm 3 training schedule and epoch numbering

## Task 6.1 --- Make stage-local and global epochs explicit

**Priority:** P1\
**Main files:** `scripts/run_two_stage_experiment.py`, training
logs/CSVs, table-generation script, paper, `report.md`

### Problem

Arm 3 uses a 5-epoch warmup followed by 15 fine-tuning epochs. The
reviewed CSV contains the 15 fine-tuning epochs, while the manuscript
refers to an "Epoch 14" peak. It is unclear whether this is stage-local
or global numbering.

### Actions

-   Inspect the actual script and raw training logs.
-   Define `stage_epoch` (within the current stage) and `global_epoch`
    (across the full 20-epoch schedule).
-   Record both in generated logs where possible.
-   Document the 5 + 15 schedule and checkpoint-selection rule.
-   Update plots, tables, and prose to use one explicit convention.
-   If historical logs cannot establish the global epoch, label it as
    stage-local or omit the ambiguous epoch number.

### Acceptance criteria

-   [ ] The 5 + 15 schedule is explicit.
-   [ ] Stage-local and global epochs are not conflated.
-   [ ] Best-checkpoint origin is traceable.
-   [ ] Logs, tables, plots, and manuscript use consistent labels.

### Tests

-   [ ] Epoch-index conversion tested.
-   [ ] Logs contain stage and global epoch fields.
-   [ ] Checkpoint metadata records selected stage and epoch.

------------------------------------------------------------------------

# Phase 7 --- Package reproducibility artifacts

## Task 7.1 --- Define an artifact availability policy

**Priority:** P1\
**Files:** `.gitignore`, `audit/BASELINE_MANIFEST.md`,
`audit/FINAL_AUDIT_REPORT.md`, `Reports/FINAL_REPRODUCIBILITY_GATE.md`,
`README.md`, `docs/RESULTS_INTEGRITY_GATE.md`

### Problem

The reviewed `.gitignore` excludes `runs/`, model checkpoints, ONNX
files, and datasets. This is reasonable for keeping Git small, but a
manifest or hash alone does not make an artifact available for
independent verification.

### Actions

For each artifact, choose one approach: - commit it if small and
appropriate; - publish as a release asset; - store in external
artifact/object storage; - provide a reproducible generation
procedure; - declare it unavailable and explain the limitation.

Do not commit large datasets or model files without considering licenses
and storage constraints.

### Minimum evidence package, where available

-   checkpoints for each reported arm;
-   SHA-256 hashes;
-   evaluation JSON/CSV;
-   ONNX files and export settings;
-   ONNX parity/comparison logs;
-   raw edge benchmark logs;
-   environment/version manifests;
-   exact training, evaluation, and benchmark commands;
-   dataset version, split definition, and class mapping;
-   random seed and experiment configurations.

For each artifact record its ID, filename/stable location, SHA-256,
size, creation date, generating command, associated experiment, and
availability status.

### Acceptance criteria

-   [ ] Every reported result has a traceable evidence record.
-   [ ] Artifact availability is explicit.
-   [ ] Hashes are supplied for obtainable binary artifacts.
-   [ ] Commands and configurations are documented.
-   [ ] Reproducibility claims match what another researcher can access
    and run.
-   [ ] Missing artifacts are declared, not implied to exist.

------------------------------------------------------------------------

# Phase 8 --- Final manuscript cleanup and release audit

## Task 8.1 --- Correct exaggerated or imprecise wording

**Priority:** P2\
**Main file:** `paper/sections/01_introduction.tex`

The reviewed introduction contains "hundreds of gigabytes of memory
bandwidth and gigawatts of power," which is not an appropriate
description of the stated warehouse edge-computing constraint.

### Actions

-   Replace it with a precise description of the relevant edge
    constraint (e.g. limited memory bandwidth, compute capacity, power
    budget, or thermal headroom).
-   Support quantitative claims with a relevant source.
-   Search the manuscript for similar exaggerated or imprecise wording.

### Acceptance criteria

-   [ ] The phrase is removed or corrected.
-   [ ] Technical claims are precise and appropriately sourced.
-   [ ] No unsupported quantitative claim is introduced.

## Task 8.2 --- Run a cross-file consistency audit

**Priority:** P1/P2

Inspect `paper/main.tex`, all section/table files, `README.md`,
`report.md`, audit reports, and scripts that generate tables/figures.

Reconcile: - experiment names and arm numbering; - dataset name, split,
image/instance counts, and class mapping; - image resolution and batch
size; - epochs and random seed; - checkpoint-selection rule; -
mAP/precision/recall definitions; - latency/FPS definitions; - benchmark
device and execution provider; - parameters, FLOPs, and model size; -
uncertainty and limitation statements.

Use machine-readable experiment records as the source of truth wherever
possible.

### Acceptance criteria

-   [ ] No conflicting values across paper, tables, README, and reports.
-   [ ] Every quantitative result has a source record.
-   [ ] Limitations are stated consistently.
-   [ ] Generated tables/figures are current.

## Task 8.3 --- Rebuild and visually inspect the PDF

**Priority:** P1

### Actions

-   Clean and rebuild the LaTeX project using its documented build
    command.
-   Check LaTeX errors and unresolved references.
-   Inspect the generated PDF page by page.
-   Check table width, figure readability, captions, citations,
    equations, and page breaks.
-   Confirm generated tables were refreshed.
-   Record the final PDF SHA-256 and source commit SHA.

### Acceptance criteria

-   [ ] PDF builds without errors.
-   [ ] No unresolved citations or references.
-   [ ] Tables and figures are legible and correctly numbered.
-   [ ] No stale generated result remains.
-   [ ] Final PDF hash and source commit are recorded.

------------------------------------------------------------------------

# Final release gate

Do not mark the repository publication-ready until all applicable P0
tasks are complete and remaining limitations are explicitly disclosed.

## P0 --- Results integrity

-   [ ] Evaluation metrics come from canonical machine-readable records.
-   [ ] Every checkpoint metric traces to a checkpoint and evaluation
    configuration.
-   [ ] Benchmark latency/FPS uses one documented convention.
-   [ ] Iteration counts and execution providers match actual records.
-   [ ] Unsupported precision/recall claims are evidenced or removed.
-   [ ] Statistical claims are supported or appropriately softened.
-   [ ] Validation/test terminology matches actual data usage.
-   [ ] Conflicting uncertainty statements are reconciled or removed.

## P1 --- Technical and reproducibility integrity

-   [ ] SS2D discretization terminology matches implemented equations.
-   [ ] Directional tests cover four directions and non-square inputs.
-   [ ] Zero-gamma identity and gradient behavior are tested
    appropriately.
-   [ ] Arm 3 epoch numbering is explicit.
-   [ ] Artifact availability is documented.
-   [ ] Reproducibility claims match accessible evidence.

## P2 --- Manuscript quality

-   [ ] Exaggerated wording is corrected.
-   [ ] Paper, tables, README, and reports are consistent.
-   [ ] Latest PDF is rebuilt and visually inspected.

------------------------------------------------------------------------

# Remediation task log

Append one entry per task to `audit/REMEDIATION_LOG.md`.

``` markdown
## Task X.Y — [Task title]

- Status: NOT STARTED | IN PROGRESS | BLOCKED | DONE
- Date:
- Branch:
- Commit:
- Files changed:
- Evidence added:
- Commands/tests run:
- Test results:
- Remaining limitations:
- Reviewer notes:
```

# Recommended first sequence

Do not edit everything at once. Follow this exact order:

1.  **Task 1.1:** Freeze the baseline.
2.  **Task 2.1:** Replace hardcoded evaluation metrics with canonical
    records.
3.  **Task 3.1:** Standardize benchmark methodology and output records.
4.  **Tasks 4.1--4.4:** Correct manuscript claims using recovered
    evidence.
5.  **Tasks 5.1--5.3:** Align SS2D math terminology and strengthen
    tests.
6.  **Task 6.1:** Clarify Arm 3 epoch provenance.
7.  **Task 7.1:** Package artifacts and update reproducibility
    documentation.
8.  **Tasks 8.1--8.3:** Final manuscript cleanup and release audit.

**Stop condition:** If original checkpoints, evaluation outputs, or
benchmark logs cannot be recovered, do not invent replacement evidence.
Record the gap and decide whether to rerun the experiment or remove the
affected claim.
