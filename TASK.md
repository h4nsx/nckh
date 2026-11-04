# TASK.md — End-to-End Research Paper and Implementation Audit

## 1. Task overview

Conduct a rigorous, evidence-based audit of the research project:

**Working title:** *A Lightweight Hybrid YOLO-Mamba Architecture for Multi-Class Warehouse Object Detection: An Edge-Feasibility Study*

The goal is to determine whether the paper's technical descriptions, implementation, training procedure, reported metrics, exported model, and edge-device measurements agree with one another and can be independently reproduced.

This is an **audit task**, not an implementation or paper-rewriting task. Do not modify model code, training configurations, datasets, checkpoints, experiment outputs, or the manuscript during the audit unless explicitly instructed. First establish what is true from the available artifacts; then report discrepancies and propose corrective actions.

The audit must distinguish:

- **Verified:** directly demonstrated by source code, logs, artifacts, or repeatable execution.
- **Partially verified:** some supporting evidence exists, but it is incomplete.
- **Unverified:** asserted in the paper or project notes without sufficient evidence.
- **Contradicted:** available evidence conflicts with the claim.
- **Not applicable / unavailable:** the required artifact or test is not present.

Never convert an assumption into a verified fact.

## 2. Primary objectives

1. Verify that the implemented architecture matches the architecture described in the paper.
2. Verify model initialization, pretrained-weight transfer, trainable parameters, and checkpoint loading.
3. Verify dataset composition, class mapping, preprocessing, sampling, and train/validation separation.
4. Reconstruct the experimental arms and determine whether comparisons isolate the intended variables.
5. Recompute reported metrics from raw predictions or evaluation outputs wherever possible.
6. Verify ONNX export correctness and numerical parity against the source framework.
7. Audit edge benchmark methodology, arithmetic, memory measurements, and performance claims.
8. Assess reproducibility: code revision, dependencies, configurations, seeds, manifests, and artifact hashes.
9. Produce a prioritized issue list and a claim-to-evidence ledger that can guide the next revision of the paper.

## 3. Required project inputs

Inspect the repository and identify the exact location and version of each available artifact. Do not assume that a file exists merely because the manuscript refers to it.

### 3.1 Manuscript and supporting documents

- Latest manuscript PDF and, if available, source files.
- Architecture diagrams and supplementary material.
- Existing architecture verification checklist or audit reports.
- Existing research workflow or project notes.

### 3.2 Source code

At minimum, locate:

- `WarehouseSS2DBlock` and all related SS2D/selective-SSM modules.
- YOLO model definition or YAML/configuration files.
- Model construction and pretrained-checkpoint loading code.
- Training scripts and experiment-specific configurations.
- Validation/evaluation scripts.
- Export scripts, especially ONNX export.
- Edge benchmark scripts.
- Utility code for preprocessing, postprocessing, metrics, and profiling.

Record the repository commit, branch, and working-tree status if Git metadata is available.

### 3.3 Data and experiment artifacts

Look for:

- Dataset directory structure and annotation files.
- Dataset split manifests, video/camera identifiers, timestamps, and frame indices.
- Class-name mappings and annotation conversion scripts.
- Training logs and validation outputs for every experimental arm.
- Model checkpoints, including pretrained and final/best checkpoints.
- Checkpoint-loading logs and missing/unexpected-key reports.
- ONNX files and export logs.
- Raw benchmark logs, memory traces, and profiling outputs.
- Environment/version records.

If an input is missing, list it explicitly and continue with the parts that can be audited. Do not fabricate substitute evidence.

## 4. Audit workflow

Execute the stages in order. Each stage must end with a concise result, evidence references, unresolved questions, and a clear gate decision.

### Stage 0 — Artifact inventory and provenance

**Tasks**

1. Inventory all relevant files and classify them by role.
2. Record file paths, sizes, timestamps where available, and SHA-256 hashes for important artifacts.
3. Identify the manuscript version being audited.
4. Record Git commit, branch, and uncommitted changes.
5. Record Python, PyTorch, Ultralytics, CUDA/CPU, ONNX, ONNX Runtime, and other relevant package versions from the actual environment.
6. Identify missing artifacts required for later stages.

**Deliverables**

- `audit/ARTIFACT_INVENTORY.md`
- `audit/artifact_manifest.csv` or equivalent machine-readable manifest
- `audit/ENVIRONMENT.md`

**Gate:** Do not claim reproducibility or artifact integrity until the required files and provenance are identified.

### Stage 1 — Dataset and preprocessing audit

**Tasks**

1. Reconstruct the dataset from the actual annotations and split files.
2. Verify the reported train and validation image/frame counts and instance counts.
3. Verify per-class counts for each split.
4. Trace source labels to the paper's classes:
   - `worker`
   - `forklift`
   - `pallet_truck`
   - `AMR`
5. Verify mappings such as `NovaCarter` to `AMR` and `Transporter` to `pallet_truck`; do not accept these mappings without evidence in the conversion code or annotation documentation.
6. Verify the frame-sampling procedure. The manuscript describes 0.5 FPS / one frame per 60 frames; inspect the source video FPS and sampling code before confirming this.
7. Verify image resizing and aspect-ratio handling, including whether 1920×1080 images are letterboxed or directly resized to 640×640.
8. Verify the minimum-box-size filter (reported as boxes of 4 pixels or smaller), including whether it is applied to training, validation, or both.
9. Check for duplicate and near-duplicate frames across splits.
10. Determine whether the split is genuinely separated by facility, camera, sequence, or video. Record the exact grouping key used to prevent leakage.
11. Compare all computed counts with the manuscript tables and prose.

**Special attention**

The manuscript reports a cross-facility validation split using Warehouse_000 for training and Warehouse_015 for validation. Verify this from the actual data, not from the paper alone.

The validation set appears to be used for checkpoint selection and hyperparameter decisions. Determine this from training/evaluation code and logs. If it is used for selection, do not describe it as an untouched independent test set.

**Deliverables**

- `audit/DATASET_AUDIT.md`
- `audit/dataset_statistics.csv`
- `audit/split_manifest` or a documented reason why one cannot be reconstructed
- A list of all discrepancies against the manuscript

**Gate:** Dataset counts, class mappings, preprocessing, and split semantics must be verified or clearly marked unresolved before accepting the paper's dataset claims.

### Stage 2 — Architecture and implementation audit

**Tasks**

1. Trace the actual model graph from model construction through the final detection head.
2. Identify precisely which YOLO11s stages remain unchanged and which modules are replaced or modified.
3. Verify the paper's description of the P4 and P5 feature stages and their spatial resolutions for a 640×640 input.
4. Inspect the `WarehouseSS2DBlock` implementation line by line.
5. Verify tensor shapes at every operation, including:
   - input and output feature maps;
   - normalization and projection;
   - split branches and gating;
   - depthwise convolution;
   - directional sequence construction;
   - selective state-space scan;
   - inverse permutations / directional aggregation;
   - output projection and residual connection.
6. Verify directional alignment: each scan direction must traverse the intended spatial order, and inverse transforms must restore the correct spatial layout.
7. Determine whether projections and state-space parameters are shared across directions or independently parameterized.
8. Inspect the state transition (`A`), input-dependent parameters (`B`, `C`, `Δ`), skip parameter (`D`), and any gating operation. Describe the implementation as a customized selective SSM/SS2D if it does not implement the canonical Mamba block.
9. Compare implementation equations and tensor dimensions with the manuscript equations.
10. Inspect the Case B and Case C configurations:
    - Case B: P5 SS2D.
    - Case C: P4 + P5 SS2D.
11. Verify whether Case C actually performs both P4 and P5 scans and how their computational costs are accounted for.
12. Verify parameter count and FLOPs using a method that accounts for custom scan operations. Record the counting tool, input shape, and whether the reported number means MACs or FLOPs.

**Zero-initialization audit — mandatory**

The manuscript contains potentially inconsistent descriptions of `W_out = 0` and zero-initialized BatchNorm affine parameters (`γ = 0`, `β = 0`). Determine exactly what the code initializes.

Test and report:

- Whether the block is an identity mapping at initialization, within numerical tolerance.
- Whether the output projection is zero-initialized.
- Whether BatchNorm scale and bias are zero-initialized.
- Whether gradients reach the normalization parameters, output projection, depthwise convolution, SSM parameters, and gate on the first backward pass.
- Whether those parameters begin updating after optimizer steps.
- Whether any initialization choice accidentally blocks learning in part or all of the branch.

Do not infer gradient behavior from the diagram. Run a minimal isolated test if the environment permits.

**Deliverables**

- `audit/ARCHITECTURE_AUDIT.md`
- `audit/architecture_shapes.md`
- `audit/initialization_and_gradient_tests.md`
- A code-to-paper mapping table with file paths and line references

**Gate:** The architecture description must match the actual forward pass and initialization behavior. Any mismatch must be reported before accepting architecture or novelty claims.

### Stage 3 — Pretrained checkpoint and training protocol audit

**Tasks**

1. Trace the exact pretrained checkpoint used by every experimental arm.
2. Inspect checkpoint loading code and logs.
3. Record matched, missing, unexpected, and shape-mismatched state-dict keys.
4. Verify which backbone, neck, regression head, and classification head tensors are retained, reinitialized, or discarded.
5. Specifically test the manuscript's claim that changing from 80 classes to 4 classes silently discards the detection head. Do not assume that all detection-head tensors are discarded: inspect tensor names and shapes and verify what the loader actually transfers.
6. Verify the number of trainable and frozen parameters for every arm.
7. Reconstruct each arm's architecture, initialization, optimizer, scheduler, learning rate, batch size, image size, epoch count, augmentation, and seed.
8. Verify whether optimizer state is retained or reset between training stages.
9. Verify the Arm 3 two-stage procedure, including the Stage 2 learning rate reported as 0.0005.
10. Verify whether a baseline two-stage control exists. If it does not, state that architecture and training protocol are confounded in comparisons between single-stage baseline and two-stage SS2D.
11. Verify checkpoint-selection logic and the epoch used for every reported result.
12. Determine whether reported final-epoch and peak-checkpoint metrics are clearly distinguished.
13. Inspect whether the runs actually converged. If not, use “final epoch” rather than “convergence” in conclusions.

**Deliverables**

- `audit/TRAINING_AUDIT.md`
- `audit/checkpoint_transfer.csv`
- `audit/experiment_matrix.csv`
- A list of confounds and missing controls

**Gate:** Do not attribute a performance difference to the SS2D architecture when the comparison also changes training procedure or checkpoint initialization without a suitable control.

### Stage 4 — Metrics and statistical claims audit

**Tasks**

1. Re-run validation for each available checkpoint using the same evaluation configuration.
2. Recompute global and per-class precision, recall, mAP@0.5, and mAP@0.5:0.95 where raw outputs and annotations permit.
3. Verify confidence threshold, IoU threshold, NMS configuration, image size, and evaluation settings.
4. Confirm whether macro metrics are unweighted class averages or instance-weighted metrics.
5. Reconcile all manuscript tables, figures, abstract values, and conclusion statements.
6. Distinguish percentage points from relative percentage changes.
7. Verify claims such as:
   - Arm 3 macro recall improvement over baseline;
   - forklift mAP improvement;
   - Arm 2 peak mAP versus final mAP;
   - Arm 4 per-class precision/recall claims.
8. Check all rounding and arithmetic.
9. Inspect whether class-level claims are supported by the corresponding tables. For example, do not report worker precision or AMR recall if the manuscript only provides per-class mAP.
10. Assess the impact of very small class counts, particularly the pallet-truck validation count.
11. Verify any claims about stability, variance, statistical significance, or seed robustness. A single seed does not establish multi-seed statistical reliability.
12. If raw predictions are unavailable, mark the relevant metrics as not independently recomputed.

**Important interpretation rule**

A result that improves recall but reduces mAP is a trade-off, not an unqualified overall improvement. State the exact metric, direction, and magnitude.

**Deliverables**

- `audit/METRICS_AUDIT.md`
- `audit/recomputed_metrics.csv`
- `audit/claim_reconciliation.csv`

**Gate:** Every headline number must be traceable to a checkpoint, evaluation configuration, and output artifact.

### Stage 5 — ONNX export and numerical parity audit

**Tasks**

1. Identify the exact checkpoint exported for each model variant.
2. Inspect export settings: opset, input shape, dynamic/static dimensions, precision, simplification, and export tool version.
3. Validate each ONNX file structurally.
4. Run ONNX inference on a fixed, documented sample set.
5. Compare source-framework and ONNX outputs before and after equivalent postprocessing.
6. Report absolute and relative numerical differences and detection-level agreement.
7. Verify that custom SS2D operations are represented correctly in the ONNX graph.
8. Count graph nodes and operator types from the actual ONNX files.
9. Check whether reported graph-node counts (including the reported 14,767-node SS2D graph and operator counts) are accurate.
10. Record file sizes and clarify whether reported “weights size” means checkpoint size, ONNX file size, or another artifact.

**Deliverables**

- `audit/ONNX_AUDIT.md`
- `audit/onnx_parity.csv`
- `audit/onnx_operator_counts.csv`

**Gate:** Do not treat a successful export as proof of numerical equivalence or edge-runtime compatibility.

### Stage 6 — Edge benchmark and resource audit

**Tasks**

1. Identify the exact hardware and software environment used for each benchmark.
2. Record CPU model, core/thread count, OS, runtime version, thread count, CPU affinity, governor, frequency behavior, and thermal/power conditions where available.
3. Inspect benchmark scripts and raw outputs.
4. Verify warm-up count, measured iteration count, input handling, synchronization, timing boundaries, and whether preprocessing/postprocessing are included.
5. Recompute median, mean, standard deviation, P90, P95, and FPS from raw latency samples where available.
6. Use one consistent FPS definition. If FPS is calculated as `1000 / mean_latency_ms`, apply that consistently and label it accordingly.
7. Reconcile the reported host and Raspberry Pi latency/FPS values.
8. Verify RSS measurements and whether they represent process RSS or total system memory.
9. Recalculate memory headroom using the stated memory budget. If the budget is 1024 MB, calculate headroom against 1024 MB rather than 1000 MB.
10. Inspect whether a 100-iteration benchmark plus warm-up is sufficient to support claims about memory leaks, long-term stability, or absence of kernel crashes. Do not make those claims without a suitably long sustained test.
11. Determine whether TensorRT results actually exist. If the paper only measured CPU/ONNX Runtime, do not present TensorRT as a completed benchmark.
12. Investigate the reported ONNX graph expansion and memory increase. Treat graph-node count and latency/RSS correlation as observations, not proof of cache misses or memory-bandwidth causation unless profiling or hardware-counter evidence supports that explanation.
13. Evaluate whether “real-time” language is justified by the measured throughput. The currently reported Raspberry Pi Case B and Case C throughput is below 1 FPS; do not describe these measurements as real-time operation.

**Deliverables**

- `audit/EDGE_BENCHMARK_AUDIT.md`
- `audit/edge_latency_recomputed.csv`
- `audit/resource_measurements.csv`
- A table separating measured results from hypotheses and future deployment targets

**Gate:** Edge claims must match the actual tested runtime and hardware. Separate portability/memory feasibility from real-time performance.

### Stage 7 — Reproducibility and reference audit

**Tasks**

1. Verify all dependency versions from lockfiles, environment files, or runtime logs.
2. Reconcile version references in the manuscript with the actual environment.
3. Check that all cited papers, software packages, datasets, and hardware references exist and support the claims attributed to them.
4. Verify that the Ultralytics version in the bibliography agrees with the reported runtime version.
5. Inspect all artifact hashes. A truncated hash is not sufficient for integrity verification.
6. Verify whether the project includes:
   - full dataset/split manifest;
   - full checkpoint hashes;
   - training and evaluation configs;
   - code commit identifier;
   - export settings;
   - benchmark scripts and raw logs.
7. Check that optimizer settings are unambiguous. If `momentum=0.937` is used with AdamW, verify whether this represents the first beta value and report the full beta pair.
8. Check scheduler, warmup, weight decay, gradient clipping, AMP/precision, augmentation, batch size, image size, and all other material training settings.
9. Check FLOPs/parameter-count methods and custom-operation accounting.
10. Check reference formatting, missing references, and unsupported “first” or “no prior work” claims.

**Deliverables**

- `audit/REPRODUCIBILITY_AUDIT.md`
- `audit/reproducibility_manifest.json` or equivalent
- `audit/REFERENCE_AUDIT.md`

**Gate:** A reader must be able to identify the exact code, data split, checkpoint, configuration, and evaluation procedure behind each reported result.

### Stage 8 — Paper claim and narrative audit

**Tasks**

1. Build a claim-to-evidence ledger for the abstract, introduction, methods, results, discussion, and conclusion.
2. Check that each claim is supported by the relevant experiment or source.
3. Flag overstatements, contradictions, ambiguous units, and unsupported causal explanations.
4. Check that “real-time,” “edge-ready,” “independent validation,” “zero leakage,” “statistically reliable,” and “convergence” are used only when justified.
5. Check that the paper's novelty is framed accurately. Related work includes Mamba-based YOLO variants; do not claim novelty merely from replacing a YOLO attention block with Mamba.
6. Assess whether the defensible contribution is more accurately framed around warehouse-specific adaptation, cross-facility validation, ONNX portability, and physical CPU edge-feasibility measurements.
7. Check that comparisons across different datasets, resolutions, hardware, and runtimes are explicitly qualified as non-identical conditions.
8. Check that the introduction does not introduce object categories absent from the dataset taxonomy unless clearly contextualized.
9. Remove or flag exaggerated quantitative language that is not supported by measurements.
10. Ensure limitations are explicit, especially small validation counts, single-seed experiments, validation-set checkpoint selection, and low measured edge throughput.

**Deliverables**

- `audit/CLAIM_EVIDENCE_LEDGER.csv`
- `audit/PAPER_CONSISTENCY_AUDIT.md`
- A prioritized manuscript correction list

**Gate:** The final narrative must not claim more than the experiments demonstrate.

### Stage 9 — Final audit synthesis

Produce one consolidated report that answers:

1. Does the implementation match the manuscript?
2. Are the dataset and split claims verified?
3. Are the experimental comparisons methodologically valid?
4. Are the headline metrics reproducible?
5. Is the ONNX export numerically consistent?
6. Are the edge measurements valid and correctly interpreted?
7. Can another researcher reproduce the results from the available artifacts?
8. Which claims can remain, which must be narrowed, and which must be removed?
9. What are the minimum additional experiments or artifacts needed before submission?

Classify every finding by:

- **Severity:** Critical / Major / Moderate / Minor.
- **Evidence status:** Verified / Partially verified / Unverified / Contradicted.
- **Location:** manuscript section/table/figure and code file/line where applicable.
- **Impact:** what claim, result, or conclusion is affected.
- **Required action:** exact correction, test, or artifact needed.

**Final deliverables**

- `audit/FINAL_AUDIT_REPORT.md`
- `audit/CLAIM_EVIDENCE_LEDGER.csv`
- `audit/ISSUE_TRACKER.csv`
- `audit/NEXT_ACTIONS.md`

## 5. Known issues to explicitly verify

These are audit leads from the current manuscript review, not established implementation facts. Confirm or reject each using the actual project artifacts.

1. The validation partition may be used for checkpoint selection and hyperparameter decisions, making “independent validation” wording misleading.
2. The paper's “zero-leakage” claim may lack a reproducible split manifest and duplicate/near-duplicate audit.
3. The class-count change from 80 to 4 may not discard every detection-head tensor as described; inspect actual state-dict transfer behavior.
4. Arm 3 changes both architecture and training protocol; a baseline two-stage control may be missing.
5. The manuscript's `W_out = 0` diagram may conflict with zero-initialized BatchNorm affine parameters in the equations/text.
6. Zero initialization may have important gradient-flow implications that need direct testing.
7. The Case C description may not fully account for both P4 and P5 scans.
8. Some reported metric improvements may mix percentage points and relative percentages.
9. Some per-class precision/recall statements may not be represented in the result tables.
10. Statistical reliability claims may be unsupported if only one seed was run.
11. FPS calculations may mix mean and median latency.
12. The methodology may mention P90/P95 while the table reports only median and mean ± standard deviation.
13. RSS headroom arithmetic may use inconsistent 1000 MB and 1024 MB budgets.
14. A short benchmark may not justify claims about memory leaks or long-duration stability.
15. TensorRT may be mentioned without corresponding measured results.
16. ONNX graph expansion may be observed, but cache/memory-bandwidth causation may not have been measured.
17. The manuscript may use “real-time” language despite sub-1-FPS Raspberry Pi measurements.
18. The Ultralytics bibliography version may not match the runtime version.
19. Truncated SHA-256 hashes may prevent artifact verification.
20. The FLOPs method may not account correctly for custom SS2D operations.

## 6. Working rules for the auditing agent

- Read the repository before proposing code changes.
- Do not silently repair discrepancies in the manuscript or implementation.
- Do not edit or overwrite original project files.
- Do not invent missing logs, metrics, hashes, dataset metadata, or benchmark results.
- Do not treat a paper statement as evidence for the implementation.
- Do not treat a code comment as proof of runtime behavior.
- Prefer executable tests and raw artifacts over prose descriptions.
- Keep an evidence trail: file path, line number, command, configuration, and output.
- When running commands, record the exact command and relevant output.
- Use small isolated tests for tensor shapes, initialization, gradients, and numerical parity before launching expensive training.
- Do not start new full training runs until the audit identifies exactly what question the run is intended to answer.
- If a test cannot be run because of hardware, dependency, or data limitations, state the limitation and provide a reproducible test procedure.
- Keep findings factual and specific. Avoid generic advice such as “improve reproducibility” without naming the missing artifact or setting.
- Do not optimize for making the paper look stronger. Optimize for making every claim defensible.
- Do not declare the project “ready” based on manuscript quality alone.

## 7. Execution strategy and priority

Use the following order to minimize wasted work:

1. Artifact inventory and provenance.
2. Dataset/split verification.
3. Architecture forward-pass, shape, initialization, and gradient tests.
4. Checkpoint-transfer and training-protocol audit.
5. Metric reconstruction and experimental-control assessment.
6. ONNX export and parity.
7. Edge benchmark reconstruction.
8. Reproducibility and reference audit.
9. Claim-to-evidence ledger and final manuscript recommendations.

If a critical mismatch is found, document it immediately. Continue other independent audit stages where possible, but do not conceal the blocker.

## 8. Definition of done

The task is complete only when:

- Every available relevant artifact has been inventoried.
- Every major paper claim has an evidence status.
- Architecture and checkpoint behavior have been traced to code and, where necessary, tested.
- Dataset counts and split semantics have been independently checked or marked unverified.
- Metrics and benchmark arithmetic have been recomputed wherever raw data permits.
- Missing controls and confounds are explicitly documented.
- Edge claims are limited to the hardware/runtime actually tested.
- A prioritized issue tracker identifies exact next actions.
- The final report clearly separates verified facts, partial evidence, assumptions, and unavailable evidence.

A successful audit may conclude that the paper is **not ready for submission**. That is an acceptable outcome if it is supported by evidence.

## 9. Suggested final report structure

Use this structure for `FINAL_AUDIT_REPORT.md`:

1. Executive summary
2. Overall readiness verdict
3. Scope and artifacts inspected
4. Critical blockers
5. Dataset and split findings
6. Architecture and implementation findings
7. Checkpoint and training findings
8. Metrics and ablation findings
9. ONNX findings
10. Edge benchmark findings
11. Reproducibility and references
12. Paper claims that are supported
13. Claims that must be narrowed or removed
14. Missing artifacts and experiments
15. Prioritized remediation plan
16. Appendix: commands, test outputs, hashes, and evidence references
