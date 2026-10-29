---
description: Audit research papers for scientific validity, evidence
  traceability, numerical consistency, reproducibility, methodological
  clarity, and defensible claims. Use when reviewing, revising, or
  preparing an academic paper for submission.
name: research-paper-audit
---

# Research Paper Audit Skill

## Purpose

Perform a rigorous, evidence-led audit of a research paper. Prioritize
scientific correctness and traceability over stylistic polishing. The
goal is to identify what is supported, what is inconsistent, what is
missing, and what must be verified before submission.

This skill is especially suitable for empirical computer-science papers,
including machine learning, computer vision, model architecture,
benchmarking, and edge-device feasibility studies. Adapt the checks to
the paper's actual discipline and claims.

## Core rules

1.  **Do not invent evidence.** Never fabricate measurements, citations,
    implementation details, dataset properties, statistical tests, or
    experimental outcomes.
2.  **Do not silently reconcile contradictions.** Report conflicting
    values and ask which source is authoritative; do not choose one
    without evidence.
3.  **Trace claims to sources.** For every material finding, identify
    the exact section, page, figure, table, equation, code file, log, or
    reference that supports it.
4.  **Separate evidence from inference.** Label findings as directly
    verified, inferred, unverified, or requiring author confirmation.
5.  **Preserve the research's scope.** Do not expand the contribution,
    claim novelty, or generalize results beyond what the evidence
    supports.
6.  **Check the actual artifacts when available.** Compare the
    manuscript with source code, configuration files, experiment logs,
    raw result exports, and cited papers. A manuscript alone cannot
    verify implementation details or reproduce results.
7.  **Do not treat plausibility as proof.** A technically plausible
    explanation is not an experimentally demonstrated result.
8.  **Do not rewrite scientific claims just to make them sound
    stronger.** Prefer precise, bounded wording over promotional
    language.
9.  **Distinguish editorial issues from validity issues.** Grammar fixes
    must not distract from numerical contradictions, methodological
    flaws, or unsupported conclusions.
10. **State limitations.** If a file, log, source, or experimental
    artifact is unavailable, explicitly say what cannot be verified.

## Required workflow

### Phase 1 --- Inventory and establish the study

Before judging the paper:

-   Identify the title, research question, claimed contributions, task,
    dataset(s), model(s), baselines, metrics, hardware, and main
    conclusions.
-   List all available artifacts: manuscript versions, appendices,
    source code, model/config files, training logs, evaluation exports,
    hardware benchmark scripts, and references.
-   Record which version is being audited. Do not assume an older
    manuscript or old metric is still authoritative.
-   Build a short claim inventory:
    -   Research question and hypothesis
    -   Claimed novelty/contribution
    -   Dataset and split claims
    -   Architecture and implementation claims
    -   Training and evaluation protocol
    -   Quantitative results
    -   Efficiency and deployment claims
    -   Limitations and generalization claims

If the task is limited to the manuscript, audit only what the manuscript
itself can support and flag implementation-dependent claims as
unverified.

### Phase 2 --- Scientific question and contribution

Check that:

-   The research question is specific and answerable by the experiments.
-   The contribution is stated in concrete, testable terms.
-   Novelty is supported by a relevant literature review, not an
    absolute statement based on a few examples.
-   Related work compares methods fairly and explains the gap the study
    addresses.
-   The proposed method is distinguishable from existing methods in a
    way the paper can demonstrate.
-   The paper does not confuse engineering integration, parameter
    reduction, feasibility evaluation, and algorithmic novelty.
-   The abstract, introduction, methods, results, and conclusion
    describe the same contribution.

Flag terms such as *novel*, *first*, *state-of-the-art*, *robust*,
*real-time*, *lightweight*, *generalizable*, and *significant* unless
the manuscript defines and supports them.

### Phase 3 --- Dataset and experimental design

Verify, where evidence is available:

-   Dataset name, version, source, license/access conditions, and task
    suitability.
-   Number of samples, classes, annotations, and split sizes.
-   Train/validation/test separation and possible leakage.
-   Whether frames from the same video, scene, camera, or recording
    session could appear across splits.
-   Class imbalance and whether aggregate metrics hide weak classes.
-   Annotation format, preprocessing, image resolution, augmentation,
    and filtering rules.
-   Whether validation data is used for model selection and whether an
    independent test set exists.
-   Whether baselines receive comparable data, resolution, training
    budget, augmentation, and checkpoint-selection rules.
-   Whether sample size and split design justify the strength of the
    conclusions.
-   Whether the paper reports random seeds, repeated runs, variance,
    confidence intervals, or other uncertainty measures when needed.

Do not call a dataset representative, statistically reliable, or
leakage-free unless the design or evidence supports that conclusion.

### Phase 4 --- Architecture and implementation traceability

For architecture papers and model modifications, check:

-   The diagram, prose, equations, configuration, and implementation
    agree.
-   Every module has a clear input/output shape and tensor dimension.
-   Layer names and indices are consistent across the paper, code,
    tables, and deployment cases.
-   The location of each inserted, replaced, or removed block is
    unambiguous.
-   Parameter counts, FLOPs/MACs, input resolution, and model variants
    are calculated under a stated convention.
-   The paper distinguishes theoretical operation counts from measured
    runtime.
-   Equations define their symbols, dimensions, assumptions, and
    recurrence/update order.
-   The described implementation matches the equations; do not assume
    equivalence from similar notation.
-   Custom kernels, scan directions, tensor reshaping, normalization,
    gating, padding, and state updates are described sufficiently to
    reproduce the method.
-   Any ablation isolates the intended design change and avoids changing
    several factors at once.
-   The baseline and proposed model use compatible definitions and
    measurement conditions.

For state-space, Mamba, or multidirectional scan modules specifically,
inspect direction ordering/alignment, sequence-to-spatial reshaping,
reverse-scan restoration, state dimensions, gating, normalization, and
whether directional outputs are aligned before fusion. Treat these as
implementation checks, not assumed defects.

### Phase 5 --- Training and evaluation protocol

Check that the paper specifies enough detail to reproduce the
experiments:

-   Software/library versions and relevant environment details.
-   Hardware model, accelerator, memory, operating system, and inference
    backend.
-   Optimizer, learning rate/schedule, batch size, epochs, weight decay,
    warmup, and stopping criteria.
-   Input size, precision, batch size, preprocessing, postprocessing,
    and confidence/IoU thresholds.
-   Checkpoint-selection rule and the exact checkpoint used for every
    table and figure.
-   Evaluation split and metric implementation.
-   Whether results are from a single run or multiple seeds.
-   Whether training, validation, export, compilation, warm-up, and
    timing are separated.
-   Whether the benchmark measures end-to-end latency or only model
    execution.

Check for ambiguous wording such as "best checkpoint" when one metric
selects the checkpoint but another metric is reported. Distinguish best
validation-loss checkpoint, peak validation metric, and final-epoch
checkpoint.

### Phase 6 --- Numerical and cross-document consistency audit

Create a consistency ledger for all important numbers. Compare values
across:

-   Abstract
-   Introduction/contribution summary
-   Methods
-   Results text
-   Tables and table captions
-   Figures and figure captions
-   Discussion
-   Conclusion
-   Appendix/supplementary materials
-   Source code/configuration
-   Raw logs or exported evaluation results

Check:

-   Exact agreement of repeated values, units, percentages, decimal
    places, and model labels.
-   Correct denominators and definitions for precision, recall, F1, mAP,
    latency, FPS, parameters, FLOPs, and memory.
-   Arithmetic in tables, averages, percentage changes, and relative
    improvements.
-   Whether reported gains use the stated baseline and the same
    metric/checkpoint.
-   Whether per-class results aggregate to the reported overall result
    when the aggregation definition permits that comparison.
-   Whether peak, best, mean, median, and final values are clearly
    distinguished.
-   Whether plotted curves agree with table values and written
    descriptions.
-   Whether units are consistent (MB vs MiB, ms vs seconds, parameters
    vs millions).
-   Whether rounding explains a small difference; do not use rounding to
    excuse material mismatches.
-   Whether all reported values can be traced to a specific experiment,
    checkpoint, script, or log.

For each contradiction, report both values and their locations. Do not
silently correct the manuscript.

### Phase 7 --- Claims, interpretation, and causal language

For every major conclusion, ask:

1.  What exact evidence supports this statement?
2.  Does the evidence measure the claimed property directly?
3.  Is the claim limited to the tested dataset, hardware, resolution,
    and protocol?
4.  Is a plausible explanation being presented as a proven cause?
5.  Are negative or mixed results acknowledged?
6.  Is the conclusion still valid when final results are used instead of
    peak/intermediate results?

Examples of common overreach:

-   Claiming real-time performance from memory feasibility or operator
    compatibility alone.
-   Claiming better spatial reasoning from detection metrics alone.
-   Claiming fewer emergency stops without a safety-event or operational
    evaluation.
-   Claiming robustness without defined perturbations or stress tests.
-   Claiming generalization from one small or correlated validation
    split.
-   Claiming statistical significance without a suitable test and
    repeated/independent observations.
-   Claiming a method is more accurate based on an intermediate peak
    while comparing other methods at their final checkpoints.
-   Claiming all variants improve a loss when the table shows that only
    some do.

Recommend narrower language when evidence is limited. Explicitly
distinguish *measured result*, *interpretation*, *hypothesis*, and
*future work*.

### Phase 8 --- Benchmark and deployment audit

For edge or systems papers, verify:

-   The exact hardware variant, RAM, power mode, cooling, OS, driver,
    and accelerator/runtime.
-   Whether the model was run on CPU, GPU, NPU, or a named inference
    provider.
-   Model export format, precision, operator support, fallback behavior,
    and compilation settings.
-   Warm-up policy, number of timed iterations, synchronization, and
    whether timing includes preprocessing/postprocessing.
-   Latency statistics reported and their definitions: mean, median,
    P90/P95, standard deviation, or range.
-   FPS calculation and whether it is derived from mean or median
    latency.
-   Peak memory vs process RSS vs device memory, and what each includes.
-   Whether results were measured on-device or estimated.
-   Whether comparisons use the same batch size, image size, precision,
    and runtime.
-   Whether power consumption, thermal throttling, or sustained
    performance are relevant and measured.
-   Whether the paper's "real-time" threshold is defined and justified.

Do not infer that a model is real-time because it runs on an edge
device. Report feasibility, compatibility, memory use, and throughput as
separate outcomes. Do not claim TensorRT, ONNX Runtime, NPU, or GPU
acceleration unless the benchmark actually used it.

### Phase 9 --- Figures, tables, equations, and references

For every figure and table:

-   Confirm the title/caption describes the content accurately.
-   Ensure axes, units, legend, class names, abbreviations, thresholds,
    and error bars are defined.
-   Check readability, resolution, numbering, and references from the
    main text.
-   Verify that captions do not claim more than the visual evidence
    demonstrates.
-   Check that tables define whether values are peak, final, average, or
    best-checkpoint results.
-   Verify equation numbering, symbol definitions, dimensions, and
    consistency between notation and prose.

For references:

-   Confirm each citation supports the specific claim attached to it.
-   Check author/title/venue/year/DOI or URL when source access permits.
-   Identify citations that are missing, irrelevant, outdated for a
    time-sensitive claim, or cited for a result they do not report.
-   Check that references cited in text appear in the bibliography and
    vice versa.
-   Do not invent bibliographic details. Mark unverified references for
    manual confirmation.
-   For claims of novelty, search the relevant literature when external
    research is in scope and state the search scope and date; never
    treat a limited search as proof that no prior work exists.

### Phase 10 --- Reproducibility and limitations

Assess whether an independent researcher could reproduce the study using
the paper and available artifacts. Check for:

-   Data access and preprocessing instructions.
-   Code, configuration, dependency versions, and checkpoints where
    available.
-   Exact experiment commands or sufficient procedural detail.
-   Clear mapping from tables to runs/checkpoints.
-   Random seeds and repeatability information.
-   Limitations that acknowledge dataset scale, split dependence,
    hardware coverage, model trade-offs, and missing experiments.
-   Clear separation between demonstrated findings and proposed future
    work.

Do not demand every possible experiment. Prioritize experiments needed
to support the paper's central claims.

## Finding classification

Assign each finding one severity:

-   **Critical** --- undermines the main result or makes a key
    conclusion unreliable; e.g., a metric is computed incorrectly, data
    leakage invalidates evaluation, or a central result cannot be
    traced.
-   **Major** --- materially affects interpretation, reproducibility, or
    a central claim; e.g., a table contradiction, unsupported
    performance claim, unclear checkpoint selection, or mismatched
    benchmark protocol.
-   **Moderate** --- creates meaningful ambiguity or weakens
    reproducibility but may not invalidate the main result.
-   **Minor** --- editorial, formatting, terminology, or low-impact
    clarity issue.

Also assign a verification status:

-   **Verified** --- checked against the relevant primary
    artifact/source.
-   **Manuscript-only** --- internally checked against the paper, but
    not independently verified.
-   **Likely issue** --- evidence suggests a problem, but the source
    artifact is missing or incomplete.
-   **Author confirmation required** --- the correct interpretation
    depends on information not available.
-   **Not checked** --- outside the current audit scope.

Do not mark an issue "verified" merely because it appears consistently
in multiple manuscript sections; consistency is not independent
validation.

## Required output format

### 1. Executive verdict

Give a concise status: - **Not ready for submission** - **Major revision
needed** - **Minor revision needed** - **No major issues found within
the audit scope**

State the most important reason and the audit scope. Avoid implying
peer-review acceptance or scientific certification.

### 2. Critical and major findings

For each finding, use:

-   **ID and title**
-   **Severity**
-   **Verification status**
-   **Location(s)** --- page/section/table/figure/equation or artifact
    path
-   **Evidence** --- quote or report only the minimum relevant content
-   **Why it matters**
-   **Required action**
-   **What would resolve it** --- e.g., source log, config, calculation,
    citation, or rerun

### 3. Numerical consistency ledger

Use a table with columns:
`Metric/claim | Location A | Location B/source artifact | Status | Action`

Include all material metrics and contradictions. If the paper has many
values, start with those used in the abstract, main result tables, and
conclusion.

### 4. Claim-to-evidence matrix

Use a table with columns:
`Claim | Evidence required | Evidence available | Status | Safer wording/action`

Include novelty, accuracy, efficiency, robustness, real-time,
generalization, and practical-impact claims when relevant.

### 5. Reproducibility gaps

List only details needed to reproduce the central experiments, separated
into: - Available and verified - Reported but not independently
verified - Missing or ambiguous

### 6. Prioritized repair plan

Order fixes by dependency: 1. Resolve source-of-truth data and numerical
contradictions. 2. Verify implementation/configuration and checkpoint
mapping. 3. Repair methodology and benchmark reporting. 4. Narrow
unsupported interpretations and conclusions. 5. Verify references and
related-work claims. 6. Polish wording, figures, and formatting.

For each action, specify the artifact needed and a completion criterion.

### 7. Submission gate

End with a short checklist of unresolved Critical/Major issues. Do not
declare the paper ready while a contradiction affecting a central claim
remains unresolved.

## Special handling for machine-learning results

-   Never mix validation and test results without clearly labeling them.
-   Never present a best epoch for one method against a final epoch for
    another unless the comparison protocol explicitly justifies it.
-   Distinguish model-selection metrics from final evaluation metrics.
-   Treat one run as one observed outcome, not proof of statistical
    significance.
-   Check whether the reported metric is macro/micro/weighted and
    whether per-class values are compatible with the aggregate.
-   Treat class-level precision/recall claims as unsupported if only mAP
    values are shown.
-   Distinguish parameter count, FLOPs, measured latency, memory use,
    energy, and operational utility; they are not interchangeable.
-   A visual example is qualitative evidence, not a substitute for
    aggregate evaluation.
-   An ablation supports a causal interpretation only to the extent that
    the comparison controls other changes.
-   Keep hypotheses for future acceleration or deployment explicitly
    hypothetical until measured.

## Special handling for research-paper editing

When asked to revise the paper:

1.  Complete or update the audit first if the requested edit could
    change scientific meaning.
2.  Preserve the author's intended contribution and terminology unless
    they are demonstrably inconsistent.
3.  Do not fabricate missing values or fill evidence gaps with plausible
    numbers.
4.  Use placeholders such as `[verify against evaluation log]` when a
    factual value is unresolved.
5.  Provide a change log that distinguishes wording edits from changes
    to scientific claims.
6.  If multiple versions exist, identify the exact version being edited
    and avoid copying stale values from earlier versions.

## Scope and stopping rules

-   If only a PDF is available, do not claim code-level,
    reference-level, or experiment-level verification unless the
    required sources were actually inspected.
-   If code/logs are supplied, inspect only the relevant files needed to
    resolve each claim; do not assume a file name proves its contents.
-   If web research is requested, use reliable primary sources where
    possible and cite the sources used. Separate external literature
    findings from the manuscript's own evidence.
-   If a conclusion depends on rerunning training, evaluating a
    checkpoint, or testing hardware, identify the required experiment;
    do not imply it was performed.
-   Prefer a small number of high-impact, actionable findings over a
    long list of speculative criticisms.
