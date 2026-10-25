# YOLO11s–Mamba Architecture Implementation Specification

> **Purpose:** This document is the implementation contract for an AI coding agent working on the research prototype. Read it before modifying code.
>
> **Research goal:** Investigate whether selective integration of a 2D State Space Model (SSM), implemented with an SS2D/Visual-Mamba-style block, into a lightweight YOLO11s detector improves warehouse-object detection while preserving deployability.
>
> **Important:** This is an experimental research implementation, not a claim that the architecture is novel or faster. Establish the baseline and verify tensor shapes, dependency compatibility, and measurements before making claims.

---

## 1. Non-negotiable implementation rules

1. **Inspect the existing repository first.** Do not assume its Ultralytics version, model YAML, layer indices, module names, training scripts, or installed Mamba dependencies.
2. **Preserve a runnable, unmodified YOLO11s baseline.** Do not edit the upstream model in place in a way that makes the baseline irreproducible.
3. **Implement only the first experiment initially: Case B — replace the deepest eligible C2PSA block with a shape-compatible SS2D block.** Do not implement the gated neck, a new detection head, extra attention modules, new loss functions, tracking, or RGB-D fusion in this milestone.
4. **Keep the original YOLO11s neck and Detect head unchanged.**
5. **Keep the high-resolution feature path unchanged.** Do not insert Mamba into P3 or other high-resolution stages in the first experiment.
6. **Do not hard-code layer indices or channel counts** until they have been verified from the actual model graph/configuration.
7. **Do not silently fall back to the baseline** if SS2D fails. Raise a clear error and explain the incompatibility.
8. **Do not claim improved accuracy, lower FLOPs, lower latency, or edge compatibility without measurements.**
9. **Keep all experimental changes configurable and reversible.** Use a custom model config and dedicated module file where practical.
10. **Make the smallest change that works.** Do not refactor unrelated training, dataset, logging, or deployment code.

---

## 2. Target model and high-level data flow

### Baseline

Use the repository's supported **YOLO11s detection model** as the control. Confirm the exact upstream YAML/configuration and package version before implementation.

### Experimental Case B

```text
RGB input: [B, 3, 640, 640]
        |
        v
YOLO11s CNN backbone
  - preserve early/high-resolution CNN stages
  - preserve P3 and P4 generation
        |
        v
Deepest backbone feature stage (normally P5 / stride 32)
  - replace the selected deepest C2PSA block with a compatible SS2D block
  - preserve the input/output tensor shape and channel count
        |
        v
Original YOLO11s neck (FPN/PAN)
  - unchanged
        |
        v
Original YOLO11s Detect head
  - unchanged
        |
        v
Per-class 2D boxes + confidence scores
```

At an input size of 640 × 640, the common detection pyramid resolutions are P3 = 80 × 80, P4 = 40 × 40, and P5 = 20 × 20 (strides 8, 16, and 32). Treat these as expectations to verify, not hard-coded assumptions. Actual channels and layer positions must come from the instantiated model.

### Tensor contract

The custom block must accept and return a feature tensor with the same shape:

```text
Input:  [B, C, H, W]
Output: [B, C, H, W]
```

For the first experiment, preserve `C`, `H`, and `W`. Do not change the neck/head interface. The module must support batch size 1 and the training batch sizes used by the project.

---

## 3. What “SS2D/Mamba block” means in this project

Use a genuine, documented 2D selective-scan/SSM implementation that is compatible with the project's PyTorch, CUDA, and deployment constraints. **Do not substitute a Transformer, ordinary self-attention, or a convolution and still call it Mamba.**

Before selecting a dependency:

- Inspect the current Python, PyTorch, CUDA, and GPU environment.
- Check the dependency's official installation and compatibility requirements.
- Check whether it supports the project's platform and whether it can run in forward/backward passes.
- Record the exact package/repository version or commit in the experiment notes.
- Check whether model export/inference will require custom operators. Export compatibility is a later validation step, not an assumption.

If no compatible implementation is available, stop and report the blocker with options. Do not invent an SS2D implementation from a paper diagram and present it as a faithful reproduction.

### Recommended block interface

Create a dedicated module with a clear name such as `WarehouseSS2DBlock`. Its interface should be compatible with the model framework's module parser.

Conceptual behavior:

1. Accept a BCHW feature map.
2. Convert/project channels as required by the chosen SS2D implementation.
3. Convert BCHW to the layout expected by that implementation.
4. Apply the genuine 2D selective-scan operation.
5. Restore BCHW layout.
6. Apply an output projection if required.
7. Use a residual path and normalization only if consistent with the selected reference implementation and the replacement's shape contract.
8. Return exactly `[B, C, H, W]`.

The above is an interface contract, not permission to improvise the mathematical core. Document the chosen block's source and design, including normalization, projections, expansion ratio, state dimension, scan directions, residual path, and initialization.

### Numerical and shape checks

Add tests for:

- Output shape equals input shape.
- Output tensor is finite (`NaN`/`Inf` absent).
- Gradients propagate to trainable parameters.
- Batch size 1 works.
- A representative small feature map works on CPU if the implementation claims CPU support; otherwise clearly mark the module as accelerator-only.
- The module can be inserted into the model and a forward pass reaches the unchanged Detect head.

Do not require CPU support from a CUDA-only selective-scan dependency. Detect and report the supported execution device clearly.

---

## 4. Repository reconnaissance — do this before editing

The coding agent must first inspect and report:

1. Repository tree and current working branch.
2. Python, PyTorch, Ultralytics, CUDA, and relevant SSM dependency versions.
3. The exact YOLO11s YAML/config used by the project.
4. The instantiated layer graph, identifying:
   - backbone and neck boundaries;
   - the deepest `C2PSA` block(s);
   - the actual feature-map shapes and channel counts at the candidate insertion point;
   - where P3/P4/P5 are produced and routed to the neck;
   - how custom modules are imported and parsed by this framework version.
5. Existing training/evaluation scripts and how baseline checkpoints are loaded.
6. Whether the chosen SS2D implementation can be installed and executed in this environment.

Deliver a short `docs/architecture_recon.md` or equivalent before implementing. If the repository does not use Ultralytics YOLO11, adapt the plan to its actual structure and explain the mapping rather than forcing these assumptions.

---

## 5. Experiment configuration and naming

Keep the baseline and experimental model configurations separate. Suggested names (adapt to repository conventions):

- `yolo11s_baseline.yaml` — exact unmodified baseline config or a documented reference to the upstream config.
- `yolo11s_ss2d_p5.yaml` — Case B, deepest eligible C2PSA replaced by SS2D.
- `models/modules/warehouse_ss2d.py` — custom block, if this matches the existing project structure.

Do not duplicate large upstream configuration files unnecessarily. If a custom YAML must repeat the full graph, document the upstream version and the exact changed layer. Ensure the custom module is registered/importable through the framework's normal model-building path; do not rely on a temporary notebook import.

Expose the experimental parameters in one explicit configuration location where feasible. Record at least:

- selected replacement layer and its verified position;
- SS2D implementation and version/commit;
- inner/state dimensions and expansion ratio;
- normalization and residual choices;
- input resolution;
- model configuration hash or revision.

Do not add a second SS2D stage yet. Case C (P4 + P5) is a later experiment after Case B is stable and measured.

---

## 6. Required implementation phases

### Phase 0 — Baseline integrity

- Build the unmodified YOLO11s model.
- Run a forward pass with a dummy input of `[1, 3, 640, 640]`.
- Confirm output format and detection pyramid shapes.
- Run a minimal training-step or framework-supported loss/backward smoke test.
- Save model summary, package versions, baseline config path/hash, and parameter count.
- Confirm the baseline can still be trained and evaluated using existing scripts.

**Acceptance:** Baseline works before any architecture changes.

### Phase 1 — Implement the SS2D block in isolation

- Add the module in a dedicated file.
- Use a genuine compatible SS2D implementation.
- Add shape/finite-value/gradient tests.
- Avoid changing unrelated files.
- Document the external dependency and any platform limitations.

**Acceptance:** Unit tests pass and the module's mathematical implementation is traceable to a stated reference.

### Phase 2 — Integrate Case B

- Locate the deepest eligible C2PSA block from the actual model graph.
- Replace only that block in a separate experimental model config.
- Keep the block's input/output channels and BCHW shape unchanged.
- Preserve the original neck, Detect head, class configuration, and preprocessing.
- Ensure model parsing, summary, forward pass, loss computation, and backward pass work.
- Add a test that fails clearly if the target layer or expected shape changes after a framework/config update.

**Acceptance:** Experimental model builds and completes forward/backward smoke tests; baseline remains unchanged and runnable.

### Phase 3 — Minimal comparison run

Train baseline and Case B under a matched protocol. Do not change the dataset, split, input size, augmentation, optimizer, schedule, batch size (unless memory constraints are documented), or checkpoint-selection rule between the two runs.

Log the actual configuration and random seed. If the full training budget is not yet known, start with a short smoke run only to validate the pipeline; do not treat smoke-run metrics as final evidence.

**Acceptance:** Both runs produce valid checkpoints and evaluation outputs, with comparable training settings documented.

### Phase 4 — Profile and decide

Compare baseline and Case B on the same evaluation split:

- mAP@0.5 and mAP@0.5:0.95;
- precision and recall;
- per-class metrics and confusion matrix;
- parameter count and FLOPs, with the measurement tool and input size stated;
- checkpoint size;
- peak memory during training/inference where available;
- inference latency/FPS, with hardware, batch size, precision, warm-up, timing method, and post-processing stated.

Do not infer real-time performance from FLOPs alone. Do not claim Jetson/RK3588 performance unless measured on that exact device/runtime. If SS2D cannot export to the target runtime, report it as a deployment limitation rather than hiding it.

**Decision:** Proceed to Case C only if Case B is technically stable and the accuracy/efficiency results justify another stage.

---

## 7. Future experiments — do not implement in the first coding milestone

These are follow-up experiments, not part of the initial implementation.

| Case | Change | When to consider it |
|---|---|---|
| A | Original YOLO11s baseline | Required control |
| B | Replace deepest eligible C2PSA with SS2D at P5 | First experimental model |
| C | Add/replace SS2D at P4 as well as P5, based on the selected B design | Only after B is stable |
| D | Keep C2PSA and add SS2D at P5 instead of replacing it | Only if replacement-versus-addition is a meaningful comparison |
| E | Context-gated neck/fusion | Only after selecting a backbone and defining a separate, controlled hypothesis |

Do **not** implement a new detection head, NMS-free/dual-label assignment, extra attention blocks, image enhancement, new loss functions, tracking/ReID, RGB-D fusion, or a second dataset pipeline in the first milestone.

---

## 8. Dataset/training assumptions

- The model is a **2D RGB object detector**.
- Use one audited primary dataset for the main ablation study.
- Verify the dataset's annotation format, class mapping, train/validation/test split, class distribution, and usable labeled frames before training.
- If extracting frames from videos, split by scene/video/sequence rather than randomly splitting adjacent frames, to reduce leakage.
- Keep a fixed class list and class-to-index mapping across baseline and experimental runs.
- Do not merge datasets just to increase image count without first resolving ontology, domain, annotation, and split differences.
- Dataset conversion/preprocessing must be deterministic and documented.
- The dataset choice is not encoded into the architecture module; keep data configuration separate.

---

## 9. Suggested code layout

Adapt to the existing repository rather than imposing a full restructure.

```text
project/
├── models/
│   ├── configs/
│   │   ├── yolo11s_baseline.yaml
│   │   └── yolo11s_ss2d_p5.yaml
│   └── modules/
│       └── warehouse_ss2d.py
├── tests/
│   ├── test_warehouse_ss2d.py
│   └── test_model_integration.py
├── scripts/
│   ├── inspect_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
├── docs/
│   ├── architecture_recon.md
│   └── experiment_log.md
└── README.md
```

If the project already has an established layout, preserve it. Avoid creating duplicate scripts when equivalent ones already exist.

---

## 10. Required tests

At minimum, implement or adapt tests that verify:

1. **Module shape:** `[B,C,H,W]` in → `[B,C,H,W]` out.
2. **Finite output:** no NaN or Inf for representative inputs.
3. **Gradient flow:** backward pass succeeds and expected parameters receive gradients.
4. **Model construction:** custom YAML/config builds through the normal framework path.
5. **Target layer validation:** replacement applies to the intended deepest block and no other layer is unintentionally modified.
6. **Pyramid/head compatibility:** neck and Detect head receive the same expected tensor shapes as in the baseline.
7. **Forward/backward integration:** a training loss can be computed and backpropagated.
8. **Baseline preservation:** original baseline still builds and runs.
9. **Clear dependency failure:** missing/incompatible SS2D dependency produces an actionable error, not a silent substitution.

Use small test inputs where possible. Full 640 × 640 model tests can be a separate integration test if resource-intensive.

---

## 11. Experiment logging and reproducibility

For every run, save:

- Git commit and working-tree status;
- model config and its hash;
- dataset version/path and split manifest;
- class mapping;
- Python/PyTorch/Ultralytics/CUDA/SS2D dependency versions;
- random seed;
- training hyperparameters and checkpoint-selection rule;
- parameter/FLOP measurement method;
- hardware, runtime, precision, batch size, warm-up and latency method;
- metrics and output checkpoint path;
- known failures or deviations from the protocol.

Never overwrite the baseline checkpoint or experiment outputs. Use separate run directories and explicit names, e.g. `baseline_seed42/` and `ss2d_p5_seed42/`.

---

## 12. Definition of done for the first coding milestone

The milestone is complete only when:

- [ ] Repository reconnaissance report is written.
- [ ] Unmodified YOLO11s baseline still builds and passes a forward/backward smoke test.
- [ ] A genuine SS2D block is implemented with a documented source/dependency.
- [ ] Case B replaces only the verified deepest eligible C2PSA block in a separate model config.
- [ ] The experimental model preserves the expected input/output and neck/head interfaces.
- [ ] Unit and integration tests pass, or any blocker is precisely documented.
- [ ] A minimal matched baseline-versus-Case-B experiment can run end-to-end.
- [ ] Metrics and environment are logged reproducibly.
- [ ] No unsupported accuracy, novelty, speed, or edge-deployment claim is made.

## 13. Instructions for the coding agent's response

Before coding, summarize the repository findings and identify any blockers. Then implement in small, reviewable steps. After each phase, report:

1. Files created or changed.
2. Why each change is necessary.
3. Commands run and their results.
4. Tests passed/failed.
5. Dependency, hardware, or export limitations.
6. The exact next step.

If a critical detail is unknown, inspect the repository or ask one focused question. Do not guess tensor channels, layer indices, package compatibility, or hardware capabilities. Do not expand scope beyond Case B without explicit approval.
