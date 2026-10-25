# Research Decision Log: Lightweight YOLO--Mamba for Warehouse Object Detection

**Status:** Working draft. Hypotheses are not validated results.

## 1. Research scope

**Working title:** A Lightweight Hybrid YOLO--Mamba Architecture for
Real-Time Multi-Class Warehouse Object Detection on Edge Devices.

**Main question:** Can selectively integrating Mamba/SS2D into YOLO11s
improve warehouse object detection while keeping the model efficient
enough for edge deployment?

### Keep the study focused

-   Task: RGB-based 2D bounding-box object detection in warehouse
    scenes.
-   Baseline: YOLO11s.
-   Main architectural direction: selectively integrate SS2D/Mamba into
    deeper feature stages.
-   Optional secondary contribution: Context-Gated Neck, only if
    experiments justify it.
-   Main study: one primary dataset.
-   Deployment: measure on the actual target edge device.

### Out of scope for v1

-   Redesigning the detection head.
-   YOLOv10-style dual-label assignment or NMS-free inference.
-   Adding transformers, several attention modules, image-enhancement
    branches, or multiple new losses.
-   RGB-D fusion, tracking, ReID, or spatial question answering.
-   Combining unrelated datasets just to increase sample count.

**Guardrail:** Every component must answer the main research question.
If it adds complexity without a testable reason, leave it out.

## 2. Proposed architecture v1

``` text
RGB image (640 × 640 × 3)
          |
          v
YOLO11s CNN backbone
  - Keep high-resolution CNN stages
  - Preserve early features for small objects
          |
          v
Selective deep-stage CNN–Mamba integration
  - P5 SS2D is the first experiment
  - P4 + P5 is a follow-up experiment
  - Decide whether SS2D replaces or supplements C2PSA
          |
          v
Multi-scale neck
  - Original YOLO11s FPN/PAN as control
  - Context-Gated Fusion as an optional experiment
          |
          v
Original YOLO11s detection head
          |
          v
2D boxes, classes, confidence scores
```

This is a design hypothesis, not a validated model.

### Verify against the actual model

Inspect the installed Ultralytics version, `yolo11.yaml`, and
instantiated `yolo11s.pt` to confirm: - Layer indices and module
names. - Actual tensor shapes, feature strides, and channel counts at
P3/P4/P5. - Exact placement/configuration of SPPF and C2PSA. - Which
layer outputs feed the neck. - SS2D input/output compatibility and
export support.

For 640 × 640 input, P3/P4/P5 are normally 80 × 80, 40 × 40, and 20 × 20
at strides 8, 16, and 32. Channel counts depend on the actual model
configuration and must be inspected rather than guessed.

## 3. Dataset decision

### First candidate to audit

**NVIDIA PhysicalAI-SmartSpaces**\
https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces

This is a candidate, not yet a final commitment. Confirm the exact
version, license, files, annotations, classes, and usable split.

### Not the primary dataset for now

**NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse**\
https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse

It is oriented toward spatial reasoning/QA and includes RGB-D-related
data and mask annotations. It may require conversion to detection
labels, so it is not the first choice for this focused 2D detection
study.

### Dataset audit checklist

-   [ ] Confirm version and license.
-   [ ] Successfully access/download the intended split.
-   [ ] Programmatically inspect images and annotations.
-   [ ] Confirm usable 2D bounding boxes exist, or document any
    conversion.
-   [ ] Choose classes based on the research scope and class
    distribution.
-   [ ] Count images, boxes, and instances per class.
-   [ ] Inspect object-size distribution, especially small objects.
-   [ ] Inspect occlusion, clutter, lighting, and reflections.
-   [ ] Identify video, sequence, scene, and camera identifiers.
-   [ ] Split by scene/video/sequence where possible to reduce
    near-duplicate frame leakage.
-   [ ] Freeze train/validation/test splits before model comparison.
-   [ ] Record filtering, frame extraction, resizing, and annotation
    conversion.
-   [ ] Never use the test set to select classes, thresholds,
    checkpoints, or architecture variants.

**Dataset rule:** Use one dataset for the main ablation study. If
SmartSpaces is unsuitable, find a better single dataset instead of
mixing sources prematurely. A single dataset improves control but does
not prove generalization to other warehouses.

## 4. Experiment cases

Run these in stages. Do not implement everything at once.

### Case A --- Original YOLO11s baseline

**Configuration:** Unmodified YOLO11s.

**Purpose:** Establish the reference accuracy, model size, compute cost,
and latency.

Record: - \[ \] mAP50 and mAP50--95. - \[ \] Precision, recall, and
per-class metrics. - \[ \] Small-object metrics, if the dataset supports
a reliable definition. - \[ \] Parameter count and FLOPs at the selected
input size. - \[ \] Model size, peak inference memory, latency, and
FPS. - \[ \] Training configuration, software version, hardware, and
training budget.

**Gate:** Do not proceed until the baseline and data pipeline are
reproducible.

### Case B --- Replace deep C2PSA with SS2D at P5

**Configuration:** Replace the deep C2PSA block with a compatible
SS2D/Mamba block; keep other components fixed as far as possible.

**Hypothesis:** Mamba at the deepest feature stage may model broad
context without the cost of applying it to high-resolution maps.

Questions: - \[ \] How do mAP, precision, and recall change? - \[ \] How
do parameters, FLOPs, memory, and latency change? - \[ \] Does the
implementation run and export to the target runtime? - \[ \] Are gains
consistent across classes and object sizes?

**Caution:** Linear theoretical complexity does not guarantee faster
inference. Kernels, memory movement, implementation, and hardware
matter.

### Case C --- SS2D at P4 and P5

**Configuration:** Start from B and add SS2D at P4.

**Hypothesis:** A second scale may improve detection, but may not
justify the extra cost.

Questions: - \[ \] Is the gain over B meaningful? - \[ \] How much do
parameters, FLOPs, memory, and latency increase? - \[ \] Does it help
small or occluded objects? - \[ \] Does it stay within the edge budget?

Keep P4 only if the measured benefit justifies its cost.

### Case D --- Keep C2PSA and add SS2D at P5

**Configuration:** Retain C2PSA and add SS2D at P5 with compatible
projection/residual connections.

**Hypothesis:** Attention and Mamba may be complementary, but the
combination may be unnecessarily expensive.

Questions: - \[ \] Does it outperform B enough to justify the extra
module? - \[ \] Does latency increase disproportionately? - \[ \] Is
there evidence of complementary benefits?

**Priority:** Lower than A--C. Run only if replacement-versus-addition
is an important question.

### Case E --- Context-Gated Neck

Use the best justified backbone from B--D. Compare: 1. Original YOLO11s
neck. 2. Simple feature fusion (add/concat) as a control. 3.
Context-Gated Fusion.

**Hypothesis:** Deep context may help modulate shallow features in
cluttered scenes, potentially improving small-object detection.

Questions: - \[ \] Does gating improve over the original neck? - \[ \]
Does it improve over simple fusion, not just add parameters? - \[ \] Are
tensor dimensions and channel broadcasting correct? - \[ \] Does the
gate suppress useful low-level details? - \[ \] Does the gain justify
extra latency and complexity?

A global pooled context vector does not preserve spatial location. The
gate could suppress useful small-object signals; measure this rather
than assuming it helps.

### Deferred --- NMS-free / dual-assignment head

Do not include this in v1. It changes a separate part of the detector
and requires a coordinated assignment/training design. Revisit only
after the core study is complete and if there is a strong justification.

## 5. Recommended run order

1.  Audit the dataset and freeze a valid split.
2.  Reproduce Case A.
3.  Implement and test Case B.
4.  Run Case C to test whether P4 is worth the cost.
5.  Review the accuracy/efficiency trade-off and discard weak variants.
6.  Run Case D only if justified.
7.  Run Case E only after selecting a backbone.
8.  Repeat important runs with fixed seeds or multiple seeds where
    compute allows.
9.  Benchmark final candidates on the actual edge device.
10. Write conclusions from results, not hypotheses.

## 6. Evaluation protocol

### Accuracy

-   [ ] mAP50 and mAP50--95.
-   [ ] Precision, recall, per-class AP.
-   [ ] Small-object metrics with a documented size definition, if
    feasible.
-   [ ] Error analysis for occlusion, clutter, and lighting when
    supported by the data.

### Efficiency

-   [ ] Parameter count.
-   [ ] FLOPs using the same input size and counting method.
-   [ ] Checkpoint/model size.
-   [ ] Peak memory.
-   [ ] End-to-end latency and FPS; report preprocessing/postprocessing
    separately where possible.
-   [ ] Inference precision mode (FP32, FP16, INT8, etc.).
-   [ ] Actual hardware and software/runtime versions.

### Fair comparison rules

-   [ ] Same dataset split, class mapping, and input resolution.
-   [ ] Same training data and augmentation unless explicitly being
    tested.
-   [ ] Comparable training budget and checkpoint-selection method.
-   [ ] Same evaluation and postprocessing procedure.
-   [ ] Record software versions, configs, hardware, and random seeds.

Do not claim Jetson FPS based on desktop-GPU results. Benchmark on the
actual device and intended runtime.

## 7. Edge-device and model-budget decisions

Previously discussed provisional targets: - Parameters: under 10M,
ideally around 4--7.5M. - FLOPs: under 30 GFLOPs at 640 × 640. - FP16
footprint: under 50 MB. - Real-time performance: measure actual
latency/FPS on the target hardware.

These are targets, not achieved results. YOLO11s may already be near or
above the parameter target depending on the exact configuration and
counting convention. Adding Mamba may exceed the budget.

-   [ ] Identify the exact edge device and accelerator.
-   [ ] Record OS, drivers, CUDA/TensorRT or other runtime versions.
-   [ ] Benchmark original YOLO11s first.
-   [ ] Verify SS2D operator support and export compatibility.
-   [ ] Measure warm-up and steady-state latency.
-   [ ] Include preprocessing/postprocessing in end-to-end measurements.
-   [ ] If a hard budget is exceeded, consider a smaller baseline or
    width reduction as a separate controlled decision.

## 8. Novelty and literature review

Do not claim novelty merely because the model combines YOLO and Mamba or
adds a gated neck.

-   [ ] Review the closest YOLO--Mamba object-detection papers.
-   [ ] Review state-space models for visual detection and small-object
    detection.
-   [ ] Review lightweight edge detectors and deployment studies.
-   [ ] Review feature-pyramid fusion and context-gating methods.
-   [ ] Compare each close method's backbone, Mamba placement, neck,
    task, dataset, and efficiency measurements.
-   [ ] Identify what has already been proposed and tested.
-   [ ] State the research gap narrowly and with evidence.
-   [ ] Define one primary contribution that can be tested by ablation.
-   [ ] Verify the target journal's scope, indexing, quartile source,
    and requirements before submission.

Q3 is a goal, not a guarantee. Suitability depends on novelty, evidence
quality, experimental rigor, journal scope, and current rankings.

## 9. Open decisions log

  ------------------------------------------------------------------------
  Decision                Current status           Next action / evidence
  ----------------------- ------------------------ -----------------------
  Primary dataset         Provisional:             Audit annotations,
                          PhysicalAI-SmartSpaces   classes, license,
                                                   split, and download
                                                   reliability

  Number of datasets      One primary dataset      Keep main ablations on
                                                   one consistent dataset

  Baseline                YOLO11s                  Reproduce standard
                                                   configuration and
                                                   record version

  First Mamba placement   P5                       Implement Case B and
                                                   compare with A

  P4 Mamba                Unconfirmed              Decide from Case C
                                                   accuracy/efficiency

  C2PSA vs SS2D           Unconfirmed              Compare replacement (B)
                                                   with addition (D) if
                                                   justified

  Context-Gated Neck      Optional                 Test only after
                                                   selecting backbone

  Detection head          Keep original YOLO11s    Do not redesign for v1
                          head                     

  Input resolution        640 × 640 initially      Revisit only with
                                                   justification

  Model budget            Provisional              Measure each model and
                                                   identify target
                                                   hardware

  Edge runtime            Unconfirmed              Identify
                                                   device/software stack
                                                   and operator
                                                   compatibility

  Novelty claim           Not established          Complete closest-paper
                                                   review first
  ------------------------------------------------------------------------

## 10. First milestone: definition of done

-   [ ] Primary dataset audited and licensed for intended use.
-   [ ] Reproducible train/validation/test split frozen.
-   [ ] Original YOLO11s baseline trained and evaluated.
-   [ ] Exact model graph and tensor shapes documented.
-   [ ] Case B implemented and compared fairly with A.
-   [ ] Accuracy, parameters, FLOPs, and latency recorded for both.
-   [ ] Results determine whether to proceed to C or revisit the design.
-   [ ] No unsupported novelty or deployment claims.

------------------------------------------------------------------------

**Working principle:** Start simple, isolate variables, measure the
trade-off, and add complexity only when evidence supports it.
