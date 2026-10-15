# Research Questions and Hypotheses

## 1. Purpose

This document defines the questions the study may answer. It is a planning document, not a claim that any hypothesis is true.

## 2. Working problem statement

Warehouse perception systems must detect multiple object classes in visually complex scenes. A practical deployment may be constrained by compute, memory, power, and latency. CNN-based detectors provide efficient local feature extraction, while Mamba/state-space models may provide a different mechanism for modeling wider spatial context. The research question is whether a carefully designed hybrid can improve the accuracy–efficiency trade-off for warehouse object detection.

This statement must be revised after reviewing relevant literature and inspecting the selected dataset.

## 3. Primary research question

**RQ1:** Can a lightweight hybrid CNN–Mamba detector achieve a competitive accuracy–efficiency trade-off for multi-class warehouse object detection compared with appropriate YOLO and lightweight detection baselines?

## 4. Secondary research questions

- **RQ2 — Placement:** Which integration point, if any, for Mamba (backbone, neck, or selected feature scales) provides a useful trade-off between accuracy and computational cost?
- **RQ3 — Fusion:** Does the proposed feature-fusion mechanism improve performance over simpler alternatives under a controlled comparison?
- **RQ4 — Data strategy:** Does using the Spatial-Intelligence-Warehouse dataset as an auxiliary source improve downstream performance on SmartSpaces, compared with SmartSpaces-only training?
- **RQ5 — Compression:** What accuracy and efficiency changes result from pruning and/or quantization?
- **RQ6 — Deployment:** Does the selected model meet a pre-defined latency or throughput target on a specified edge device under a reproducible benchmark protocol?
- **RQ7 — Robustness:** How does performance vary by object size, occlusion, scene density, lighting, and other conditions that can be reliably annotated or measured?

Not every question must be included in the final paper. Keep only those that can be answered with the available data, compute, time, and hardware.

## 5. Hypotheses to test

| ID | Hypothesis | Evidence required | Status |
|---|---|---|---|
| H1 | A selected CNN–Mamba hybrid improves a pre-defined detection metric over a comparable CNN-only baseline. | Controlled experiment with matched data, input size, training budget, and evaluation protocol. | [HYPOTHESIS] |
| H2 | The selected integration point provides a better accuracy–efficiency trade-off than alternative placements. | Placement ablation with parameters, FLOPs/MACs, latency, and accuracy. | [HYPOTHESIS] |
| H3 | A lightweight design reduces computational cost while maintaining an acceptable detection-performance level. | Matched baseline comparison and an explicitly defined acceptable-performance criterion. | [HYPOTHESIS] |
| H4 | Auxiliary pretraining or auxiliary data use improves downstream detection on SmartSpaces. | SmartSpaces-only versus auxiliary-data strategy, with leakage controls and matched evaluation. | [HYPOTHESIS] |
| H5 | Compression improves deployment efficiency with a measurable and reported accuracy cost. | Before/after model measurements using the same hardware and runtime protocol. | [HYPOTHESIS] |
| H6 | The model meets a pre-defined real-time target on the target device. | End-to-end or clearly scoped latency measurements, device details, and timing protocol. | [HYPOTHESIS] |

## 6. Scope boundaries

### In scope for the initial paper
- Multi-class 2D object detection.
- Warehouse-oriented imagery.
- Comparison with suitable YOLO/lightweight baselines.
- CNN–Mamba integration, if the literature review supports a defensible design.
- Accuracy and computational-efficiency evaluation.
- Edge deployment only if a specific device is available.

### Potential extensions, not automatically in scope
- Multi-camera tracking and re-identification.
- 3D bounding-box prediction.
- Bird's-eye-view reconstruction.
- RGB-D multimodal detection.
- Vision-language question answering.
- Synthetic-to-real transfer using SimReady assets.
- Power/energy profiling if the equipment supports trustworthy measurement.

Adding too many tasks may weaken the central research question. Do not include an extension merely because a dataset contains the relevant annotation.

## 7. Operational definitions to settle

- **Lightweight:** define using model size, parameter count, FLOPs/MACs, measured memory, latency, or a stated combination. Avoid using the term without a metric.
- **Real-time:** define a target latency/FPS and the exact device, input resolution, batch size, runtime, and measurement scope.
- **Competitive accuracy:** define the comparison set and the metric before seeing final results.
- **Warehouse object:** define the included classes using the verified dataset taxonomy.
- **Edge device:** name the actual hardware and its relevant specifications.

## 8. Decision log

| Date | Decision or question | Evidence | Decision owner | Status |
|---|---|---|---|---|
| TBD | Final detection task and classes | Official dataset schema and project scope | Research team | [UNKNOWN] |
| TBD | Primary baseline | Literature review and available implementation | Research team | [UNKNOWN] |
| TBD | Mamba integration point | Related work and pilot experiments | Research team | [UNKNOWN] |
| TBD | Target edge hardware | Hardware availability and deployment objective | Research team | [UNKNOWN] |
