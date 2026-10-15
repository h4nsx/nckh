# Edge Deployment Plan

## Purpose

Define a reproducible deployment evaluation. Do not use “real-time” in the final claim until the target and measured results support it.

## 1. Target device

| Field | Value |
|---|---|
| Device name/model | TBD |
| CPU | TBD |
| GPU/NPU | TBD |
| RAM | TBD |
| Operating system | TBD |
| Power mode | TBD |
| Runtime/backend | TBD |
| Driver/runtime versions | TBD |
| Power measurement available? | TBD |

Choose hardware based on actual availability and the intended use case. Do not claim a deployment result on a device that was not tested.

## 2. Export/deployment candidates

Depending on model and hardware support:
- Native PyTorch.
- ONNX Runtime.
- TensorRT.
- OpenVINO.
- Other documented runtime appropriate to the device.

These are alternatives, not a promise that every backend supports the architecture. Verify custom Mamba operations, operator coverage, numerical behavior, and export support before selecting a runtime.

## 3. Precision and compression options

- FP32 reference.
- FP16, if supported and numerically valid.
- INT8 PTQ or QAT, if supported.
- Structured pruning, if it reduces actual runtime cost on the chosen backend.

Record exact settings. A smaller weight file does not guarantee lower latency. Sparse weights may not accelerate inference without hardware/runtime support.

## 4. Timing protocol

Record:
1. Input resolution and batch size.
2. Preprocessing inclusion/exclusion.
3. Model-only or end-to-end timing.
4. Warm-up iteration count.
5. Number of measured iterations.
6. Synchronization method for asynchronous devices.
7. Whether decoding, NMS, and output transfer are included.
8. Latency summary: median (p50), p95, and optionally mean/std.
9. Throughput/FPS definition.
10. Peak memory measurement method.
11. Device power mode and background workload.

Avoid reporting a single best latency as the model's typical performance.

## 5. Deployment acceptance criteria

Set thresholds before final evaluation:
- Maximum latency: TBD.
- Minimum throughput/FPS: TBD.
- Maximum memory: TBD.
- Minimum acceptable mAP: TBD.
- Maximum acceptable accuracy drop after compression: TBD.

These are project targets, not results. Choose realistic thresholds after inspecting the use case and device.

## 6. Deployment result table

| Model/export | Precision | Runtime | Input | Batch | mAP | p50 latency | p95 latency | FPS | Peak memory | Model size |
|---|---|---|---|---:|---|---|---|---|---|---|
| Baseline | TBD | TBD | TBD | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| Proposed | TBD | TBD | TBD | 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| Compressed proposed | TBD | TBD | TBD | 1 | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. Deployment checklist

- [ ] The exact checkpoint is identified.
- [ ] Export succeeds and outputs are validated against the reference implementation.
- [ ] Numerical differences are checked.
- [ ] Test data and preprocessing are fixed.
- [ ] Warm-up and timing repetitions are specified.
- [ ] Runtime, driver, and device details are recorded.
- [ ] Accuracy is re-evaluated after export/quantization.
- [ ] Latency and memory are measured on the target device.
- [ ] Failures and unsupported operators are documented.
- [ ] Reproduction instructions are saved.
