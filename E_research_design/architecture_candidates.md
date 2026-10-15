# Architecture Candidates

## Purpose

Document possible designs before implementation. The paper must not describe a candidate as the final proposed architecture until a design is selected and evaluated.

## 1. Design objective

Investigate whether combining CNN-based local feature extraction with Mamba/state-space contextual modeling can improve the accuracy–efficiency trade-off for multi-class warehouse object detection.

The architecture should be justified against:
- the closest related papers;
- the shape and resolution of feature maps;
- compute and memory budgets;
- implementation/runtime support;
- ablation evidence.

## 2. Candidate integration strategies

| Candidate | Concept | Potential benefit | Main risk | Evidence required |
|---|---|---|---|---|
| A — Backbone insertion | Replace or augment selected backbone blocks with a CNN–Mamba block | Context modeling during feature extraction | May add substantial cost at high spatial resolution | Compare placement, latency, and accuracy |
| B — Neck insertion | Add Mamba to selected multi-scale feature-fusion stages | Context after multi-scale features are formed | Feature-map processing and fusion may be expensive | Compare against unchanged neck and simpler fusion |
| C — Low-resolution feature only | Apply Mamba only to deeper, lower-resolution feature maps | Potentially reduce sequence length and compute | May miss fine details for small objects | Per-size AP and efficiency measurements |
| D — Parallel branches | CNN and Mamba process features in parallel, then fuse | Explicit local/global feature complementarity | Extra parameters, memory, and synchronization | Compare sum, concatenation, and gated/learned fusion where feasible |
| E — Sequential hybrid block | CNN features feed a Mamba block, followed by projection/residual connection | Simple integration and easier ablation | Context block may not add useful information | Compare with CNN-only block under matched settings |

These are conceptual options. Verify that each design is not already covered by closely related literature before presenting it as novel.

## 3. Fusion mechanisms to investigate

- Element-wise addition after compatible projections.
- Channel concatenation followed by a projection/convolution.
- Lightweight learned weighting or gating.
- Residual fusion.

Do not test every option without a hypothesis. Choose a small set based on literature and parameter/FLOP budget.

## 4. Design constraints

- Keep tensor dimensions and channel alignment explicit.
- Document how 2D feature maps are converted to the sequence/layout expected by the chosen Mamba implementation.
- Check memory behavior at the intended input size.
- State whether positional/spatial information is retained and how.
- Avoid using FLOPs alone as proof of runtime efficiency.
- Record dependency versions and licenses for any external implementation.
- Check export compatibility before promising deployment to TensorRT, ONNX Runtime, OpenVINO, or another backend.

## 5. Proposed block specification template

### Block name
TBD — do not finalize the name before the design is selected.

### Input
- Tensor shape: TBD.
- Feature stride: TBD.
- Channel count: TBD.

### Operations
1. Local feature extraction: TBD.
2. State-space/context operation: TBD.
3. Fusion/projection: TBD.
4. Residual/normalization/activation: TBD.

### Output
- Tensor shape: TBD.
- Channel count: TBD.

### Intended benefit
[HYPOTHESIS] TBD.

### Added cost
- Parameters: TBD.
- FLOPs/MACs: TBD.
- Runtime/memory impact: TBD.

### Closest prior designs
TBD, with citations.

### Required ablations
TBD.

## 6. Selection criteria

Choose a candidate only after:
1. comparing it to the closest published methods;
2. confirming the implementation is feasible in the chosen framework;
3. testing a small pilot run;
4. measuring the actual added compute/memory;
5. writing down the expected benefit and a result that would falsify it.

## 7. Architecture diagram checklist

The final diagram should identify:
- input image and resolution;
- backbone stages and feature strides;
- where the Mamba block is placed;
- CNN and Mamba paths if parallel;
- fusion operations;
- neck and detection head;
- output classes and bounding boxes.

Every module in the diagram must correspond to the implementation and configuration used in experiments.
