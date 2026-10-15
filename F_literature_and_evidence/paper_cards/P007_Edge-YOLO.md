# Paper Analysis Card: Edge-YOLO

## Identity
- Source ID: P007
- Canonical title: Edge-YOLO: Lightweight Infrared Object Detection Method Deployed on Edge Devices
- Authors: Junqing Li, Jiongyao Ye
- Year: 2023
- Venue: Applied Sciences (MDPI), Vol. 13, Issue 7, Article 4402, 2023 (15 pages)
- DOI/arXiv/URL: https://doi.org/10.3390/app13074402
- Peer-review status: Peer-reviewed Journal
- BibTeX key: li2023edgeyolo
- Local PDF: `d:\nckh\Reports\Related work\Edge-YOLO Lightweight Infrared Object Detection Method Deployed on Edge Devices.pdf`
- Date analyzed: 2026-10-15

## Research problem
- Problem addressed: Standard YOLO models (e.g. YOLOv5m) are computationally heavy and memory-intensive, hindering real-time inference on low-power edge NPU platforms.
- Motivation: Develop an ultra-lightweight detector for thermal/infrared road scenes that reduces compute while preserving target localization accuracy.
- Research questions: How much can computation and model size be reduced through ShuffleNet-inspired blocks, attention, and lightweight upsampling without compromising detection performance?

## Method
- Detector/backbone/neck/head: Replaces YOLOv5m backbone with lightweight **ShuffleBlock** units and a strip depthwise convolutional attention module.
- Mamba/SSM use and insertion point: Pure convolutional with strip depthwise attention (useful reference for strip/directional feature aggregation).
- Fusion method: Uses **CAU-Lite** as an efficient lightweight upsampling operator in the feature pyramid; optimized with **EX-IoU** bounding box loss.
- Lightweight/compression methods: ShuffleNet channel shuffle, depthwise convolutions, strip attention.
- Training method: Trained on infrared datasets using SGD, cosine decay, PyTorch.
- Implementation details: Exported to ONNX and deployed natively on the **Rockchip RK3588** embedded NPU.

## Data
- Dataset(s): FLIR thermal infrared dataset, road scenes.
- Real/synthetic/mixed: Real infrared imagery.
- Classes: People, bicycles, cars, dogs.
- Split protocol: Standard published FLIR split.
- Potential leakage or domain issues: Infrared thermal signatures vs RGB visible textures.

## Evaluation
- Accuracy metrics: mAP@0.5, Precision, Recall.
- Efficiency metrics: Computational complexity (FLOPs), Model size (MB), Inference latency (ms), FPS.
- Hardware/runtime: **Rockchip RK3588 NPU**, desktop GPU.
- Main reported results:
  - Compared with YOLOv5m:
    - **70.3% reduction in computational complexity (GFLOPs)**.
    - **71.6% reduction in model size (MB)**.
    - **44.4% increase in inference speed (FPS)** on the RK3588 embedded platform while matching baseline accuracy.
- Are the results directly comparable to this project? Yes, provides a practical benchmark for edge NPU deployment.

## Findings
- Main contribution stated by authors: ShuffleBlock backbone integration, strip depthwise attention, CAU-Lite upsampling, and physical RK3588 NPU deployment.
- Author-stated limitations: Evaluated specifically on single-channel infrared imagery.
- Limitations observed during reading: Tested on an older baseline (YOLOv5m); modern YOLOv8/v11 models offer stronger baseline efficiency.
- Most relevant figure/table/section: Table comparing YOLOv5m vs Edge-YOLO on RK3588 NPU.
- What this paper does NOT prove: Does not evaluate multi-scale global context modeling.

## Relevance to current project
- Related RQ/Hypothesis: RQ1, RQ6 (edge hardware deployment), `E_research_design/deployment_plan.md`.
- Closest method or baseline: Embedded hardware deployment reference.
- Design ideas worth investigating: Strip depthwise attention and lightweight upsamplers (CAU-Lite) to eliminate memory bottlenecks in the neck.
- Potential overlap with our proposed idea: Edge deployment on embedded hardware.
- Research-gap implications: Confirms that edge hardware requires architectural pruning of standard YOLO necks to unlock maximum NPU throughput.
- Claims supported by this paper: Replacing dense convolutions with depthwise shuffle blocks significantly accelerates inference on embedded edge accelerators.

## Integrity check
- [x] Full paper inspected where available.
- [x] Metadata verified.
- [x] Numbers transcribed accurately.
- [x] Source result distinguished from reproduced result.
- [x] Limitations and caveats recorded.
