# Paper Analysis Card: Lightweight Edge Object Detection Survey (Payal Mittal)

## Identity
- Source ID: P005
- Canonical title: A comprehensive survey of deep learning-based lightweight object detection models for edge devices
- Authors: Payal Mittal
- Year: 2024
- Venue: Artificial Intelligence Review (Springer), Vol. 57, Article 242, August 2024 (49 pages)
- DOI/arXiv/URL: https://doi.org/10.1007/s10462-024-10877-1
- Peer-review status: Peer-reviewed Comprehensive Survey
- BibTeX key: mittal2024lightweightsurvey
- Local PDF: `d:\nckh\Reports\Related work\A comprehensive survey of deep learning-based lightweight object detection models for edge devices.pdf`
- Date analyzed: 2026-10-15

## Research problem
- Problem addressed: Designing deep learning object detectors capable of running accurately, quickly, and with minimal latency on resource-constrained edge hardware.
- Motivation: Systematize lightweight backbones, architectural optimizations, hardware constraints, and deployment bottlenecks across embedded platforms.
- Research questions: Which architectural design rules and compression strategies actually translate into reduced physical latency on edge devices?

## Method
- Detector/backbone/neck/head: Surveys all dominant lightweight backbones (MobileNet v1-v4, ShuffleNet v1-v2, GhostNet, EfficientNet, PeleeNet, SqueezeNet) paired with single-stage and two-stage detection heads.
- Mamba/SSM use and insertion point: Analyzes linear-complexity alternatives to dense self-attention.
- Fusion method: Reviews PANet, BiFPN, NAS-FPN, and lightweight cross-scale connections.
- Lightweight/compression methods: Quantization (PTQ/QAT), structured/unstructured pruning, knowledge distillation, neural architecture search (NAS), and low-rank tensor decomposition.
- Training method: Comprehensive review of training workflows, loss functions, and optimization policies for edge models.
- Implementation details: Details embedded deployment runtimes including TensorRT, ONNX Runtime, TFLite, OpenVINO, and hardware-specific NPU toolchains.

## Data
- Dataset(s): MS COCO, Pascal VOC, ImageNet, VisDrone, BDD100K.
- Real/synthetic/mixed: Real imagery across surveillance, automotive, and robotics domains.
- Classes: Multi-class benchmarks.
- Split protocol: Standard published protocol analysis.
- Potential leakage or domain issues: Emphasizes the need for strict train/val/test separation in edge benchmarks.

## Evaluation
- Accuracy metrics: AP, mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.
- Efficiency metrics: Parameters, FLOPs/MACs, Model Size (MB), Latency (ms), FPS, Energy/Power (W).
- Hardware/runtime: Edge CPUs (ARM Cortex), Edge GPUs (NVIDIA Jetson Nano/TX2/Xavier), NPUs, FPGAs.
- Main reported results: Demonstrates that **FLOPs alone do not predict inference latency** on edge hardware. Memory Access Cost (MAC) and operator fragmentation dictate real-world latency.
- Are the results directly comparable to this project? Yes, provides the core theoretical and methodological guidelines for our edge benchmark protocol.

## Findings
- Main contribution stated by authors: 49-page systematic taxonomy of lightweight backbones, edge hardware capabilities, deployment pipelines, and practical trade-off rules.
- Author-stated limitations: Survey of existing literature up to 2024.
- Limitations observed during reading: Pre-dates wide deployment of Mamba-specific TensorRT plugins.
- Most relevant figure/table/section: Table of lightweight backbones, Section on hardware-specific latency bottlenecks, Guidelines for edge deployment.
- What this paper does NOT prove: Does not evaluate custom SSM kernels on embedded hardware.

## Relevance to current project
- Related RQ/Hypothesis: RQ1, RQ5 (compression), RQ6 (edge deployment), and `E_research_design/deployment_plan.md`.
- Closest method or baseline: Foundational survey reference.
- Design ideas worth investigating: Designing models that minimize memory access overhead and avoid highly fragmented multi-branch graph topologies.
- Potential overlap with our proposed idea: Background and deployment principles.
- Research-gap implications: Highlights that while CNN and ViT lightweight models are thoroughly mapped, hardware-aware hybrid CNN-Mamba architectures on edge platforms remain an open research frontier.
- Claims supported by this paper: Edge deployment demands careful consideration of memory bandwidth and hardware-supported primitives.

## Integrity check
- [x] Full paper inspected where available.
- [x] Metadata verified.
- [x] Numbers transcribed accurately.
- [x] Source result distinguished from reproduced result.
- [x] Limitations and caveats recorded.
