# Paper Analysis Card: LEAF-YOLO

## Identity
- Source ID: P006
- Canonical title: LEAF-YOLO: Lightweight Edge-Real-Time Small Object Detection on Aerial Imagery
- Authors: Nghiem Van Quang, Nguyen Huy Hoang, Hoang Minh Son
- Year: 2024
- Venue: Intelligent Systems with Applications (Elsevier), Vol. 23, Article 200388, 2024 (22 pages)
- DOI/arXiv/URL: https://doi.org/10.1016/j.iswa.2024.200388
- Peer-review status: Peer-reviewed Journal
- BibTeX key: quang2024leafyolo
- Local PDF: `d:\nckh\Reports\Related work\LEAF-YOLO Lightweight Edge-Real-Time Small Object Detection on Aerial.pdf`
- Date analyzed: 2026-10-15

## Research problem
- Problem addressed: Accurate, real-time detection of small objects in aerial imagery under strict compute and latency constraints of edge platforms (e.g. UAV-mounted devices).
- Motivation: Large deep learning models achieve high accuracy on desktop GPUs but cannot run in real time on embedded devices.
- Research questions: Can a lightweight aggregating fusion mechanism achieve state-of-the-art small object detection on edge devices with sub-5M parameters?

## Method
- Detector/backbone/neck/head: Two models proposed: **LEAF-YOLO** (standard) and **LEAF-YOLO-N** (nano). Uses Lightweight-Efficient Aggregating Fusion (LEAF) neck and optimized multi-scale feature extractors.
- Mamba/SSM use and insertion point: Pure convolutional lightweight baseline (provides a baseline for comparison against Mamba models).
- Fusion method: Lightweight-Efficient Aggregating Fusion (LEAF) block minimizing redundant channel projections while merging multi-scale features.
- Lightweight/compression methods: Channel pruning, depthwise convolutions, optimized kernel sizes.
- Training method: Trained on VisDrone and TinyPerson benchmarks using PyTorch with Mosaic and standard data augmentations.
- Implementation details: Deployed and timed on NVIDIA Jetson AGX Xavier.

## Data
- Dataset(s): VisDrone2019-DET, TinyPerson.
- Real/synthetic/mixed: Real UAV aerial photographs.
- Classes: Multi-class vehicle/pedestrian taxonomy (VisDrone), human instances (TinyPerson).
- Split protocol: Standard validation splits (VisDrone2019-DET-val).
- Potential leakage or domain issues: Extreme scale variations and severe occlusion.

## Evaluation
- Accuracy metrics: AP, AP50, AP75, AP_S, AP_M, AP_L.
- Efficiency metrics: Parameters (M), Latency (ms), FPS.
- Hardware/runtime: NVIDIA Jetson AGX Xavier (TensorRT / PyTorch).
- Main reported results:
  - **LEAF-YOLO-N (Nano):** **1.2M parameters**, achieving **39.7% AP50** and **21.9% AP50:95**. Runs in real time (>30 FPS) on Jetson AGX Xavier.
  - **LEAF-YOLO (Standard):** **4.28M parameters**, achieving **48.3% AP50** and **28.2% AP50:95**.
  - On TinyPerson: Achieves **23.0% AP50**, outperforming models with $\ge 20\text{M}$ parameters.
- Are the results directly comparable to this project? Yes, provides strong benchmark evidence for nano-scale edge models.

## Findings
- Main contribution stated by authors: Ultra-compact architectures (<5M params) specifically optimized for small-object perception running >30 FPS on embedded Jetson platforms.
- Author-stated limitations: Focuses on aerial top-down imagery.
- Limitations observed during reading: Limited receptive field in dense cluttered scenes due to purely local convolutional kernels.
- Most relevant figure/table/section: Model architecture diagrams and VisDrone benchmark comparison table.
- What this paper does NOT prove: Does not evaluate state-space or global sequence modeling.

## Relevance to current project
- Related RQ/Hypothesis: RQ1 (lightweight baseline comparison), RQ6 (edge deployment).
- Closest method or baseline: Strong lightweight CNN baseline.
- Design ideas worth investigating: Aggregating feature fusion without excessive channel re-projection.
- Potential overlap with our proposed idea: Target parameter budget (<10M params) and Jetson edge deployment.
- Research-gap implications: Confirms that pure CNNs hit an accuracy ceiling on complex spatial context, motivating our hybrid Mamba integration.
- Claims supported by this paper: Models under 5M parameters can run at real-time speeds (>30 FPS) on NVIDIA Jetson embedded hardware.

## Integrity check
- [x] Full paper inspected where available.
- [x] Metadata verified.
- [x] Numbers transcribed accurately.
- [x] Source result distinguished from reproduced result.
- [x] Limitations and caveats recorded.
