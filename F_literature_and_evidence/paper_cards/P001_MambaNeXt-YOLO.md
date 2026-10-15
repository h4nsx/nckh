# Paper Analysis Card: MambaNeXt-YOLO

## Identity
- Source ID: P001
- Canonical title: MambaNeXt-YOLO: A Hybrid State Space Model for Real-time Object Detection
- Authors: Xiaochun Lei, Siqi Wu, Weilin Wu, Zetao Jiang
- Year: 2025
- Venue: Image and Vision Computing (Under Review) / arXiv:2506.03654v3
- DOI/arXiv/URL: https://arxiv.org/abs/2506.03654
- Peer-review status: Preprint / Under Review
- BibTeX key: lei2025mambanext
- Local PDF: `d:\nckh\Reports\Related work\MambaNeXt-YOLO A Hybrid State Space Model for Real-time Object.pdf`
- Date analyzed: 2026-10-15

## Research problem
- Problem addressed: High computational complexity ($O(N^2)$) of Transformer self-attention hinders deployment for real-time edge detection, while CNNs suffer from limited receptive fields. Pure SSM detectors (like MambaYOLO) lack explicit local-global synergy and advanced multi-scale feature fusion.
- Motivation: Exploit Mamba's linear time complexity $O(N)$ with explicit convolutional local feature modeling and asymmetric multi-scale feature pyramids.
- Research questions: Can a hybrid CNN-Mamba block with adaptive gated fusion outperform pure CNN YOLO variants (v8, v10, 11) on edge devices under real-time constraints?

## Method
- Detector/backbone/neck/head: Single-stage detector with custom hybrid backbone, MAFPN neck, and decoupled anchor-free head predicting at P3, P4, P5.
- Mamba/SSM use and insertion point: MambaNeXt Block embedded in all stages of backbone and inside MAFPN neck. Uses SS2D (2D selective scan) with state dimension $d_{\text{state}}=16\text{–}32$, conv dimension 3, SSM expansion ratio 2.0.
- Fusion method: **ResGate Block**: Parallel linear projections $U, V$, depthwise conv with GELU, Hadamard product gating $Z = \text{GELU}(\text{DWConv}(U)+U) \odot V$, plus residual addition.
- Lightweight/compression methods: Vision Clue Merge (VCM) for space-to-depth downsampling with $1\times 1$ pointwise convs without normalization; reparameterized multi-size kernels in neck.
- Training method: SGD optimizer, initial lr 1e-2, batch size 16, 1000 epochs, 640x640 resolution, trained on single RTX 3090.
- Implementation details: PyTorch, TensorRT FP16 export for edge evaluation.

## Data
- Dataset(s): PASCAL VOC (VOC 2007 + VOC 2012 trainval), DOTA v1.5 (aerial remote sensing).
- Real/synthetic/mixed: Real imagery.
- Classes: 20 classes (VOC), 16 classes (DOTA).
- Split protocol: VOC 2007 test set used for validation/testing.
- Potential leakage or domain issues: Standard benchmark split; no cross-scene domain shift.

## Evaluation
- Accuracy metrics: mAP@0.5 (AP50), mAP@0.5:0.95 (AP).
- Efficiency metrics: Parameters (M), FLOPs (G), FPS.
- Hardware/runtime: NVIDIA RTX 3090, NVIDIA Jetson Xavier NX, NVIDIA Jetson Orin NX (TensorRT FP16, batch=1).
- Main reported results:
  - VOC: **85.1% AP50 / 66.6% AP** with **7.1M parameters** and **22.4 GFLOPs**.
  - Edge Inference: **31.9 FPS on Jetson Orin NX**, **19.5 FPS on Jetson Xavier NX**, **34.6 FPS on RTX 3090**.
- Are the results directly comparable to this project? Yes, serves as the primary direct architecture and speed benchmark.

## Findings
- Main contribution stated by authors: MambaNeXt hybrid block, MAFPN neck, and real-time edge deployment validation on Jetson Orin/Xavier NX.
- Author-stated limitations: Evaluated only on generic PASCAL VOC and DOTA; did not explore multi-sensor or temporal tracking.
- Limitations observed during reading: Tested at 640x640 with batch=1; custom CUDA kernel dependency for selective scan.
- Most relevant figure/table/section: Figure 1 (Architecture), Figure 2 (MambaNeXt block), Table 7 (Inference FPS across NVIDIA devices).
- What this paper does NOT prove: Does not prove performance on industrial warehouse assets, extreme occlusions, or edge CPUs.

## Relevance to current project
- Related RQ/Hypothesis: Direct reference for RQ1, RQ2 (placement), RQ3 (fusion), and RQ6 (deployment).
- Closest method or baseline: Primary closest competing method.
- Design ideas worth investigating: ResGate fusion mechanism, VCM downsampling block, MAFPN neck.
- Potential overlap with our proposed idea: Uses hybrid CNN-Mamba for YOLO; our work differentiates by targeting multi-class warehouse perception (SmartSpaces) under strict edge memory and occlusion constraints.
- Research-gap implications: Confirms viability of hybrid CNN-Mamba, establishing our baseline for warehouse scenes.
- Claims supported by this paper: Mamba achieves linear context modeling, and hybrid CNN-Mamba outperforms pure CNN YOLO models at small parameter budgets.

## Integrity check
- [x] Full paper inspected where available.
- [x] Metadata verified.
- [x] Numbers transcribed accurately.
- [x] Source result distinguished from reproduced result.
- [x] Limitations and caveats recorded.
