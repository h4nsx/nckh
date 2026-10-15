# Paper Analysis Card: UAVDet

## Identity
- Source ID: P003
- Canonical title: UAVDet: A CNN–Mamba hybrid network for efficient small object detection in UAV imagery
- Authors: Yiming Yang, Feng Guo, Pei Niu
- Year: 2026
- Venue: Computer Vision and Image Understanding (CVIU), Vol. 264, 104637, 2026
- DOI/arXiv/URL: https://doi.org/10.1016/j.cviu.2026.104637
- Peer-review status: Peer-reviewed Journal
- BibTeX key: yang2026uavdet
- Local PDF: `d:\nckh\Reports\CNN - Mamba\UAVDet A CNN–Mamba hybrid network for efficient small object detection.pdf`
- Date analyzed: 2026-10-15

## Research problem
- Problem addressed: Small object detection in complex aerial imagery where aggressive downsampling causes loss of fine-grained spatial features.
- Motivation: Combine CNN inductive bias for local patterns with Mamba's long-range sequence modeling while tailoring feature pyramid flow toward small targets.
- Research questions: Can integrating Mamba into Cross Stage Partial (CSP) modules improve small object perception while drastically reducing parameter count?

## Method
- Detector/backbone/neck/head: Reconfigured backbone depth/width, Cross Stage Partial Mamba (CSPMB) module, Tiny-focused Feature Pyramid Network (TFPN), and dedicated small-object detection heads.
- Mamba/SSM use and insertion point: Embedded within the **Cross Stage Partial Mamba (CSPMB)** modules across intermediate and deep backbone stages.
- Fusion method: Bidirectional multi-scale feature flow re-balanced specifically for small and tiny objects (P2, P3, P4), completely eliminating the large-object prediction branch.
- Lightweight/compression methods: Elimination of large prediction head and pruning of redundant channels.
- Training method: PyTorch framework, cosine annealing lr, trained on RTX 4090.
- Implementation details: Depthwise separable convolutions integrated within CSPMB blocks.

## Data
- Dataset(s): VisDrone2019-DET, UAVDT, DroneVehicle.
- Real/synthetic/mixed: Real aerial UAV imagery.
- Classes: Pedestrians, cars, vans, buses, trucks, motors, bicycles.
- Split protocol: Standard official benchmark splits.
- Potential leakage or domain issues: High visual clutter, severe scale variation, camera angle changes.

## Evaluation
- Accuracy metrics: AP, AP50, AP_S, AP_M, AP_L.
- Efficiency metrics: Parameter count (M), GFLOPs, FPS.
- Hardware/runtime: NVIDIA RTX 4090.
- Main reported results:
  - VisDrone2019: **+4.5% AP and +5.0% AP_S** over baseline.
  - Parameter Reduction: **84.9% parameter reduction** compared to standard baseline.
  - Inference Speed: **53 FPS on RTX 4090** (exceeding 30 FPS real-time threshold).
- Are the results directly comparable to this project? Yes, provides strong architectural evidence for CSP-based Mamba integration.

## Findings
- Main contribution stated by authors: CSPMB module, TFPN architecture, reconfiguration of backbone for small-object detail preservation.
- Author-stated limitations: Evaluated on high-end desktop GPU (RTX 4090); edge embedded benchmark (e.g. Jetson) was not reported in detail.
- Limitations observed during reading: Removal of the large-object head works well for aerial views, but warehouse scenes contain both large objects (forklifts, racks) and small items (cartons, pallets).
- Most relevant figure/table/section: CSPMB block diagram, TFPN feature flow diagram, and VisDrone ablation tables.
- What this paper does NOT prove: Does not prove performance when large objects occupy a substantial portion of the field of view.

## Relevance to current project
- Related RQ/Hypothesis: RQ1, RQ2 (CSP placement), RQ7 (small vs large object robustness).
- Closest method or baseline: Direct competitor in hybrid CNN-Mamba detection.
- Design ideas worth investigating: Integrating Mamba directly into CSP/C2f blocks (CSPMB).
- Potential overlap with our proposed idea: Combining CSP with Mamba.
- Research-gap implications: Confirms CSP-Mamba synergy, but highlights the need to retain multi-scale heads (P3, P4, P5) for warehouse domains containing both small and massive objects.
- Claims supported by this paper: Mamba effectively suppresses complex background clutter while boosting small object feature representation.

## Integrity check
- [x] Full paper inspected where available.
- [x] Metadata verified.
- [x] Numbers transcribed accurately.
- [x] Source result distinguished from reproduced result.
- [x] Limitations and caveats recorded.
