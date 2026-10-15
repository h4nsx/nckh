# Paper Analysis Card: Hybrid-YOLO (Chen & Er)

## Identity
- Source ID: P002
- Canonical title: A hybrid architecture based on structured state space sequence model and convolutional neural network for real-time object detection
- Authors: Jie Chen, Meng Joo Er
- Year: 2024
- Venue: Engineering Applications of Artificial Intelligence (EAAI), Vol. 130, 2024
- DOI/arXiv/URL: https://doi.org/10.1016/j.engappai.2023.107657
- Peer-review status: Peer-reviewed Journal
- BibTeX key: chen2024hybridyolo
- Local PDF: `d:\nckh\Reports\CNN - Mamba\A hybrid architecture based on structured state space sequence model and.pdf`
- Date analyzed: 2026-10-15

## Research problem
- Problem addressed: Balancing high processing speed and low latency on edge devices while capturing long-range spatial dependencies in complex background environments.
- Motivation: Exploit Mamba's linear scaling in early backbone stages while mitigating Mamba's directional perception limitation.
- Research questions: Can structured state-space sequence modeling in the backbone improve detection in complex/cluttered scenes without violating edge compute budgets?

## Method
- Detector/backbone/neck/head: Single-stage YOLO architecture with Mamba-enhanced backbone, Ghost-inspired cost-efficient module, and PANet neck.
- Mamba/SSM use and insertion point: Mamba strategically embedded in the **early stages** of the backbone network.
- Fusion method: Combined with a lightweight spatial attention mechanism to restore 2D spatial-directional context.
- Lightweight/compression methods: Ghost-module-inspired parameter-reduction modules.
- Training method: Standard PyTorch training on MS COCO and DUO datasets.
- Implementation details: Open-source implementation available at https://github.com/chenjie04/Hybrid-YOLO.

## Data
- Dataset(s): MS COCO, Pascal VOC, DUO (Detecting Underwater Objects benchmark).
- Real/synthetic/mixed: Real imagery.
- Classes: 80 classes (COCO), underwater marine classes (DUO).
- Split protocol: Standard COCO trainval/test-dev splits.
- Potential leakage or domain issues: Underwater dataset tests extreme visual degradation (turbidity, color distortion).

## Evaluation
- Accuracy metrics: AP, AP50, AP75, AP_S, AP_M, AP_L.
- Efficiency metrics: Parameters, FLOPs, Latency/FPS.
- Hardware/runtime: Standard GPU and embedded deployment benchmarks.
- Main reported results:
  - MS COCO: Achieves **41.1 AP** at nano scale (+1.6 AP over baseline).
  - DUO: Achieves **69.5 AP** (+0.3 AP over YOLO11).
- Are the results directly comparable to this project? Yes, provides evidence for early-stage backbone Mamba placement.

## Findings
- Main contribution stated by authors: Early-stage Mamba backbone insertion, lightweight spatial attention for directional perception, Ghost-inspired cost-efficiency.
- Author-stated limitations: Primarily evaluated on generic and underwater domains.
- Limitations observed during reading: Spatial attention adds slight memory overhead that must be balanced on low-RAM edge devices.
- Most relevant figure/table/section: Backbone architecture diagrams and COCO nano comparison table.
- What this paper does NOT prove: Does not evaluate multi-camera warehouse settings.

## Relevance to current project
- Related RQ/Hypothesis: RQ1, RQ2 (backbone placement), RQ3 (spatial attention vs gating).
- Closest method or baseline: Direct hybrid baseline.
- Design ideas worth investigating: Pairing Mamba with spatial attention to resolve 1D scanning directional blindness.
- Potential overlap with our proposed idea: Hybrid CNN-Mamba concept; our approach differs in fusion block design (ResGate) and warehouse deployment focus.
- Research-gap implications: Demonstrates that early backbone Mamba placement is viable if directional priors are restored.
- Claims supported by this paper: Mamba enhances feature representation in cluttered backgrounds while preserving real-time throughput.

## Integrity check
- [x] Full paper inspected where available.
- [x] Metadata verified.
- [x] Numbers transcribed accurately.
- [x] Source result distinguished from reproduced result.
- [x] Limitations and caveats recorded.
