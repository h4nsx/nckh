# Source Registry

## Purpose

A single source-of-truth list for papers, dataset documentation, repositories, and technical references. Assign a stable ID to every source and reuse it across notes and evidence matrices.

## Source status

Use one of:
- `TO_FIND`
- `FOUND`
- `FULL_TEXT_AVAILABLE`
- `ANALYZED`
- `CITED`
- `REJECTED`
- `NEEDS_VERIFICATION`

## Registry

| Source ID | Title | Authors | Year | Type/status | Venue or repository | DOI/URL | Access date | Peer-review status | Local filename | Notes |
|---|---|---|---:|---|---|---|---|---|---|---|
| P001 | MambaNeXt-YOLO: A Hybrid State Space Model for Real-time Object Detection | Xiaochun Lei, Siqi Wu, Weilin Wu, Zetao Jiang | 2025 | Paper / ANALYZED | Image and Vision Computing (under review) / arXiv:2506.03654 | https://arxiv.org/abs/2506.03654 | 2026-10-15 | Preprint / Under review | Reports/Related work/MambaNeXt-YOLO A Hybrid State Space Model for Real-time Object.pdf | Closest competing hybrid CNN-Mamba detector; ResGate, MAFPN, Jetson Orin benchmarks |
| P002 | A hybrid architecture based on structured state space sequence model and convolutional neural network for real-time object detection | Jie Chen, Meng Joo Er | 2024 | Paper / ANALYZED | Engineering Applications of Artificial Intelligence | https://doi.org/10.1016/j.engappai.2023.107657 | 2026-10-15 | Peer-reviewed | Reports/CNN - Mamba/A hybrid architecture based on structured state space sequence model and.pdf | Hybrid-YOLO; early backbone Mamba insertion + spatial attention |
| P003 | UAVDet: A CNN–Mamba hybrid network for efficient small object detection in UAV imagery | Yiming Yang, Feng Guo, Pei Niu | 2026 | Paper / ANALYZED | Computer Vision and Image Understanding | https://doi.org/10.1016/j.cviu.2026.104637 | 2026-10-15 | Peer-reviewed | Reports/CNN - Mamba/UAVDet A CNN–Mamba hybrid network for efficient small object detection.pdf | CSPMB (Cross Stage Partial Mamba) and TFPN for small object detection |
| P004 | Ultralytics YOLO Evolution: An Overview of YOLO27, YOLO26, YOLO11, YOLOv8, and YOLOv5 Object Detectors | Ranjan Sapkota, Manoj Karkee | 2026 | Paper / ANALYZED | arXiv:2510.09653 | https://arxiv.org/abs/2510.09653 | 2026-10-15 | Preprint | Reports/Yolo/Ultralytics YOLO Evolution...pdf | Comprehensive YOLOv5-v27 taxonomy; NMS-free inference, anchor-free decoupled heads |
| P005 | A comprehensive survey of deep learning-based lightweight object detection models for edge devices | Payal Mittal | 2024 | Paper / ANALYZED | Artificial Intelligence Review | https://doi.org/10.1007/s10462-024-10877-1 | 2026-10-15 | Peer-reviewed | Reports/Related work/A comprehensive survey of deep learning-based lightweight object detection models for edge devices.pdf | Comprehensive 49-page survey on lightweight backbones, MAC, and edge hardware bottlenecks |
| P006 | LEAF-YOLO: Lightweight Edge-Real-Time Small Object Detection on Aerial Imagery | Nghiem Van Quang, Nguyen Huy Hoang, Hoang Minh Son | 2024 | Paper / ANALYZED | Intelligent Systems with Applications | https://doi.org/10.1016/j.iswa.2024.200388 | 2026-10-15 | Peer-reviewed | Reports/Related work/LEAF-YOLO Lightweight Edge-Real-Time Small Object Detection on Aerial.pdf | LEAF-YOLO (4.28M) & LEAF-YOLO-N (1.2M) deployed on Jetson AGX Xavier |
| P007 | Edge-YOLO: Lightweight Infrared Object Detection Method Deployed on Edge Devices | Junqing Li, Jiongyao Ye | 2023 | Paper / ANALYZED | Applied Sciences | https://doi.org/10.3390/app13074402 | 2026-10-15 | Peer-reviewed | Reports/Related work/Edge-YOLO Lightweight Infrared Object Detection Method Deployed on Edge Devices.pdf | ShuffleBlock + strip attention deployed on Rockchip RK3588 |
| D001 | PhysicalAI-SmartSpaces dataset documentation | NVIDIA | 2025/2026 | Dataset docs | Hugging Face | https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces | 2026-10-15 | Documentation | N/A | Status: BLOCKED (size/storage; selective subset required). See dataset_access_status.md |
| D002 | PhysicalAI-Spatial-Intelligence-Warehouse documentation | NVIDIA | 2025 | Dataset docs | Hugging Face | https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse | 2026-10-15 | Documentation | N/A | Status: BLOCKED (download/auth issue). See dataset_access_status.md |

## Registration rules

- Never invent DOI, publication year, author list, or venue.
- Use the canonical title from the paper.
- Record arXiv ID and journal/conference DOI separately when multiple versions exist.
- Record whether the source is peer-reviewed, a preprint, documentation, or a blog.
- If a source is retracted, corrected, or superseded, note that status.
- Save the exact version consulted when possible.
- Do not cite search snippets when the primary source can be inspected.

## File naming convention

`[SOURCE_ID]_[short-title]_[year].pdf`

Example format only: `P001_short-title_2025.pdf`

## BibTeX linkage

Add a stable BibTeX key to each analyzed paper, e.g. `author2025shorttitle`. Record the key in the paper-analysis card and `references_bib_guide.md`. Do not generate fake entries to make LaTeX compile; use clearly marked placeholders only during skeleton setup.
