# Research Claim Audit

**Document ID:** `F_literature_and_evidence/research_claim_audit.md`  
**Date:** October 2026  
**Status:** Canonical Audit Log & Verification Record  
**Associated Project:** *A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices*  

---

## 1. Audit Objective and Protocol

This document provides a systematic, line-by-line audit of every major empirical, architectural, and contextual claim made in the preliminary research design and literature synthesis reports.

Each claim is assigned one of four verification statuses:
* **Verified:** The original primary source was inspected; the exact numerical values, hardware platform, precision, and conditions match the claim without distortion.
* **Partially Verified:** The claim is generally supported by the literature, but numbers require clarification, conditions (e.g., batch size, FP16 vs. FP32, NMS inclusion) were omitted, or the claim was overstated.
* **Unverified:** The primary source cannot be confirmed with available documents or requires independent empirical execution in this project.
* **Contradicted:** The primary source contradicts the stated claim or reports opposing evidence under realistic edge conditions.

---

## 2. Comprehensive Claim Audit Table

### Table 2.1: Audit of Architectural and Empirical Claims

| Claim ID | Exact Claim in Project Drafts | Original Source & Identifier | Exact Evidence Location (Table/Fig/Sec/Page) | Directly Supported? | Verification Status | Corrected & Defensible Wording |
|:---|:---|:---|:---|:---:|:---:|:---|
| **CLM-01** | *MambaNeXt-YOLO achieves 85.1% AP50 on PASCAL VOC with 7.1M parameters and 22.4 GFLOPs.* | Lei et al. (2025), P001, arXiv:2506.03654 | Table 1 & Table 2 (VOC 2007+2012 test benchmark) | Yes | **Verified** | MambaNeXt-YOLO achieves 85.1% mAP@0.5 and 66.6% mAP@0.5:0.95 with 7.1M parameters and 22.4 GFLOPs at $640 \times 640$ on the PASCAL VOC test set. |
| **CLM-02** | *MambaNeXt-YOLO runs at 31.9 FPS on Jetson Orin NX and 19.5 FPS on Xavier NX.* | Lei et al. (2025), P001, arXiv:2506.03654 | Table 7 (Inference FPS comparison across platforms) | Yes | **Verified** | Lei et al. report 31.9 FPS on NVIDIA Jetson Orin NX and 19.5 FPS on Xavier NX using TensorRT FP16 at batch size 1 ($640 \times 640$). Model barely clears the 30 FPS threshold on Orin NX. |
| **CLM-03** | *Hybrid-YOLO achieves 41.1 AP on COCO nano and 69.5 AP on DUO.* | Chen & Er (2024), P002, DOI: 10.1016/j.engappai.2023.107657 | Section 4 (Experimental Results), Table 3 & Table 5 | Yes | **Verified** | Chen & Er report 41.1 mAP@0.5:0.95 on COCO-nano and 69.5 mAP on the DUO marine dataset with ~3.8M parameters and 10.5 GFLOPs. |
| **CLM-04** | *Hybrid-YOLO runs at ~45 ms (~22 FPS) on Jetson Xavier NX.* | Chen & Er (2024), P002, DOI: 10.1016/j.engappai.2023.107657 | Section 4.3 (Latency analysis) | Partially | **Partially Verified** | Authors report desktop GPU timings and an embedded baseline (~22 FPS on embedded Jetson Xavier), but exact TensorRT vs. PyTorch engine settings and pre/post-processing timing boundaries were not detailed. |
| **CLM-05** | *UAVDet achieves 84.9% parameter reduction and 53 FPS on RTX 4090.* | Yang et al. (2026), P003, DOI: 10.1016/j.cviu.2026.104637 | Table 2 & Table 4 (VisDrone-DET benchmark) | Yes | **Verified** | Yang et al. achieve 4.2M parameters (-84.9% vs. baseline) and 53 FPS on desktop NVIDIA RTX 4090. *Crucial qualification:* Evaluated on desktop GPU, not embedded edge devices; large-object detection head was pruned. |
| **CLM-06** | *LEAF-YOLO-N has 1.2M parameters, 39.7% AP50, and runs >30 FPS on Jetson AGX Xavier.* | Quang et al. (2024), P006, DOI: 10.1016/j.iswa.2024.200388 | Table 4 & Section 4.4 (Deployment evaluation) | Yes | **Verified** | LEAF-YOLO-N achieves 39.7% AP50 with 1.2M parameters on VisDrone2019 and runs at $>30\text{ FPS}$ (<30 ms) on NVIDIA Jetson AGX Xavier using TensorRT FP16. |
| **CLM-07** | *Edge-YOLO achieves 70.3% FLOP reduction and 44.4% speed increase on RK3588 NPU.* | Li & Ye (2023), P007, DOI: 10.3390/app13074402 | Table 3 & Section 3.3 (Hardware benchmark on RK3588) | Yes | **Verified** | Li & Ye report 70.3% FLOP reduction vs. YOLOv5m and 44.4% higher throughput on the Rockchip RK3588 NPU (28.5 ms, ~35 FPS). *Limitation:* Tested on single-channel infrared FLIR imagery. |
| **CLM-08** | *YOLOv8-Small has 11.2M parameters, 28.6 GFLOPs, and 44.9% AP on COCO.* | Jocher et al. (2023) / Sapkota & Karkee (2026), P004 | Table 1 & Section 3 (Benchmark comparison) | Yes | **Verified** | YOLOv8-Small baseline has ~11.1M--11.2M parameters, 28.6 GFLOPs at $640 \times 640$, and 44.9% mAP@0.5:0.95 on COCO val2017. |
| **CLM-09** | *YOLO11-Small has 9.4M parameters, 21.5 GFLOPs, and 47.0% AP on COCO.* | Ultralytics (2024) / Sapkota & Karkee (2026), P004 | Table 1 & Section 3 (Benchmark comparison) | Yes | **Verified** | YOLO11-Small features 9.4M parameters, 21.5 GFLOPs at $640 \times 640$, and 47.0% mAP@0.5:0.95 on COCO val2017. |
| **CLM-10** | *YOLOv8-Nano / Small achieve ~122 FPS and ~67 FPS on Jetson Orin NX (FP16 TRT).* | Ultralytics / Sapkota & Karkee (2026), P004, Lei et al. (2025), P001 | Benchmark tables on Orin NX (batch=1, FP16) | Partially | **Partially Verified** | Supported under isolated engine execution (pure model forward pass), but real-world throughput drops when decoding, image scaling, and NMS are included. |
| **CLM-11** | *RT-DETR-R18 has 20M params, 60 GFLOPs, and runs at 38.5 ms (~26 FPS) on Jetson Orin NX.* | Zhao et al. (2024) / Mittal (2024), P005 | Section 4 & Table of DETR models | Yes | **Verified** | RT-DETR-R18 has 20.0M parameters and 60.0 GFLOPs; fails the real-time threshold (>30 FPS) on embedded edge SoCs. |
| **CLM-12** | *MobileViT-S runs at 52.6 ms (~19 FPS) on Jetson Orin NX due to quadratic self-attention.* | Mehta & Rastegari (2022) / Mittal (2024), P005 | Hardware analysis of transformer unfold/fold | Partially | **Partially Verified** | Latency penalty verified; however, slowdown is caused not only by $\mathcal{O}(N^2)$ math, but also by memory layout conversions (patch unfolding/folding) which thrash embedded DRAM. |
| **CLM-13** | *No prior study has evaluated YOLO-Mamba architectures on warehouse object detection.* | Systematic literature search across 26 papers & databases | Workspace literature review across IEEE, ACM, Elsevier, arXiv | Conditionally Supported | **Partially Verified (Requires Date & Scope Qualifier)** | *Audited wording:* "To the best of our knowledge as of October 2026, no peer-reviewed or preprint study has evaluated a hybrid YOLO-Mamba detector on multi-class indoor warehouse logistics benchmarks." (Avoid absolute "first" without qualifiers). |
| **CLM-14** | *Low-resolution Mamba tensors (P4, P5) remain entirely within Jetson Orin NX L2 cache.* | Architectural conjecture in preliminary draft | Hardware architecture specification | No direct proof | **Contradicted as Fact / Downgraded to Hypothesis** | NVIDIA Jetson Orin NX has 4MB of shared L2 cache. Intermediate activation tensors and state expansions may fit partially, but cache eviction depends on concurrent kernel memory access and OS overhead. Must be marked as an empirical hypothesis requiring Nsight Systems profiling. |
| **CLM-15** | *The proposed model will achieve >35 FPS (<28.5 ms) on Jetson Orin NX.* | Architectural proposal | Preliminary hypothesis | No (Not yet trained or compiled) | **Unverified (Hypothesis)** | Must be labeled strictly as `[HYPOTHESIS H6]`. Cannot be stated as an established result. |
| **CLM-16** | *The proposed model will achieve 50 MB FP16 footprint.* | Architectural proposal | Preliminary hypothesis | Needs definition | **Unverified (Requires Operational Metric)** | A 7M parameter model in FP16 weighs ~14 MB for weights alone. 50 MB can refer to peak runtime CUDA memory, but requires explicit specification. |

---

## 3. High-Risk Claim Sanitization Log

The following phrases from early drafts have been formally flagged and sanitized:

1. **"Our model is guaranteed to run real-time on edge devices."**  
   *Sanitized:* "The proposed architecture is designed to target a latency budget of $\le 33.3\text{ ms}$ ($\ge 30\text{ FPS}$) on NVIDIA Jetson Orin NX under TensorRT FP16, subject to empirical validation."
2. **"Mamba is unconditionally faster than Transformers and CNNs."**  
   *Sanitized:* "While Mamba provides theoretical linear computational complexity $\mathcal{O}(N)$ compared to Transformer self-attention $\mathcal{O}(N^2)$, its real-world edge throughput depends heavily on memory access cost (MAC) and custom fused kernel availability."
3. **"State-of-the-art warehouse perception has been achieved."**  
   *Sanitized:* "The design aims to achieve competitive accuracy–efficiency trade-offs against established YOLOv8, YOLO11, and lightweight CNN baselines on the SmartSpaces benchmark."
4. **"Low-resolution Mamba states avoid DRAM thrashing by residing in L2 cache."**  
   *Sanitized:* "Restricting Mamba blocks to lower spatial resolutions ($40 \times 40$ and $20 \times 20$) substantially reduces sequence lengths, reducing memory bandwidth pressure compared to early-stage insertion."
