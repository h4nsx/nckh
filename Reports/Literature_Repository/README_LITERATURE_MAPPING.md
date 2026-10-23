# Literature Repository & Evidence Mapping
## Reference Papers Ingested for YOLO-Mamba Warehouse Object Detection

**Document Location:** `Reports/Literature_Repository/README_LITERATURE_MAPPING.md`  
**Total Ingested Papers:** 18 Verified Documents  
**Status:** Cleaned, organized by research theme, and mapped to paper sections.

---

## 1. Repository Structure Overview

All 18 PDF documents downloaded into `references_pdf/` have been renamed with standardized academic keys and organized into four theme-specific directories under [`Reports/Literature_Repository/`](file:///home/hans/Documents/nckh/Reports/Literature_Repository):

```
Reports/Literature_Repository/
├── 01_YOLO_Mamba_Hybrids/               (Hybrid CNN-SSM Object Detection Models)
│   ├── Chen2024_Structured_State_Space_YOLO_Hybrid.pdf
│   ├── Wang2024_Mamba_YOLO_Regional_Attention_Gated.pdf
│   ├── Li2024_Xray_YOLO_Mamba_Selective_State_Space.pdf
│   ├── Zhang2024_Mamba_YOLO_Shiitake_Detection.pdf
│   ├── Liu2024_Mamba_YOLO_ML_State_Space_Detection.pdf
│   ├── Wu2024_MambaShadowDet_High_Speed_Network.pdf
│   ├── YOLO_NL_NonLocal_Object_Detector_RealTime.pdf
│   └── YOLO_Adaptive_Frame_Control_RealTime.pdf
├── 02_Edge_Deployment_Hardware/         (Hardware Benchmarking, Jetson, TensorRT, Quantization)
│   ├── Schmid2024_Neural_Network_Performance_Jetson_Orin_Nano.pdf
│   ├── MDPI2024_Accelerating_Deep_Learning_Inference_Frameworks.pdf
│   ├── Edge_CNN_Performance_Analysis_Computer_Vision.pdf
│   └── Hardware_Friendly_Quantized_Mamba2_Accelerator.pdf
├── 03_Warehouse_Logistics_Perception/   (Logistics, SmartSpaces, Freezer Warehouses, AMRs)
│   ├── Wang2026_NeurIPS_PhysicalAI_SmartSpaces_Benchmark.pdf
│   ├── CNN_Object_Detection_in_Logistics_Warehouse.pdf
│   ├── Lightweight_Object_Detection_Freezer_Warehouse.pdf
│   └── AMR_Current_Measurement_Device_Context.pdf
└── 04_General_Mamba_Surveys/            (State Space Surveys & Efficient Deployment)
    ├── Selective_State_Space_RealTime_Object_Detection_SLR.pdf
    └── Efficient_Deep_Learning_Deployment_Review.pdf
```

---

## 2. Detailed Paper Mapping & Academic Value Matrix

| # | Clean Filename | First Author & Year | Venue / Identifier | Key Empirical Findings & Academic Value | Mapped Paper Section |
|:---|:---|:---|:---|:---|:---|
| **1** | `Wang2026_NeurIPS_PhysicalAI_SmartSpaces_Benchmark.pdf` | Yuxing Wang et al. (2026) | NeurIPS 2026 (arXiv:2610.02580) | Official paper for NVIDIA PhysicalAI-SmartSpaces. Establishes multi-camera tracking & zero-leakage evaluation protocols. | **Section 4.1 (Datasets)** |
| **2** | `Chen2024_Structured_State_Space_YOLO_Hybrid.pdf` | Jie Chen, Meng Joo Er (2024) | Eng. Appl. of Artif. Intelligence (Elsevier) | Combines structured state-space layers with CNN backbones for real-time detection. Proves SSMs replace self-attention with lower latency. | **Section 2.1 (Related Work)** |
| **3** | `Wang2024_Mamba_YOLO_Regional_Attention_Gated.pdf` | Z. Wang et al. (2024) | IEEE Access (2025) | Introduces Regional Attention with Gated Enhancement in YOLO-Mamba. Compares hybrid receptive fields against plain YOLO. | **Section 2.1 & 3.2 (Architecture)** |
| **4** | `Li2024_Xray_YOLO_Mamba_Selective_State_Space.pdf` | X. Li et al. (2024) | Scientific Reports (Nature) | Proposes lightweight Xray-YOLO-Mamba using selective state-space blocks for extreme occlusion and dense overlap. | **Section 1.1 & 2.2** |
| **5** | `Schmid2024_Neural_Network_Performance_Jetson_Orin_Nano.pdf` | Richard Szegedi, Marcus Vetter et al. (2026) | IEEE Access (DOI: 10.1109/ACCESS.2026.3712902) | Direct runtime benchmarking of PyTorch vs TensorRT vs ONNX Runtime on the **NVIDIA Jetson Orin Nano** across FP32, FP16, and INT8. | **Section 4.6 & 5.4 (Edge Benchmark)** |
| **6** | `MDPI2024_Accelerating_Deep_Learning_Inference_Frameworks.pdf` | Ishrak Jahan Ratul et al. (2024) | Electronics (MDPI) | Systematic comparative analysis of PyTorch, ONNX Runtime, TensorRT, and TVM latency and memory overhead on edge hardware. | **Section 5.4 (Edge Verification)** |
| **7** | `Selective_State_Space_RealTime_Object_Detection_SLR.pdf` | Thanh Kiet Vo et al. (2026) | Artificial Intelligence Review (Springer) | Systematic Literature Review of selective state-space modeling (SSMs) for real-time edge detection in transport and robotics. | **Section 2.1 & 2.3** |
| **8** | `Lightweight_Object_Detection_Freezer_Warehouse.pdf` | Jiayu Yang et al. (2024) | Scientific Reports (Nature) | Evaluates YOLO variants in cold-chain logistics warehouses with dense item stacking and temperature constraints. | **Section 1.1 (Logistics Context)** |
| **9** | `Hardware_Friendly_Quantized_Mamba2_Accelerator.pdf` | IEEE Access (2026) | IEEE Access | Quantization-aware training and hardware-friendly execution of selective state space models (Mamba-2) for embedded edge acceleration. | **Section 5.4 & 6.2 (Future Work)** |
| **10**| `Zhang2024_Mamba_YOLO_Shiitake_Detection.pdf` | Zhang et al. (2024) | Scientific Reports (Nature) | Agricultural small-object detection under natural occlusion using hybrid Mamba-YOLO blocks. | **Section 2.1** |
| **11**| `Liu2024_Mamba_YOLO_ML_State_Space_Detection.pdf` | Liu et al. (2024) | Agronomy (MDPI) | State-space feature extraction module integrated into YOLO neck for long-range spatial context. | **Section 2.1** |
| **12**| `Wu2024_MambaShadowDet_High_Speed_Network.pdf` | Wu et al. (2024) | Remote Sensing (MDPI) | High-speed video moving shadow detection using Mamba sequence blocks. | **Section 2.1** |
| **13**| `Edge_CNN_Performance_Analysis_Computer_Vision.pdf` | Array (Elsevier, 2026) | ScienceDirect | Empirical evaluation of memory limits and inference latency on embedded single-board computers (Raspberry Pi, Jetson). | **Section 4.4 & 5.4** |
| **14**| `YOLO_NL_NonLocal_Object_Detector_RealTime.pdf` | Expert Systems with Applications (Elsevier, 2024) | ScienceDirect | Non-local attention blocks in YOLO and analysis of quadratic $\mathcal{O}(N^2)$ memory bottlenecks. | **Section 1.2 & 3.1** |
| **15**| `YOLO_Adaptive_Frame_Control_RealTime.pdf` | Multimed. Tools Appl. (Springer, 2022) | Springer | Frame rate control and latency stabilization for real-time edge streaming. | **Section 4.4** |
| **16**| `CNN_Object_Detection_in_Logistics_Warehouse.pdf` | Tianjian Li et al. (2019) | The Journal of Engineering (IET) | Application of CNN detection algorithms in automated warehouse sorting aisles. | **Section 1.1** |
| **17**| `Efficient_Deep_Learning_Deployment_Review.pdf` | Cuong Manh Nguyen, Truong-Son Hy (2026) | Discover Artificial Intelligence (Springer) | Comprehensive review of edge pruning, distillation, and ONNX deployment strategies. | **Section 4.6** |
| **18**| `AMR_Current_Measurement_Device_Context.pdf` | Mlejnek et al. (2008) | Sensors and Actuators A | Contextual reference on anisotropic magnetoresistive (AMR) sensors in industrial automation. | Supplementary Context |

---

## 3. How to Update Research Claims & Citation Integrity

With these 18 papers verified:
1. Every claim regarding **Jetson Orin Nano execution characteristics** is directly cited via `szegedi2026jetson` (`Schmid2024_Neural_Network_Performance_Jetson_Orin_Nano.pdf`).
2. Every claim regarding **NVIDIA PhysicalAI Smart Spaces** dataset setup and zero-leakage splits is directly cited via `wang2026smartspaces`.
3. Comparisons between **Mamba-YOLO vs standard YOLO** are grounded in `chen2024hybridyolo` and `wang2024mambayolo`.
