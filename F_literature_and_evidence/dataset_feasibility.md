# Dataset Feasibility & Annotation Audit

**Document ID:** `F_literature_and_evidence/dataset_feasibility.md`  
**Date:** October 2026  
**Status:** Canonical Dataset Specification & Feasibility Report  
**Associated Project:** *A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices*  

---

## 1. Executive Feasibility Assessment

Modern deep learning research in warehouse perception requires high-fidelity, multi-class benchmarks with verified ground-truth bounding boxes, extreme scale variation, and realistic occlusion.

This audit evaluates the feasibility of the candidate datasets:
1. **Primary Candidate:** `NVIDIA PhysicalAI-SmartSpaces` (MTMC Tracking benchmark).
2. **Auxiliary Candidate:** `NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse` (Multi-sensor spatial benchmark).
3. **Contingency / Fallback Candidates:** `Roboflow Warehouse / Logistics Benchmarks` and synthetic subsets.

---

## 2. Dataset Feasibility Matrix

### Table 2.1: Dataset Evaluation Across Project Requirements

| Evaluation Dimension | PhysicalAI-SmartSpaces (Primary) | PhysicalAI-Spatial-Intelligence-Warehouse (Auxiliary) | Contingency: Open-Source Warehouse Datasets |
|:---|:---|:---|:---|
| **Target Role in Study** | Primary benchmark for multi-class 2D object detection. | Auxiliary pretraining / spatial representation transfer. | Contingency fallback if Hugging Face blockers persist. |
| **Current Operational Status** | **BLOCKED (Storage volume & download method)** | **BLOCKED (Unauthenticated LFS batch access)** | **AVAILABLE (Open-source public repositories)** |
| **Data Modality** | Multi-camera 1080p 30 FPS video streams (MP4) + calibration. | Multi-modal: RGB, depth, camera extrinsics/intrinsics, point clouds. | Static RGB 2D images (JPEG/PNG) with YOLO/COCO annotations. |
| **Target Class Taxonomy** | Humans/Workers, Forklifts, Pallet Trucks, AMRs, Pallets, Packages/Totes. | Warehouse racks, pallets, bins, forklift paths, obstacles. | Forklifts, pallets, boxes, workers, safety gear. |
| **Annotation Format** | JSON / MOTChallenge format (bounding boxes with tracking IDs). | JSON spatial layout, 3D bounding boxes, mesh representations. | Standard YOLO format (`.txt` normalized xywh) or COCO JSON. |
| **Scale Disparity Feasibility** | **High:** Wide-area overhead views capture both full forklifts and small cartons. | **Medium:** Spatial focus on navigation geometry. | **Medium to High:** Varies by collection angle. |
| **Split Leakage Safeguards** | **Built-in:** Modular scenes (`Warehouse_000` to `Warehouse_017`) allow strict zero-leakage cross-scene splitting. | Scene-based directories. | Often collected in single facilities; high risk of background correlation. |
| **Storage & Bandwidth Feasibility** | **Extreme risk if unguided:** >280 hours of video (>500 GB). Selective subsetting required. | Large LFS archive. | Manageable (1--5 GB). |
| **License & Commercial Viability** | CC-BY-4.0 (Permissive with attribution). | CC-BY-4.0 (Permissive with attribution). | CC-BY / Open Academic. |

---

## 3. Subsetting Strategy for PhysicalAI-SmartSpaces

Because downloading the full 280-hour SmartSpaces repository is computationally intractable and unnecessary for 2D object detection, the study will implement a **Targeted Warehouse Subsetting Protocol**:

```
+----------------------------------------------------------------------------------------------------+
|                             SMARTSPACES TARGETED SUBSET PROTOCOL                                   |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [STEP 1: Scene-Level Isolation (Zero-Leakage Splitting)]                                          |
|    - Training Set:   Warehouse_000, Warehouse_001, Warehouse_002, Warehouse_003                    |
|    - Validation Set: Warehouse_015                                                                 |
|    - Test Set:       Warehouse_016, Warehouse_017                                                  |
|    * STRICT RULE: No camera stream or scene in Train may appear in Val or Test.                    |
|                                                                                                    |
|  [STEP 2: Filtered Asset Ingestion]                                                                |
|    - Download ONLY `videos/*.mp4` and `ground_truth.json` for the selected warehouse scenes.       |
|    - EXCLUDE all `depth_maps/*.h5` files (saving >85% storage).                                    |
|    - EXCLUDE corrupted stream: `scene_071/camera_0649` (officially flagged).                       |
|                                                                                                    |
|  [STEP 3: Temporal Sub-Sampling (Redundancy Elimination)]                                          |
|    - Continuous 30 FPS video generates massive frame redundancy.                                   |
|    - Extract keyframes at 1 frame per second (1 FPS) or 2 FPS across cameras.                       |
|    - Estimated Benchmark Size: ~15,000 to 25,000 high-diversity annotated frames.                  |
|    - Resulting Disk Footprint: < 15 GB (feasible for edge research workstation).                   |
|                                                                                                    |
|  [STEP 4: Annotation Conversion Pipeline]                                                          |
|    - Parse `ground_truth.json` bounding boxes.                                                     |
|    - Convert $[x_{\text{min}}, y_{\text{min}}, w, h]$ to normalized $[x_c, y_c, w_n, h_n]$.       |
|    - Validate bounding box bounds ($0 \le x_c, y_c, w_n, h_n \le 1.0$) and discard occluded boxes  |
|      with visibility $< 0.10$.                                                                     |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Annotation Schema & Class Harmonization

To guarantee consistency across baseline and ablation models, the following canonical class mapping is established:

### Table 4.1: Unified Warehouse Class Taxonomy

| Class ID | Class Name | Typical Pixel Area (at 640x640) | Scale Category | Evaluation Significance |
|:---:|:---|:---:|:---:|:---|
| **0** | `worker` / `person` | $1,200 - 8,000\text{ px}^2$ | Medium | Worker safety & AMR collision prevention. |
| **1** | `forklift` | $20,000 - 90,000\text{ px}^2$ | Large | Heavy machinery; dominant perspective occluder. |
| **2** | `pallet_truck` / `agv` | $8,000 - 30,000\text{ px}^2$ | Medium/Large | Mobile industrial transport robot. |
| **3** | `pallet` (wood/plastic) | $3,000 - 18,000\text{ px}^2$ | Medium | Primary logistical unit load; prone to stacking clutter. |
| **4** | `carton` / `box` | $400 - 4,000\text{ px}^2$ | Small/Medium | High visual diversity, variable textures, dense packing. |
| **5** | `tote` / `bin` | $500 - 3,500\text{ px}^2$ | Small/Medium | Plastic sorting container inside shelf compartments. |
| **6** | `tag` / `barcode` | $100 - 800\text{ px}^2$ | **Tiny (<32x32)** | Critical fine-grained inventory marker; primary test for SSM neck. |

---

## 5. Contingency Plan: Unblocking & Fallback Execution

If Hugging Face API rate limits or authentication issues delay downloading `PhysicalAI-SmartSpaces`:

1. **Local Authentication Procedure:** The lead researcher will execute `huggingface-cli login` in an external shell using a private Hugging Face Read Token (ensuring tokens are never committed to git).
2. **Selective Downloader Script:** A Python utility using `huggingface_hub.hf_hub_download` will fetch only `Warehouse_000` ground-truth JSON and sample MP4s to test conversion before scaling.
3. **Fallback Baseline Sanity Check:** If external network restrictions prevent downloading within 48 hours, baseline sanity runs (E01) will commence using a validated public 2D warehouse benchmark (e.g., Roboflow Warehouse Logistics dataset) with matched classes to verify code compilation and training loop mechanics before swapping in the full SmartSpaces benchmark.
