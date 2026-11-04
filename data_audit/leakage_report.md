# LEAKAGE AND PROVENANCE REPORT (Stage 2 Audit)

- **Date:** 2026-11-04
- **Auditor Role:** Autonomous Research Implementation Auditor (SKILLS.md Stage 2)
- **Dataset:** NVIDIA PhysicalAI-SmartSpaces Micro-Benchmark
- **Status:** **STAGE 2 AUDIT COMPLETE (PASS WITH DOCUMENTED CLASS IMBALANCE)**

---

## 1. Dataset Provenance & Split Integrity

| Criterion | Specification | Audited Reality | Evidence | Status |
|---|---|---|---|---|
| **Source Dataset** | NVIDIA PhysicalAI-SmartSpaces (MTMC Tracking 2025) | Downloaded via Hugging Face Hub (`nvidia/PhysicalAI-SmartSpaces`) | `scripts/build_micro_dataset.py` | **PASS** |
| **Partitioning Protocol** | Zero-Leakage Cross-Facility Disjoint Split | **Train:** `Warehouse_000` (Cameras 0000, 0001)<br>**Val:** `Warehouse_015` (Cameras Camera, Camera_01) | Exact facility separation verified across all 400 images. | **PASS** |
| **Cross-Split Image Overlap** | 0 duplicate images across splits | MD5 perceptual check: **0 duplicates** between train & val. | `data_audit/dataset_manifest.csv` | **PASS** |
| **Sampling Interval** | 0.5 FPS (1 frame every 60 video frames @ 30 FPS) | Subsample step `60` verified in dataset builder. | `scripts/build_micro_dataset.py`: lines 72, 148, 158 | **PASS** |
| **Image Resolution & Geometry** | Uniform $640 \times 640$ pixels | All 400 images verified at exact `(640, 640)` dimensions. | PIL dimension check across all images. | **PASS** |
| **Label Integrity & Bounding Boxes** | Normalized coordinates in $[0, 1]$, $w > 0, h > 0$, no NaN/Inf | 16,655 total bounding boxes audited. **0 anomalies** detected. | Strict numerical audit script. | **PASS** |

---

## 2. Quantitative Instance Distribution & Class Imbalance

| Target Class ID & Name | Train Instances ($n=12{,}959$) | Val Instances ($n=3{,}696$) | Total Instances | Val Share (%) | Audit Finding / Imbalance Risk |
|---|---|---|---|---|---|
| **0: `worker`** | 11,399 (87.96%) | 3,488 (94.37%) | 14,887 | 94.37% | Dominant majority class. Stable statistical support. |
| **1: `forklift`** | 600 (4.63%) | 150 (4.06%) | 750 | 4.06% | High industrial importance. Sufficient validation sample ($n=150$). |
| **2: `pallet_truck`** | 236 (1.82%) | **13 (0.35%)** | 249 | **0.35%** | **CRITICAL TAIL CLASS**: Extreme sample sparsity ($n=13$). |
| **3: `amr`** | 724 (5.59%) | 45 (1.22%) | 769 | 1.22% | Moderate support ($n=45$). |

### Tail-Class Risk Analysis (`pallet_truck`):
1. The validation partition contains only **13 bounding box instances** of `pallet_truck` across 100 images.
2. In object detection evaluation, a single missed or misclassified instance shifts the per-class average precision ($\text{mAP}$) by $\sim 7.7\%$.
3. **Traceability Confirmation:** The paper explicitly acknowledges this in Table III and Section 4.1 footnote: *"Notice on rare tail class: pallet_truck contains only 13 validation instances (0.35% of total labels), exhibiting high sample variance."*
4. **Requirement Satisfied:** The paper does not claim false precision or statistical significance on `pallet_truck`.

---

## 3. Spatial and Temporal Data Leakage Assessment

- **Facility Independence:** `Warehouse_000` (training) and `Warehouse_015` (validation) are distinct simulated facility geometries featuring different aisle layouts, shelf configurations, textures, and lighting vectors.
- **Camera Independence:** Disjoint camera trajectories ensure no visual background memorization occurs between training and validation.
- **Temporal Separation:** Since train and val originate from completely separate warehouse scenes, zero temporal leakage (frame-to-frame trajectory overlap) is guaranteed.
- **Validation vs Test Set Framing:** The 100 images from `Warehouse_015` serve as the validation set for model selection. Consistent with SKILLS.md Section 2.3 and 5.1, this set must be accurately reported as a **validation set** (not an independent unseen test partition), which is strictly adhered to in the paper text and tables.

---

## 4. Exit Gate G2 Verdict: PASS
The dataset is clean, traceable, completely free of spatial/temporal leakage, and the extreme class imbalance on `pallet_truck` ($n=13$) is rigorously documented.

