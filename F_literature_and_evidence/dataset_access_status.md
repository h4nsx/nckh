# Dataset Access Status and Download Blockers

> **Document Status:** Active Tracking Log  
> **Last Checked:** 2026-10-18  
> **Overall Dataset Status:** **UNBLOCKED / IN PROGRESS**  
> *(Note: Hugging Face authentication established. Selective subsetting protocol engaged to handle high-volume video footprint without disk exhaustion).*

---

## 1. Summary of Affected Datasets

| Dataset Identifier | Canonical HF URL | Intended Research Role | Access Status | License | Gating / Auth Required? |
|---|---|---|---|---|---|
| **PhysicalAI-Spatial-Intelligence-Warehouse** | [HF Repo](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse) | Auxiliary dataset (pretraining / multi-task / spatial representation) | **UNBLOCKED** | CC-BY-4.0 | Authenticated via local Hugging Face OAuth token (`votuanhung`) |
| **PhysicalAI-SmartSpaces** | [HF Repo](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | Primary benchmark dataset (multi-class 2D warehouse object detection) | **UNBLOCKED / SELECTIVE DOWNLOAD** | CC-BY-4.0 | Authenticated; selective subsetting protocol omitting HDF5 depth maps active |

---

## 2. Detailed Dataset Reports

### 2.1 PhysicalAI-Spatial-Intelligence-Warehouse

- **Official URL:** `https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse`
- **Intended Role in Study:** Candidate auxiliary source for pretraining or domain transfer to improve spatial reasoning and robustness in warehouse environments.
- **Current Access Status:** **UNBLOCKED**.
- **Date Checked:** 2026-10-18.
- **Access / Gating Investigation:**
  - **Gated Status:** Verified via Hugging Face repository metadata. The repository is publicly visible under the **CC-BY-4.0** license.
  - **Access Resolution:** User authenticated successfully via browser OAuth (`votuanhung`). API and CLI calls now proceed without rate limits.
  - **Security Rule:** **Do not hardcode or commit Hugging Face access tokens into source control or documentation.**

---

### 2.2 PhysicalAI-SmartSpaces

- **Official URL:** `https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces`
- **Intended Role in Study:** Primary target benchmark for multi-class 2D object detection (humans, forklifts, pallet trucks, AMRs, cartons/pallets).
- **Current Access Status:** **UNBLOCKED / SELECTIVE DOWNLOAD IN PROGRESS**.
- **Date Checked:** 2026-10-18.
- **Volume & Storage Investigation:**
  - **Dataset Scope:** Over **280 hours** of 1080p 30 FPS video across nearly **1,800 camera streams** and dozens of scenes.
  - **Subsets Available:**
    - `MTMC_Tracking_2025` / `MTMC_Tracking_2026` (JSON format annotations, camera calibration, 1080p MP4 videos, and optional heavy HDF5 depth maps).
    - Partitioned into modular scenes: Train (`Warehouse_000` to `Warehouse_014`), Validation (`Warehouse_015`, `Warehouse_016`), Test (`Warehouse_017`+).
  - **Selective Download Protocol:**
    - **Do NOT download the full repository.**
    - Restrict download strictly to **Warehouse scenes only** (`Warehouse_000` for train, `Warehouse_015` for validation).
    - Omit `depth_maps/` HDF5 files (~185 GB per scene); download only `videos/` and `ground_truth.json` / `calibration.json` (~3.3 GB per scene).
    - Extract sampled frames (e.g. 1 frame per second) to build a manageable 2D detection benchmark in standard YOLO format without redundant frames.

---

## 3. Explicit Recovery Criteria & Status

1. **Access Confirmed:** Authenticated session verified (`votuanhung` logged in via HF OAuth). ✅ **RESOLVED**
2. **Selective Strategy Defined:** Target scenes selected (`Warehouse_000` and `Warehouse_015`), omitting `.h5` depth maps. ✅ **RESOLVED**
3. **Download Script Ready:** Dedicated selective download and extraction utility prepared. 🔄 **IN PROGRESS**
4. **Annotation Format Conversion:** `ground_truth.json` mapping to YOLO format (`class x_center y_center width height`). 🔄 **PENDING EXECUTION**
5. **Minimal Data-Loading Test:** Dataset verified in PyTorch loader with zero integrity errors. 🔄 **PENDING EXECUTION**

---

## 4. Current Decoupled Workplan

While datasets are blocked, research progress continues independently on:
- **Literature Review & Evidence Synthesis:** Analyzing prior YOLO and Mamba detection papers, populating `F_literature_and_evidence/source_registry.md`.
- **LaTeX Infrastructure:** Maintaining the clean IEEE/standard paper skeleton in `paper/`.
- **Experiment Planning & Ablation Specifications:** Refining experimental matrices and metric definitions in `E_research_design/`.
- **Data Conversion Design:** Drafting parser logic for converting MOTChallenge/JSON formats to standard YOLO bounding box formats (`class_id x_center y_center width height`).
