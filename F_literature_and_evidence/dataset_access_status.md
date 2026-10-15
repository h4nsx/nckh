# Dataset Access Status and Download Blockers

> **Document Status:** Active Tracking Log  
> **Last Checked:** 2026-10-15  
> **Overall Dataset Status:** **BLOCKED — download/access issue**  
> *(Note: This indicates the current operational status for this project environment, not a permanent unavailability of the datasets).*

---

## 1. Summary of Affected Datasets

| Dataset Identifier | Canonical HF URL | Intended Research Role | Access Status | License | Gating / Auth Required? |
|---|---|---|---|---|---|
| **PhysicalAI-Spatial-Intelligence-Warehouse** | [HF Repo](https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse) | Auxiliary dataset (pretraining / multi-task / spatial representation) | **BLOCKED** | CC-BY-4.0 | No gating agreement detected, but requires HF account authentication / token for bulk LFS API access |
| **PhysicalAI-SmartSpaces** | [HF Repo](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) | Primary benchmark dataset (multi-class 2D warehouse object detection) | **BLOCKED** | CC-BY-4.0 | High-volume video dataset (>280h, ~1,800 cameras); full download blocked by extreme storage and bandwidth footprint |

---

## 2. Detailed Dataset Reports

### 2.1 PhysicalAI-Spatial-Intelligence-Warehouse

- **Official URL:** `https://huggingface.co/datasets/nvidia/PhysicalAI-Spatial-Intelligence-Warehouse`
- **Intended Role in Study:** Candidate auxiliary source for pretraining or domain transfer to improve spatial reasoning and robustness in warehouse environments.
- **Current Access Status:** **BLOCKED — download/access issue**.
- **Date Checked:** 2026-10-15.
- **Access / Gating Investigation:**
  - **Gated Status:** Verified via Hugging Face repository metadata. The repository is publicly visible under the **CC-BY-4.0** license and does not enforce a formal gated access approval wall (i.e. no click-through user vetting form is mandated).
  - **Access Blocker:** Automated anonymous API or git-lfs clone commands fail or rate-limit when attempting bulk downloads without a valid Hugging Face User Access Token (Read).
  - **Security Rule:** **Do not hardcode or commit Hugging Face access tokens into source control or documentation.**
- **Attempted Methods:**
  - Web metadata inspection via HTTP requests (successful).
  - Local CLI batch download without authenticated session (blocked / deferred to prevent token leakage and uninspected multi-gigabyte disk usage).
- **Exact Error / Limitation:** Unauthenticated batch fetching of high-volume LFS assets returns rate limits or requires authenticated session credentials (`huggingface-cli login`).
- **Next Troubleshooting Steps:**
  1. Researcher logs in locally via `huggingface-cli login` in an external interactive terminal using a private personal read token.
  2. Test downloading only the metadata manifest (`train.json` or `val.json`) and a single scene archive or 10-sample subset to inspect schema before initiating full download.

---

### 2.2 PhysicalAI-SmartSpaces

- **Official URL:** `https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces`
- **Intended Role in Study:** Primary target benchmark for multi-class 2D object detection (humans, forklifts, pallet trucks, AMRs, cartons/pallets).
- **Current Access Status:** **BLOCKED — download/access issue**.
- **Date Checked:** 2026-10-15.
- **Volume & Storage Investigation:**
  - **Dataset Scope:** Over **280 hours** of 1080p 30 FPS video across nearly **1,800 camera streams** and dozens of scenes.
  - **Subsets Available:**
    - `MTMC_Tracking_2024` (MOTChallenge format text annotations, 1080p MP4 videos). Note: Official docs advise excluding corrupted stream `scene_071/camera_0649`.
    - `MTMC_Tracking_2025` / `MTMC_Tracking_2026` (JSON format annotations, camera calibration, 1080p MP4 videos, and optional heavy HDF5 depth maps).
    - Partitioned into modular scenes: Train (`Warehouse_000` to `Warehouse_014`), Validation (`Warehouse_015`, `Warehouse_016`, `Lab_000`, `Hospital_000`), Test (`Warehouse_017`+).
  - **Storage Feasibility:** Downloading the entire 280-hour repository is estimated in the hundreds of gigabytes (or terabytes if including HDF5 depth maps). A full download is computationally intractable and prohibited by project guidelines.
- **Selective Download Strategy (Proposed):**
  - **Do NOT download the full repository.**
  - Restrict download strictly to **Warehouse scenes only** (e.g. `Warehouse_000` for train, `Warehouse_015` for validation).
  - Omit `depth_maps/` HDF5 files; download only `videos/` and `ground_truth.json` / `calibration.json`.
  - Extract sampled frames (e.g. 1 frame per second or 2 FPS) to build a manageable 2D detection benchmark without redundant consecutive video frames.
- **Exact Error / Limitation:** Attempting an unguided full download risks disk exhaustion and network timeout. Subsetting requires targeted scripted downloading using `hf_hub_download` or `snapshot_download(allow_patterns=...)`.
- **Next Troubleshooting Steps:**
  1. Formulate a minimal download script targeting a single training scene (`Warehouse_000`) and single validation scene (`Warehouse_015`).
  2. Filter exclusively for RGB MP4 files and corresponding JSON annotation files, omitting multi-gigabyte depth archives.

---

## 3. Explicit Recovery Criteria

The blocker status will remain **BLOCKED** until all of the following criteria are verified empirically:

1. **Access Confirmed:** Authenticated or unrestricted connection established without HTTP 401/403 or rate-limiting errors.
2. **Minimal Sample Downloaded:** Exactly one small scene or a minimal set of sample frames downloaded to local workspace (e.g. `< 500 MB`).
3. **Archive/File Integrity Checked:** Checksums or archive integrity verified (e.g. video playable, JSON parses with zero syntax errors).
4. **Annotation Format Inspected:** Verified coordinate conventions (top-left vs center, pixel vs normalized coordinates, class taxonomy mapping).
5. **Minimal Data-Loading Test Passed:** A lightweight PyTorch/torchvision Dataset script successfully loads an image and returns matching bounding boxes and class IDs.

---

## 4. Current Decoupled Workplan

While datasets are blocked, research progress continues independently on:
- **Literature Review & Evidence Synthesis:** Analyzing prior YOLO and Mamba detection papers, populating `F_literature_and_evidence/source_registry.md`.
- **LaTeX Infrastructure:** Maintaining the clean IEEE/standard paper skeleton in `paper/`.
- **Experiment Planning & Ablation Specifications:** Refining experimental matrices and metric definitions in `E_research_design/`.
- **Data Conversion Design:** Drafting parser logic for converting MOTChallenge/JSON formats to standard YOLO bounding box formats (`class_id x_center y_center width height`).
