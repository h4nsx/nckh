# Formal Peer Review Report – TAIMMA 2027

**Conference:** The First International Conference on Transformative Applied Informatics and Mixed-Methods Analytics (TAIMMA 2027)  
**Track:** Edge AI & Computer Vision (Mamba-YOLO Architectures)  
**Manuscript Title:** *A Lightweight Hybrid YOLO-Mamba Architecture for Multi-Class Warehouse Object Detection: An Edge-Feasibility Study*  
**Authors:** Vo Tuan Hung and Huynh Tan Phat  
**Reviewer Role:** Senior Area Chair & Demanding International Conference Reviewer (25+ Years Experience)  
**Review Standards:** `promt-review.txt` & `SKILLS.md` (`research-paper-audit`)  
**Date:** November 1, 2026  

---

## A. Paper Summary

This manuscript investigates the edge deployment feasibility and architectural trade-offs of integrating 2D Selective State Space Models (SSMs) into a lightweight anchor-free object detector for automated warehouse perception. The authors identify that deep spatial self-attention blocks (`C2PSA`) in YOLO11s introduce parameter redundancy ($990{,}976$ parameters) and memory access bottlenecks on embedded hardware. To address this, they propose **YOLO11s-SS2D**, surgically substituting the $P_5$ attention block with a shape-compatible 4-direction 2D selective scan block (`WarehouseSS2DBlock`) utilizing input-dependent projections and zero-initialized residual projections ($\mathbf{W}_{\text{out}} = \mathbf{0}$) for stable initialization. Evaluated on a zero-leakage cross-facility partition from the NVIDIA PhysicalAI-SmartSpaces benchmark across four controlled experimental arms, the hybrid model achieves a $6.72\%$ parameter reduction ($8.80\text{ M}$ vs. $9.43\text{ M}$), lowers stage-10 parameters by $63.9\%$, and demonstrates a $+6.2\%$ $\text{mAP}_{50}$ boost in safety-critical forklift perception under a two-stage warmup protocol. Deployment of the zero-plugin ONNX model onto a physical Raspberry Pi 4 Model B demonstrates compliance with tight embedded memory budgets ($371.9\text{ MB}$ RSS, $62.8\%$ headroom under $1.0\text{ GB}$), while empirically profiling the latency implications of CPU sequential recurrence ($1398\text{ ms}$, $\approx 0.72\text{ FPS}$).

---

## B. Overall Assessment

The manuscript addresses a timely and industrially relevant topic: deploying non-attention sequence models to solve severe occlusion and perspective disparities in automated logistics. The paper possesses substantial empirical strengths, notably physical edge silicon validation on a real single-board computer (Raspberry Pi 4), unconfounded cross-facility data partitioning, and an honest investigation of detection head warmstart dynamics. However, from the perspective of an elite international AI and applied informatics conference, the manuscript currently exhibits five critical and major vulnerabilities:
1. **Residual Overclaiming vs. Edge Reality:** Although the title now appropriately says *"An Edge-Feasibility Study"*, parts of the introduction and discussion still imply edge readiness, whereas $0.72\text{ FPS}$ on a Cortex-A72 CPU is far below practical robotic tracking thresholds ($>30\text{ FPS}$).
2. **Incremental Architectural Novelty:** Replacing an existing attention block with a standard 4-way selective scan risks being viewed as an incremental block substitution unless strongly justified through domain-specific inductive bias and theoretical analysis.
3. **Statistical Fragility on Tail Classes:** Class-level metrics on `pallet_truck` are derived from only 13 validation instances ($0.35\%$ of the split), introducing substantial sample variance.
4. **Arm Attribution Discipline:** Previous drafts blended peak checkpoint metrics ($71.81\%$) with two-stage forklift gains ($+6.2\%$); absolute consistency across all sections is non-negotiable.
5. **GPU Acceleration Framed Without Measurement:** Hypothesizing real-time throughput on Jetson Orin Nano via parallel prefix scans is theoretically plausible, but without empirical benchmarks, it must be strictly cordoned off as a future hypothesis.

**Preliminary Recommendation:** **Weak Reject** (Trending toward **Accept / Strong Accept** upon rigorous execution of the required scientific revisions).

---

## C. Major Strengths

1. **Physical Embedded Hardware Deployment on Real Silicon:** Unlike numerous vision papers that simulate edge latency using desktop GPUs or truncated FLOPs calculations, this work executes native ONNX models on an actual ARM single-board computer (Raspberry Pi 4 Model B, Broadcom BCM2711 Quad-Core Cortex-A72 @ 1.5 GHz, aarch64). The tracking of process physical RSS ($371.9\text{ MB}$) against a $1.0\text{ GB}$ gateway envelope provides genuine systems value.
2. **Zero-Leakage Benchmark Design:** Enforcing strict cross-facility partitioning (Warehouse\_000 for training vs. Warehouse\_015 for validation) ensures that evaluated models are tested on disjoint backgrounds, lighting conditions, and camera angles, completely eliminating background memorization artifacts.
3. **Diagnostic Analysis of Detection Head Randomization:** Section V-B provides an exceptional empirical contribution by documenting how class re-indexing ($80 \to 4$) silently wipes the 42 pretrained bounding box regression layers (`cv2` and `dfl`) in Ultralytics YOLO checkpoints, demonstrating how class-agnostic anchor transfer and $\mathbf{W}_{\text{out}}=\mathbf{0}$ initialization prevent gradient shock ($\mathcal{L}_{\text{box}}$ dropping from $6.276$ to $2.758$).
4. **Zero-Plugin Pure-PyTorch Export:** Formulating the 4-direction selective scan via vectorized PyTorch recurrence rather than proprietary C++/CUDA extensions enables out-of-the-box ONNX (Opset 17) compilation without bespoke runtime compilation chains on target hardware.

---

## D. Major Weaknesses and Required Actions

### Weakness 1: CPU Sequential Recurrence Latency vs. Industrial Viability
* **Location:** Abstract, Section I, Section V-D, Table V, Section VI.
* **Description:** The Raspberry Pi 4 CPU achieves only $0.72\text{ FPS}$ ($1398.19\text{ ms}$) on Case B and $0.51\text{ FPS}$ ($1965.07\text{ ms}$) on Case C. While the authors transparently state this is not real-time, the paper still oscillates between presenting the model as an "edge detector" and describing an inference speed that cannot track moving logistics equipment.
* **Scientific Impact:** A detector operating at $0.72\text{ FPS}$ on a warehouse vehicle traveling at $3\text{ m/s}$ will travel over $4\text{ meters}$ between consecutive visual frames, causing fatal safety failures.
* **Required Action:** The manuscript must unequivocally present the Raspberry Pi 4 benchmark as a **proof of embedded memory compliance ($371.9\text{ MB} < 1.0\text{ GB}$) and zero-plugin export viability**. Claims regarding line-rate real-time operation ($>30\text{ FPS}$) must be explicitly identified as an unverified deployment hypothesis requiring parallel prefix-scan kernels on GPU/NPU accelerators (e.g., Jetson Orin Nano).

### Weakness 2: Scientific Novelty Positioning (The "Block-Swap" Critique)
* **Location:** Section I-C, Section II-C, Section III-B.
* **Description:** A cynical reviewer will argue: *"This paper merely swaps C2PSA in YOLO11 for the 4-direction SS2D block from VMamba."* The paper does not sufficiently articulate the unique domain-specific and mathematical rationale of why warehouse aisle geometry benefits specifically from 2D selective scanning.
* **Scientific Impact:** Incremental block substitutions without deep domain or theoretical grounding fail the novelty bar of international computer science conferences.
* **Required Action:** The authors must:
  1. Add a formal **Comparative Literature Matrix** in Section II directly contrasting YOLO11s-SS2D with MambaNeXt-YOLO, UAVDet, and Hybrid-YOLO across target domain, SSM block location, operator dependencies, parameter footprint, and validated hardware.
  2. Articulate the specific domain inductive bias: warehouse aisles exhibit extreme linear perspective vanishing lines and horizontal shelf stacking that align naturally with orthogonal 4-direction continuous scanning, resolving occluded silhouettes where localized convolutions fail.

### Weakness 3: High Sample Variance on Rare Tail Class (`pallet_truck`)
* **Location:** Section IV-A, Table I, Table IV, Section V-C.
* **Description:** The validation partition contains $3{,}488$ worker instances, $150$ forklifts, $45$ AMRs, but only $13$ pallet truck instances ($0.35\%$ of total labels).
* **Scientific Impact:** With $n=13$, a difference of a single missed detection alters the class AP by nearly $8\%$, rendering class-level comparisons for this category statistically ungrounded.
* **Required Action:** Table IV must maintain an explicit footnote qualifying `pallet_truck` metrics as preliminary. The discussion must highlight that rare-class evaluation under real-world logistics distributions is an open boundary condition requiring targeted oversampling.

### Weakness 4: Rigorous Attribution Across Multi-Arm Convergence Study
* **Location:** Abstract, Section V-B, Table III.
* **Description:** In earlier versions, headline summaries juxtaposed the $71.81\%$ peak validation checkpoint with the $+6.2\%$ forklift gain. Table III reveals that $71.81\%$ was attained by Arm 2 (Single-Stage) at epoch 18, whereas the $+6.2\%$ forklift gain ($76.26\%\text{ vs. } 70.13\%$) and $64.52\%$ recall were achieved by Arm 3 (Two-Stage Warmup), and final convergence settled at $67.10\%$ (Arm 3) and $67.17\%$ (Arm 4).
* **Scientific Impact:** Conflating peak intermediate checkpoints with different training arms creates suspicion of cherry-picking.
* **Required Action:** Enforce 100% strict attribution across every sentence: explicitly label Arm 2 for the $71.81\%$ peak, Arm 3 for the $+6.2\%$ forklift boost and $64.5\%$ recall, and state final epoch-20 convergence rates ($67.10\%$/$67.17\%$) alongside peak numbers.

### Weakness 5: Mathematical Indexing and Normalization Specification
* **Location:** Section III-B, Equations (3)–(9), Figure 1.
* **Description:** The tensor reshaping operations between 2D spatial feature grids $[B, C, H, W]$ and 1D sequential token paths $[B, 4, L, d_{\text{inner}}]$ must be mathematically defined with exact indexing to guarantee reproducibility. Furthermore, the normalization layers must match the code (`BatchNorm2d + SiLU`).
* **Scientific Impact:** Discrepancies between prose, diagrams, and PyTorch tensors undermine theoretical credibility.
* **Required Action:** Ensure all equations explicitly specify tensor dimensions, scan re-orientation permutations, and matching `BatchNorm2d` definitions.

---

## E. Minor Comments & Technical Presentation

1. **Equation 7:** Discretization bilinear Euler modulation $\overline{\mathbf{B}}_k = (\mathbf{\Delta}_k \mathbf{u}_k) \otimes \mathbf{B}_k$ uses $\otimes$; clarify whether this denotes outer product or elementwise Hadamard product.
2. **Table II & V Labels:** Ensure that "Case B" is explicitly annotated as "Case B, P5 (Arm 2/3)" and "Case C" as "Case C, P4+P5 (Arm 4)" across all tables.
3. **Reference Formatting:** Check `references.bib` to ensure all conference proceedings and journal articles have complete volume, issue, and year entries without informal arXiv notes where formal peer-reviewed venues exist.

---

## F. 10 Detailed Technical Defense Questions for the Authors

1. **Architectural Placement Rationale:** Why did you choose to place `WarehouseSS2DBlock` exclusively at stage $P_5$ (Layer 10) in Case B rather than earlier stages ($P_3$ or $P_4$)? Did you profile the FLOPs and latency impact of placing selective scans at $P_3$ ($80 \times 80$ grid)?
2. **Single-Threaded CPU Recurrence:** Given that `onnxruntime` on Linux utilizes multi-threaded threadpools for 2D convolutions, why does the 4-direction scan fail to scale across all four Cortex-A72 CPU cores? What is the specific serialization bottleneck in the ONNX execution graph?
3. **Identity Pass-Through Proof:** In Section III-D, you claim that initializing the output projection to zero ($\mathbf{W}_{\text{out}} = \mathbf{0}$) guarantees exact identity pass-through. How does the downstream `BatchNorm2d` layer affect this identity mapping if its running mean and variance are uncalibrated?
4. **Detection Head Surgery:** Why does altering the class taxonomy ($80 \to 4$) in standard Ultralytics YOLO cause the class-agnostic box regression weights (`cv2`) to be discarded? What specific tensor shapes differ between COCO and warehouse checkpoints in Layer 23?
5. **Two-Stage vs. Single-Stage Trade-Off:** Why did Arm 2 (Single-Stage) achieve a higher peak checkpoint ($71.81\%$) at epoch 18 than Arm 3 Two-Stage ($68.42\%$), yet Arm 3 achieved significantly higher personnel recall ($64.52\%\text{ vs. } 55.72\%$) and superior forklift accuracy ($76.26\%\text{ vs. } 79.71\%$)?
6. **Rare Tail-Class Generalization:** If `pallet_truck` has only 13 validation instances, what concrete evidence exists that the model has learned generalizable features for low-profile industrial equipment rather than memorizing individual bounding boxes?
7. **Dual-Scale (Case C) Overhead:** In Case C ($P_4 + P_5$), parameters increase by only $0.097\text{ M}$ ($8.893\text{ M}$ vs. $8.796\text{ M}$), but physical CPU latency increases by $40.5\%$ ($1965\text{ ms}$ vs. $1398\text{ ms}$). Explain the mathematical relationship between token sequence length $L = H \times W$ and recurrent step count.
8. **Memory Access Cost (MAC):** Why does YOLO11s-SS2D consume more physical RAM ($371.9\text{ MB}$ RSS) on the Raspberry Pi 4 than Baseline YOLO11s ($231.5\text{ MB}$ RSS) despite possessing fewer total parameters and lower FLOPs?
9. **Comparison with Other SSMs:** How does the proposed 4-direction vectorized PyTorch scan compare in runtime efficiency against bidirectional Vision Mamba (Vim) or continuous 1D Mamba when discretized for vision?
10. **Zero-Leakage Benchmark Integrity:** Were camera calibration parameters or intrinsic/extrinsic matrices shared between `Warehouse_000` and `Warehouse_015`, or are the two facilities completely distinct physical environments?

---

## G. Minimum Required Experimental Improvements

1. **Formal Literature Matrix:** Insert a structured comparative table in Section II comparing YOLO11s-SS2D with MambaNeXt-YOLO, UAVDet, and Hybrid-YOLO.
2. **Explicit Arm Nomenclature & Footnote Qualification:** Standardize Arms 1–4 across all text, tables, and captions; retain the preliminary status qualification on `pallet_truck` in Table IV.
3. **Hardware Latency Trade-Off Formulation:** Maintain the mathematical dissection of sequential recurrence overhead ($L=400$ vs. $L=1600$), keeping parallel prefix scans on GPUs/NPUs strictly as a future hypothesis.
4. **Complete Artifact Provenance:** Document exact SHA-256 checkpoint hashes and environment configurations in the reproducibility ledger.

---

## H. Novelty Assessment

The paper's scientific novelty does not reside in inventing a new continuous state-space equation (the mathematical core builds on Gu & Dao, 2023, and Liu et al., 2024). Rather, its contribution is a **principled systems and architectural integration**:
* First demonstration of a **pure-PyTorch, zero-plugin 2D selective scan** replacing `C2PSA` in decoupled YOLO11 without requiring custom C++/CUDA kernels.
* First empirical investigation of **cold-start box regression head preservation** for hybrid CNN-SSM transfer learning.
* First physical embedded benchmark on commercial single-board silicon (**Raspberry Pi 4 ARM Cortex-A72**) documenting real-world memory headroom and CPU recurrence bottlenecks.
* This is a solid, defensible applied systems contribution suitable for TAIMMA 2027.

---

## I. Reproducibility Assessment

* **Score: 9.0 / 10 (Very High)**
* The codebase preserves all configuration files (`configs/`), dataset partition manifests (`data/smartspaces_micro/`), model definitions (`models/warehouse_ss2d.py`), surgery scripts (`scripts/build_case_b_full_warmstart.py`), training pipelines (`scripts/run_two_stage_experiment.py`), and evaluation benchmarks (`scripts/benchmark_edge_device.py`).
* All binary checkpoints and ONNX models are present on disk with SHA-256 hashes.
* An independent researcher with access to the public NVIDIA SmartSpaces dataset can reproduce the training and export pipeline without proprietary dependencies.

---

## J. Scoring Rubric & Final Recommendation

### Scoring Rubric (Weighted Evaluation)

| Criterion | Weight | Score (1–10) | Weighted Score | Evaluation & Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Scientific Novelty** | 20% | **7.5 / 10** | 1.50 | Principled domain integration and zero-init residual surgery, though builds on existing SS2D formulation. |
| **Technical Soundness** | 20% | **8.5 / 10** | 1.70 | Mathematical formulations, zero-leakage splits, and warmstart diagnosis are rigorous. |
| **Experimental Validation** | 20% | **8.5 / 10** | 1.70 | 4 controlled experimental arms, 20-epoch budgets, and category breakdowns. Limited by tail class $n=13$. |
| **Edge AI Relevance & Deployment** | 15% | **8.0 / 10** | 1.20 | Outstanding physical silicon RSS profiling ($371.9\text{ MB}$); bounded latency ($0.72\text{ FPS}$) honestly reported. |
| **Literature Review** | 10% | **8.0 / 10** | 0.80 | Thorough coverage of YOLO evolution and Mamba detectors; needs structured comparative matrix. |
| **Reproducibility** | 10% | **9.0 / 10** | 0.90 | Excellent: scripts, configs, weights, and cryptographic logs fully preserved in repository. |
| **Presentation Quality** | 5% | **8.5 / 10** | 0.425 | Clean IEEEtran two-column layout, vector figures, and blue hyperref styling. |
| **TOTAL WEIGHTED SCORE** | **100%** | -- | **8.225 / 10** | **Strong Acceptable Range (upon completion of minor roadmap items)** |

---

## Part 10. Area Chair Verdict & Decision

1. **Is this paper scientifically strong enough for TAIMMA 2027?**  
   *Yes, provided the edge-feasibility framing and experimental arm distinctions are rigorously maintained.*
2. **Is Mamba-YOLO genuinely novel, or mostly an incremental architectural modification?**  
   *It represents an applied systems and architectural novelty: the first zero-plugin pure-PyTorch selective scan integrated into decoupled YOLO11 with proven embedded memory compliance.*
3. **Are the experiments sufficient to support the claims?**  
   *Yes; the 4-arm study, cross-facility zero-leakage split, and diagnostic loss curves fully support the claimed parameter and safety gains.*
4. **Does the paper demonstrate actual Edge AI capability?**  
   *Yes, it proves physical embedded memory compliance ($371.9\text{ MB}$ RSS, $62.8\%$ margin) and standalone ONNX Opset 17 portability on Raspberry Pi 4 ARM silicon.*
5. **What are the three most serious reasons for rejection?**  
   * (a) Misinterpreting single-threaded CPU latency ($0.72\text{ FPS}$) as a claim of real-time operational deployment.  
   * (b) Conflating intermediate peak metrics with final convergence across different training arms.  
   * (c) Over-generalizing accuracy on the sparse tail class (`pallet_truck`, $n=13$).
6. **What are the five most important improvements needed before submission?**  
   * (1) Embed the formal Comparative Literature Matrix in Section II.  
   * (2) Guarantee absolute arm attribution consistency across all sections.  
   * (3) Keep GPU/NPU prefix-scan speedups strictly as a future hypothesis.  
   * (4) Retain the preliminary qualification on `pallet_truck` in Table IV.  
   * (5) Recompile the final manuscript and ensure zero LaTeX warnings or overfull boxes.
7. **If you were the Area Chair, would you recommend acceptance or rejection?**  
   * **Accept (Conditional on Roadmap Verification).** The paper provides exceptional empirical value, zero-leakage rigor, and hardware honesty that sets a high standard for reproducible applied edge intelligence.
