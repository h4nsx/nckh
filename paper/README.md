# Paper Directory Overview and Build Instructions

## Working Paper Title
**A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices**

> **Note:** This title is provisional and will be revised once the literature review and experimental results are established.

---

## 1. Directory Structure

```text
paper/
├── main.tex                          # Main document entry point (working article template)
├── references.bib                    # BibTeX bibliography file (verified entries only)
├── sections/                         # Modular section files
│   ├── 01_introduction.tex           # Section 1: Introduction, Problem, Gap, Contributions
│   ├── 02_related_work.tex           # Section 2: Related Work & Comparative Synthesis
│   ├── 03_methodology.tex            # Section 3: Proposed Architecture, Formulations, Complexity
│   ├── 04_experimental_setup.tex     # Section 4: Datasets, Baselines, Training & Edge Protocol
│   ├── 05_results_discussion.tex     # Section 5: Comparative Benchmarks, Ablations, Limitations
│   └── 06_conclusion.tex             # Section 6: Summary, Impact, Future Directions
├── figures/                          # Directory for vector diagrams and qualitative visuals
│   └── .gitkeep
├── tables/                           # Directory for external table inputs (if needed)
│   └── .gitkeep
└── README.md                         # This file
```

---

## 2. Compilation Instructions

This project currently uses standard LaTeX packages compatible with modern TeX distributions (TeX Live, MacTeX, MiKTeX):

### Standard Two-Pass Build via `latexmk`:
```bash
cd paper
latexmk -pdf main.tex
```

### Manual Build Sequence:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

> **Target Venue Notice:** This skeleton uses the standard `article` document class as a neutral working template. Once a target publication venue is selected (e.g., IEEE Conference/Transactions, Springer LNCS, ACM), update `main.tex` to load the official class file (such as `IEEEtran.cls`) and adhere to its specific author guidelines, citation style, and layout rules.

---

## 3. Authoring and Evidence Discipline

- **No Fabricated Information:** Do not insert fake citations, fabricated benchmark numbers, simulated hardware latencies, or unverified dataset statistics.
- **Traceable Claims:** Keep all factual claims tied to source IDs in `F_literature_and_evidence/source_registry.md` and `F_literature_and_evidence/claim_evidence_matrix.md`.
- **Experimental Verification:** Experimental results must come from verified logs created via `templates/experiment_record.md`.
- **Placeholders:** Unresolved decisions are explicitly flagged with `TODO` markers indicating the responsible planning document.
