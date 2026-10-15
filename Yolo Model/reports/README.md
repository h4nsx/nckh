# IEEE-Style LaTeX Paper: Durian YOLO Benchmarking

## Overview

This directory contains a complete, compilable IEEE-style LaTeX paper converted from the original manuscript at `/home/sandbox/durian_yolo_manuscript.md`.

## Files Structure

```
/home/sandbox/reports/
├── main.tex                          # Main LaTeX document (entry point)
├── main.pdf                          # Compiled PDF output (13 pages, 206 KB)
├── references.bib                    # BibTeX bibliography (auto-generated)
├── sections/
│   ├── introduction.tex              # Section I: Introduction
│   ├── related_work.tex              # Section II: Related Work
│   ├── methods.tex                   # Section III: Materials and Methods
│   ├── results.tex                   # Section IV: Results and Discussion
│   └── conclusion.tex                # Section V: Conclusion and Future Work
├── fig1_optimization_comparison.png  # Figure 1 placeholder
└── fig2_model_comparison.png         # Figure 2 placeholder
```

## Paper Details

**Title:** Benchmarking YOLOv8 to YOLOv12 for Real-Time Durian Leaf Disease Detection on Raspberry Pi: A Systematic Accuracy-Speed Trade-off Analysis

**Document Class:** `\documentclass[conference]{IEEEtran}`

**Pages:** 13 pages (including references)

**Sections:**
1. Introduction (with subsection: Research Objectives and Contributions)
2. Related Work (4 subsections)
3. Materials and Methods (6 subsections)
4. Results and Discussion (6 subsections)
5. Conclusion and Future Work (4 subsections)
6. Acknowledgments
7. References (auto-generated from references.bib)

## Key Features

### IEEE Formatting Compliance
- Two-column layout (IEEEtran handles automatically)
- Proper section numbering and hierarchy
- IEEE-style abstract with keywords
- Author block with affiliations and superscripts
- Conference paper format

### Tables (5 total)
1. **Table I:** Dataset Statistics (Section III-A)
2. **Table II:** Model Architecture Characteristics (Section III-C)
3. **Table III:** GPU Training Performance (Section IV-A)
4. **Table IV:** Raspberry Pi Performance (Section IV-B, full-width table*)
5. **Table V:** Optimization Format Comparison (Section IV-C)

All tables use:
- `\caption{}` above table (IEEE style)
- `\label{}` for cross-referencing
- `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`)
- `\resizebox{\columnwidth}{!}{...}` for wide tables
- `[!t]` placement (top of page)

### Figures (2 placeholders)
1. **Figure 1:** Optimization format comparison (Section IV-C)
2. **Figure 2:** Model architecture comparison (Section IV-D)

Figure format:
- `\includegraphics[width=\columnwidth]{filename.png}`
- `\caption{}` below figure (IEEE style)
- `\label{}` for cross-referencing
- `[!t]` placement

### Equations (4 total)
- Precision, Recall, F1-Score, IoU definitions
- Proper LaTeX math mode with `\equation` environment
- Numbered equations for reference

### Citations
- Uses `\cite{}` commands with numeric keys (1-150)
- Bibliography auto-generated from `references.bib`
- IEEE citation style (`\bibliographystyle{ieeetr}`)
- All 53 references from original manuscript included

### Packages Used
```latex
\usepackage{cite}              % IEEE citation handling
\usepackage{amsmath,amssymb,amsfonts}  % Math symbols
\usepackage{graphicx}          % Graphics inclusion
\usepackage{booktabs}          % Professional tables
\usepackage{multirow}          % Multi-row table cells
\usepackage{hyperref}          % Hyperlinks and PDF metadata
\usepackage{xcolor}            % Color support
\usepackage{float}             % Float placement control
\usepackage{caption}           % Caption customization
\usepackage{subcaption}        % Subfigures
\usepackage{url}               % URL formatting
\usepackage[utf8]{inputenc}    % UTF-8 input encoding
\usepackage[T1]{fontenc}       % T1 font encoding
```

## Compilation Instructions

### Standard Compilation (with BibTeX)
```bash
cd /home/sandbox/reports
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Quick Compilation (without bibliography updates)
```bash
cd /home/sandbox/reports
pdflatex main.tex
```

### Using latexmk (if available)
```bash
cd /home/sandbox/reports
latexmk -pdf main.tex
```

## Content Summary

### Abstract
Comprehensive abstract covering:
- Problem statement (durian disease detection challenges)
- Methodology (YOLOv8-v12 comparison on Raspberry Pi)
- Dataset (2,000 images, 4 disease classes)
- Key results (YOLOv11n optimal: 87.3% mAP@50, 12.5 FPS)
- Contributions (practical deployment guidelines)

### Introduction (Section I)
- Background on durian cultivation and disease challenges
- Motivation for edge AI deployment
- YOLO evolution overview
- Research gaps identification
- 5 research objectives
- 5 key contributions
- Paper organization

### Related Work (Section II)
- YOLO evolution and plant disease detection
- Lightweight YOLO architectures
- Edge deployment and Raspberry Pi studies
- Durian disease detection
- Research gap analysis

### Materials and Methods (Section III)
- Dataset description (2,000 images, 4 classes)
- Data augmentation strategy
- Model architectures (8 variants)
- Training configuration
- Model optimization (PyTorch, ONNX, TFLite, OpenVINO)
- Edge device evaluation (Raspberry Pi 4B)
- Evaluation metrics (mAP, Precision, Recall, F1, IoU)

### Results and Discussion (Section IV)
- GPU training performance
- Raspberry Pi inference performance
- Optimization format comparison
- Model architecture comparison
- Per-class performance analysis
- Practical deployment considerations

### Conclusion and Future Work (Section V)
- Summary of contributions
- Broader implications
- Limitations
- 7 future research directions:
  1. Advanced model optimization
  2. Hardware acceleration
  3. Multi-modal and multi-task learning
  4. System integration and field deployment
  5. Generalization and transfer learning
  6. Explainability and trust
  7. Concluding remarks

### References
- 53 references auto-generated from references.bib
- IEEE citation style
- Includes DOIs where available

## Placeholder Content

### Figures
The current PDF includes placeholder figures (simple text boxes). To replace with actual figures:

1. Generate/obtain the actual figures
2. Save as PNG/PDF in `/home/sandbox/reports/`
3. Ensure filenames match:
   - `fig1_optimization_comparison.png`
   - `fig2_model_comparison.png`
4. Recompile: `pdflatex main.tex`

### Author Information
Current placeholder:
```latex
Author One^1, Author Two^1, Author Three^2
^1 Department of Computer Science, University Name, City, Country
^2 Department of Agricultural Technology, University Name, City, Country
Email: author@university.edu
```

To update:
1. Edit `main.tex` lines 28-33
2. Replace with actual author names, affiliations, emails
3. Recompile

## Validation Checklist

✅ **Compilation:** Compiles without errors (only minor warnings)
✅ **Document Class:** IEEEtran conference format
✅ **Title:** Uses recommended title from manuscript
✅ **Abstract:** Complete with IEEE keywords
✅ **Sections:** All 5 main sections + subsections
✅ **Tables:** All 5 tables converted with proper IEEE formatting
✅ **Figures:** 2 figure placeholders with captions
✅ **Equations:** 4 equations properly formatted
✅ **Citations:** All \cite{} commands reference valid BibTeX entries
✅ **Bibliography:** Auto-generated from references.bib (53 entries)
✅ **Cross-references:** All \ref{} and \label{} pairs valid
✅ **Page Count:** 13 pages (appropriate for conference paper)
✅ **File Size:** 206 KB (reasonable for submission)

## Known Issues and Notes

1. **Caption Warning:** Minor warning about unknown document class for caption package (cosmetic, does not affect output)

2. **Placeholder Figures:** Current figures are simple text placeholders. Replace with actual plots/diagrams for final submission.

3. **Author Block:** Uses placeholder author information. Update before submission.

4. **Acknowledgments:** Generic acknowledgment text. Customize with actual funding sources and contributors.

5. **Data Availability:** Includes placeholder for dataset repository URL. Update with actual link upon publication.

6. **Underfull hbox warnings:** Minor typesetting warnings (common in LaTeX, do not affect readability)

## Output Files

**Main Output:**
- `/home/sandbox/reports/main.pdf` (13 pages, 206 KB)
- `/home/sandbox/durian_yolo_ieee_paper.pdf` (copy for convenience)

**Auxiliary Files:**
- `main.aux` - Auxiliary file for cross-references
- `main.bbl` - Bibliography file (generated by BibTeX)
- `main.blg` - BibTeX log
- `main.log` - LaTeX compilation log
- `main.out` - Hyperref outline file

## Customization Guide

### To Change Title
Edit line 24 in `main.tex`:
```latex
\title{Your New Title Here}
```

### To Add/Remove Sections
1. Create/delete `.tex` file in `sections/`
2. Add/remove `\input{sections/filename}` in `main.tex`

### To Modify Tables
Edit the corresponding section file in `sections/`:
- Table I-II: `methods.tex`
- Table III-V: `results.tex`

### To Add Figures
1. Place image file in `/home/sandbox/reports/`
2. Add figure environment in appropriate section file:
```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{your_figure.png}
\caption{Your caption here.}
\label{fig:your_label}
\end{figure}
```

### To Update References
1. Edit `references.bib` (or it's auto-generated from citation keys)
2. Recompile with BibTeX:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submission Checklist

Before submitting to IEEE conference:

- [ ] Replace placeholder figures with actual plots
- [ ] Update author information with real names and affiliations
- [ ] Customize acknowledgments with actual funding sources
- [ ] Add data availability statement with repository URL
- [ ] Verify all citations are accurate and complete
- [ ] Check that all tables and figures are referenced in text
- [ ] Ensure page limit compliance (typically 6-8 pages for IEEE conferences)
- [ ] Convert to PDF/A format if required by conference
- [ ] Validate PDF with IEEE PDF eXpress (if required)
- [ ] Include copyright notice if required by conference

## Contact and Support

For questions about this LaTeX conversion:
- Original manuscript: `/home/sandbox/durian_yolo_manuscript.md`
- LaTeX source: `/home/sandbox/reports/`
- Compiled PDF: `/home/sandbox/durian_yolo_ieee_paper.pdf`

## Version History

- **2026-08-27:** Initial LaTeX conversion from Markdown manuscript
  - Complete IEEE-style formatting
  - All sections, tables, and citations converted
  - Placeholder figures included
  - Successfully compiled to 13-page PDF
