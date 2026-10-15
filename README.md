# YOLO-Mamba Research Workspace

## Project purpose

This workspace contains the research-planning, evidence-management, and paper-structure documents for the working research topic:

**A Lightweight Hybrid YOLO-Mamba Architecture for Real-Time Multi-Class Warehouse Object Detection on Edge Devices**

The title is provisional. The final title, research gap, architecture, and contributions must be revised after the literature review and experiments.

## Research status

- **Stage:** Research preparation / literature review.
- **Proposed task:** Multi-class 2D warehouse object detection.
- **Dataset Access Status:** **BLOCKED — download/access issue** (See [`F_literature_and_evidence/dataset_access_status.md`](F_literature_and_evidence/dataset_access_status.md) for full blocker report, recovery criteria, and troubleshooting).
- **Primary dataset candidate:** NVIDIA PhysicalAI-SmartSpaces (Blocked: extreme size / selective subset strategy required).
- **Auxiliary dataset candidate:** NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse (Blocked: download authentication / rate limit issue).
- **Optional extension:** NVIDIA PhysicalAI-SimReady-Warehouse-01.
- **Core idea under investigation:** Efficiently combine local visual feature extraction (CNN) and long-range/contextual modeling (Mamba or another state-space approach), then evaluate the accuracy–efficiency trade-off.
- **No results have been established yet.** Do not invent results, citations, dataset statistics, hardware performance, or novelty claims.

## Document map

### E — Research design
- `E_research_design/research_questions_and_hypotheses.md`
- `E_research_design/research_gap_matrix.md`
- `E_research_design/research_gap_validation.md`
- `E_research_design/research_problem_and_architectural_comparison.md`
- `E_research_design/architecture_candidates.md`
- `E_research_design/architecture_decision_log.md`
- `E_research_design/baseline_plan.md`
- `E_research_design/experiment_matrix.md`
- `E_research_design/experimental_protocol_draft.md`
- `E_research_design/ablation_plan.md`
- `E_research_design/deployment_plan.md`
- `E_research_design/threats_to_validity.md`

### F — Literature, sources, and evidence
- `F_literature_and_evidence/dataset_access_status.md`
- `F_literature_and_evidence/dataset_feasibility.md`
- `F_literature_and_evidence/research_claim_audit.md`
- `F_literature_and_evidence/literature_review_protocol.md`
- `F_literature_and_evidence/literature_review_template.md`
- `F_literature_and_evidence/source_registry.md`
- `F_literature_and_evidence/claim_evidence_matrix.md`
- `F_literature_and_evidence/references_bib_guide.md`
- `F_literature_and_evidence/reproducibility_manifest.md`

### G — Paper and LaTeX
- `G_paper_and_latex/paper_outline.md`
- `G_paper_and_latex/latex_project_spec.md`
- `G_paper_and_latex/writing_and_citation_rules.md`
- `G_paper_and_latex/antigravity_agent_instructions.md`

### Templates
- `templates/paper_analysis_card.md`
- `templates/experiment_record.md`
- `templates/dataset_card.md`

## Recommended workflow

1. Read this README and `G_paper_and_latex/antigravity_agent_instructions.md`.
2. Set up a LaTeX skeleton using `G_paper_and_latex/paper_outline.md` and `G_paper_and_latex/latex_project_spec.md`.
3. Collect papers and register each source in `source_registry.md`.
4. Analyze every relevant paper using `templates/paper_analysis_card.md`; consolidate findings in `literature_review_template.md`.
5. Update the research-gap matrix only when supported by multiple sources.
6. Inspect the official dataset cards, licenses, annotation schemas, and downloaded files before locking the dataset plan.
7. Select a baseline and architecture only after the evidence review.
8. Run experiments and record configurations/results. Never write predicted outcomes as measured results.
9. Draft the paper section by section, keeping claims connected to the evidence matrix.
10. Before submission, verify all references, numbers, figures, tables, and claims.

## Evidence labels

Use these labels consistently:
- **[SOURCE]** Explicitly stated in a cited source.
- **[MEASURED]** Produced by an experiment in this project.
- **[DERIVED]** Calculated from recorded data using a stated method.
- **[HYPOTHESIS]** A testable expectation, not yet verified.
- **[PROPOSAL]** A design choice under consideration.
- **[UNKNOWN]** Not yet verified.
- **[CONFLICT]** Sources or experiments disagree.

## Non-negotiable research rules

- Never fabricate papers, DOIs, citations, results, hardware specifications, or dataset counts.
- Distinguish the source's reported results from results reproduced by this project.
- Do not claim novelty just because YOLO, Mamba, and a warehouse dataset are combined.
- Do not claim real-time performance without measured latency/FPS on specified hardware and a clear timing protocol.
- Do not compare FPS values measured under incompatible settings as if they were directly comparable.
- Preserve train/validation/test separation and prevent leakage across video sequences, cameras, scenes, or near-duplicate frames.
- Check dataset licenses and permitted use before publication or redistribution.
- Treat every proposed research contribution as provisional until supported by literature and experiments.
