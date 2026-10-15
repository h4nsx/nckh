# Research Gap Matrix

## Purpose

Use this matrix to identify a defensible research gap from evidence across multiple studies. A gap is not established by a single paper omitting a feature. It must be compared against relevant prior work and checked for newer studies.

## Rules

1. Record verifiable facts and cite sources using stable IDs from `source_registry.md`.
2. Distinguish author-stated limitations from your own interpretation.
3. Do not mark a gap as confirmed until the search strategy and evidence are sufficient.
4. Search recent work before claiming that something is novel.
5. If the gap is not supported, revise the question rather than forcing the original idea.

## Comparison matrix

| Source ID | Year/status | Task/domain | Base detector | Mamba/SSM location | Fusion method | Lightweight strategy | Dataset | Metrics | Hardware benchmark | Author-stated limitations | Potential relevance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P001 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P002 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| P003 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Candidate gap categories to investigate

These are search directions, not confirmed gaps.

### G1 — Efficient Mamba integration
- What integration locations have already been tested?
- Do papers report actual device latency, or only FLOPs/parameters?
- Is the Mamba component efficient at the target image resolution and feature-map sizes?
- Does the design introduce memory or runtime overhead not reflected in FLOPs?

### G2 — Warehouse-specific detection
- Which studies evaluate warehouse classes and crowded scenes?
- Are the datasets real, synthetic, or mixed?
- Are object size, occlusion, or scene-level results reported?
- Are class taxonomies and evaluation splits comparable?

### G3 — Data transfer or auxiliary pretraining
- Has the exact dataset pair already been used together?
- Are their annotations and class definitions compatible?
- Does auxiliary data improve generalization when evaluated on a held-out target split?
- Could apparent gains result from leakage, duplicate scenes, or distribution overlap?

### G4 — Deployment-oriented evaluation
- Which studies test on real edge hardware?
- Are latency, throughput, peak memory, and power measured?
- Are preprocessing and postprocessing included?
- Are results reproducible on the same device/runtime?

### G5 — Compression and accuracy trade-off
- Which compression methods are already applied to YOLO-Mamba variants?
- Does structured pruning help the actual runtime, or only reduce parameter count?
- What is the impact of PTQ/QAT on accuracy and latency?
- Does hardware/runtime support the chosen precision or sparsity pattern?

## Gap statement template

> Existing studies have investigated [well-supported summary of prior work; cite multiple sources]. However, the reviewed evidence indicates that [specific, narrowly scoped limitation] remains insufficiently evaluated under [defined task, data, hardware, or protocol]. This study therefore investigates [specific method/question] and evaluates it using [metrics and baselines].

Do not complete this paragraph until the supporting papers have been reviewed.

## Gap validation checklist

- [ ] Search terms and databases recorded.
- [ ] Recent papers included through the date of the search.
- [ ] Closest competing methods identified.
- [ ] Related papers compared by architecture, data, metrics, and hardware.
- [ ] Claimed gap is not merely a change of dataset or name.
- [ ] Gap is feasible to test with available compute and data.
- [ ] The contribution can be falsified by experiments.
- [ ] The wording avoids absolute claims such as “the first” unless exhaustively verified.

## Current conclusion

**Research gap status: [UNKNOWN].** The gap must be filled after the literature review. Do not use a placeholder gap as a confirmed claim in an abstract or introduction.
