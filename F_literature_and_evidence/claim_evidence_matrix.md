# Claim–Evidence Matrix

## Purpose

Every factual statement in the paper that depends on prior knowledge or project results should be traceable to evidence. This matrix reduces unsupported claims and makes writing/review easier.

## Evidence labels

- **SOURCE:** stated by a paper, dataset card, or official documentation.
- **MEASURED:** measured in this project.
- **DERIVED:** calculated from a recorded source/result.
- **HYPOTHESIS:** not yet verified.
- **PROPOSAL:** intended method or experiment.
- **UNKNOWN:** not verified.
- **CONFLICT:** evidence disagrees.

## Matrix

| Claim ID | Proposed claim | Claim type | Evidence/source ID | Exact location (page/section/table/line) | Supporting evidence | Contradicting evidence | Confidence | Paper section | Action |
|---|---|---|---|---|---|---|---|---|---|
| CL001 | TBD | SOURCE | P001 | TBD | TBD | TBD | TBD | Related Work | Verify |
| CL002 | TBD | MEASURED | EXP001 | TBD | TBD | TBD | TBD | Results | Add after experiment |
| CL003 | The proposed model is novel | UNKNOWN | TBD | TBD | TBD | TBD | Low | Introduction | Do not claim yet |

## Claim review checklist

For each claim:
1. Is it factual, interpretive, or a proposal?
2. Is the source primary and credible?
3. Does the source actually support the wording?
4. Is the claim too broad compared with the evidence?
5. Does another source contradict it?
6. Does it need a date qualifier?
7. Is it a result from the source or from this project?
8. Is the citation attached to the correct sentence?
9. Does the claim require a fresh literature search?
10. Should it be softened or removed?

## High-risk claims requiring special verification

- “The first method to…”
- “No prior work has…”
- “State-of-the-art.”
- “Real-time.”
- “Lightweight.”
- “Generalizes to real-world warehouses.”
- “Significantly improves.”
- Any numerical performance claim.
- Any claim that one dataset has a particular number of images/classes/splits.
- Any statement about license or permitted commercial use.

## Wording examples

Avoid: “Mamba is always faster than attention.”

Prefer: “The cited work reports [specific result] under [specific setting]. Whether that advantage holds for the feature sizes and runtime in this study remains to be evaluated.”

Avoid: “Our model is state-of-the-art.”

Prefer: “Under the evaluation protocol used in this study, the proposed model achieved [measured result] compared with [named baselines].”

Avoid: “The dataset proves real-world generalization.”

Prefer: “The evaluation covers [verified data conditions]. Performance on real-world warehouse footage remains untested unless such footage is included.”
