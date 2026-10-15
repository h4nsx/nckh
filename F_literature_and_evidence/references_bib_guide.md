# References and BibTeX Guide

## Purpose

Maintain accurate, consistent references for LaTeX. The `.bib` file should contain verified metadata, not guessed fields.

## Workflow

1. Find the canonical publisher, DOI, conference, or arXiv record.
2. Verify title, authors, year, venue, DOI/identifier, and version.
3. Export BibTeX from a trusted source where available.
4. Check capitalization of technical names such as YOLO, Mamba, NVIDIA, and dataset names.
5. Use a stable citation key.
6. Test that every `\\cite{...}` key exists.
7. Remove unused references before submission if required by the venue.

## Suggested key convention

`firstauthorYYYYshorttopic`

Examples below are patterns only, not real citation entries:
- `gu2023mamba`
- `liu2024visionmodel`
- `author2025lightweightdetector`

Always verify the actual paper metadata before using a key.

## Citation integrity rules

- Never fabricate a DOI or BibTeX entry.
- Do not cite an abstract if the full paper supports a more precise claim.
- Cite the original method paper for method definitions and the implementation source for code-specific behavior.
- Cite dataset documentation for schema, split, and license claims.
- Cite the exact source of a number.
- Distinguish preprints from peer-reviewed versions.
- Avoid citation stacking where the sources do not support the same claim.
- Keep a one-to-one mapping between paper-analysis cards and source registry IDs.

## BibTeX skeleton

Use this only as a structural example and replace all placeholder values with verified metadata.

```bibtex
@article{verifiedkey2025example,
  author  = {Verified Author},
  title   = {Verified Paper Title},
  journal = {Verified Journal},
  year    = {2025},
  doi     = {VERIFIED-DOI}
}
```

Do not leave example records in a submitted paper.

## LaTeX checks

- Compile the project and inspect unresolved citation warnings.
- Check that citations point to the intended paper.
- Ensure references render in the required venue style.
- Verify all figures/tables cited in text exist and have the correct numbering.
- Recheck bibliography metadata before submission.
