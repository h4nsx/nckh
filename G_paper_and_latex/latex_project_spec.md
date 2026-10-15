# LaTeX Project Specification

## Purpose

Provide a maintainable LaTeX skeleton that can be generated or edited by an AI coding agent. The selected venue template takes precedence over this generic structure.

## 1. Suggested directory structure

```text
paper/
├── main.tex
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_experimental_setup.tex
│   ├── 05_results_discussion.tex
│   └── 06_conclusion.tex
├── figures/
├── tables/
├── appendix/
└── README.md
```

## 2. Template selection

Before submission:
1. Choose a target venue or use a neutral working template.
2. Download the official author template.
3. Follow its class file, citation style, page limit, and figure/table requirements.
4. Do not manually recreate an official template if a class/style file exists.

Do not assume IEEE, Springer, or another format until the venue is chosen.

## 3. Main document responsibilities

`main.tex` should:
- load the official or working document class;
- load only necessary packages;
- set title/authors/affiliations;
- include the abstract and keywords in the format required by the class;
- include section files in order;
- load bibliography using the correct style;
- include appendices if required.

Each section file should contain only that section's content, not a second document preamble.

## 4. Figure and table conventions

- Use vector PDF for diagrams when practical.
- Use readable labels and consistent notation.
- Include units and measurement conditions in tables.
- State whether values are from this project or copied from cited work.
- Every figure/table must be referenced and discussed in the text.
- Do not use decorative graphics that do not convey information.
- Include a text description/caption that communicates the main point.
- Respect dataset and image licenses.

Suggested filenames:
- `figures/architecture_overview.pdf`
- `figures/dataset_examples.pdf`
- `figures/qualitative_results.pdf`
- `tables/main_results.tex`
- `tables/ablation_results.tex`

These are suggested names only; create them only when the artifact exists.

## 5. Citation and bibliography

- Use verified BibTeX entries in `references.bib`.
- Use citation commands supported by the selected template.
- Avoid hard-coded reference numbers.
- Resolve missing citations and bibliography warnings before finalizing.
- Keep references in a consistent style and verify DOI/URL metadata.

## 6. Build checks

- Compile from a clean working directory.
- Inspect all warnings, especially undefined references/citations.
- Check overfull/underfull boxes and figure placement.
- Confirm all referenced files exist.
- Verify equations, symbols, units, and table values.
- Ensure the PDF matches the selected template.
- Check page limits and supplementary-material rules.

## 7. Placeholder policy

Use clear placeholders such as `TODO: verify dataset version`. Do not insert fake author names, fake citations, fake numbers, or example results into a submission draft.

## 8. Suggested LaTeX commands

Exact commands depend on the selected class. Keep reusable macros minimal and avoid redefining standard commands unless necessary. Use `\\label{}` and `\\ref{}` for figures, tables, equations, and sections where supported.
