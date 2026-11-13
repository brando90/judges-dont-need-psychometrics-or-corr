# Correlation Alone Does Not Identify a Judge's Conditional Rating Distribution

Research draft in the bundled AAAI-27 author kit. No venue submission has been made. The current [four-page PDF](00_main_aaai_2027.pdf) includes the completed controlled synthetic study and public historical results, including null findings, followed by the references and an appendix that explains the ranked probability score (RPS) with intuition, propriety and worked examples ([8-appendix-rps.tex](8-appendix-rps.tex); every example number is printed by [rps_appendix_examples.py](rps_appendix_examples.py)). The 2–4 page limit excludes references; whether the appendix counts toward it should be checked against the workshop call before submission. Implementation is complete; the parent owns the requested multi-provider review and main publication.

Layout follows agents-config (Rules 35/49): one venue dir under `paper_latex_and_notes/`, main file `00_main_aaai_2027.tex`, numbered section files, `00_main.bib`, `math_commands.tex`, `figures/`, committed PDF, aux files in `_build/`. Style files `aaai2027.sty` and `aaai2027.bst` come from https://aaai.org/authorkit27/.

From the repository root:

```bash
make -C paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL
```

The default target creates `_build/`, runs pdfLaTeX, BibTeX and two further pdfLaTeX passes, and copies the PDF alongside the sources. `make latexmk` is an alternative. The bundled AAAI style requires pdfTeX; Tectonic uses an incompatible engine. The measured build used TeX Live 2026 with `newtx`, `booktabs`, `natbib`, AMS fonts/math and their dependencies. The build completed without errors or unresolved citations. All three rendered pages were inspected; five underfull-box warnings remain, with no clipping or overlap observed.

See the [build and render receipt](../../experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/results/paper_build.json), [full study report](../../experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/RESULTS_conditional_identification.md), and [primary-source citation audit](../../experiments/01_conditional_binomial_distribution_identification_start_08Nov2026/CITATIONS.md). The build receipt records source/PDF hashes, engine version and page-image hashes. `inspect_paper.py` reproduces rendering using PyMuPDF; images are runtime evidence outside the repository.
