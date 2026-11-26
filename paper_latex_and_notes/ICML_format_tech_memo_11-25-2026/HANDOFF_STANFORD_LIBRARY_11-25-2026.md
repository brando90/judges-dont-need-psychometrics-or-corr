# Hand-off: Stanford library deposit of the 11-25-2026 technical memo

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/HANDOFF_STANFORD_LIBRARY_11-25-2026.md>

**TLDR:** The memo is finished, built and committed to `main`; it is not deposited. A deposit in the Stanford Digital Repository (SDR) needs an account holder, and whoever deposits uses this note. It names the two files to upload, which are committed in the `deposit/` folder next to it, with their hashes, the metadata to paste and the decisions only Brando can make.

## 1. What to upload

Both files are committed in `paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/deposit/`, so a depositor can download them from GitHub, and `SHA256SUMS.txt` in that folder lists both hashes.

1. PDF, `ce-not-correlation-memo-11-25-2026.pdf` (https://github.com/brando90/judges-dont-need-psychometrics-or-corr/raw/main/paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/deposit/ce-not-correlation-memo-11-25-2026.pdf). It is a byte-for-byte copy of `memo_11-25-2026.pdf` in this folder, 40 pages. SHA-256: `ab4f92200190d7b08b5140a118f81cc144c6ede3a172374160ce60af13caa179`.
2. Source archive, `ce-not-correlation-memo-sources-11-25-2026.zip` (https://github.com/brando90/judges-dont-need-psychometrics-or-corr/raw/main/paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/deposit/ce-not-correlation-memo-sources-11-25-2026.zip). It holds the 50 tracked source files of this folder: the LaTeX source, the generated figures and tables, the bibliography, the ICML kit, the build and check scripts, the log of every number the figure script prints, and a README. `make pdf` rebuilds the PDF from it. `make` also regenerates the figures and tables, which needs the rest of the repository. SHA-256: `3dc80f8d51efc6bfbbf4273af6a049de8e2e6e08fc31a03ce5d9f7aebffdf10f`.

The earlier deposit used the same pair (a PDF and a source archive), so this follows that route. If either file is rebuilt, replace it in `deposit/` and record the new hash here and in `SHA256SUMS.txt`.

## 2. Metadata to paste

- **Title:** Learn Judges Like Any Other Model: Cross-Entropy, Not Correlation, for Predicting Human Rating Distributions
- **Author:** Brando Miranda, Department of Computer Science, Stanford University (brando9@stanford.edu)
- **Date and version:** 11-25-2026, version 1, preprint (technical memo)
- **Keywords:** LLM-as-a-judge, proper scoring rules, ranked probability score, cross-entropy, evaluation, scaling laws
- **License:** CC BY 4.0, as for the 11-23-2026 deposit (see decision 1 below)
- **Related item:** continues and extends the earlier preprint "What Correlation Can and Cannot Identify in Human-Rating Judges", DOI 10.25740/bh388th6008 (<https://purl.stanford.edu/bh388th6008>)
- **Code and data:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr>. Every experiment result the memo reads was committed at or before commit `2bd73d9`. The memo's own files are in the commit that last changed `memo_11-25-2026.pdf` (find it with `git log -1 --format=%h -- paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/memo_11-25-2026.pdf`).
- **Length and format:** 40 pages, ICML 2026 style (preprint option). The main text ends on page 9, the references are pages 10 to 11, and Appendices A to H are pages 12 to 40.

**Abstract (copied from the memo):**

Teams validate AI judges by their correlation with human ratings. Correlation sees only conditional means, and only up to a positive affine map. It cannot tell a judge with the right means and the wrong rating distributions from a correct one. We treat a judge as a probability model of human ratings, fit it with cross-entropy and score it with proper scores, chiefly the ranked probability score (RPS). An exact three-item example matches every conditional mean, so Pearson correlation is 1, yet it has mean total variation distance 0.20 and cross-entropy 1.065 against 0.915 nats. Across synthetic targets and human ratings of theorems, homework, images, dialogue, helpfulness and inference, cross-entropy fitting usually beats Pearson fitting on held-out RPS (lower is better); SummEval summaries are the exception, where at 768 ratings the cross-entropy fit selects the uniform forecast (RPS 0.284) and the Pearson fit scores 0.144. Frozen frontier judges with a calibration head cross over at 4–8 ratings, one per item, in all 25 judge-by-dataset cells. Matched low-rank adaptation (LoRA) of Qwen3-4B and Llama-3.1-8B favors cross-entropy in all seven setups at the largest budget, by 0.013 to 0.076 RPS on raw forecasts and 0.0002 to 0.0043 after one shared calibration. A saturating power law fits the held-out RPS curve of cross-entropy-fit judges in 9 of 11 panels (median exponent 0.66 on RPS and 0.82 on cross-entropy where the floor is identified) and that of Pearson-fit judges in 1 of 11 (exponent 0.09, extrapolated floor). Correlation remains competitive below about eight ratings, when the mean barely varies, and for choosing a single judge from some pools. Judges are an ordinary supervised learning problem: choose a proper loss, hold out data and fit the learning curve.

## 3. Decisions only Brando can make

1. **License.** The 11-23 deposit is CC BY 4.0. Keep the same unless you want something else.
2. **Acknowledgments.** The memo thanks Sanmi Koyejo for repeated technical feedback that shaped experiments 11, 15 to 17 and 22. Confirm he is comfortable with that line, or delete it and rebuild before depositing.
3. **AI-use statement.** The memo has none, like the earlier deposit. Add one only if the library or a later venue asks for it, and rebuild so the PDF and its hash stay matched.
4. **New object or new version.** This is a different document that builds on the 11-23 preprint, so a new deposit that lists the earlier DOI as a related item is the cleaner record than a new version of the old one.
5. **Folder numbering.** The duplicate number 30 is resolved: the Cursor figures report that came in through pull request 2 was renumbered to experiment 32 on 11-25-2026, under the repository's duplicate-number rule, and Appendix F lists it as 32. One active and one archived folder both begin with 11. Appendices C and F say which one each table means. Renumbering the active one would change paths the memo and the paper cite, so leave it unless you want that rebuild.

## 4. After the deposit

1. Put the new Stanford record and DOI in the repository README next to the 11-23 DOI. Do not change the PDF, since its hash is now the deposited one.
2. Replace the GitHub link to the memo with the DOI wherever it is cited, including the "current draft" link in the blog post `_posts/2026-11-25-bitter-judges-lesson.md` of the website repository (`brando90/brandomiranda`).
3. Record the final PDF and archive hashes, the record URL and the DOI in a `DEPOSIT_11-25-2026.md` next to this file, in the format of experiment 27's `DEPOSIT_11-23-2026.md`.
