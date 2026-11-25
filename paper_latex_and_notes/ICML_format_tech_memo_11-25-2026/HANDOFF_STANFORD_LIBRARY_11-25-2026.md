# Hand-off: Stanford library deposit of the 11-25-2026 technical memo

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/HANDOFF_STANFORD_LIBRARY_11-25-2026.md>

**TLDR:** The memo is finished, built and committed to `main`; it is not deposited. A deposit in the Stanford Digital Repository needs a person with the account, so whoever deposits uses this note. It has the two files with their hashes, the metadata to paste, and the decisions only Brando can make. Nobody has been contacted about it.

## 1. What to upload

1. PDF, deposit name `ce-not-correlation-memo-11-25-2026.pdf`. It is a byte-for-byte copy of `memo_11-25-2026.pdf` in this folder. SHA-256: `385fe8caa1d52c2c8a0bc771565bfdddf4f807b3b18e4ed92ba373bf460d63dc`.
2. Source archive, `ce-not-correlation-memo-sources-11-25-2026.zip`. It holds the LaTeX source, the generated figures and tables, the bibliography, the ICML kit, the build and check scripts and a README. `make pdf` rebuilds the PDF from it. SHA-256 of the archive delivered with this note: `114a203d3cb432ebb69df358001ee59a77f5fdaa34ed85e5f43812b61ad3dc78`. If the archive is rebuilt, record its new hash here.

The earlier deposit used the same pair (a PDF and a source archive), so this follows that route.

## 2. Metadata to paste

- **Title:** Learn Judges Like Any Other Model: Cross-Entropy, Not Correlation, for Predicting Human Rating Distributions
- **Author:** Brando Miranda, Department of Computer Science, Stanford University (brando9@stanford.edu)
- **Date and version:** 11-25-2026, version 1, preprint (technical memo)
- **Keywords:** LLM-as-a-judge, proper scoring rules, ranked probability score, cross-entropy, evaluation, scaling laws
- **License:** CC BY 4.0, as for the 11-23-2026 deposit (see decision 1 below)
- **Related item:** continues and extends the earlier preprint "What Correlation Can and Cannot Identify in Human-Rating Judges", DOI 10.25740/bh388th6008 (<https://purl.stanford.edu/bh388th6008>)
- **Code and data:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr>, built from commit `ce9e209`; the memo files are added by the commit that adds this note (find it with `git log --diff-filter=A -- paper_latex_and_notes/ICML_format_tech_memo_11-25-2026/memo_11-25-2026.pdf`)
- **Length and format:** 23 pages, ICML 2026 style; main text through page 9, references on pages 9 and 10, Appendices A to E on pages 11 to 23

**Abstract (copied from the memo):**

Teams choose and validate AI judges by their correlation with human ratings. Correlation sees only conditional means, up to a positive affine map, so it cannot tell a judge with the right means and the wrong rating distributions from a correct one. We instead treat a judge as a probability model of expert ratings, fit it with cross-entropy, and score it with proper scores such as the ranked probability score (RPS). First, an exact three-item example matches every conditional mean (Pearson correlation 1) yet has total variation distance 0.20 and cross-entropy 1.065 against 0.915 nats. Second, across the repository's experiments on synthetic targets and on human ratings of theorems, homework, images, dialogue, helpfulness and inference, cross-entropy fitting beats Pearson fitting on held-out RPS. Frozen frontier judges with a calibration head cross over at 4–8 ratings, one per item, in all 25 judge-by-dataset cells, and matched low-rank adaptation (LoRA) of Qwen3-4B and Llama-3.1-8B favors cross-entropy in all seven setups at the largest budget. Third, cross-entropy error falls as a saturating power law in the number of ratings with exponent about 0.6–0.9, while Pearson RPS curves are flat, non-monotone or still declining at the largest budget tested. Correlation remains competitive below about eight ratings, when the mean barely varies, and for choosing among scalar-vote judges. Judges are an ordinary supervised learning problem: choose a proper loss, hold out data, and fit the learning curve.

## 3. Decisions only Brando can make

1. **License.** The 11-23 deposit is CC BY 4.0. Keep the same unless you want something else.
2. **Acknowledgments.** The memo thanks Sanmi Koyejo for repeated technical feedback that shaped experiments 11, 15 to 17 and 22. Confirm he is comfortable with that line, or delete it and rebuild before depositing.
3. **AI-use statement.** The memo has none, like the earlier deposit. Add one only if the library or a later venue asks for it, and rebuild so the PDF and its hash stay matched.
4. **New object or new version.** This is a different document that builds on the 11-23 preprint, so a new deposit that lists the earlier DOI as a related item is the cleaner record than a new version of the old one.
5. **Folder numbering.** The duplicate number 30 is resolved: the Cursor figures report that came in through pull request 2 was renumbered to experiment 32 on 11-25-2026, under the repository's duplicate-number rule, and Appendix C lists it as 32. One active and one archived folder both begin with 11; Appendix C notes it, and renumbering the active one would change paths the memo and the paper cite, so leave it unless you want that rebuild.

## 4. After the deposit

1. Put the new Stanford record and DOI in the repository README next to the 11-23 DOI. Do not change the PDF, since its hash is now the deposited one.
2. Replace the GitHub link to the memo with the DOI wherever it is cited, including the blog's "current draft" link.
3. Record the final PDF and archive hashes in a `DEPOSIT_11-25-2026.md` next to this file, in the format of experiment 27's `DEPOSIT_11-23-2026.md`.
