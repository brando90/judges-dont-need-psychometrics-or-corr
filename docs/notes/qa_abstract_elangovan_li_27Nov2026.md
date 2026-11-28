# QA of the Elangovan/Li paragraph, the abstract and the intro (27 Nov 2026, late evening)

**TLDR:** The requested adversarial reviewers ("Fable 5.2", "Codex Vals Ultra") could not be run from the box without subscription credits or a new credential, so this QA was done by hand: a hostile area-chair pass and a claim-by-claim trace of every number and every Elangovan/Li statement to its source. 11 findings; 10 fixed, 1 kept as a stated risk. Both LaTeX editions compile with 0 errors, 0 undefined citations or references and no "??" (venue 13 pages, arXiv 26 pages). API spend \$0.

## Were the requested reviewers reachable?

| reviewer asked for | what was tried | result |
|---|---|---|
| Fable 5.2 | Stanford gateway `https://aiapi-prod.stanford.edu/v1/models` (the endpoint in Exp 46's `api_panel.json`) | HTTP 401 "No api key passed in"; no key on the box (no gateway, OpenAI or Anthropic key in the environment) |
| Fable 5.2 | Vercel AI Gateway public catalog `https://ai-gateway.vercel.sh/v1/models` | lists `anthropic/claude-fable-5.2`, but calling it needs an AI Gateway key; none is configured. A Vercel connector tool can mint one, but creating a credential was not authorized, so it was not used |
| Fable 5.2 | Claude Code CLI (how Exp 19's Fable 5.2 review ran) | not installed on the box, and subscription credits are excluded |
| Codex Vals Ultra | same two gateway catalogs; `codex` CLI | no model with that name in either catalog (closest: `openai/gpt-5.3-codex`); `codex` CLI not installed; prior repo QA ran "Codex, effort ultra" through the subscription CLI, which is excluded |

Spend: \$0 (logged in `docs/STATE_OF_EVIDENCE.md`). To run them, the box needs a gateway key (Stanford or Vercel AI Gateway); with one, the review prompt is the scope below.

## Hostile area-chair pass (by hand)

| # | finding | disposition |
|---|---|---|
| F1 | The abstract said the binned JSD groups items "by the human majority"; Eq. 1 of Elangovan et al. bins ordinal items by the human *median* and nominal ones by the most frequent label. | Fixed: the abstract names the method only; the intro and related work say "median or majority". |
| F2 | "Left open: item-level strictly proper scores for ordinal or nominal human ratings" reads as a novelty claim, but Lyu et al. (2026) already score judges with annotation-level RPS and Brier, and one-rating unbiasedness is classical. | Fixed: the abstract says these scores are "adopted by concurrent work", and the open question is whether they *change conclusions about real judges*. The paragraph says "also used concurrently by Lyu et al." |
| F3 | "Left open is whether ..." overstates: Guerdan et al. (2025) already show that validation-metric choice changes judge selection. | Softened to "Less settled is whether ..."; the supplement already calls Guerdan et al. the closest evidence. Kept as a stated risk (R2). |
| F4 | "maps moved across datasets lost skill in 78 of 82": the 82 include 2 rubric shifts within SummEval. | Fixed: "across datasets or rubrics" (abstract and intro). |
| F5 | The "100–200 held-out ratings" figure Brando asked to reuse was not a frontier-judge figure: Exp 46 cites it from Exp 47, which measures linear heads. | Fixed by measurement: Exp 56 (protocol frozen at b487c4b, then run) gives about 100 single ratings per task for every frontier judge to beat the base rate. Abstract and intro use Exp 56; Exp 46's README now points to it. |
| F6 | The paragraph said Li et al. report "majority agreement only" on human data; App. D.1 also compares an aggregate win rate with human votes. | Fixed: "majority agreement and an aggregate win rate (Sec. 5.3, App. D.1)". |
| F7 | The related-work paragraph on Li et al. said they are scored "against preference distributions built from ten annotations" without saying the annotations are an LLM teacher's, and listed Brier as if distributional. | Fixed: the votes are Gemini-2.5-Flash persona annotations (Sec. 4), and Brier/ECE use the binarized majority (App. C.1). Same correction in Exp 54's coding note (count unchanged, Li kept as P_strict, the conservative choice). |
| F8 | The intro Evidence item and the supplement's "What is new" item still led with, or leaned on, the cross-entropy-vs-correlation linear heads, which the advisor called circular. | Fixed: the intro item now leads with frontier-judge evidence and calls the linear-head comparison circular and a check only; the supplement's "This supports the evaluation claim ..." sentence is replaced by a circularity statement. |
| F9 | "Elangovan et al. leave rankings alone": they already show metric-dependent judge rankings (Tables 2–3 captions). Claiming "rank reversals" as new would contradict them. | Fixed: our increment is stated as a *significance-tested* reversal between correlation and a proper score (paragraph, related work). |
| F10 | "Correlation is squared error after the best affine refit" (identity stated loosely). | Fixed: "Correlation sees only a judge's best affine refit" in the abstract; the exact identity stays in the intro and supplement. |
| F11 | The paragraph and abstract cite Li et al. with different years (2025 arXiv vs 2026 in the bib). | Fixed: "Li et al. (2026)" throughout, matching `li2026judging`. |

## Claim trace: abstract

| claim (abstract) | source | check |
|---|---|---|
| Judges are validated by correlation or agreement | Exp 43/54 audit (29 of 30 direct-score papers report correlation; 13 of 13 pairwise report agreement/accuracy/kappa) | matches supplement §practice |
| Target $p^*(r\mid$item, pool, rubric$)$ | paper Sec. "What Scalar Criteria Identify" | unchanged |
| Binned JSD scores judges against the distribution | Elangovan Sec. 3.2, Eqs. 1–2, p. 7 | matches |
| Squared error to per-item vote means, pairwise | Li Sec. 5.1 p. 7 ("MSE between the predicted $p_\theta(x)$ and ... $\hat p(x)$"), Sec. 2.1 (pairwise) | matches |
| Strictly proper scores estimable from one rating per item; adopted concurrently | paper Sec. "Evaluate Conditional Probability Forecasts"; Exp 52; Lyu et al. per supplement | matches |
| Correlation sees only the best affine refit | $r^2 = 1-\mathrm{MSE}_{\text{aff}}/\mathrm{Var}$, supplement "What Is New Here" item 2 | matches |
| Sampled scores: correlation prefers the conditional-mean judge | Lean-checked, $r_T = r_D^2 < r_D$ (supplement item 2) | matches |
| Spearman vs Brier skill reverse one pair on SNLI and MNLI, Holm-significant | Exp 46 README result 1: Spearman +0.021 / +0.034, Brier skill −0.039 / −0.098, Holm p = 0.025 / 0.042 | matches |
| Four DICES-350 judges with 80–82% majority agreement below the base rate | Exp 46 result 2 and per-judge table (skill −0.096 to −0.177, CIs < 0; top-1 0.803–0.823); Exp 55 H1 (all three rules) | matches |
| Two judges tied on agreement differ by 0.175 in skill | Exp 46 result 3 (top-1 difference +0.001 [−0.022, +0.023]; skill −0.175 [−0.200, −0.152]) | matches |
| Recalibration removes these gaps | Exp 46 result 4 (0 significant reversals; every judge's skill positive after calibration) | matches |
| With in-domain labels correlation suffices to choose | Exp 46 H3 frozen reading; Exp 53 (calibrated difference −0.0003 to +0.0098); Exp 41 (≤ 0.0024 nRPS) | interpretation, consistent |
| About 100 ratings per task before every judge beat the base rate | Exp 56 README (n = 100: every judge's mean calibrated skill > 0 on all 5 sets; n = 50: 4 of 5 sets) | matches |
| Maps across datasets or rubrics lose skill in 78 of 82, 33 worse than none | Exp 48 README A1 skill terms (78 of 82 CI below 0 vs in-domain; 33 of 82 worse than raw) | matches |

## Claim trace: intro (changed sentences)

| claim | source | check |
|---|---|---|
| Elangovan pool all human ratings and sampled judge labels within bins sharing a human median or majority, compare by JSD | Elangovan Sec. 3.2, Eq. 1 (bins), Sec. 3 p. 4 (20 samples, T = 1.0) | matches |
| Li score pairwise probabilities by squared error to each item's ten-vote mean; train with Brier and log rewards | Li Sec. 5.1, Sec. 4 ("1K prompts ... each with 10 annotations"), Eqs. 1–2 | matches |
| Wang et al. report squared error to averaged human preference | Exp 43 `papers.py` coding of Wang, Zhang & Choi; supplement paragraph | matches the audit (paper not re-read tonight) |
| Binned divergence is neither item-level nor proper | Elangovan Eq. 2 (per-bin pooled histograms); Exp 54 | matches |
| Li's votes come from an LLM teacher; ordinal left to future work | Li Sec. 4 p. 5; App. E p. 21 | matches |
| 21 of 25 judge-set cases below uncalibrated with 25 ratings | Exp 56 summary table | matches |
| Linear heads within 0.01 skill after calibration | Exp 53 README (−0.0003 to +0.0098) | matches |

## Claim trace: paragraph for the advisor

Every sentence is pinned in the appendix list of [elangovan_li_summary_27Nov2026.md](elangovan_li_summary_27Nov2026.md) (items 1–17, with quotes and page numbers checked against the PDFs page by page with `pdftotext -f N -l N`). Two page numbers were corrected during the check (Elangovan Secs. 4.2 and 4.3 are on p. 10, not pp. 8–9).

## Build

`pdflatex → bibtex → pdflatex × 2` for both editions: 0 errors, 0 undefined citations, 0 undefined references, no "??" in the PDF text. Venue edition 13 pages (unchanged; references still start on p. 6); arXiv edition 26 pages (25 before, the added prior-work and Exp 56 sentences).

## Remaining risks

- R1. Exp 56 uses one calibrator family and 50 draws; "about 100" is a mean-over-draws statement (at n = 100, 8 of 25 cases are still below their uncalibrated skill on average).
- R2. Guerdan et al. (2025) already show that validation-metric choice changes judge selection; the abstract's "less settled" framing and the supplement's credit are the mitigation.
- R3. Li et al. was read in its arXiv v1; the bib cites the Findings of EMNLP 2026 version, whose section numbers could differ.
- R4. Wang, Zhang & Choi (2025) was not re-read tonight; the intro's one clause about it rests on the Exp 43 coding.

## Mega QA triage, 28 Nov 2026 (Grok Bot)

Reviewers: Codex CLI (gpt-6-astra, reasoning effort ultra, read-only sandbox, Brando's "vals" subscription) on main at 929bc19, output verbatim in `qa_codex_vals_28Nov2026.md`. Fable 5.2 via Claude Code did **not** run: every Anthropic CLI login on the Mac was logged out or expired (`qa_fable52_28Nov2026.md`). Codex verdict: warning 2 conceded explicitly; warning 1 acknowledged, but the introduction still claimed an opening that cited work occupies. All 20 findings accepted; the ones marked "verified" were checked against the source before fixing. Numbers used in fixes come only from committed READMEs/JSON.

| ID | Sev. | Finding | Decision | Fix |
|---|---|---|---|---|
| F01 | major | Intro still claims item-level proper ordinal/nominal scoring as open; Lyu App. E has preliminary transfer and label-budget studies | accept (verified, arXiv 2610.37868 App. E) | Intro: proper scores "already in use", we adopt them; increment = real-judge evidence, identification, audit. Related work: Lyu App. E credited, Exps 48/56 framed as the analogous question for post-hoc maps of frozen judges. Summary paragraph reworded |
| F02 | major | "Correlation suffices" not established; point reversals remain after calibration | accept (verified, Exp 46 kendall.csv: calibrated MNLI, correlations pick Astra, proper score picks Opus) | Abstract/intro: "no significant reversal remained … correlation-based selection was not contradicted"; intro names the MNLI point reversal. Earlier line 47 of this file is superseded |
| F03 | major | "About 100 ratings" is about means and the first tested budget; "per task" ambiguous | accept | "every judge's mean calibrated skill first exceeds the base rate at 100 single ratings per dataset, the smallest tested budget at which it does"; 50 draws stated |
| F04 | major | Li App. D.2 controlled ablation omitted when we say their comparison confounds method and split | accept (verified) | Related work cites D.2 numbers (0.7913/0.1321 vs 0.8575/0.1015) and what they show |
| F05 | major | Summary mischaracterizes Li's calibration baselines and OOD evaluation | accept | Summary item 17: only temperature scaling uses held-out labels; contextual/batch are label-free; PandaLM OOD acknowledged; claim narrowed to post-hoc map transfer/budget |
| F06 | major | "Without in-domain labels" undefined | accept | Abstract separates calibration labels from held-out evaluation ratings ("when calibration labels are scarce … report a proper score on held-out target ratings") |
| F07 | major | Shift population mixed | accept | Abstract/intro: "other datasets, rubrics or pooled sources"; intro adds 21 of 33 harms are NLI↔DICES (different label sets). Per-type table already in Exp 48 README |
| F08 | major | Primary preregistered Pearson null not disclosed (same as Claude's Exp 46 erratum) | accept | Abstract and intro lead with the Pearson null; Spearman reversals labeled secondary/ordered-coded. Recorded in STATE_OF_EVIDENCE |
| F09 | major | NLI ordinal coding is an assumption | accept (verified, Exp 46 PROTOCOL line 26) | "with NLI labels coded as ordered" in abstract and intro |
| F10 | major | Sampled-score theorem stated beyond its hypotheses (Pearson, conditional independence, nondegeneracy) | accept (verified, PlanClaims.lean) | Abstract/intro now say Pearson, independence given the item, and the nondegeneracy conditions |
| F11 | major | "Affine identity predicts" the calibrated gap overstated | accept | "consistent with the two heads differing mainly in calibration" |
| F12 | major | Audit counts include bin-level Elangovan and Li's teacher votes | accept | Related work labels both counts conservative, broad codes and names the two borderline papers |
| F13 | minor | 0.0024 / 0.01 are maxima of seed means | accept (verified, Exp 41/53 JSON: 0.00244 mean, seed max 0.00315; 0.0098, seed max 0.0126) | Related work, calibration appendix, supp and intro reworded as largest five-seed mean |
| F14 | minor | 78/82, 33/82 are significance counts; summary said "cross-dataset"; G-Eval hidden | accept (verified, point counts 82/82, 34/82) | "significantly" added; "datasets, rubrics or pooled sources" |
| F15 | minor | 0.175 is normalized-RPS skill; "tied" too strong | accept | Intro: "normalized-RPS skill", "agreement differs by 0.001"; abstract: "nearly equal agreement" |
| F16 | minor | Li App. E says framework applies to nominal/ordinal, defers only empirical study | accept | Related work and intro reworded |
| F17 | minor | Summary drops clipping caveat of Prop. 1 | accept | Summary item 11 adds the clipped maximizer |
| F18 | minor | "Scores judges against it" over-identifies both papers with p* | accept | Abstract: "pooled human and judge labels", "per-item vote means" |
| F19 | minor | Exp 06 values are test Spearman of differently trained fits | accept | Related work states the quantity, seeds and test tasks |
| F20 | minor | FineWeb-Edu thresholds a classifier trained on LLM scores | accept | Intro line 7 reworded |

Rejected: none. Not done: a predefined regret/equivalence criterion for "selection adequacy" (F02 remedy) would need a new preregistered analysis; the paper now avoids the sufficiency claim instead.
