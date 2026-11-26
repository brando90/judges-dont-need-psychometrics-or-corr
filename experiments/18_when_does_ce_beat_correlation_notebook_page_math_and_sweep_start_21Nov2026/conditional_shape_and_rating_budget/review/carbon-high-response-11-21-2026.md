# Gemini Carbon High reviewer response (11-21-2026) — QA corrections 11-26-2026

**QA correction (11-26-2026):** Read the [dated correction and evidence limits](../../QA_CORRECTIONS_11-26-2026.md) before the historical text below; original scored tables remain unchanged.

Date: 11-21-2026

Model response with Markdown whitespace normalized. The [archived packet](carbon-high-input-11-21-2026.txt.gz) preserves the input. The [builder’s resolution](REVIEW_11-21-2026.md) links the original receipt.

**Verdict:** **FIX REQUIRED**
**Severity:** **Minor** (no primary RPS conclusion, mathematical derivation, or interval is invalidated, but there is one cross-file numerical contradiction, one IEEE-754 tie-breaking bug in secondary Spearman evaluation, and one protocol-vs-output omission, plus one minor denominator-wording ambiguity).

---

### Concrete Errors and Corrections

1. **Cross-file numerical contradiction in reported held-out Spearman correlation (`0.691` vs. `0.692`)**
   - **Files:** `README.md` (`## Evidence from historical expert ratings`, *"Can a Pearson-selected judge win at test correlation?"*) vs. `historical_rating_budget/RESULTS_11-21-2026.md` (`## Did training with Pearson win on held-out Pearson or Spearman?`, secondary metric table).
   - **Issue:** For the identical cell (`15` training families, `2` ratings per training artifact, `ce` selector), `README.md` reports mean within-split test Spearman as **`0.691`** (`0.691 vs 0.632`), whereas `historical_rating_budget/RESULTS_11-21-2026.md` reports **`0.692`** (`0.692` vs. `0.632`), both citing `historical_rating_budget/scores.csv` at three decimal places.
   - **Correction:** Reconcile `README.md` and `historical_rating_budget/RESULTS_11-21-2026.md` against the authoritative float in `historical_rating_budget/scores.csv` (`train_families=15, train_ratings_per_artifact=2, selector=ce, spearman`) after fixing Issue #2 below.

2. **IEEE-754 floating-point tie-breaking bug in empirical test means before `rankdata` (violates tied-rank Spearman protocol)**
   - **Files:** `historical_rating_budget/analyze.py` (`histograms()` and `test_panel()`, lines 66–67 and 85–89) vs. `historical_rating_budget/PROTOCOL_11-21-2026.md` (`## Fixed comparisons and metrics`: *"Pearson and tied-rank Spearman correlations compare expected and empirical mean ratings"*).
   - **Issue:** In `scripts/run.py` (lines 125–126), empirical test means are computed via `test_r.mean(axis=1)` specifically because `# Direct integer totals preserve exact ties for midranks and Kendall tau-b.` By contrast, `historical_rating_budget/analyze.py` computes `observed = p @ R` from the 6-bin normalized histogram `p = np.bincount(row, minlength=6) / len(row)` (where test artifacts have $k_i \in \{2, 3\}$ ratings) and passes `observed` directly to `scipy.stats.rankdata(observed)`, which tests exact float equality (`==`). Across rating multisets of size 2 and 3 on $\{0,\dots,5\}$, `p @ R` splits exact rational ties by 1 ULP (`2.22e-16` to `4.44e-16`):
     - Exact mean `2`: `(0, 1, 5)` gives `0x1.fffffffffffffp+0` (`1.99999999999999978`), whereas `(1, 2, 3)`, `(2, 2, 2)`, `(1, 3)`, `(2, 2)` give `0x1.0000000000000p+1` (`2.0`).
     - Exact mean `7/3`: `(1, 3, 3)` gives `0x1.2aaaaaaaaaaabp+1`, whereas `(0, 2, 5)`, `(0, 3, 4)`, `(1, 1, 5)`, `(1, 2, 4)`, `(2, 2, 3)` give `0x1.2aaaaaaaaaaaap+1`.
     - Exact mean `4`: `(2, 5, 5)` and `(3, 4, 5)` give `0x1.fffffffffffffp+1` (`3.99999999999999956`), whereas `(4, 4, 4)`, `(3, 5)`, `(4, 4)` give `0x1.0000000000000p+2` (`4.0`).
     - Under pairwise SIMD summation, exact mean `10/3` also splits (`0x1.aaaaaaaaaaaaap+1` vs. `0x1.aaaaaaaaaaaabp+1`).
     As a result, `rankdata(observed)` breaks true empirical-mean ties rather than assigning midranks.
   - **Correction:** In `historical_rating_budget/analyze.py`, compute empirical test means directly from integer sums—e.g., `observed = np.array([row.mean() for row in ratings])[te]` (0 broken ties across all multisets of size 2 and 3 on $\{0,\dots,5\}$), or pass exact integer scaled means `np.array([row.sum() * (6 // len(row)) for row in ratings])[te]` to `rankdata`.

3. **Protocol-specified descriptive contrast to the constant baseline omitted from code and report**
   - **Files:** `historical_rating_budget/PROTOCOL_11-21-2026.md` (`## Fixed comparisons and metrics`) vs. `historical_rating_budget/analyze.py` (lines 137, 179–187) and `historical_rating_budget/RESULTS_11-21-2026.md` (line 36).
   - **Issue:** `PROTOCOL_11-21-2026.md` states: *"Also report the contrast to RPS selection and the constant on identical test artifacts as descriptive controls."* However, `analyze.py` tracks `estimates` only for `OBJECTIVES = ("ce", "pearson", "rps")` and writes `paired_rps.csv` only for `subtrahend_selector` in `("ce", "rps")`, omitting any paired difference against `constant`. Meanwhile, `RESULTS_11-21-2026.md` reports only absolute RPS columns for `RPS-selected` and `Constant` in its table while labeling the link `[All paired RPS contrasts](../historical_rating_budget/paired_rps.csv)`.
   - **Correction:** Either add the paired descriptive contrast against `constant` to `analyze.py` / `paired_rps.csv` (and report the descriptive contrasts to RPS selection and constant in `RESULTS_11-21-2026.md`, noting whether constant is unweighted descriptive only or family-reweighted on both train and test), or explicitly document the protocol deviation in `RESULTS_11-21-2026.md`.

4. **Minor denominator-wording ambiguity on unequal-$k_i$ test splits (clarification recommended)**
   - **Files:** `README.md` (`## Evidence from historical expert ratings`) and `historical_rating_budget/RESULTS_11-21-2026.md` (`## Held-out rating distributions` and `## Did training with Pearson win on held-out Pearson or Spearman?`).
   - **Issue:** `historical_rating_budget/PROTOCOL_11-21-2026.md` and `MATH_11-21-2026.md` explicitly define split-level losses with **equal artifact weights** ($\frac{1}{|\text{te}|}\sum_{i \in \text{te}} \frac{1}{k_i}\sum_{j=1}^{k_i} \ell(q_i, y_{ij})$), which `analyze.py` implements (`rps.mean(0)`, `ce.mean(0)`, `observed.mean()`). Because historical test artifacts have $k_i \in \{2, 3\}$ ratings (30 have 2, 45 have 3), equal-artifact averaging differs from equal-rating pooling ($\frac{1}{\sum_{i \in \text{te}} k_i}\sum_{i,j}$). `README.md` and `RESULTS_11-21-2026.md` omit the equal-artifact weighting detail while defining CE/RPS *"for an individual rating"* alongside *"24–87 ratings"* per test split.
   - **Correction:** State briefly in `README.md` and `RESULTS_11-21-2026.md` that individual-rating losses are averaged within each test artifact first and then equally across test artifacts within each split (matching `PROTOCOL_11-21-2026.md`).

---

### Audit Summary by Review Criterion

- **Mathematical identities, derivatives, and bounds (`MATH_11-21-2026.md`, `README.md`, `input_ablation/README.md`):** **Verified exact.**
  - Standardized Pearson loss $1 - \rho = \frac{1}{2n}\sum_i (z_i - t_i)^2$ (with population-divisor $s_m, s_h$), both rating/category expansions of $\rho$ (including unequal $k_i$), CE logit gradient $(q_i(r) - \hat p_i(r))/n$, and Pearson logit gradient $\frac{\rho z_i - t_i}{n s_m} q_i(r)(r - m_i)$ (verified analytically and via central finite differences to $2 \times 10^{-10}$) are exact.
  - Nullspace dimension ($6 - 2 = 4$ mean- and mass-preserving shape directions per interior distribution), population correlation bound $\operatorname{Corr}(m(X), Y) = \operatorname{Corr}(m(X), \mu(X))\sqrt{\operatorname{Var}(\mu(X))/\operatorname{Var}(Y)}$, $k$-rating variance $\operatorname{Var}(\mu(X)) + \mathbb E[\operatorname{Var}(Y\mid X)]/k$, 1-rating empirical TVD expectation $1 - \sum_r p(r)q(r)$, Bhattacharyya/Markov bound $(\mathbb E_X[B(X)^k])^n$, Jensen comparison $(\mathbb E B(X))^{nk} \le (\mathbb E[B(X)^k])^n$, and all three rows of the distribution-correlation table are exact.
  - Exact rational enumeration confirms every reported figure in the 6-category nonlinear-mean example (`0.121293` nats, `1/360` RPS, `1/6` TVD; `11,664` sequences, Bayes error `809/1944 ≈ 0.416152263`, bounds `2/9 ≈ 0.222222222` and `0.478651439`) and all 20 cells of `input_ablation/README.md` (including `4(0.06)^2/5 = 0.00288` RPS excess and `0.789` task-only Pearson).
- **Loss definitions and histogram identities (`scripts/run.py`, `historical_rating_budget/analyze.py`):** **Verified exact.**
  - Both `Q**2 + (1 - 2*Q)*P` (`run.py`) and `(Q - P)**2 + P*(1 - P)` (`analyze.py`) are algebraically identical to $\frac{1}{k_i}\sum_{j=1}^{k_i}(Q(t) - \mathbf{1}\{y_{ij} \le t\})^2$ because indicator variables satisfy $I^2 = I$.
  - Candidate probabilities in `run.py` are strictly positive ($\min q = 2/75 > 0$), sum to 1, satisfy $\mathbb E[R\mid x,z] = 2.5 + 0.42 g(x)$ and $\Delta\operatorname{Var}(R\mid x,z) = +0.96 s z$, and yield exact population RPS shape penalty $4(0.08)^2/5 = 0.00512$.
- **Reported differences, counts, and cross-table arithmetic (`README.md`, `results.md`, `RESULTS_11-21-2026.md`):** **Consistent except for Issue #1.**
  - All primary RPS differences, roundings (4-decimal in `README.md` vs. 5-decimal in `RESULTS_11-21-2026.md`), exploratory budget interactions (`-0.01718`, `-0.01412`), signed biases, fixed-128-rating contrasts, validation rating counts (`2, 8, 32, 128, 512`), oracle-vs-default constant finite-sample shift (`0.00509` across all 9 varying rows), record/denominator counts (`32,256` rows; `4,608` instances; `75` artifacts; `195` ratings; `1,125` forecasts; `3,750` candidate correlations per budget; `12–30` test artifacts), and `Population TVD = 0.00000` / `0.16000` (confirming 256/256 exact recovery at `128 × 16`) check out.
- **Split handling and ties:** **Sound except for Issue #2.**
  - Synthetic `train`, `val`, and `test` sets are drawn independently per replicate; bootstrap intervals resample the 256 independent replicates. Historical splits enforce family-level disjointness (15 train vs. 4 stratified test families: 3 of 14 ordinary, 1 of 5 placeholder), while explicitly disclosing rater overlap, 60/75 unresolved reference hashes, 27 uniform parse fallbacks, and prior dataset exposure. Candidate tie tolerances (`1e-12`) and lexicographic fallback rules match both protocols.
- **Conditional family uncertainty, fitting vs. selection, and exact vs. empirical scope:** **Accurately framed.**
  - The 1,000-draw $\text{Exp}(1)$ family-level Bayesian bootstrap with full candidate reselection (`analyze.py`) matches `PROTOCOL_11-21-2026.md` and is properly scoped as conditional on the candidate pool, rater panel, split schedule, and subsampled masks rather than external population coverage (and honestly reports that the 15-family intervals span zero).
  - The package cleanly separates fixed-pool selection (`historical_rating_budget`), finite-family global enumeration (`PROTOCOL.md` / `results.md`), earlier nonconvex 48-parameter fitting (`README.md`), and exact population/decision-problem examples (`MATH_11-21-2026.md`, `input_ablation/README.md`), explicitly refraining from universal human sample-size claims.
