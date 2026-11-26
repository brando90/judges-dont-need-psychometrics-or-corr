# How to read the results of the squared-difference note, 11-26-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/HOW_TO_READ_RESULTS.md>

**Data.** Every number is analytic or exact rational arithmetic on small discrete laws written in `check_squared_difference_vs_distribution.py` (the six-rating pair $p=(\tfrac12,0,0,0,0,\tfrac12)$ and $q=(0,0,\tfrac12,\tfrac12,0,0)$, the pairs on $\{0,\dots,4\}$, and random joint pmfs drawn with fixed numpy `default_rng` seeds 1, 2, 7, 8, 9 and 10). No human ratings are used, so no train, validation or test split applies and there is no prior exposure. "Random joint pmf" means integer weights drawn uniformly from 0 to 5 (some forced to zero) and normalised, so every weight is an exact fraction.

**Metrics.**

- Squared difference $E[(X-Y)^2]$: mean of the squared gap between two random variables $X$ and $Y$ on the 0 to 4 or 0 to 5 rating scale. Range 0 to 25 on the 0 to 5 scale. Lower is better; 0 means $X=Y$ almost surely. Reference value in the worked example: 12.5 for the truthful report, 6.5 for the best point mass.
- Mean difference $E[X-Y]$: signed gap of the two means. Zero says nothing about the shape of either law.
- Standardized squared difference $E[(Z_X-Z_Y)^2]=2(1-\rho)$, where $Z_X$ is $X$ minus its mean divided by its standard deviation. Range 0 to 4. Lower is better; 0 means correlation 1.
- Brier distance $\sum_r(p(r)-q(r))^2$: squared Euclidean distance between two probability vectors. Range 0 to 2. Lower is better; 0 means $p=q$. It equals the excess expected Brier score of reporting $q$ when the truth is $p$.
- Total variation distance $\tfrac12\sum_r|p(r)-q(r)|$: range 0 to 1; 1 means disjoint supports.
- Minimum over couplings of $E[(X-Y)^2]$: the smallest squared difference any joint law with the two given marginals can reach (a squared Wasserstein distance, computed by an exact monotone coupling and cross-checked with a linear program). Range 0 to 25.
- "Max gap": largest absolute difference between a quantity computed from its definition and the closed form, over all random instances; 0 for the exact checks, about $10^{-15}$ for the floating-point ones.

**What is subtracted.** Every difference subtracts the same metric on the same laws: for example the squared difference of the matched report minus that of the point mass, both against the same human law $p$. No two different metrics are ever subtracted. Figures 1 and 2 never put two metrics on one axis.

**Uncertainty.** There is none to report: no check is Monte Carlo. The random joint pmfs only supply many different instances of exact identities; each check prints how many it ran and the largest gap.

**How the compared reports were chosen.** No judge is trained. Reports are written by hand (the equal-mean pairs, the point masses, $Y=2X$), searched exhaustively where a search appears (the four-direction count is an exact rank; the pair with equal mean and variance was found by enumerating all pmfs on $\{0,\dots,4\}$ with denominator 8 and the same mean and second moment as $(\tfrac14,0,\tfrac12,0,\tfrac14)$, which returned exactly two other pmfs; the first, $(\tfrac18,\tfrac38,\tfrac18,\tfrac18,\tfrac14)$, is used), or computed in closed form (the minimum over couplings has an exact monotone-coupling cost, and the linear program is only a cross-check).
