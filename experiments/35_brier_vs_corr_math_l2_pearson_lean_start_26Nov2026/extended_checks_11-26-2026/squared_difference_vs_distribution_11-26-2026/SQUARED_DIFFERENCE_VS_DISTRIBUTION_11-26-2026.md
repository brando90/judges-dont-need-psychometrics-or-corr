# Optimizing a difference of random variables does not match a distribution (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md>

**TLDR:** A difference of two random variables compares them through one coupling and sees only a few moments of it, so driving a mean difference, a squared difference, or a correlation to its optimum does not make two laws equal. The raw squared difference splits exactly as $(\mu_X-\mu_Y)^2+(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y(1-\rho)$; a mean difference constrains one number and leaves $K-2$ free directions for a pmf on $K$ ratings (four for $K=6$); correlation is blind to location and scale. A loss on the whole predicted distribution (Brier, log-loss, ranked probability score) is what identifies the conditional law. 10 of 10 deterministic checks PASS (exact fractions, one linear program); the Lean 4 file compiles with no `sorry`. Synthetic and exact arithmetic only; no human ratings.

Part of experiment 35's extended checks ([`../README.md`](../README.md)). The page that asked for this is page three of the 13:01 photographs, transcribed in [`../../assets/TRANSCRIPTION_DERIVATION_PAGES_ROUND2_11-26-2026.md`](../../assets/TRANSCRIPTION_DERIVATION_PAGES_ROUND2_11-26-2026.md); page one's boxed result $E[(Z_X-Z_Y)^2]=2(1-\rho)$ is the starting identity. Companion notes: [`../brier_proper_6437_style_11-26-2026/`](../brier_proper_6437_style_11-26-2026/BRIER_PROPER_6437_STYLE_11-26-2026.md) (Brier is proper; correlation is not) and [`../../DERIVATION.md`](../../DERIVATION.md).

## Notation (the PS on page one)

Following the PS on page one: $\tilde X=X-\mu_X$ is the **centered** variable and $Z_X=\tilde X/\sigma_X$ is the **standardized** variable, with $\sigma_X>0$; $\rho=\mathrm{corr}(X,Y)$. The earlier pages used the tilde for the standardized variable; `../../DERIVATION.md` was changed to the new convention and this note uses it throughout.

$p_X$ and $p_Y$ are the laws of $X$ and $Y$. **Distribution matching** means $p_X=p_Y$. A **coupling** is a joint law of $(X,Y)$ with those two marginals. A **fixed pairing** means the coupling is set by the data, as in a judge-training set where each judge output is paired with the human rating of the same item. Ratings live on $\lbrace0,\dots,K-1\rbrace$; $K=6$ is the 0 to 5 scale.

## What page three asks

Show that optimizing a difference of random variables does not yield the "optimal" pdf or pmf, meaning distribution matching. This is the step after page two's closing note, which says that Brier versus $E[(X-Y)^2]$ is "obviously very different" because one is a difference at the distribution level and the other at the individual values of the random variables (the expectation there is written with plain $X$ and $Y$, no tildes). Two objectives appear on page three: the squared difference $E[(X-Y)^2]$ and the mean difference $E[X-Y]$. Page one's boxed result is the standardized squared difference, which is correlation. The six results below say exactly what each objective identifies.

## Results

**R1 (page one's identity, two terms longer).** For $X,Y$ with finite positive variances,

$$E[(X-Y)^2]=(\mu_X-\mu_Y)^2+E[(\tilde X-\tilde Y)^2],\qquad E[(\tilde X-\tilde Y)^2]=\sigma_X^2+\sigma_Y^2-2\mathrm{cov}(X,Y)=(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y(1-\rho),$$

and $E[(Z_X-Z_Y)^2]=2(1-\rho)$. Hence

$$E[(X-Y)^2]=\underbrace{(\mu_X-\mu_Y)^2}_{\text{location gap}}+\underbrace{(\sigma_X-\sigma_Y)^2}_{\text{scale gap}}+\underbrace{2\sigma_X\sigma_Y(1-\rho)}_{\text{coupling term}}.$$

*Proof.* Write $X-Y=(\tilde X-\tilde Y)+(\mu_X-\mu_Y)$; the cross term vanishes because $E[\tilde X-\tilde Y]=0$. Expand $E[(\tilde X-\tilde Y)^2]=\sigma_X^2+\sigma_Y^2-2\mathrm{cov}(X,Y)$ and use $\mathrm{cov}=\rho\sigma_X\sigma_Y$. Dividing $\tilde X-\tilde Y$ by the scales gives the standardized form as on page one. $\square$

So the squared difference depends on the pair through five numbers: two means, two standard deviations and one correlation. No moment of order three or higher of either marginal appears.

**R2 (mean difference; counting).** $E[X-Y]=0$ if and only if $\mu_X=\mu_Y$, a single equation. The pmfs on $\lbrace0,\dots,K-1\rbrace$ with a given mean $m$ are the solutions of $\sum_r q(r)=1$ and $\sum_r r\,q(r)=m$ with $q\ge 0$. The two equations are linearly independent for $K\ge2$, so the solution set has dimension $K-2$, and for $0<m<K-1$ it contains strictly positive pmfs, so it is a $(K-2)$-dimensional polytope: four directions for $K=6$, none for $K=2$. More generally, matching normalization and the first $j$ moments leaves $K-1-j$ free directions (the rows $(r^k)_{r=0,\dots,K-1}$ are independent because the nodes are distinct); for $K=6$ that is $4,3,2,1,0$ for $j=1,\dots,5$. Matching a mean and a variance together still leaves three. An example with the same mean $5/2$: $p=(\tfrac12,0,0,0,0,\tfrac12)$ and $q=(0,0,\tfrac12,\tfrac12,0,0)$ have variances $25/4$ and $1/4$, total variation distance $1$ and Brier distance $\sum_r(p(r)-q(r))^2=1$.

**R3 (standardized difference = correlation).** $E[(Z_X-Z_Y)^2]=2(1-\rho)=0$ if and only if $\rho=1$ if and only if $Y=\mu_Y+(\sigma_Y/\sigma_X)(X-\mu_X)$ almost surely. So $p_Y$ is the law of a positive affine image of $X$. That equals $p_X$ only if the scale ratio is $1$ and the means agree, since a different scale changes the variance and a different location changes the mean. Example: $X\sim(\tfrac14,\tfrac12,\tfrac14,0,0)$ on $\lbrace0,\dots,4\rbrace$ and $Y=2X\sim(\tfrac14,0,\tfrac12,0,\tfrac14)$: $\rho=1$, $Z_X=Z_Y$ exactly, yet the Brier distance between the laws is $3/8$ and the total variation distance is $1/2$.

**R4 (fixed pairing identifies only the conditional mean).** Let $Z$ be the item, $X$ the human rating (random given $Z$) and $f(Z)$ a judge's single number. Then

$$E[(X-f(Z))^2]=E[\mathrm{Var}(X\mid Z)]+E\big[(E[X\mid Z]-f(Z))^2\big].$$

*Proof.* Add and subtract $E[X\mid Z]$; the cross term has conditional expectation zero given $Z$ (tower property). $\square$

Consequences: the minimizer is $f^\star(Z)=E[X\mid Z]$; the minimum is $E[\mathrm{Var}(X\mid Z)]$, which is positive unless the human rating is deterministic given the item, so the squared difference cannot reach $0$ here; and any two rating laws with the same conditional mean give the same minimizer. For the pair in R2 the best single number is $5/2$ for both laws, with irreducible squared error $25/4$ for $p$ and $1/4$ for $q$: the judge never learns which law it faces.

**R5 (a free coupling does see the law).** Over all couplings of two marginals with finite variances,

$$\min_{\pi}E_\pi[(X-Y)^2]=(\mu_X-\mu_Y)^2+(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y(1-\rho_{\max}),$$

where $\rho_{\max}$ is the largest correlation any coupling attains (the comonotone, or monotone-rearrangement, coupling in one dimension); the minimum is $0$ if and only if $p_X=p_Y$.

*Proof.* R1 with $\rho$ replaced by the coupling's correlation: minimizing the squared difference maximizes $\rho$. The minimum is zero only if the means agree, the standard deviations agree and $\rho_{\max}=1$. A correlation of one means $Y=a+bX$ almost surely with $b>0$; equal means and standard deviations force $b=1$ and $a=0$, so $p_Y=p_X$. The converse takes $Y=X$. $\square$

So the claim "a squared difference can never do distribution matching" needs the qualifier *when the pairing is fixed by the data*. If the coupling is itself a decision variable (optimal transport), the minimum sees location, scale and shape. A judge trained on paired data does not choose the coupling. Even the minimum over couplings can be driven purely by shape: the laws $(\tfrac14,0,\tfrac12,0,\tfrac14)$ and $(\tfrac18,\tfrac38,\tfrac18,\tfrac18,\tfrac14)$ on $\lbrace0,\dots,4\rbrace$ share mean $2$ and variance $2$ and still have minimum squared difference $1/2$.

**R6 (independent draws reward collapse).** If a stochastic judge draws $Y\sim q$ independently of the human rating $X\sim p$, then

$$E[(X-Y)^2]=\sigma_p^2+\sigma_q^2+(\mu_p-\mu_q)^2=\sigma_p^2+E_{Y\sim q}[(Y-\mu_p)^2].$$

This is linear in $q$, so it is minimized by a point mass at the rating nearest $\mu_p$. The truthful $q=p$ scores $2\sigma_p^2$ and is a minimizer only if $p$ lives on the rating or ratings nearest its own mean (a point mass, or an even split of two adjacent ratings). In the sense of the 6.437 notes this is an improper scoring rule. Example: $p=\tfrac12\delta_0+\tfrac12\delta_5$. The truthful $q=p$ scores $12.5$; point masses at $0,\dots,5$ score $12.5,\,8.5,\,6.5,\,6.5,\,8.5,\,12.5$; with continuous support the point mass at $5/2$ would score $25/4$.

**R7 (what does identify the law).** The excess expected Brier score of a forecast $q$ against the truth $p$ is $\sum_r(p(r)-q(r))^2$, zero only at $q=p$ (strictly proper; see the companion note), and the excess log-loss is $\mathrm{KL}(p\Vert q)$. The ranked probability score uses the $K-1$ cumulative events and is strictly proper as well. All three look at every outcome's probability, which is why they can pin down a distribution on $K$ outcomes ($K-1$ free numbers) while a mean, a variance and a correlation cannot.

## How this sits against the page

1. **The headline claim.** "Optimizing differences of RVs does not yield the optimal pmf" is right for the mean difference (R2), for correlation (R3), for the squared difference under a fixed pairing (R4), and for the squared difference between independent draws (R6, which rewards collapse). For the raw squared difference it needs the qualifier "when the pairing is fixed by the data", because with a free coupling the minimum is zero exactly when the laws are equal (R5). The page's "if $X=Y$ then yes, the squared difference is zero too" is true, and so is the converse: zero forces $X=Y$ almost surely.
2. **"6 degrees of freedom, only 1 equation."** There are six probabilities, but they must sum to one, so five are free; the mean equation removes one more. That leaves four free directions (R2). The conclusion, infinitely many pmfs, is the same. For a pdf the space is infinite-dimensional, which is the page's "and even more".
3. **"Ignoring all higher-order moments."** The squared difference uses two moments of each marginal and the correlation (R1); the mean difference uses one. A pmf on $K$ points is determined only by its moments up to order $K-1$ (R2). The page's circled labels are hard to read, so the transcription records the marks as seen; in this note the objectives are named in words.
4. **A sentence for the paper.** A difference of two random variables scores one coupling of them, while a distribution is a property of their marginals: a mean, a squared difference and a correlation constrain at most two moments plus the coupling, while distribution matching constrains $K-1$ numbers.

## Intuition

Picture the human ratings for one item as a coin that lands on zero or on five. Picture a judge whose rating is a coin that lands on two or on three. Both coins have the same average, two and a half, so a mean difference of zero says they match. They share nothing else: no outcome in common, total variation distance one. A squared difference is no better. For a single judge number it is minimized at the average whatever the shape, so the judge never learns which coin the human has. If the judge instead draws its own rating, the squared difference is smaller when the judge says two or three every time, because a judge that is as spread out as the human adds its own spread to the error; the score rewards collapsing to the mean, the opposite of copying the human's distribution. Correlation is the same story one level up: it forgets location and scale, so a judge whose ratings are exactly twice the human's has correlation one and a different distribution. Only a score defined on the whole predicted distribution checks every outcome's probability, and it takes $K-1$ numbers to describe a distribution on $K$ outcomes; a mean is one of them.

## Worked numbers

- Equal-mean pair, $K=6$ ($p$ against $q$ of R2): mean $5/2$ for both; variances $25/4$ and $1/4$; total variation distance $1$; Brier distance $1$; minimum squared difference over couplings $4$ (the scale gap $(5/2-1/2)^2$; the correlation term is $0$ because any two two-point laws with equal weights differ by an affine map); independent coupling $13/2$.
- Correlation one, different laws ($Y=2X$): means $1$ and $2$; variances $1/2$ and $2$; $\rho=1$; Brier distance $3/8$; total variation $1/2$.
- Equal mean and variance, different shape: minimum squared difference over couplings $1/2$, with $\rho_{\max}=7/8$.
- Free directions for $K=6$ after matching normalization and $j$ moments: $4,3,2,1,0$ for $j=1,\dots,5$.
- Independent draws, $p=\tfrac12\delta_0+\tfrac12\delta_5$: truthful $12.5$; best point mass on the integer grid $6.5$; continuous support $6.25$.

## Checks, figures and Lean

- [`check_squared_difference_vs_distribution.py`](check_squared_difference_vs_distribution.py): ten deterministic checks (exact fractions; one linear program), output in [`results/check_squared_difference_vs_distribution.out`](results/check_squared_difference_vs_distribution.out): (1) the moment expansion R1; (2) zero squared difference forces $X=Y$; (3) the equal-mean pair; (4) free directions after matching moments; (5) correlation one with different laws; (6) the minimum over couplings against a linear program; (7) the fixed-pairing decomposition R4; (8) independent draws R6; (9) excess Brier equals the squared distance; (10) raw, centered and standardized identities.
- [`MeanMatching.lean`](MeanMatching.lean): Lean 4 and Mathlib proofs of the concrete facts (equal means, different laws, Brier distance, independent-draw scores, $\mathrm{cov}^2=\mathrm{var}\,\mathrm{var}$ for $Y=2X$).
- Figures: [`figures/fig1_equal_mean_different_laws.png`](figures/fig1_equal_mean_different_laws.png) (the six-rating pair and the $K=3$ simplex with the mean-1 line coloured by Brier distance) and [`figures/fig2_standardized_and_independent_draws.png`](figures/fig2_standardized_and_independent_draws.png) (the $Y=2X$ pair; expected squared difference between independent draws for seven reports).
- [`HOW_TO_READ_RESULTS.md`](HOW_TO_READ_RESULTS.md): data, metrics, differences, uncertainty and how reports were chosen.

## Reference

- Squared Euclidean distance as excess Brier risk, properness and locality: [`../brier_proper_6437_style_11-26-2026/BRIER_PROPER_6437_STYLE_11-26-2026.md`](../brier_proper_6437_style_11-26-2026/BRIER_PROPER_6437_STYLE_11-26-2026.md); page one's derivation, typeset: [`../../DERIVATION.md`](../../DERIVATION.md).
- The one-dimensional minimum over couplings (monotone rearrangement; Fréchet–Hoeffding comonotone coupling) is standard optimal-transport background; see C. Villani, *Topics in Optimal Transportation* (2003). Check 6 verifies the one-dimensional case numerically against a linear program.
