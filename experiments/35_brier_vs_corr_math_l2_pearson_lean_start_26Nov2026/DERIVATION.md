# Derivation: standardized squared error, Pearson correlation, and excess Brier risk

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/DERIVATION.md>

**TLDR:** After centering and scaling, expected squared error equals $2(1-\mathrm{Corr}(X,Y))$. Restoring the original units adds a squared bias and a squared scale gap. Excess Brier risk is the squared distance between two distributions; scalar squared error and Pearson correlation compare scalar random variables, and those two comparisons agree, up to a factor of 2, when the outcome is binary.

This note records the population algebra from Brando's notebook derivation (11-26-2026). The matching formal statements are finite-sum identities in `BrierCorr.lean`. The numerical checks are in `results.txt`. Definitions of each printed column are in `HOW_TO_READ_RESULTS.md`.

## Notation (changed 11-26-2026)

Following the PS on notebook page 1 (as I read the handwriting: "I wish I would've defined X~ as centered RV. & Z_{X,i}^{(a)} [reading uncertain] standardized RV. Let's change it."), a tilde now marks the **centered** variable, $\tilde X=X-\mu_X$, and $Z_X=\tilde X/\sigma_X$ is the **standardized** variable. The handwritten pages, their transcriptions, `results.txt` and `experiments.py` were written with the tilde for the standardized variable and are left as they were; wherever they print `X~` for a standardized variable, read $Z_X$ here.

## 1. Standardized variables

Let $X$ and $Y$ be real random variables with means $\mu_X=\mathbb{E}[X]$ and $\mu_Y=\mathbb{E}[Y]$ and with positive variances $\mathrm{Var}(X)=\sigma_X^2>0$ and $\mathrm{Var}(Y)=\sigma_Y^2>0$. Write $\sigma_X=\sqrt{\mathrm{Var}(X)}$ and $\sigma_Y=\sqrt{\mathrm{Var}(Y)}$ for the positive square roots. The centered variables $\tilde X,\tilde Y$ and the standardized variables $Z_X,Z_Y$ are

$$
\tilde X=X-\mu_X,\quad \tilde Y=Y-\mu_Y,\qquad Z_X=\frac{\tilde X}{\sigma_X}=\frac{X-\mu_X}{\sigma_X},\quad Z_Y=\frac{\tilde Y}{\sigma_Y}=\frac{Y-\mu_Y}{\sigma_Y}.
$$

Linearity of expectation gives

$$
\mathbb{E}[Z_X]=\frac{1}{\sigma_X}\bigl(\mathbb{E}[X]-\mu_X\bigr)=0,
$$

and the same calculation gives $\mathbb{E}[Z_Y]=0$.

The variance is then the second moment. The law of the unconscious statistician (LOTUS) applies the function $g(x)=((x-\mu_X)/\sigma_X)^2$ under the law of $X$:

$$
\mathrm{Var}(Z_X)=\mathbb{E}[Z_X^2]=\frac{\mathbb{E}[(X-\mu_X)^2]}{\sigma_X^2}=\frac{\mathrm{Var}(X)}{\sigma_X^2}=\frac{\mathrm{Var}(X)}{\mathrm{Var}(X)}=1.
$$

The same identity gives $\mathbb{E}[Z_Y^2]=1$.

Pearson correlation is the remaining cross moment:

$$
\mathrm{Corr}(X,Y)=\frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y}=\mathbb{E}[Z_XZ_Y].
$$

## 2. Standardized squared error

Expand the square and use the two unit second moments:

$$
\begin{aligned}
\mathbb{E}[(Z_X-Z_Y)^2]
&=\mathbb{E}[Z_X^2]+\mathbb{E}[Z_Y^2]-2\mathbb{E}[Z_XZ_Y]\\
&=1+1-2\mathrm{Corr}(X,Y)\\
&=2\bigl(1-\mathrm{Corr}(X,Y)\bigr).
\end{aligned}
$$

Standardization has already removed location and positive scale. A map $X\mapsto aX+b$ with $a>0$ therefore leaves both $Z_X$ and the correlation unchanged, and it leaves $\mathbb{E}[(Z_X-Z_Y)^2]$ unchanged. In `BrierCorr.lean` this invariance is `pearson_affine_invariant`: for $a>0$, the Pearson correlation of $ax+b$ with $y$ equals the Pearson correlation of $x$ with $y$.

The finite-sum form proved by `standardized_mse_corr` assumes $n\neq 0$ and $\sum_{i=1}^n x_i^2=\sum_{i=1}^n y_i^2=n$, and concludes

$$
\frac1n\sum_{i=1}^n(x_i-y_i)^2=2\left(1-\frac1n\sum_{i=1}^n x_i y_i\right).
$$

If those vectors are centered as well as scaled, $(1/n)\sum_i x_i y_i$ is the Pearson correlation of the original sample, with the $1/n$ second-moment convention used throughout `BrierCorr.lean`.

## 3. Unstandardized mean squared error

Write $X-Y=(X-\mu_X)-(Y-\mu_Y)+(\mu_X-\mu_Y)$. The centered piece has mean zero, so the cross term drops and

$$
\mathbb{E}[(X-Y)^2]=(\mu_X-\mu_Y)^2+\mathbb{E}\bigl[((X-\mu_X)-(Y-\mu_Y))^2\bigr].
$$

In the centered variables this reads $\mathbb{E}[(X-Y)^2]=(\mu_X-\mu_Y)^2+\mathbb{E}[(\tilde X-\tilde Y)^2]$.

Expand the second term:

$$
\begin{aligned}
\mathbb{E}\bigl[((X-\mu_X)-(Y-\mu_Y))^2\bigr]
&=\sigma_X^2+\sigma_Y^2-2\sigma_X\sigma_Y\mathrm{Corr}(X,Y)\\
&=(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y\bigl(1-\mathrm{Corr}(X,Y)\bigr).
\end{aligned}
$$

Therefore

$$
\mathbb{E}[(X-Y)^2]=(\mu_X-\mu_Y)^2+(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y\bigl(1-\mathrm{Corr}(X,Y)\bigr).
$$

The first term is squared bias. The second is the squared gap of scales. The third is the standardized discrepancy of Section 2, restored to the original product of standard deviations. In `BrierCorr.lean`, `mse_decomposition` is this identity for length-$n$ rational vectors: $s_x$ and $s_y$ are any positive rationals whose squares equal the $1/n$ variances, and the correlation is the $1/n$ covariance divided by $s_x s_y$.

## 4. Distributional Brier risk and scalar comparison

Let $p^{\ast}$ be a probability distribution on a finite set and let $\hat q$ be a forecast of the same coordinates, with $\sum_y p^{\ast}(y)=1$. The Brier score of a forecast $q$ on a realized class is the squared Euclidean distance from $q$ to the corresponding one-hot vector. Its excess over the Bayes score depends on the forecast only through the probability vectors:

$$
D_{\mathrm{Brier}}(p^{\ast}\|\hat q)
=\mathbb{E}_{p^{\ast}}[\mathrm{BS}(\hat q)]-\mathbb{E}_{p^{\ast}}[\mathrm{BS}(p^{\ast})]
=\sum_y\bigl(p^{\ast}(y)-\hat q(y)\bigr)^2.
$$

The same squared distance is $\int (p^{\ast}(y)-\hat q(y))^2\,dy$ when the objects being compared are densities and the integral exists. Either form scores the forecast at the level of the whole distribution.

$\mathbb{E}[(X-Y)^2]$ and $\mathbb{E}[(Z_X-Z_Y)^2]$ score pairs of scalar values. Pearson correlation is the scalar cross moment that remains after those values are centered and scaled. Excess Brier risk can be positive for two distributional forecasts that share a mean, because the sum runs over every coordinate of $p^{\ast}-\hat q$.

The binary outcome is the case in which the scalar and the distribution describe the same object. For $y\in\lbrace0,1\rbrace$ and a scalar forecast $q$, `binary_brier_eq_two_l2` is the pointwise identity

$$
(q-y)^2+\bigl((1-q)-(1-y)\bigr)^2=2(q-y)^2.
$$

The left-hand side is the two-class Brier score of the probability vector $(q,1-q)$. The right-hand side is twice the scalar squared error. Here the scalar $q$ is the full distribution of a binary rating. On a larger ordinal alphabet, a scalar forecast specifies a mean, and the Brier score still requires a distribution on every class.

`brier_decomposition` is the finite-sum statement of the display above: if the coordinates of $p$ sum to 1, then the excess expected Brier score of $q$ equals $\sum_j(q_j-p_j)^2$.

## 5. Theorem index

| Notebook step | Theorem in `BrierCorr.lean` |
| --- | --- |
| $\mathbb{E}[(Z_X-Z_Y)^2]=2(1-\mathrm{Corr}(X,Y))$, under sum of squares equal to $n$ | `standardized_mse_corr` |
| $\mathbb{E}[(X-Y)^2]=(\mu_X-\mu_Y)^2+(\sigma_X-\sigma_Y)^2+2\sigma_X\sigma_Y(1-\mathrm{Corr}(X,Y))$ | `mse_decomposition` |
| Correlation unchanged by $x\mapsto ax+b$ for $a>0$ | `pearson_affine_invariant` |
| Excess Brier risk equals $\sum_y(p^{\ast}(y)-\hat q(y))^2$ | `brier_decomposition` |
| Two-class Brier score equals twice scalar squared error | `binary_brier_eq_two_l2` |

`BrierCorr.lean` is Lean 4.36.0, with no Mathlib and no `sorry`. The check in `lean_check.txt` reports only `propext`, `Classical.choice`, and `Quot.sound` for these theorems. The development uses rational finite sums. It does not formalize a general integral, a square-root function on the rationals, or LOTUS as a measure-theoretic lemma. Standard deviations enter as positive rationals whose squares equal the variances.
