# Transcription of two handwritten derivation pages (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/assets/TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md>

Date: 11-26-2026. Two photographs of handwritten pages, supplied to the session at about 12:03 PST (PNG, 3,024 by 4,032 pixels each). The exact supplied bytes are kept next to this file and were compared byte for byte with the attachments after copying.

| Page | File | Bytes | SHA-256 of supplied bytes |
| --- | --- | --- | --- |
| 1, ruled notebook page | [`derivation_page_1_ruled_notebook_11-26-2026.png`](derivation_page_1_ruled_notebook_11-26-2026.png) | 16,500,022 | `d4f80d8cd2f38328375b49e2e97a3fc75cb07f1d875e3f79c37b8210aa3f353a` |
| 2, loose-leaf page | [`derivation_page_2_loose_leaf_11-26-2026.png`](derivation_page_2_loose_leaf_11-26-2026.png) | 17,531,373 | `70eee48439006389419c81972d8409a13ed9f86501cff79ac3456cb9b09a2fdf` |

The Google Drive copy of the two photographs is pending: the Drive connector refuses uploads above about 3 MB per call. A typeset version of the derivation is [`../DERIVATION.md`](../DERIVATION.md); its Lean counterpart is `standardized_mse_corr` in [`../BrierCorr.lean`](../BrierCorr.lean) and, over Mathlib, `standardized_mse` in [`../extended_checks_11-26-2026/lean/ProperScoring/Basic.lean`](../extended_checks_11-26-2026/lean/ProperScoring/Basic.lean).

## Conventions

- `[?]` marks a reading I am not sure of. `[struck: ...]` marks writing with a line through it. `[scribbled out]` marks ink that blots out the writing underneath, so it cannot be read.
- `X~` is X with a tilde above it. The pages set `Z_X = X~` and `Y~ = Z_Y`, so a tilde means a standardized variable (the z-score).
- `mu_X*` is the mean with a small raised mark that looks like a star in some places and a degree sign in others; I could not tell which.
- Subscripts on `E` and `Var` are written as they appear; the tildes in the subscripts on page 2 are hard to see.
- The pen, desk, laptop edge, cable and second pen in the photographs are not page text. Page 1 is mostly blank below the last equation.

## Page 1 (ruled notebook page)

Top line, with two scribbled-out marks at the left margin before it:

```text
2(1 - xy) = 2 - 2xy = 1 + 1 [scribbled-out symbol] - xy - xy
```

Below it, three struck-through lines. Parts are covered by a mass of overlapping scribbled ovals and cannot be read:

```text
[struck: Putnam Axiom: -> [scribbled out] ... hard at college level, to [?]]
[struck: [mass of overlapping ovals over two lines of writing, not legible] -> MATH, GSM8K [?]]
[struck: -> functional variations show that the models do not generalize even outside variables (except that at [?] the individual question).]
```

Left margin and right side, around the same height as the struck lines (their exact pairing is not clear):

```text
Var[X]            (left margin, with a scribbled-out mark before it)
[struck: Var[X~]] (left margin, scribbled out)

Right side: three fractions with denominators sigma_X, sigma_Y and sigma_X sigma_Y:
   [sigma_X [?]] / sigma_X   +   [sigma_Y [?]] / sigma_Y   -   2 cor(X, Y) [?] / (sigma_X sigma_Y)
   (numerators partly overwritten; exponents unclear)

Z_X = X~ ,   Y~ = Z_Y                         none iid.
```

The question, written across two lines with a small `eps` above `(iid)`:

```text
Q: why is Brier proper while l2 loss (iid eps) is "gaussian
noise"?
```

(A stray mark that looks like a `13` sits above "is".) Then the three equations that fill the rest of the writing, with a curly brace under the first one pointing down to the next line:

```text
Cov[X, Y] = E_{X,Y}[ (X - mu_X*) (Y - mu_Y*) ]

Corr(X, Y) = E_{X,Y}[ ((X - mu_X*) / sigma_X)  ((Y - mu_Y*) / sigma_Y) ]

Corr(X, Y) = E_{X,Y}[ Z_X [scribbled-out symbol] Z_Y ]
```

## Page 2 (loose-leaf page, blue rules and red margin)

```text
(a + b)^2 = a^2 + 2ab + b^2  ==  1^2 + 2ab + [scribbled-out symbol] 1^2
(a - b)^2 = a^2 [scribbled-out symbol] - 2ab + b^2  ==  1 - 2ab + 1^2

E_{X,Y}[(X~ - Y~)^2] = E_{X,Y}[ X~^2 - X~ Y~ + Y~^2 ]
                     = E_X[X~^2] + E_Y[Y~^2] - 2 E_{X,Y}[X~ Y~]

Z_X = X~ = (X - mu_X*) / sigma_X          E[X~] = E_X[(X - mu_X*)] / sqrt( E_X[(X - mu_X*)^2] )

Var_X[X~] = Var_X[(X - mu_X*)] / sqrt( Var[X~] )  =  [scribbled out]
```

The next line has two small annotations written above it, `x^2 - 2 mu* x + mu*^2` above the first expectation and `x^2 - mu^2` above the second, and a short remark to its right:

```text
[scribbled-out mark] Var[X] = E_X[(X - mu_X*)^2] = E[X^2] - E[X]^2
                                       (1/sigma_X) (E[X] - mu*) = 0 .
```

```text
E_X[X~] = E_X[ (X - mu_X*) / sigma_X ] = (1/sigma_X) E[(X - mu_X*)]
   |_> centered, so it's mean is zero.   ~~> 0

Var_X~[X~] = E_X~[ [scribbled out] (X~ - mu_X~*)^2 ] = E_X~[X~^2]
           = E_X[ (X - mu_X*)^2 / sigma_X^2 ]  =  E_X[ ((X - mu_X*)^2) / sigma_X^2 ]
```

Two notes sit beside and under that last line: `& 1/sigma_X == constant.` at the right, and, under an upward arrow, `more details, but centering doesn't really chg pmf/pdf of a 1-1 invertible op RV.` (I read "chg" as "change").

```text
          = E_X[(X - mu_X)^2] / sigma_X^2  =  Var[X] / Var[X]  =  1 .

Var[X~] = 1                  -> Brier.   D_Brier(p* || q^) = [scribbled-out symbol] sum_{x in X} (p*(x) - q^(x))^2
                                                              or  integral_X (p*(x) - q^(x))^2 dx
```

An arrow runs from the word "Brier" in the last line of the page up to the "-> Brier." above it. The last lines, starting with a star in the margin:

```text
* imho at this point Brier vs E_{X,Y}[(X~ - Y~)^2] is obviously very
different. One is a diff at the distribution level the other is at
the individual values of the RVs.
```

The last line, "the individual values of the RVs.", is written smaller, below the line above it.
