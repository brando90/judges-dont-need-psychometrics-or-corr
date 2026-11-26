# Transcription of three handwritten derivation pages, round 2 (11-26-2026)

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/assets/TRANSCRIPTION_DERIVATION_PAGES_ROUND2_11-26-2026.md>

Date: 11-26-2026. Three photographs of handwritten pages, supplied to the session at about 13:01 PST with no accompanying text (PNG, 3,024 by 4,032 pixels each). The exact supplied bytes are kept next to this file and were compared byte for byte with the attachments after copying. The 12:03 photographs and their transcription ([`TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md`](TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md)) are left as they were.

| Page | File | Bytes | SHA-256 of supplied bytes |
| --- | --- | --- | --- |
| 1, extended (the 12:03 ruled notebook sheet, now written down to the boxed result and a PS) | [`derivation_page_1_extended_ruled_notebook_round2_11-26-2026.png`](derivation_page_1_extended_ruled_notebook_round2_11-26-2026.png) | 17,832,813 | `45c858b26d3f85bfc837a68a4b7725f9c10b73db7440fe1aa85a0be74e9aea6d` |
| 2, loose-leaf page, re-photographed | [`derivation_page_2_loose_leaf_rephotographed_round2_11-26-2026.png`](derivation_page_2_loose_leaf_rephotographed_round2_11-26-2026.png) | 16,989,921 | `40445339eba5c79d867125d567b3e8a7b8ece3162778c55e2aa69ef03b0232fd` |
| 3, new: "the next step" | [`derivation_page_3_next_step_round2_11-26-2026.png`](derivation_page_3_next_step_round2_11-26-2026.png) | 16,725,992 | `1b6e5624a128e0ce323a1aa13ce8e12fe7de89fdfe94cfa683056ea3085686b0` |

The Google Drive copies of the three photographs are downscaled viewing copies (the Drive connector refuses uploads above about 3 MB per call); the exact bytes are the files above.

## How these relate to the 12:03 photographs

None of the three files has the same bytes as the 12:03 pair. Each of pages 1 and 2 was aligned onto its 12:03 photograph (SIFT keypoints, RANSAC homography; script and result in [`round2_photo_comparison_11-26-2026/`](round2_photo_comparison_11-26-2026/)) and the ink present in only one photograph was marked:

- **Page 2** is the same sheet re-photographed from a different position (3,045 of 3,419 keypoint matches are RANSAC inliers). Every line of writing is in both photographs. The only differences in the diff image are the ruled lines, the paper edge, the punched holes and the pens, which misalign because the sheet is curled and the camera moved. No writing was added or removed. Its content is the 12:03 page 2; the frozen transcription applies unchanged.
- **Page 1** is the same ruled sheet (230 of 521 matches are inliers). The writing from the top of the sheet down to the three `Cov`/`Corr` lines is in both photographs. New in this photograph: the many overlapping ovals drawn over the `Q:` line and over the three `Cov`/`Corr` lines, the short identities in the right margin, the whole lower derivation from `E[(X - Y)^2] = ...` down to the boxed `2(1 - corr(X, Y))`, and the PS. The lower half of the sheet, blank at 12:03, is now written.
- **Page 3** has no 12:03 counterpart.

## Conventions

- `[?]` marks a reading I am not sure of. `[struck: ...]` marks writing with a line through it. `[scribbled out]` marks ink that blots out the writing underneath, so it cannot be read. `[ovals]` marks writing that is covered by many overlapping hand-drawn ovals (the same cancelling gesture as the struck lines at the top); the text under the ovals is still legible unless said otherwise.
- `X~` is X with a tilde above it. **As written on these pages the tilde marks the standardized variable** (the z-score), as at 12:03. The PS on page 1 asks to change this: the tilde should mean the centered variable and a separate Z the standardized one (see below). The transcription records the pages as written; the typeset derivation [`../DERIVATION.md`](../DERIVATION.md) and the new note use the new convention.
- `mu_X*` is the mean with a small raised mark that looks like a star in some places and a degree sign or arrowhead in others.
- The pens, desk, notebook edge and other sheets in the photographs are not page text. On page 3, part of another sheet with a red-ink circled `Q:` and part of another sheet with blue-ink writing are visible at the right edge.

## Page 1 (extended ruled notebook sheet)

**Top part (unchanged from 12:03).** The top line, the three struck-through lines with their ovals, the left-margin `Var[X]` marks, `Z_X = X~ , Y~ = Z_Y` and the `none iid` note are as in the 12:03 transcription. The first two small marks at the left of the top line are two short illegible tokens before `2(1 - xy) = 2 - 2xy = 1 + 1 [scribbled out] - xy - xy`; in this photograph that top line is also crossed by several long strokes.

**Cancelled by ovals (written at 12:03, now covered).** The question and the three equations are still legible under the ovals:

```text
[ovals] Q: why is Brier proper while l2 loss (iid eps) is "gaussian noise"?
[ovals] Cov[X, Y]   = E_{X,Y}[ (X - mu_X*) (Y - mu_Y*) ]
[ovals] Corr(X, Y)  = E_{X,Y}[ ((X - mu_X*) / sigma_X) ((Y - mu_Y*) / sigma_Y) ]
[ovals] Corr(X, Y)  = E_{X,Y}[ Z_X [scribbled out] Z_Y ]
```

Below them, two lines that are also crossed by ovals and wavy strokes (the first is incomplete, the second is the Brier definition):

```text
[ovals] E_{X,Y}[(X - Y)^2] =                                   (nothing follows the equals sign)
[ovals] Brier = D_B(p || q^) = sum_{x in X} (p(x) - q^(x))^2
```

**Right-margin notes, small and clean (new).** These supply the two identities used in the derivation below. Some are cut by the right edge of the sheet:

```text
[short scribbled-out token]
= E[XY] - E[X] E[Y]                 (the last factors are small, overlapped, and cut by the edge)
Cov(X, Y) = E_{X,Y}[ (X - mu_X) (Y - mu_Y) ]
Var[X] = E[X^2] - mu_X^2
       = E[X^2] - E[X]^2           (followed by two marks that look like "X  Y" [?])
(cov(X, Y) + mu ...                  (cut by the edge)
2 ( [scribbled out] )               (and, below it, "- [scribbled out]")
```

Two small annotations with wavy underlines sit above the next block: `sigma_X^2 + mu_X^2` over the first expectation and `sigma_Y^2 + mu_Y^2` over the second (the values of `E[X^2]` and `E[Y^2]`).

**Lower derivation (new).**

```text
E_{X,Y}[(X - Y)^2] = E_{X,Y}[ X^2 + Y^2 - 2XY ] = E_X[X^2] + E_Y[Y^2] - 2 E_{X,Y}[XY]

[scribbled out]      = sigma_X^2 + mu_X^2 + sigma_Y^2 + mu_Y^2 - 2 ( cov(X, Y) + mu_X mu_Y )
```

Boxed:

```text
E_{X,Y}[(X - Y)^2] = sigma_X^2 + sigma_Y^2 - 2 cov(X, Y) + [blot] mu_X^2 + mu_Y^2 - 2 mu_X mu_Y
```

Then the standardized version (tildes as written, meaning standardized; a scribble sits under the `E(X~, Y~)` in the first line):

```text
E_{X,Y}[(X~ - Y~)^2] = E_{X,Y}[ X~^2 + Y~^2 - 2 X~ Y~ ] = 1 + 1 - 2 E(X~, Y~)

   = 1 + 1 - 2 E_{X,Y}[ ((X - mu_X) / sigma_X) ((Y - mu_Y) / sigma_Y) ]
   = 2 - 2 cov / (sigma_X sigma_Y)  =  2 ( 1 - corr(X, Y) )
```

Boxed (large box around the whole line):

```text
E_{X,Y}[(X~ - Y~)^2] = 2 ( 1 - corr(X, Y) )
```

**PS (below the box).**

```text
PS: [scribbled-out word] I wish I would've defined X~ as centered RV. & Z_{X,i}^{(a)} [?]
standardized RV. Let's change it.
```

I read the new symbol as a `Z` with a small subscript `X, i` and a small superscript `(a)`; the subscript and superscript are tiny, so `Z_{X,i}^{(a)}` is uncertain, and what matters is that the standardized variable gets its own symbol `Z`.

## Page 2 (loose-leaf page, re-photographed)

Same writing as the 12:03 page 2; see [`TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md`](TRANSCRIPTION_DERIVATION_PAGES_11-26-2026.md). In this sharper photograph the tildes over `X` in the expansion lines are clear (for example `Var_X[X~] = Var_X[(X - mu_X*)] / sqrt(Var[X~])`), and the last lines read:

```text
Var[X~] = 1                  -> Brier.   D_Brier(p* || q^) = [scribbled out] sum_{x in X} (p*(x) - q^(x))^2
                                                              or  integral_X (p*(x) - q^(x))^2 dx

* imho at this point Brier vs E_{X,Y}[(X - Y)^2] is obviously very different. One is a diff at the
distribution level the other is at the individual values of the RVs.
```

**Erratum to the 12:03 transcription (which is left unchanged).** In the closing note the expectation is `E_{X,Y}[(X - Y)^2]` with plain `X` and `Y`: there is no tilde over either letter in this photograph or in the 12:03 photograph (compare the clearly tilded `Var[X~] = 1` a few lines above it). The 12:03 transcription wrote `E_{X,Y}[(X~ - Y~)^2]` there, which is a misreading. So the note compares Brier with the squared difference of the raw random variables, which is the same object that page 3 analyzes, and not with the squared difference of the standardized variables. Sentences in earlier summaries that describe the note as "Brier versus the squared error of the z-scores" are corrected in the updated plan and summary.

## Page 3 (new, loose-leaf page with blue rules, red margin and a punched hole at the left)

```text
btw Now the next step would be to show optimizing diff.
RVs doesn't yield the "optimal" pdf/pmf / distribution matching.
Consider any pair of RVs with pdfs p_X(.), p_Y(.).

So if E_{X,Y}[(X - Y)^2] = 0 (circled mark A)
according to our derivations we have:   V_X + V_Y

(circled mark B)  E_{X,Y}[(X - Y)^2] = mu_X^2 + mu_Y^2 - 2 mu_X mu_Y + sigma_X^2 + sigma_Y^2 - 2 cov(X, Y)
                                                                   ^ an arrow starts here, under the "+" before sigma_X^2

if X = Y  ==>  yes [scribbled out] squared diff of X, Y
is zero too.

But if we have (circled mark A) E[x - y] = 0, we have: [the arrow ends here] is zero, and we have
6 degrees ["variables"] of freedom, but only 1 equation, so it has an
infinite set of pdfs that satisfy it
(and even more! cuz (circled mark B) is ignoring all
higher order moments, so (circled mark B) can never do
distribution matching.
```

Readings that need a caveat:

1. **The circled marks.** There are two kinds. Mark A (a small triangle or arrowhead inside the circle) follows `E[(X - Y)^2] = 0` and precedes `E[x - y] = 0`. Mark B (a dot or short tick inside the circle, probably a small 1) sits to the left of the expansion equation and before "is ignoring all higher order moments" and before "can never do distribution matching". I cannot tell whether mark A is a small 2, a delta or an arrowhead, so I do not assign numbers to the circles; the note [`../extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/`](../extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md) names the objectives in words (mean difference, squared difference, standardized squared difference) and its conclusions do not depend on which circle is which.
2. **`V_X + V_Y`.** Written small to the right of "we have:", above the variance terms of the next equation: I read it as naming `sigma_X^2 + sigma_Y^2` as the two variances.
3. **The arrow.** It starts at the `we have:` of the last paragraph and ends under the `+` between `- 2 mu_X mu_Y` and `+ sigma_X^2` in the equation above. I read it as pointing at the mean terms `mu_X^2 + mu_Y^2 - 2 mu_X mu_Y`, which equal `(mu_X - mu_Y)^2` and are zero when `E[x - y] = 0`; the tip lands at the boundary between the mean terms and the variance terms, so the variance terms are the other possible target.
4. **`E[x - y]` is written with lowercase `x` and `y`** in the last paragraph and I read them as the random variables `X` and `Y`.
5. **`6 degrees ["variables"] of freedom`.** The word "variables" is written small, in quotation marks, above the line between "degrees" and "of".
6. **Line ends.** The first two lines run to the right edge of the sheet and end `optimizing diff.` and `distribution matching` (the last word is written `matchg` with a dot; I read "matching").

## What was checked on these pages

The identities on page 1 (the three-term expansion, `E[X^2] = sigma_X^2 + mu_X^2`, `E[XY] = cov + mu_X mu_Y`, and `E[(X~ - Y~)^2] = 2(1 - corr)` with the tildes as standardization) are correct as written. The one slip on these pages is on page 2, whose first expansion line writes the cross term as `- X~ Y~`; the next line has the correct `- 2 E[X~ Y~]`, and page 1's new derivation has the factor two throughout.

The page 3 argument is worked out, with counterexamples, the degrees-of-freedom count, the minimum over couplings and deterministic checks, in [`../extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md`](../extended_checks_11-26-2026/squared_difference_vs_distribution_11-26-2026/SQUARED_DIFFERENCE_VS_DISTRIBUTION_11-26-2026.md). In short: after normalization and the mean equation there are four free directions for six outcomes, not six; the claim "can never do distribution matching" holds for the mean difference, for correlation (which sees the laws only up to a positive affine map) and for the squared difference when the data fix the pairing of the two variables; and if the pairing is free, the smallest squared difference over all pairings is zero only when the two laws are equal.
