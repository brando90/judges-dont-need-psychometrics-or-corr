# Notebook transcription · Cross-entropy versus correlation · 11-21-2026

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/18_when_does_ce_beat_correlation_notebook_page_math_and_sweep_start_21Nov2026/conditional_shape_and_rating_budget/assets/handwritten_ce_vs_correlation.md>

Date: 11-21-2026 (transcribed). The main notebook page has no visible date.

**TLDR:** The author asks for the equation behind fitting a judge with Pearson correlation, next to the cross-entropy expression, and for a mathematical or numerical check of what is linear. The page also lists four possible sets of inputs to a judge.

**Source:** [Original notebook photo](handwritten_ce_vs_correlation.jpg), uploaded as `Photo 1.jpg` (attachment filename `1-Photo-1.jpg`), 1,356,551 bytes. SHA-256: `8fe2cf2438e5ad7ddd0988f9b5fdc82a08ebb5c84257d3aa7581a36f82f10af2`.

## Reading the page

Square-bracketed material describes marks or records uncertain readings. Line breaks have been reflowed. Subscripts, superscripts, dots, and arrows are typeset where legible. The transcription retains the order of the arguments, the sign shown, and the author's unanswered question. The page uses CE for cross-entropy and Corr for correlation.

### Upper part, in blue

> Goal: CE vs Corr. Academics will like [crossed out] showing the linear decomposition.
>
> Fit judge with CE vs Corr.

“CE” and “Corr” are circled above the respective choices, with circled “1” and “2” beneath the labels. “CE vs Corr.” is underlined.

> Judges are just distributions, but over ratings/score.
>
> [At right] $p_j(\overset{r}{\cdot}\mid\mathrm{task})$ or $p_j(\overset{s}{\cdot}\mid\mathrm{task})$.
>
> eg. task in FSC/VB is
>
> i) $x,\hat y,y^\star$ or
>
> ii) $x,\hat y$ or
>
> iii) $x$ or
>
> iv) $\hat y,y^\star$

The following note begins to the left of that list and continues underneath it.

> note: $x^\star$ vs $x$. $x$ denotes it does come from “true” distribution, but isn't “perfect”
>
> eg it's a input to Solvers/$P_{[\mathrm{Astra?}]}(\cdot)$, so limited info. eg not all [“pbls”?] to no leak theorems/FS.
>
> [Above “theorems/FS”:] (unproven) ≡ formal specs
>
> → Currently [“do” or “doing”] i but would be nice to test all of them (empirically & theoretically eg verify it with lean or at least simulations!)

“FSC/VB” and “FS” are reproduced as written. The second abbreviation after “Solvers/” and the phrase before “leak” are difficult to distinguish.

### Middle, in blue

$$
p_j(r\mid x,y^\star,\hat y;\theta)
\underset{\mathrm{metric}}{\approx}
p_j^\star(r\mid x,y^\star,y;H\text{ or human etc}).
$$

> [From $\theta$, arrow to:] $\theta$ ≡ changing weights/prompt/harness etc.
>
> [Small notes around the right-hand arguments:] humans [arrow toward $H$]; or personas; (synthetic) could be! [An arrow below “personas” leads to a partially illegible note containing “per” and “h?”].

The right-hand output in this first equation looks like a plain $y$. A dark mark beside the right-hand $j$ is unclear. The subsequent CE expression more clearly includes a hat. Neither has been silently harmonized.

> metric [circled: CE]

$$
\mathrm{CE}\Big(
p_j^\theta(\overset{r}{\cdot}\mid x,y^\star,\hat y;\theta),
p_j^\star(\overset{r}{\cdot}\mid x,y^\star,\hat y;h)
\Big)
$$

> [Above $h$, linked by arrow:] human or persona.

$$
=\sum_{v\in V}\;\longrightarrow\;\sum_{r\in R}
p_j^\star(r\mid x,y^\star,\hat y;h)
\log p_j^\theta(r\mid x,y^\star,\hat y;\theta).
$$

Under the first sum, the author writes $r\in R$, then “eg $r\in\{0,1,\ldots,5$” (no clear closing brace). The first summation index looks like $v\in V$. The horizontal arrow between sums is visible. There is a small overwrite near the final $\theta$. No leading minus sign is visible. The log term has a large underbrace. These are marks on the source, not a corrected loss definition.

### Lower part, in black

> TODO: replace [arrow to the underbraced log term] or [arrow to the full CE expression] with say pearson corr,
>
> Show linearity: ideally clearly & correct in mathematical notation!
>
> [Below “clearly”, arrow to:] mathy/[img?]!
>
> [Below “correct”, arrow to:] lean, sympy, [“numerically” or “numerical”] checked!

Another arrow leads from “linearity” to the final line.

$$
\mathrm{Corr}\Big(
p_j^\theta(\overset{r}{\cdot}\mid x,y^\star,\hat y;\theta),
p_j^\star(\overset{r}{\cdot}\mid x,y^\star,\hat y;h)
\Big) =\;?\quad\text{TODO!}
$$

The framing above is the author's question. In particular, the handwritten Corr expression leaves open what is being correlated across which observations. The [mathematical derivation](../MATH_11-21-2026.md) specifies those choices. A sliver of a different sheet is visible at the top right of the original photograph and does not contain a complete legible passage.
