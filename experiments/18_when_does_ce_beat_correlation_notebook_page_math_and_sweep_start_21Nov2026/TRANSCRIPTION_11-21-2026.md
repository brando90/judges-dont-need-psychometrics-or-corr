# Transcription 11-21-2026: Brando's notebook page (CE vs Corr) and Sanmi's Discord message of 11-16-2026 5:15 PM

**Date of this note:** 11-21-2026. **Transcribed by:** ultimate_brando9 (Brando's AI assistant), model Claude Sonnet 5.5 in the Claude Code harness, reading the three photos below. Brando to check the marked unclear spots.

Originals (unchanged bytes, as Brando sent them on 11-21-2026 11:20 AM):
- [images/2026-11-21_notebook_page_IMG_0008.jpg](images/2026-11-21_notebook_page_IMG_0008.jpg) (the handwritten page, from `IMG_0008.jpg`)
- [images/2026-11-21_sanmi_discord_2026-11-16_part1_IMG_0010.png](images/2026-11-21_sanmi_discord_2026-11-16_part1_IMG_0010.png) (Discord screenshot, from `IMG_0010.png`: the message start, points 1 to 3)
- [images/2026-11-21_sanmi_discord_2026-11-16_part2_IMG_0009.png](images/2026-11-21_sanmi_discord_2026-11-16_part2_IMG_0009.png) (Discord screenshot, from `IMG_0009.png`: the end of point 3, then points 4 and 5 and the framing paragraphs)

---

## 1. The handwritten page (word for word, top to bottom)

Conventions: `[unclear: ...]` = my best reading is a guess; `[struck]` = crossed out on the page; `(circled)` = circled on the page. Math is first as written (plain), then in LaTeX. A hat over a letter is written `y^` in plain text.

**Line 1-2:** "Goal: CE vs Corr. Academics will like [struck: one crossed-out word, unreadable] showing the linear decomposition."

**Under it:** (circled) "CE" with (circled) "1" below it, and (circled) "Corr" with (circled) "2" below it.

**Line 3:** "Fit judge with CE vs Corr." (the words "CE vs Corr." are underlined, with an arrow).

**Right margin, next to lines 3-4:** "p_j( . ^r | task )" "or" "p_j( . ^s | task )". (a small `r` or `s` sits above the dot: the judge's distribution over ratings r, or over scores s.)

**Line 4:** "Judges are just distribution, but over ratings/score." ("ratings" and "score" underlined)

**Line 5:** "e.g. task in FSC/VB is  i) x, y^, y*   or   ii) x, y^   or   iii) x   or"  (FSC = formal specification/code task; VB = VeriBench, my reading)

**Lines 6-7:** "note: x* vs x. x denotes it does come from "true" distribution, but isn't "perfect" iv) y^, y*" then "e.g. it's a input to Solver/P_[unclear: Autoval or Lean](.), so limited [unclear: hmb / ability]. y not all PBs [unclear] no lack theorems/FS." Above "theorems/FS", a bracket labelled "(unproven) = formal specs". [unclear: the end of this line; my reading is roughly "so not all problems [have] no lack [of] theorems / formal specs"]

**Lines 8-9:** "-> currently doing i but would be nice to test all of them (empirically & theoretically e.g. verify it with Lean or at least simulations!)" with a note to the right: "{synthetic could be!}".

**Main relation (line 10), plain:**
`p_j(r | x, y*, y^; theta)  ~approx (metric)~  p*_j(r | x, y*, y^; H or human etc)`
with, above the right side: "human" (pointing to `H`), "or personas", and below right an arrow to "? ... per h ?" [unclear].
LaTeX: $p_j(r \mid x, y^\ast, \hat y;\, \theta) \;\overset{\text{metric}}{\approx}\; p^\ast_j(r \mid x, y^\ast, \hat y;\, h)$ where $h$ is a human (or persona).

**Line 11:** "theta == change weights/prompt/harness etc."  ($\theta$ = changing weights, prompt, harness, etc.)

**Metric CE (circled "CE" after the word "metric"), plain:**
`CE( p_j^theta( . ^r | x, y*, y^; theta), p*_j( . ^r | x, y*, y^; h) )`
`= Sum_{v in V}  ->  Sum_{r in R}  p*_j(r | x, y*, y^; h)  log p_j^theta(r | x, y*, y^; theta)`
Under the sums: "v in V" and, wavy underline, "r in R", then "e.g. r in {0, 1, ..., 5}". A brace under `p*_j(r|...) log p_j^theta(r|...)`; from the brace, arrows go back to the left side `CE(` and to the right end of the CE line (the whole expression).
Beside the CE line: "human or persona" with an arrow to `h`.
LaTeX (as written, no minus sign on the page; the standard CE has a leading minus):
$$\mathrm{CE}\big(p^\theta_j(\cdot\mid x,y^\ast,\hat y;\theta),\,p^\ast_j(\cdot\mid x,y^\ast,\hat y;h)\big)=\sum_{v\in V}\ \sum_{r\in R} p^\ast_j(r\mid x,y^\ast,\hat y;h)\,\log p^\theta_j(r\mid x,y^\ast,\hat y;\theta),\quad R=\{0,1,\dots,5\}$$
(The outer $\sum_{v\in V}$ is not defined on the page; my reading is a sum over the items/tasks $v$ in a set $V$. [unclear])

**Bottom block:**
"TODO: replace [arrow up to the CE term]  or  [arrow up to the right end of the CE line] with say Pearson corr,"
"Show linearity: ideally [unclear: clearly / checkly] & [underlined: correct] in mathematical notation!"
"-> Lean, sympy, numerically checked!"
An arrow from "[unclear: math/mag!]" under "clearly" points down to the last line, and another arrow from "linearity" points to it.

**Last line, plain:** `Corr( p_j^theta( . | x, y*, y^; theta), p*_j( . | x, y*, y^; h) ) = ?  TODO!`
LaTeX: $\mathrm{Corr}\big(p^\theta_j(\cdot\mid x,y^\ast,\hat y;\theta),\,p^\ast_j(\cdot\mid x,y^\ast,\hat y;h)\big)=\;?\quad\text{TODO!}$

Spots I could not read with confidence: the struck-out word in line 1; "P_[Autoval/Lean]" and the end of the note on lines 6-7; "? ... per h ?"; "clearly/checkly" and the small word under it; the meaning of the outer sum over $v \in V$.

---

## 2. Sanmi's Discord message, 11/16/2026 5:15 PM (word for word, from the two screenshots)

sanmi, 11/16/2026 5:15 PM:

> Read the brief. The identification point is right and cleanly shown (correlation can't separate distributions that share means). My worry is that the bottom line stretches this into a general "train and select by loss" claim the evidence doesn't yet support.
>
> 1. Human data. The frontier mixture comparison is null. The exp 14 open-pool win compares two judges that both lose to the constant forecast (0.277 and 0.327 vs 0.184). The best-single-judge rows are closest to how people actually use correlation, and there correlation selection does better: −0.025 in exp 12 (CI just touches 0) and −0.032 in the 44-judge pool (CI excludes 0). The CE-selected single judges (0.205, 0.211) are worse than the constant, probably because the 1e-6 floor drives CE selection on three-sample histograms. That result belongs in the bottom line. Can you add an RPS-selected arm on the human ratings? It's the loss you're arguing for, and it's cheap to run.
>
> 2. Toy baselines. A Pearson-trained softmax only pins the mean, so its distribution is bound to lose on RPS; it ends up worse than the constant in all four scenarios. The fair comparison is score-then-calibrate, and at large n that gap is a few percent of the irreducible RPS. The head is also worse than the constant histogram at small n even though the constant is nested in it (b = 0), so it probably needs regularization. I'd add an affine-calibrated or MSE-trained baseline, and report RPS against the constant and the truth instead of as ratios of excess.
>
> 3. Small n. Fitting 78 weights to a scale-free objective on 16 tasks will overfit, especially with one ridge shared across objectives and Pearson fits that never converge. That supports "don't fit flexible models by maximizing correlation," which is a much smaller claim.

(second screenshot; the same message continues, shown as a reply to the first part)

> 4. Use. Distributional accuracy is the right target when the judge's output gets read as a rating or a distribution. For ranking uses, rank agreement is what matters. It's also worth asking whether disagreement on these artifacts is item signal or rater severity. With 2–3 raters we can't tell, and if it's mostly rater error then the true score (expected rating) is the target.
>
> 5. References: Appendix D is Lin's concordance correlation and the ICC consistency vs agreement split, and ordinal likelihood models with rater facets are proper-loss judges. This is a psychometric argument, so I'd drop "don't need psychometrics" from the framing.
>
> On framing, your narrower version seems close to me. Something like: correlation measures association with mean ratings and can't certify agreement in level and scale, let alone spread; when a judge's output is read as a rating or distribution, select and report with a proper scoring rule like RPS and keep correlation as a ranking diagnostic; on the historical ratings the two rules aren't distinguishable yet.
>
> On prompts vs weights: prompts and harness. That's where people tune against agreement in practice, and it's coarse selection, the regime where the single-judge rows already lean toward correlation. Let's run it after the workshop deadline[cut off at the bottom edge of the screenshot; the message may continue]

Notes on the screenshots: both show the Discord DM with "sanmi", phone clock 11:17. The first screenshot ends in the middle of point 3; the second starts with the end of point 3 (the same two lines) and then the "4." block, which Discord shows as a continuation (reply link) of the 5:15 PM message. The last visible words are "Let's run it after the workshop deadline" with the rest cut off.
