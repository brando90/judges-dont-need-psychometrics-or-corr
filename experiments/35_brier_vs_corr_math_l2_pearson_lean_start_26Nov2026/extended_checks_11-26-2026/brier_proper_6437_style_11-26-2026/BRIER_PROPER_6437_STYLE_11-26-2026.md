# The Brier score is proper, and local only for two outcomes: the 6.437 argument redone for Brier

**Doc link:** <https://github.com/brando90/judges-dont-need-psychometrics-or-corr/blob/main/experiments/35_brier_vs_corr_math_l2_pearson_lean_start_26Nov2026/extended_checks_11-26-2026/brier_proper_6437_style_11-26-2026/BRIER_PROPER_6437_STYLE_11-26-2026.md>

**TLDR:** Redoing the log-loss properness proof from the MIT 6.437 notes (Inference and Information) with the Brier cost gives the exact identity $\mathbb{E}[\mathrm{Brier}\mid y]=\lVert q-p\rVert^2+(1-\lVert p\rVert^2)$. The Brier score is therefore strictly proper, and the price of a dishonest report is exactly its squared Euclidean distance from the truth. The Brier score is local only when there are two outcomes; for three or more it is not, as Theorem 1 of the notes requires. The ranked probability score (RPS) inherits both facts. All checks pass (8 of 8), and the Lean 4 file compiles with no `sorry`.

**Where this sits.** The identity itself is already claim C1 of [`../DERIVATIONS.md`](../DERIVATIONS.md) and Lean theorem `brier_excess` in [`../lean/ProperScoring/Basic.lean`](../lean/ProperScoring/Basic.lean). RPS properness is C3 there (`rps_strictly_proper`), and the distribution-level reading is Section 4 of [`../../DERIVATION.md`](../../DERIVATION.md). This note adds four things: the proof written step for step in the style of the 6.437 notes; the affine family $A\cdot\mathrm{Brier}+B(x)$; the locality question; and negative controls showing that local costs other than log-loss are not proper.

**Notation.** As in the parent notes, $x\in\{0,\dots,K-1\}$ is a rating ($K=6$ in this project, $L=K$ in the 6.437 notation), $y$ is the item being rated, $p=p_{x|y}(\cdot\mid y)$ is the true rating distribution of that item, and $q$ is the distribution a judge reports. $\lVert v\rVert^2=\sum_a v(a)^2$. All expectations are over $x\sim p$ with $y$ held fixed.

## 1. Statement

**Definition 1 (6.437).** A cost $C(\cdot,\cdot)$ is *proper* if, for every $y$,
$p_{x|y}(\cdot\mid y)=\arg\min_{q\ge 0,\ \sum_a q(a)=1}\ \mathbb{E}[C(x,q)\mid y=y]$.
It is *strictly* proper when this minimiser is unique.

**Claim 1′.** The Brier cost
$$C(x,q)=A\sum_{a}\bigl(q(a)-\mathbb{1}\{x=a\}\bigr)^2+B(x),\qquad A>0,\ B\ \text{arbitrary},$$
is strictly proper.

## 2. Proof, in the style of (5) and (6)

Two facts replace Gibbs' inequality. Because the expectation is taken with respect to $p_{x|y}$, $\mathbb{E}[\mathbb{1}\{x=a\}\mid y=y]=p(a)$. And an indicator squared is itself, $\mathbb{1}\{x=a\}^2=\mathbb{1}\{x=a\}$. Then

$$
\begin{aligned}
\mathbb{E}[C(x,q)\mid y=y]
&= A\sum_a \Bigl(q(a)^2-2q(a)\,\mathbb{E}[\mathbb{1}\{x=a\}\mid y]+\mathbb{E}[\mathbb{1}\{x=a\}^2\mid y]\Bigr)+\mathbb{E}[B(x)\mid y] && (5')\\
&= A\sum_a \bigl(q(a)^2-2q(a)p(a)+p(a)\bigr)+\mathbb{E}[B(x)\mid y] \\
&= A\sum_a \bigl(q(a)-p(a)\bigr)^2 + A\Bigl(1-\sum_a p(a)^2\Bigr)+\mathbb{E}[B(x)\mid y] && (6')\\
&\ge A\Bigl(1-\sum_a p(a)^2\Bigr)+\mathbb{E}[B(x)\mid y]. && (7')
\end{aligned}
$$

Line (6′) completes the square: $q^2-2qp+p=(q-p)^2+p-p^2$, and $\sum_a p(a)=1$. Line (7′) holds because a sum of squares is nonnegative and $A>0$. Equality holds in (7′) if and only if $q(\cdot)$ is identically $p_{x|y}(\cdot\mid y)$, and the right side of (7′) does not depend on $q$. So $p$ is the unique minimiser, and the Brier cost is strictly proper. This is the same conclusion as Claim 1: minimising the expected Brier cost returns exactly the posterior.

With $A=1$ and $B=0$ the proof gives the exact identity

$$\mathbb{E}[\mathrm{Brier}(x,q)\mid y]=\lVert q-p\rVert^2+\bigl(1-\lVert p\rVert^2\bigr). \tag{8'}$$

The second term is the best attainable expected cost, the Brier analogue of the entropy. The excess of any report over that floor is exactly the squared Euclidean distance $\lVert q-p\rVert^2$. For log-loss, the excess over the entropy is $\mathrm{KL}(p\,\Vert\,q)$, and Gibbs' inequality is the statement that it is nonnegative.

**The same proof as bias plus variance.** For any number $f$ and random variable $Z$, $\mathbb{E}(f-Z)^2=(f-\mathbb{E}Z)^2+\operatorname{Var}Z$. Apply it to each indicator $Z_a=\mathbb{1}\{x=a\}$ with $f=q(a)$. Then $\mathbb{E}Z_a=p(a)$ and $\operatorname{Var}Z_a=p(a)(1-p(a))$, and summing over $a$ gives (8′), since $\sum_a p(a)(1-p(a))=1-\lVert p\rVert^2$. The Brier score is squared error on each one-hot coordinate, and squared error is minimised by the mean, which for an indicator is a probability.

**Empirical version.** Replace $p$ by the frequency vector $f$ of $N$ observed ratings $x_1,\dots,x_N$ of the item. Since $f$ is a probability vector, (8′) with $p\to f$ reads $\tfrac1N\sum_i\mathrm{Brier}(x_i,q)=\lVert q-f\rVert^2+1-\lVert f\rVert^2$, so the empirical minimiser is $f$ itself. Its error is $\mathbb{E}\lVert f-p\rVert^2=\sum_a\operatorname{Var}f(a)=\sum_a p(a)(1-p(a))/N=(1-\lVert p\rVert^2)/N$, so the distance shrinks like $1/\sqrt N$.

## 3. Locality: proper, but local only for two outcomes

**Definition 2 (6.437).** $C$ is *local* if $C(x,q)=\varphi(x,q(x))$ for some $\varphi$: only the probability assigned to the outcome that occurred counts.

Expanding the square splits the Brier cost into a part about the outcome and a part about everything else:
$$\mathrm{Brier}(x,q)=\bigl(1-q(x)\bigr)^2+\sum_{a\ne x}q(a)^2 .$$

*Two outcomes ($L=2$).* The other coordinate is $1-q(x)$, so $\mathrm{Brier}(x,q)=2(1-q(x))^2$ and $\varphi(x,\nu)=2(1-\nu)^2$. The Brier score is local.

*Three or more outcomes ($L\ge3$).* The second term depends on how the remaining mass is spread, so the Brier score is not local. An exact counterexample with $L=3$, outcome $x$ = the first label, and $q(x)=\tfrac12$ in both reports:
$$q=(\tfrac12,\tfrac12,0):\ \ \tfrac14+\tfrac14+0=\tfrac12,\qquad q'=(\tfrac12,\tfrac14,\tfrac14):\ \ \tfrac14+\tfrac1{16}+\tfrac1{16}=\tfrac38 .$$
Same outcome, same $q(x)$, different costs, so no $\varphi$ exists.

*Consistency with Theorem 1.* Theorem 1 says that for $L\ge3$ log-loss is the only smooth local proper cost. The Brier score is smooth and proper, so for $L\ge3$ it must fail to be local, and it does. For $L=2$ the hypothesis of the theorem fails, and every cost is local, proper or not, because $q(x)$ determines the whole of $q$. So the answer to "show Brier is local in this style of argument" is: **the Brier score is proper, but local only for two outcomes.** It cannot be shown local for $L\ge3$, because that is false.

*Negative controls.* Locality without properness is easy and useless. The local cost $(1-q(x))^2$ has expected value $\sum_a p(a)(1-q(a))^2$, whose minimiser over the simplex is the water-filling solution $q(a)=\max(0,1-c/p(a))$, with $c$ chosen so that $q$ sums to 1. That is not $p$. For $p=(0.2,0.5,0.3)$ it is $(0.0323,0.6129,0.3548)$. The local cost $1-q(x)$ is minimised at the corner of the most likely rating. Both reward exaggeration (check vii).

## 4. Corollary: the ranked probability score (RPS)

Ratings are ordinal. With $F_q(k)=\sum_{a\le k}q(a)$, the ranked probability score (RPS) is
$$\mathrm{RPS}(x,q)=\sum_{k=0}^{K-2}\bigl(F_q(k)-\mathbb{1}\{x\le k\}\bigr)^2 ,$$
and the repository's normalised RPS divides this by $K-1$. Each term is a binary Brier score (up to the factor 2 of Section 3) on the nested event $\{x\le k\}$, so RPS is a sum of $K-1$ binary Brier scores. Apply the bias plus variance step to $Z_k=\mathbb{1}\{x\le k\}$, whose mean is $F_p(k)$:
$$\mathbb{E}[\mathrm{RPS}(x,q)\mid y]=\sum_{k=0}^{K-2}\bigl(F_q(k)-F_p(k)\bigr)^2+\sum_{k=0}^{K-2}F_p(k)\bigl(1-F_p(k)\bigr). \tag{9'}$$
The second sum does not depend on $q$. The first vanishes only if $F_q(k)=F_p(k)$ for every $k$, and the map $q\mapsto(F_q(0),\dots,F_q(K-2))$ is injective ($q(0)=F(0)$, $q(k)=F(k)-F(k-1)$, $q(K-1)=1-F(K-2)$). So RPS is strictly proper. It is not local: with $K=3$, outcome $x$ the first label, $(\tfrac12,\tfrac12,0)$ scores $\tfrac14$ and $(\tfrac12,0,\tfrac12)$ scores $\tfrac12$. Dividing by $K-1$ changes neither conclusion.

## 5. Intuition

**A point and the corners of a triangle.** Picture every possible report as a point in a triangle, or in its higher-dimensional cousin, the probability simplex. The corners are the certain reports: all the mass on one rating. When a rating actually comes in, it is one of those corners. The Brier score is simply the squared distance from your report to the corner that happened. Now imagine the truth throwing ratings at you many times. The corners land with frequencies given by the true distribution, and their average position, their center of mass, is the true distribution itself. A classical fact about clouds of points is that the average squared distance to the cloud is smallest at its center of mass, and grows by exactly the squared distance to that center as you move away. That one fact is the whole proof.

**The log-loss version of the same idea.** Log-loss measures surprise: the less probability you gave to what happened, the more surprised you are. Averaged over what the truth produces, your surprise is the cross-entropy, and Gibbs' inequality says it can never be lower than the entropy, the surprise you would feel if you had reported the truth. The gap is the Kullback-Leibler divergence. For the Brier score the gap is a plain squared distance instead.

**Shading the report never pays.** Suppose you believe the truth but report something a little more confident, or a little more cautious. The identity says you pay, on average, exactly the squared distance between what you said and what you believe. There is no clever hedge that earns a discount. That is what "proper" means: the scoring rule makes honesty the best policy.

**Why this matters for learning a judge.** In practice we never see the true rating distribution of an item. We only see items and the ratings people gave them. Because the expected score is minimised by the truth, we can replace the expectation by an average over observed item and rating pairs and minimise that instead. We never need to know the true distribution to do it, and with enough ratings, and a judge class rich enough to contain the truth, the minimiser converges to the human rating distribution. For one item's histogram of ratings the error shrinks like one over the square root of the number of ratings. Correlation offers no such guarantee: any positive rescaling and shift of a judge's scores has the same correlation, so its optimum is a whole family and cannot single out the distribution (claims C7 and C8 in [`../DERIVATIONS.md`](../DERIVATIONS.md)).

**Honest versus local.** Log-loss and the Brier score are both honest. Log-loss is also local, in the sense of Definition 2: it looks only at the probability you gave to what actually happened. The Brier score and the ranked probability score look at the whole report. That is a feature for ordinal ratings: the ranked probability score rewards putting mass near the true rating, so a judge that says four when the answer is five loses less than one that says zero. The plain Brier score does not do this. It treats the ratings as unordered labels, so a certain report of four and a certain report of zero cost exactly the same, namely 2, when the answer is five; only the cumulative form feels the order. For three or more outcomes, Theorem 1 says you cannot have both properties in a smooth rule other than log-loss. Locality is bought only with the logarithm.

**A worked example.** Take two ratings, with the truth putting seventy percent on the first and thirty on the second. Reporting the truth gives an expected Brier score of 0.42 and an expected log-loss of 0.6109 nats. Hedging to fifty-fifty raises the Brier score to 0.50 and the log-loss to 0.6931. Overclaiming to ninety-ten also raises the Brier score to 0.50, but raises the log-loss to 0.7645. The two mistakes are the same distance from the truth, so the Brier score charges them the same excess, 0.08: it is a symmetric bowl. Log-loss charges 0.0823 for the hedge and 0.1537 for the overclaim. It is a lopsided bowl that punishes overconfidence hardest, because it goes to infinity as the probability on the actual rating goes to zero. Figure 1 draws both bowls.

| Report $q$ | Expected Brier | Excess $\lVert q-p\rVert^2$ | Expected log-loss (nats) | Excess $\mathrm{KL}(p\Vert q)$ |
| --- | --- | --- | --- | --- |
| (0.7, 0.3) | 0.4200 (= 21/50) | 0.0000 | 0.6109 | 0.0000 |
| (0.5, 0.5) | 0.5000 (= 1/2) | 0.0800 | 0.6931 | 0.0823 |
| (0.9, 0.1) | 0.5000 (= 1/2) | 0.0800 | 0.7645 | 0.1537 |

All numbers are printed by [`check_brier_proper_6437_style.py`](check_brier_proper_6437_style.py) in [`results/check_brier_proper_6437_style.out`](results/check_brier_proper_6437_style.out).

## 6. Checks and formal proofs

The script runs eight checks, all PASS: (i) identity (8′) on 10,000 random pairs for each $K\in\{2,3,6,10\}$, maximum gap $8.9\times10^{-16}$; (ii) the $A,B$ family; (iii) for $K=3$, 6 and 10, each of 20,000 random perturbations of $p$ raises the expected Brier score, and a grid of step 1/200 on the $K=3$ simplex has its minimum exactly at $p=(0.2,0.5,0.3)$; (iv) the empirical minimiser is the frequency vector, and $\mathbb{E}\lVert f-p\rVert^2$ over $(1-\lVert p\rVert^2)/N$ is between 1.004 and 1.012 for $N=10$ to $10^6$; (v) identity (9′) at $K=6$; (vi) locality at $K=2$ and the exact counterexamples at $K=3$; (vii) the two negative controls; and the worked example.

[`Brier6437.lean`](Brier6437.lean) proves in Lean 4.36.0 with Mathlib v4.36.0, with no `sorry`: identity (8′) (`expBrier_identity`), strict propriety (`brier_proper`), the $A,B$ family (`affine_brier_proper`), $\mathrm{Brier}=2(1-q(x))^2$ and locality at $L=2$ (`brier_two_eq`, `brier_two_local`), and the $L=3$ counterexample together with the statement that no $\varphi$ exists (`brier_three_not_local`). The receipt is [`lean_check.txt`](lean_check.txt).

Figures: [`figures/fig1_k2_brier_vs_logloss.png`](figures/fig1_k2_brier_vs_logloss.png) (the two bowls), [`figures/fig2_k3_brier_excess_simplex.png`](figures/fig2_k3_brier_excess_simplex.png) (the excess as squared distance on the $K=3$ simplex), [`figures/fig3_locality_k3_vs_k2.png`](figures/fig3_locality_k3_vs_k2.png) (a cloud at $K=3$, one curve at $K=2$).

Reference: MIT 6.437, Inference and Information, lecture notes on proper cost functions and log-loss (Definition 1, Claim 1, Definition 2, Claim 2, Theorem 1).
