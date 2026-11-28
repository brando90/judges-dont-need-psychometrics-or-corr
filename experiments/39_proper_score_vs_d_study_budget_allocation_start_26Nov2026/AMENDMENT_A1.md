# Amendment A1 (11-26-2026, America/Los_Angeles)

Declared after a 2-replication smoke test on chaosnli_snli, and **before** the 150-replication run.

The smoke test showed that two different designs can produce *exactly* the same test ranked probability score (RPS) under the primary predictor, because that predictor is a global choice among 27 frozen transforms of frozen judge probabilities. Once two designs select the same transform, the test forecast is identical.

**Change:** add a secondary predictor and selector, `ps_mix`. It minimizes categorical cross-entropy jointly over the same 27 transforms and a mixture weight $\alpha\in\lbrace 0,0.25,0.5,0.75,1\rbrace$ in $q=(1-\alpha)q_{\mathrm{transform}}+\alpha\widehat{\pi}$, where $\widehat{\pi}$ is the Laplace-smoothed pooled histogram of the training ratings. Deployment metrics `rps_mix`, `ce_mix`, `tvd_mix` and a mixture oracle are reported alongside the primary ones.

**Unchanged:** the primary predictor, the G/D-U1 and G/D-U2 rules, seeds, splits, budgets, and the meaning of "beats." Primary claims use the original 27-transform CE-fit only.

**Erratum, 11-27-2026.** The text above says "Laplace-smoothed". The code (`study_a.py`) adds 0.5 per category, which is Jeffreys (Krichevsky-Trofimov) smoothing, not Laplace (add-one) smoothing. The experiment's results were computed with the code, so only the label in this amendment was wrong.

