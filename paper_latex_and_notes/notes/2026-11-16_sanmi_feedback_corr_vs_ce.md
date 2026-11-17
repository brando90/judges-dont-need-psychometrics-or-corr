Sanmi Koyejo's feedback on Brando's corr vs CE judge brief, pasted by Brando from Slack on 2026-11-16 (saved unchanged below; numbered 1–5 Slack screenshot saved at [sanmi_feedback_11-16-2026_slack.jpg](sanmi_feedback_11-16-2026_slack.jpg)).

Brando — 9:47 AM
Ran out of time to curate perfectly, but it was curated quite a bit. My corr vs CE simulated results https://claude.ai/artifact/Rg2hYLxYMGJAte7jcV7kKV all feedback is welcome (and likely needed so I can put it in my defense! I really like this idea)!
Correlation vs Loss Judges
Correlation vs loss for training LLM judges
Correlation vs Loss Judges
sanmi — 5:15 PM
Read the brief. The identification point is right and cleanly shown (correlation can't separate distributions that share means). My worry is that the bottom line stretches this into a general "train and select by loss" claim the evidence doesn't yet support.

Human data. The frontier mixture comparison is null. The exp 14 open-pool win compares two judges that both lose to the constant forecast (0.277 and 0.327 vs 0.184). The best-single-judge rows are closest to how people actually use correlation, and there correlation selection does better: −0.025 in exp 12 (CI just touches 0) and −0.032 in the 44-judge pool (CI excludes 0). The CE-selected single judges (0.205, 0.211) are worse than the constant, probably because the 1e-6 floor drives CE selection on three-sample histograms. That result belongs in the bottom line. Can you add an RPS-selected arm on the human ratings? It's the loss you're arguing for, and it's cheap to run.

Toy baselines. A Pearson-trained softmax only pins the mean, so its distribution is bound to lose on RPS; it ends up worse than the constant in all four scenarios. The fair comparison is score-then-calibrate, and at large n that gap is a few percent of the irreducible RPS. The head is also worse than the constant histogram at small n even though the constant is nested in it (b = 0), so it probably needs regularization. I'd add an affine-calibrated or MSE-trained baseline, and report RPS against the constant and the truth instead of as ratios of excess.

Small n. Fitting 78 weights to a scale-free objective on 16 tasks will overfit, especially with one ridge shared across objectives and Pearson fits that never converge. That supports "don't fit flexible models by maximizing correlation," which is a much smaller claim.
sanmi — 5:15 PM
Use. Distributional accuracy is the right target when the judge's output gets read as a rating or a distribution. For ranking uses, rank agreement is what matters. It's also worth asking whether disagreement on these artifacts is item signal or rater severity. With 2–3 raters we can't tell, and if it's mostly rater error then the true score (expected rating) is the target.

References: Appendix D is Lin's concordance correlation and the ICC consistency vs agreement split, and ordinal likelihood models with rater facets are proper-loss judges. This is a psychometric argument, so I'd drop "don't need psychometrics" from the framing.

On framing, your narrower version seems close to me. Something like: correlation measures association with mean ratings and can't certify agreement in level and scale, let alone spread; when a judge's output is read as a rating or distribution, select and report with a proper scoring rule like RPS and keep correlation as a ranking diagnostic; on the historical ratings the two rules aren't distinguishable yet.

On prompts vs weights: prompts and harness. That's where people tune against agreement in practice, and it's coarse selection, the regime where the single-judge rows already lean toward correlation. Let's run it after the workshop deadline.
