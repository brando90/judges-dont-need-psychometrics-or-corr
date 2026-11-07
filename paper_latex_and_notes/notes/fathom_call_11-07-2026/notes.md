# Fathom call: "Impromptu Call", 2026-11-07

Recording and transcript: **https://fathom.video/calls/893310061**

Participants: Brando Miranda and Sanmi (labmate).

## Key points (timestamped)

- [01:12](https://fathom.video/calls/893310061?timestamp=72): Context. Brando is collecting human judgments on whether theorem-proving agents write good theorems, and wants to know whether an LLM judge "works".
- [05:36](https://fathom.video/calls/893310061?timestamp=336): The community default is to correlate judge scores with human scores.
- [06:02](https://fathom.video/calls/893310061?timestamp=362): Correlations are dangerous because they invite spurious conclusions and don't capture the full human "taste" distribution.
- [06:35](https://fathom.video/calls/893310061?timestamp=395) / [06:46](https://fathom.video/calls/893310061?timestamp=406) / [07:04](https://fathom.video/calls/893310061?timestamp=424): Pearson measures only linear relationships, Spearman only rank order. Each captures *some* distributional property, not all of them.
- [09:35](https://fathom.video/calls/893310061?timestamp=575): The proposal is to fit KL divergence between judge and human score distributions, the standard ML distribution-matching move ("that's why LLMs work").
- [14:27](https://fathom.video/calls/893310061?timestamp=867) / [14:51](https://fathom.video/calls/893310061?timestamp=891): Formalization. A judge is p(R | Y, Y*), where Y is the agent output, Y* the gold reference, and R a score in {0..5} or [0,1]. Replace argmax_θ corr with argmin_θ KL.
- Psychometrics: labmates use metrics from psychometrics. Brando's position is that we don't need them, because this is a normal ML problem.
- How the idea came up: Brando was trying to work out how many human-rated pairs are needed (via a learning bound) and noticed that the correlation objective was the wrong thing to bound.
- [27:40](https://fathom.video/calls/893310061?timestamp=1660): KL is the training objective but isn't an intuitive reporting metric.
- [28:32](https://fathom.video/calls/893310061?timestamp=1712) / [28:57](https://fathom.video/calls/893310061?timestamp=1737): Report MSE on normalized scores (0–5 → 0–1). It penalizes 0-vs-5 disagreements more than 4-vs-5, unlike accuracy.
- Reuse existing KL machinery (Adam etc.) rather than inventing new training metrics.

## Notable quotes (from the Fathom transcript, lightly cleaned)

> "You basically don't fit correlations. You do what we do in machine learning in general. You just fit KL divergence, because that just computes the difference between distributions."

> "Pearson measures linear correlations... [Spearman] just tells you that the ordering of your things is correct."

> "I just mean a probability distribution over the rankings, over the scores."

> "If humans have some taste, just match the taste distribution. That's it. That's all I'm saying."

> "Pearson is linear. Kendall only measures ranking, while KL divergence actually measures the entire distribution."

> "At the very least, we'll stop wasting time on the wrong metric."

> "There's always a chance that in practice this doesn't make a big difference." (Brando's own caveat)

Note: timestamps link to sections of the Fathom AI summary. The raw transcript export is mostly one undivided block.
