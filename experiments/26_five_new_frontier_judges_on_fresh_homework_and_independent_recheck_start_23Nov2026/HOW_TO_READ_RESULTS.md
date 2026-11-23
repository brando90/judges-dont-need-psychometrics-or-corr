# How to read the experiment 26 results | 11-23-2026

- **Data:** 160 UNT short answers, 10 assignments (16 answers each), two graders, grades 0 to 5. Leave-one-assignment-out: each of the 10 folds holds out one assignment and trains on the other nine.
- **Judges:** each reports six grade probabilities per answer (one call). Unparseable output becomes the uniform forecast and stays in the analysis (Claude Opus 5.5: 4, GLM 5.3: 1).
- **Training budget `B`:** number of individual human ratings used to choose a calibration (4 to 256). Primary condition: two ratings per answer, so `B/2` answers. Secondary: one rating per answer (256 not run, only 144 training answers exist).
- **CE fit / Pearson fit:** from 27 fixed transforms of the judge's probabilities, pick the one with lowest mean training CE, or highest Pearson correlation of predicted mean with observed mean grade.
- **Comparators:** uniform, uncalibrated judge, training histogram (0.5 pseudo-count). 200 seeded training draws per fold.
- **Metrics:** held-out RPS (primary, lower is better), CE in nats, TVD. Each is averaged over the 16 held-out answers using their two-grader histogram, then over draws; the 10 fold values are the data.
- **Sign:** every difference is comparator minus CE fit, so positive means CE is better.
- **Intervals:** 95% two-sided, standard deviation of the ten fold differences times sqrt(1/10 + 1/9), t with 9 degrees of freedom.
- **Formal rule:** descending from 256; a budget passes when the one-sided lower bounds against the Pearson fit and against uniform are both above 0 (alpha 0.01 per judge); stop at the first failure. Rows after a failure are descriptive only.
- **Limit of correlation:** Pearson sees only the per-answer predicted mean (up to level and scale), not spread or shape; see the first page of the PDF.
