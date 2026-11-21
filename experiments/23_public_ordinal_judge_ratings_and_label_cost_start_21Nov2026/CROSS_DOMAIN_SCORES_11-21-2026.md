# Distribution scores across the public human-rating benchmarks | 11-21-2026

As of 11-21-2026. All numbers are held-out test losses from 64 matched draws, one human rating per training item. The first two comparisons were prespecified together. The remaining comparisons were added after the first two results, with separate protocols published before fitting each archive. Each row uses the largest tested budget within its task. All three losses are lower-is-better. Uniform assigns probability 0.2 to each of five grades.

| Held-out rating population | Test units | Training grades | Fitted procedure | Ranked probability score (RPS) | Cross-entropy (CE), nats | Total variation distance (TVD) |
|---|---|---:|---|---:|---:|---:|
| [SummEval · factual consistency](results/gpt4_consistency/scores.csv) | 40 articles / 640 summaries | 768 | CE fit | 0.28367 | 1.60944 | 0.76719 |
| [SummEval · factual consistency](results/gpt4_consistency/scores.csv) | 40 articles / 640 summaries | 768 | Pearson fit | 0.14361 | 1.93348 | 0.49402 |
| [SummEval · factual consistency](results/gpt4_consistency/scores.csv) | 40 articles / 640 summaries | 768 | Uniform | 0.28367 | 1.60944 | 0.76719 |
| [SummEval · factual consistency](results/gpt4_consistency/scores.csv) | 40 articles / 640 summaries | 768 | Training histogram | 0.07243 | 0.54723 | 0.22961 |
| [SummEval · factual consistency](results/gpt4_consistency/scores.csv) | 40 articles / 640 summaries | 768 | Pearson + calibration | 0.05429 | 0.42135 | 0.17096 |
| [KonIQ-10k · image quality](results/image_quality/scores.csv) | 2,015 photographs | 6,144 | CE fit | 0.10678 | 1.27885 | 0.37369 |
| [KonIQ-10k · image quality](results/image_quality/scores.csv) | 2,015 photographs | 6,144 | Pearson fit | 0.30723 | 4.34565 | 0.78809 |
| [KonIQ-10k · image quality](results/image_quality/scores.csv) | 2,015 photographs | 6,144 | Uniform | 0.13325 | 1.60944 | 0.53019 |
| [KonIQ-10k · image quality](results/image_quality/scores.csv) | 2,015 photographs | 6,144 | Training histogram | 0.10541 | 1.17923 | 0.29698 |
| [KonIQ-10k · image quality](results/image_quality/scores.csv) | 2,015 photographs | 6,144 | Pearson + calibration | 0.09215 | 1.07173 | 0.23373 |
| [USR · Topical-Chat, published scores](usr/results/topical_usr/scores.csv) | 24 contexts / 120 responses | 180 | CE fit | 0.19640 | 1.61182 | 0.58748 |
| [USR · Topical-Chat, published scores](usr/results/topical_usr/scores.csv) | 24 contexts / 120 responses | 180 | Pearson fit | 0.32028 | 4.75098 | 0.65633 |
| [USR · Topical-Chat, published scores](usr/results/topical_usr/scores.csv) | 24 contexts / 120 responses | 180 | Uniform | 0.19653 | 1.60944 | 0.59000 |
| [USR · Topical-Chat, published scores](usr/results/topical_usr/scores.csv) | 24 contexts / 120 responses | 180 | Training histogram | 0.19777 | 1.62575 | 0.59069 |
| [USR · Topical-Chat, published scores](usr/results/topical_usr/scores.csv) | 24 contexts / 120 responses | 180 | Pearson + calibration | 0.18400 | 1.59276 | 0.56444 |
| [USR · Topical-Chat, lexical features](usr/results/topical_lexical/scores.csv) | 24 contexts / 144 responses | 216 | CE fit | 0.20104 | 1.60944 | 0.58889 |
| [USR · Topical-Chat, lexical features](usr/results/topical_lexical/scores.csv) | 24 contexts / 144 responses | 216 | Pearson fit | 0.33732 | 4.59291 | 0.67465 |
| [USR · Topical-Chat, lexical features](usr/results/topical_lexical/scores.csv) | 24 contexts / 144 responses | 216 | Uniform | 0.20104 | 1.60944 | 0.58889 |
| [USR · Topical-Chat, lexical features](usr/results/topical_lexical/scores.csv) | 24 contexts / 144 responses | 216 | Training histogram | 0.20019 | 1.62212 | 0.59431 |
| [USR · Topical-Chat, lexical features](usr/results/topical_lexical/scores.csv) | 24 contexts / 144 responses | 216 | Pearson + calibration | 0.19103 | 1.59282 | 0.56876 |
| [USR · PersonaChat, lexical features](usr/results/persona_lexical/scores.csv) | 24 contexts / 120 responses | 180 | CE fit | 0.16738 | 1.56489 | 0.52826 |
| [USR · PersonaChat, lexical features](usr/results/persona_lexical/scores.csv) | 24 contexts / 120 responses | 180 | Pearson fit | 0.22510 | 3.42375 | 0.58795 |
| [USR · PersonaChat, lexical features](usr/results/persona_lexical/scores.csv) | 24 contexts / 120 responses | 180 | Uniform | 0.18875 | 1.60944 | 0.62833 |
| [USR · PersonaChat, lexical features](usr/results/persona_lexical/scores.csv) | 24 contexts / 120 responses | 180 | Training histogram | 0.14634 | 1.40990 | 0.52930 |
| [USR · PersonaChat, lexical features](usr/results/persona_lexical/scores.csv) | 24 contexts / 120 responses | 180 | Pearson + calibration | 0.13865 | 1.38280 | 0.51211 |
| [MovieLens 100K · submitted stars](movielens/results/all_users/scores.csv) | 400 titles | 768 | CE fit | 0.17801 | 1.61032 | 0.35442 |
| [MovieLens 100K · submitted stars](movielens/results/all_users/scores.csv) | 400 titles | 768 | Pearson fit | 0.29405 | 3.97783 | 0.56136 |
| [MovieLens 100K · submitted stars](movielens/results/all_users/scores.csv) | 400 titles | 768 | Uniform | 0.17824 | 1.60944 | 0.36522 |
| [MovieLens 100K · submitted stars](movielens/results/all_users/scores.csv) | 400 titles | 768 | Training histogram | 0.17442 | 1.55586 | 0.32010 |
| [MovieLens 100K · submitted stars](movielens/results/all_users/scores.csv) | 400 titles | 768 | Pearson + calibration | 0.17382 | 1.55436 | 0.31529 |

RPS averages squared cumulative probability error over four grade thresholds (0–1). CE is negative log probability per individual rating in nats. TVD compares the forecast to an item's empirical grade frequencies (0–1). The generative language benchmark uses archived GPT-4 probabilities. For the other applications, fixed feature-based judges use metadata, text features, or published metric scores as specified in each protocol. Paired uncertainty belongs to comparisons on the same held-out observations; comparing absolute scores across different human populations also reflects differences in their grade distributions.

[Primary comparisons and intervals](BUDGET_TABLES_11-21-2026.md) · [Dialogue methods and results](usr/RESULTS_11-21-2026.md) · [MovieLens methods and results](movielens/RESULTS_11-21-2026.md) · [Cross-domain figure](figures/cross_domain_test_scores.pdf)
