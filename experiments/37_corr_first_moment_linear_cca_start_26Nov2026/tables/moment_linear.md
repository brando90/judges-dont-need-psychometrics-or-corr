# Held-out human-human association

One seeded pair of ratings per test item. Pearson and Spearman use the numeric grade. Quadratic gain is the increase in in-sample R² from adding the square of the first rating, on the same test pairs. It is a description of the test pairs, not a trained model.

| Dataset | Test items with a pair | Pearson | Spearman | Linear R² | Quadratic R² gain | Pearson of half means |
| --- | --- | --- | --- | --- | --- | --- |
| summeval_expert | 1280 | 0.7254 [0.6964, 0.7531] | 0.7113 [0.6766, 0.7427] | 0.5262 | 0.0000 | undefined (n=0) |
| summeval_crowd | 1280 | 0.5255 [0.4679, 0.5792] | 0.5217 [0.4689, 0.5721] | 0.2761 | 0.0014 | 0.7172 (n=1280) |
| chaosnli_snli_mnli | 624 | 0.4788 [0.4123, 0.5455] | 0.4733 [0.4061, 0.5404] | 0.2292 | 0.0005 | 0.9783 (n=624) |
| chaosnli_alpha | 307 | 0.5955 [0.4943, 0.6867] | 0.5955 [0.5064, 0.6922] | 0.3546 | -0.0000 | 0.9888 (n=307) |
| dices990 | 198 | 0.2278 [0.0844, 0.3671] | 0.2318 [0.0876, 0.3839] | 0.0519 | 0.0045 | 0.8957 (n=198) |
| veribench_fsc_exp09 | 24 | 0.7772 [0.5732, 0.9252] | 0.7988 [0.5878, 0.9589] | 0.6041 | 0.0002 | undefined (n=1) |
