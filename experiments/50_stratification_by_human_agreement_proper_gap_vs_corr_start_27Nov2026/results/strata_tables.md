## Arm A: Experiment 41 heads (raw and matched gap per entropy quintile, seed means; positive = CE-trained head better)

| view | binning | raw gap Q1 | Q2 | Q3 | Q4 | Q5 | Q5 - Q1 (seed mean) | seeds Q5-Q1 CI > 0 / < 0 | R1 | matched gap Q1 / Q5 | Pearson (Pearson-trained head) pooled / mean within-bin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popquorn_offensiveness|K2 | split halves | +0.1561 | +0.1630 | +0.1596 | +0.1120 | +0.0497 | -0.1065 | 0 / 5 | no | +0.0000 / +0.0002 | 0.142 / 0.118 |
| popquorn_offensiveness|K3 | split halves | +0.1442 | +0.1391 | +0.1045 | +0.0488 | +0.0386 | -0.1056 | 0 / 5 | no | +0.0003 / +0.0003 | 0.148 / 0.120 |
| popquorn_offensiveness|K5 | split halves | +0.1477 | +0.1343 | +0.1051 | +0.0777 | +0.0572 | -0.0904 | 0 / 5 | no | +0.0004 / -0.0001 | 0.163 / 0.113 |
| popquorn_politeness|K2 | split halves | -0.0085 | -0.0073 | +0.0001 | -0.0015 | -0.0005 | +0.0080 | 3 / 0 | yes | +0.0026 / +0.0027 | 0.417 / 0.406 |
| popquorn_politeness|K3 | split halves | +0.0167 | +0.0225 | +0.0352 | +0.0386 | +0.0362 | +0.0195 | 2 / 0 | no | +0.0028 / +0.0006 | 0.407 / 0.385 |
| popquorn_politeness|K5 | split halves | +0.0489 | +0.0809 | +0.0790 | +0.0817 | +0.0831 | +0.0342 | 5 / 0 | yes | +0.0015 / +0.0005 | 0.432 / 0.425 |
| summeval_expert|K2 | same ratings | +0.0738 | +0.0792 | +0.0697 | +0.0358 | -0.0235 | -0.0974 | 0 / 5 | no | +0.0004 / -0.0006 | 0.198 / 0.176 |
| summeval_expert|K3 | same ratings | +0.1378 | +0.1384 | +0.1392 | +0.0355 | +0.0010 | -0.1368 | 0 / 5 | no | +0.0004 / -0.0004 | 0.183 / 0.134 |
| summeval_expert|K5 | same ratings | +0.1440 | +0.1452 | +0.0883 | +0.0561 | +0.0605 | -0.0835 | 0 / 5 | no | +0.0032 / -0.0014 | 0.219 / 0.134 |
| summeval_crowd|K2 (no signal) | split halves | +0.0341 | +0.0377 | +0.0363 | +0.0283 | +0.0298 | -0.0042 | 0 / 0 | no | -0.0001 / +0.0000 | -0.021 / -0.024 |
| summeval_crowd|K3 (no signal) | split halves | +0.1027 | +0.1060 | +0.0993 | +0.0991 | +0.0954 | -0.0073 | 0 / 1 | no | -0.0001 / +0.0000 | -0.018 / -0.021 |
| summeval_crowd|K5 (no signal) | split halves | +0.1296 | +0.1315 | +0.1266 | +0.1283 | +0.1283 | -0.0013 | 0 / 0 | no | -0.0001 / +0.0001 | -0.041 / -0.044 |
| dices990|K2 | split halves | +0.1589 | +0.1496 | +0.0819 | +0.0123 | -0.0303 | -0.1892 | 0 / 5 | no | +0.0008 / +0.0010 | 0.274 / 0.129 |
| dices990|K3 | split halves | +0.1429 | +0.1252 | +0.0514 | +0.0154 | -0.0197 | -0.1626 | 0 / 5 | no | +0.0008 / +0.0005 | 0.278 / 0.156 |
| chaosnli_snli_mnli|K3 | split halves | +0.0663 | +0.0599 | +0.0636 | +0.0563 | +0.0419 | -0.0244 | 0 / 4 | no | +0.0005 / -0.0001 | 0.139 / 0.161 |
| chaosnli_alphanli|K2 (no signal) | split halves | -0.0026 | +0.0028 | +0.0053 | +0.0032 | +0.0013 | +0.0039 | 1 / 0 | no | -0.0000 / -0.0001 | -0.041 / -0.033 |

R1: supported on 2 of 12 views with signal; significantly the opposite way (Q5 gap smaller, >=3 seeds) on 9.
R2: within-bin Pearson below pooled Pearson in 494 of 800 (view, seed, bin, head) cells.

R2 (Arm B): within-bin Pearson below the judge's pooled Pearson in 76 of 135 (set, bin, judge) cells.

## Arm B: real LLM judges per entropy quintile

| set | binning | judge | S Q1 -> Q5 | skill Q1 -> Q5 | Pearson Q1 -> Q5 | top-1 Q1 -> Q5 |
|---|---|---|---|---|---|---|
| chaosnli_snli | split halves | gpt-6-astra | 0.141 -> 0.660 | 0.781 -> -0.005 | 0.993 -> 0.834 | 0.992 -> 0.696 |
| chaosnli_snli | split halves | claude-opus-5-5 | 0.187 -> 0.582 | 0.709 -> 0.114 | 0.959 -> 0.862 | 0.936 -> 0.671 |
| chaosnli_snli | split halves | kimi-k3 | 0.158 -> 0.679 | 0.754 -> -0.034 | 0.990 -> 0.700 | 0.975 -> 0.628 |
| chaosnli_snli | split halves | glm-5.3 | 0.164 -> 0.605 | 0.744 -> 0.078 | 0.989 -> 0.810 | 1.000 -> 0.686 |
| chaosnli_snli | split halves | deepseek-v4.1-flash | 0.145 -> 0.685 | 0.774 -> -0.043 | 0.986 -> 0.692 | 0.994 -> 0.632 |
| chaosnli_mnli | split halves | gpt-6-astra | 0.386 -> 0.781 | 0.368 -> -0.154 | 0.931 -> 0.631 | 0.842 -> 0.642 |
| chaosnli_mnli | split halves | claude-opus-5-5 | 0.378 -> 0.658 | 0.381 -> 0.027 | 0.926 -> 0.702 | 0.867 -> 0.625 |
| chaosnli_mnli | split halves | kimi-k3 | 0.425 -> 0.748 | 0.304 -> -0.105 | 0.859 -> 0.508 | 0.775 -> 0.560 |
| chaosnli_mnli | split halves | glm-5.3 | 0.377 -> 0.690 | 0.383 -> -0.020 | 0.929 -> 0.605 | 0.872 -> 0.586 |
| chaosnli_mnli | split halves | deepseek-v4.1-flash | 0.448 -> 0.763 | 0.267 -> -0.128 | 0.842 -> 0.581 | 0.769 -> 0.628 |
| chaosnli_alphanli | split halves | gpt-6-astra | 0.046 -> 0.595 | 0.907 -> -0.191 | 0.997 -> 0.517 | 1.000 -> 0.700 |
| chaosnli_alphanli | split halves | claude-opus-5-5 | 0.083 -> 0.492 | 0.832 -> 0.016 | 0.963 -> 0.589 | 0.975 -> 0.762 |
| chaosnli_alphanli | split halves | kimi-k3 | 0.051 -> 0.609 | 0.896 -> -0.218 | 0.992 -> 0.479 | 1.000 -> 0.650 |
| chaosnli_alphanli | split halves | glm-5.3 | 0.062 -> 0.598 | 0.873 -> -0.196 | 0.980 -> 0.401 | 0.988 -> 0.631 |
| chaosnli_alphanli | split halves | deepseek-v4.1-flash | 0.240 -> 0.640 | 0.511 -> -0.280 | 0.794 -> 0.380 | 0.894 -> 0.644 |
| dices350 | split halves | gpt-6-astra | 0.162 -> 0.349 | 0.158 -> -0.358 | 0.731 -> 0.216 | 0.943 -> 0.693 |
| dices350 | split halves | claude-opus-5-5 | 0.141 -> 0.283 | 0.267 -> -0.101 | 0.789 -> 0.184 | 0.936 -> 0.707 |
| dices350 | split halves | kimi-k3 | 0.159 -> 0.347 | 0.176 -> -0.348 | 0.675 -> 0.115 | 0.929 -> 0.640 |
| dices350 | split halves | glm-5.3 | 0.156 -> 0.327 | 0.191 -> -0.269 | 0.724 -> 0.120 | 0.914 -> 0.721 |
| dices350 | split halves | deepseek-v4.1-flash | 0.156 -> 0.355 | 0.189 -> -0.379 | 0.744 -> 0.169 | 0.948 -> 0.702 |
| dices990 | split halves | gpt-6-astra | 0.101 -> 0.332 | 0.253 -> -0.187 | 0.330 -> 0.472 | 1.000 -> 0.669 |
| dices990 | split halves | claude-opus-5-5 | 0.095 -> 0.269 | 0.296 -> 0.038 | 0.436 -> 0.489 | 1.000 -> 0.650 |
| dices990 | split halves | kimi-k3 | 0.101 -> 0.340 | 0.250 -> -0.215 | 0.239 -> 0.413 | 0.992 -> 0.656 |
| dices990 | split halves | glm-5.3 | 0.098 -> 0.317 | 0.276 -> -0.135 | 0.480 -> 0.373 | 1.000 -> 0.606 |
| dices990 | split halves | deepseek-v4.1-flash | 0.101 -> 0.343 | 0.256 -> -0.227 | 0.212 -> 0.385 | 0.992 -> 0.598 |
| summeval_consistency | same ratings | g-eval-gpt-4|sampled20 | 0.213 -> 0.237 | -7.712 -> 0.151 | 0.347 -> 0.645 | 0.319 -> 0.249 |
| summeval_relevance | same ratings | g-eval-gpt-4|sampled20 | 0.198 -> 0.200 | -0.406 -> -0.450 | 0.714 -> 0.433 | 0.267 -> 0.300 |

| set | top-1 falls Q1 -> Q5 for every judge | Kendall tau Q1 vs Q5 ranking: by top-1 / by S | Kendall tau(Pearson ranking, S ranking) per bin Q1..Q5 | best by Pearson / best by S, per bin |
|---|---|---|---|---|
| chaosnli_snli | yes | +0.20 / -0.60 | +0.6, +0.8, +0.8, +0.6, +0.8 | gpt-6-astra/gpt-6-astra; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5 |
| chaosnli_mnli | yes | -0.20 / +0.40 | +0.6, +0.8, +0.8, +0.8, +0.2 | gpt-6-astra/glm-5.3; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5 |
| chaosnli_alphanli | yes | +0.11 / +0.20 | +1.0, +1.0, +1.0, +0.8, +0.8 | gpt-6-astra/gpt-6-astra; gpt-6-astra/gpt-6-astra; gpt-6-astra/gpt-6-astra; gpt-6-astra/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5 |
| dices350 | yes | -0.20 / +0.60 | +0.4, +0.4, -0.2, +0.2, +0.0 | claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; gpt-6-astra/claude-opus-5-5; gpt-6-astra/claude-opus-5-5; gpt-6-astra/claude-opus-5-5 |
| dices990 | yes | +0.12 / +0.60 | +0.4, +1.0, +0.6, +0.2, +0.4 | glm-5.3/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5; claude-opus-5-5/claude-opus-5-5 |
| summeval_consistency | yes | n/a (1 judge) | nan, nan, nan, nan, nan | None/None; None/None; None/None; None/None; None/None |
| summeval_relevance | no | n/a (1 judge) | nan, nan, nan, nan, nan | None/None; None/None; None/None; None/None; None/None |
