### ChaosNLI alphaNLI binary

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 0.6966 | 0.5017 | 0.5035 | -0.066 |
| 2 | ce-trained | 0.6964 | 0.5016 | 0.5032 | -0.065 |
| 4 | ce-trained | 0.6960 | 0.5014 | 0.5028 | -0.056 |
| 8 | ce-trained | 0.6959 | 0.5014 | 0.5027 | -0.061 |
| 16 | ce-trained | 0.6958 | 0.5013 | 0.5026 | -0.067 |
| 32 | ce-trained | 0.6955 | 0.5012 | 0.5024 | -0.057 |
| 1 | pearson-trained | 0.6987 | 0.5027 | 0.5055 | -0.052 |
| 2 | pearson-trained | 0.6967 | 0.5018 | 0.5036 | -0.043 |
| 4 | pearson-trained | 0.6960 | 0.5014 | 0.5028 | -0.030 |
| 8 | pearson-trained | 0.6968 | 0.5018 | 0.5035 | -0.037 |
| 16 | pearson-trained | 0.6963 | 0.5016 | 0.5031 | -0.042 |
| 32 | pearson-trained | 0.6959 | 0.5013 | 0.5027 | -0.034 |

### SummEval crowd Likert 1-5

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.3276 | 0.7187 | 0.7077 | -0.051 |
| 2 | ce-trained | 1.3273 | 0.7186 | 0.7073 | -0.049 |
| 4 | ce-trained | 1.3272 | 0.7183 | 0.7072 | -0.047 |
| 1 | pearson-trained | 8.6509 | 1.0120 | 1.2395 | -0.057 |
| 2 | pearson-trained | 7.3792 | 1.0110 | 1.2388 | -0.057 |
| 4 | pearson-trained | 7.4972 | 1.0092 | 1.2378 | -0.052 |

### DICES-990 3-level ordinal

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 0.7467 | 0.6294 | 0.4445 | 0.087 |
| 2 | ce-trained | 0.7447 | 0.6283 | 0.4432 | 0.112 |
| 4 | ce-trained | 0.7438 | 0.6279 | 0.4427 | 0.128 |
| 8 | ce-trained | 0.7437 | 0.6279 | 0.4426 | 0.122 |
| 16 | ce-trained | 0.7433 | 0.6276 | 0.4422 | 0.129 |
| 32 | ce-trained | 0.7429 | 0.6274 | 0.4420 | 0.134 |
| 1 | pearson-trained | 1.0532 | 0.6883 | 0.5246 | 0.087 |
| 2 | pearson-trained | 1.1894 | 0.7208 | 0.5704 | 0.104 |
| 4 | pearson-trained | 1.1820 | 0.7208 | 0.5704 | 0.117 |
| 8 | pearson-trained | 1.1619 | 0.7174 | 0.5655 | 0.107 |
| 16 | pearson-trained | 1.1598 | 0.7158 | 0.5633 | 0.112 |
| 32 | pearson-trained | 1.1757 | 0.7164 | 0.5641 | 0.116 |

### ChaosNLI SNLI+MNLI 3-class

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.0533 | 0.6268 | 0.6373 | 0.061 |
| 2 | ce-trained | 1.0523 | 0.6265 | 0.6369 | 0.072 |
| 4 | ce-trained | 1.0523 | 0.6264 | 0.6370 | 0.081 |
| 8 | ce-trained | 1.0529 | 0.6266 | 0.6375 | 0.079 |
| 16 | ce-trained | 1.0530 | 0.6267 | 0.6375 | 0.074 |
| 32 | ce-trained | 1.0529 | 0.6266 | 0.6374 | 0.079 |
| 1 | pearson-trained | 4.9269 | 0.7103 | 0.9387 | 0.066 |
| 2 | pearson-trained | 3.9205 | 0.7105 | 0.9387 | 0.081 |
| 4 | pearson-trained | 4.3045 | 0.7112 | 0.9399 | 0.089 |
| 8 | pearson-trained | 3.7623 | 0.7112 | 0.9396 | 0.087 |
| 16 | pearson-trained | 3.9491 | 0.7105 | 0.9389 | 0.084 |
| 32 | pearson-trained | 4.3640 | 0.7107 | 0.9392 | 0.087 |

### POPQUORN offensiveness 1-5

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.3076 | 0.8026 | 0.6541 | 0.120 |
| 2 | ce-trained | 1.3004 | 0.7994 | 0.6512 | 0.116 |
| 4 | ce-trained | 1.3008 | 0.7996 | 0.6516 | 0.125 |
| 8 | ce-trained | 1.2983 | 0.7986 | 0.6503 | 0.134 |
| 1 | pearson-trained | 5.1142 | 0.9938 | 0.8973 | 0.112 |
| 2 | pearson-trained | 4.3290 | 1.0021 | 0.9048 | 0.124 |
| 4 | pearson-trained | 4.1808 | 1.0053 | 0.9078 | 0.129 |
| 8 | pearson-trained | 4.1948 | 1.0053 | 0.9078 | 0.139 |

### POPQUORN politeness 1-5

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.5286 | 0.8277 | 0.7682 | 0.339 |
| 2 | ce-trained | 1.5296 | 0.8286 | 0.7687 | 0.339 |
| 4 | ce-trained | 1.5283 | 0.8281 | 0.7682 | 0.339 |
| 1 | pearson-trained | 8.9809 | 0.9829 | 1.1669 | 0.329 |
| 2 | pearson-trained | 9.9721 | 0.9821 | 1.1659 | 0.328 |
| 4 | pearson-trained | 9.2428 | 0.9830 | 1.1668 | 0.328 |

### VeriBench FSC R_v1 (0-5, all5)

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.9172 | 0.9951 | 0.8524 | 0.643 |
| 2 | ce-trained | 1.7854 | 0.9830 | 0.8289 | 0.659 |
| 4 | ce-trained | 1.7304 | 0.9512 | 0.8153 | 0.533 |
| 1 | pearson-trained | 11.5095 | 1.1302 | 1.1710 | 0.505 |
| 2 | pearson-trained | 10.5257 | 1.1161 | 1.1627 | 0.425 |
| 4 | pearson-trained | 10.8509 | 1.1195 | 1.1648 | 0.292 |

### VeriBench FSC pooled (0-5)

| k | objective | CE | sqrt(RPS) | Brier | Pearson r |
| --- | --- | --- | --- | --- | --- |
| 1 | ce-trained | 1.2946 | 0.8151 | 0.6042 | 0.325 |
| 2 | ce-trained | 1.2659 | 0.8109 | 0.5938 | 0.313 |
| 1 | pearson-trained | 6.8135 | 1.1577 | 0.9312 | 0.294 |
| 2 | pearson-trained | 7.7680 | 1.1965 | 0.9640 | 0.282 |
