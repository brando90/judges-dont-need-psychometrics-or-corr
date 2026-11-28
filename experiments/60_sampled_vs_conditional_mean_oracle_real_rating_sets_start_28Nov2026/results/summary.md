# Experiment 60 summary (ORACLE CHECK on empirical human rating laws; no language model involved)

set | rho_hat [95%] | r(sampled) [95%] | r(mean judge) [95%] | sqrt(rho_hat) | finite-half prediction | difference r(mean) - r(sampled) [95%]
---|---|---|---|---|---|---
chaosnli_alphanli | 0.651 [0.618, 0.682] | 0.651 [0.618, 0.682] | 0.802 [0.781, 0.822] | 0.807 | 0.802 | +0.151 [+0.140, +0.163]
dices350 | 0.192 [0.166, 0.217] | 0.192 [0.166, 0.217] | 0.424 [0.391, 0.453] | 0.438 | 0.424 | +0.232 [+0.226, +0.236]
dices990 | 0.181 [0.160, 0.201] | 0.181 [0.160, 0.201] | 0.401 [0.374, 0.425] | 0.425 | 0.401 | +0.220 [+0.214, +0.225]

set | matched rule | skill truthful [95%] | skill collapsed [95%] | risk collapsed minus risk truthful [95%] (positive = truthful better) | Monte Carlo max |difference| from closed form
---|---|---|---|---|---
chaosnli_alphanli | brier | +0.644 [+0.611, +0.676] | +0.529 [+0.476, +0.582] | +0.0571 [+0.0463, +0.0683] | 0.0076
dices350 | nrps | +0.167 [+0.142, +0.192] | -0.475 [-0.535, -0.416] | +0.1470 [+0.1380, +0.1567] | 0.0107
dices990 | nrps | +0.148 [+0.128, +0.168] | -0.492 [-0.539, -0.444] | +0.1359 [+0.1259, +0.1456] | 0.0138

## Decisions
- H1: supported
- H2: supported (3 of 3 sets within 0.02)
- H3: supported
