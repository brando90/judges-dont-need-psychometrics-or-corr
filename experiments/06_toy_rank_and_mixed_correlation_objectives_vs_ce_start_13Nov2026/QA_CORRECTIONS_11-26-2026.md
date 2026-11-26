# Experiment 06: verified QA corrections (11-26-2026)

**Date: 11-26-2026.** Codex CLI, GPT-6 Astra ultra (Vals account). CPU Python/raw-output checks; no Lean checking, no new model calls. Original protocols, result tables and raw outputs remain preserved.

**g2-M2 — CONFIRMED evidence limitation; historical reconstruction remains open.** Old assurance: exact-CE refits preserved reported results. Corrected interpretation 11-26-2026: the experiment 08 aggregate receipt records **337 + 1 = 338** changed tuned fits, one ridge-choice change and **95,291/95,291** converged refits. Its arithmetic is consistent, but individual exact-refit outputs are absent, so the historical assurance cannot be independently rerun from that receipt alone. This does not demonstrate a wrong conclusion; all original measurements remain preserved.

**g2-L1 — CONFIRMED, correlation-only tables.** Old `results/tables.md` contains standalone Spearman and Pearson tables. Corrected 11-26-2026: use the full matched panel below, covering all 132 scenario/training-size/judge rows. Ranked probability score (RPS) excess is squared cumulative error averaged over five thresholds versus known truth [0,1]; cross-entropy (CE) excess is KL(true || forecast), nats; lower better. Total variation distance (TVD) against truth was not recorded, so it is explicitly unavailable. Spearman and Pearson compare forecast means with true conditional means and follow the proper scores. Each row averages 100 seeds with 500 fresh held-out tasks/seed; constants/undefined correlations retain the original zero aggregation convention. No uncertainty was newly inferred. These are the same saved fitted judges, fixed family/ridge/budget, not new optimization runs.

## Recomputed evidence and matched metric panel (11-26-2026)

Date: 11-26-2026. Read-only CPU reconstruction; no model calls.

g2-M2: changed tuned fits 338 = {'0.01': 337, '0.1': 1, '1': 0, '10': 0, '100': 0} ; converged 95291 / 95291
Historical individual refit outputs and earlier 1e-17 reproduction receipt absent; this verifies receipt arithmetic only.
All saved scenarios, training sizes and judges: 100 seeds/cell, 500 held-out tasks/seed. RPS and CE are EXCESS over truth (CE excess = KL). TVD against truth was not recorded; tvd_from_constant has a different target and is not substituted. Correlation NaNs use the original zero convention.

| Dataset / judge | RPS excess | CE excess = KL (nats) | TVD | Spearman | Pearson |
|---|---:|---:|---:|---:|---:|
| A, n=4, ce | 0.010561955 | 0.135814013 | not recorded | 0.100987649 | 0.148132471 |
| A, n=4, kendall | 0.010330607 | 0.131109625 | not recorded | 0.080276293 | 0.094243855 |
| A, n=4, pearson | 0.009597533 | 0.128559065 | not recorded | -0.017823896 | -0.005412897 |
| A, n=4, pearson_spearman | 0.009601389 | 0.128556778 | not recorded | 0.052246952 | 0.059040163 |
| A, n=4, rps | 0.009436249 | 0.125738296 | not recorded | 0.100633205 | 0.148316944 |
| A, n=4, spearman | 0.009593071 | 0.128461955 | not recorded | 0.021258199 | 0.020354008 |
| A, n=8, ce | 0.011600090 | 0.144778548 | not recorded | 0.147141534 | 0.170356043 |
| A, n=8, kendall | 0.012232457 | 0.140409445 | not recorded | 0.133030022 | 0.138595721 |
| A, n=8, pearson | 0.009634823 | 0.129050725 | not recorded | 0.056530672 | 0.063679873 |
| A, n=8, pearson_spearman | 0.009606613 | 0.128547817 | not recorded | 0.070218237 | 0.082377639 |
| A, n=8, rps | 0.009408324 | 0.124045637 | not recorded | 0.143144263 | 0.165434586 |
| A, n=8, spearman | 0.009710751 | 0.129122718 | not recorded | 0.043183584 | 0.048588403 |
| A, n=16, ce | 0.012331780 | 0.147974830 | not recorded | 0.228279475 | 0.263615568 |
| A, n=16, kendall | 0.014801645 | 0.164551333 | not recorded | 0.163512161 | 0.165713630 |
| A, n=16, pearson | 0.009612872 | 0.129179784 | not recorded | 0.025624951 | 0.045306693 |
| A, n=16, pearson_spearman | 0.009660899 | 0.130045783 | not recorded | 0.030583148 | 0.035686408 |
| A, n=16, rps | 0.009146996 | 0.119407682 | not recorded | 0.229892997 | 0.264938829 |
| A, n=16, spearman | 0.009890385 | 0.131444208 | not recorded | -0.004179872 | -0.008198482 |
| A, n=20, ce | 0.011311323 | 0.140292782 | not recorded | 0.246533273 | 0.258948386 |
| A, n=20, kendall | 0.015054908 | 0.179314284 | not recorded | 0.176173327 | 0.184869913 |
| A, n=20, pearson | 0.009718376 | 0.131626323 | not recorded | 0.046070936 | 0.036069896 |
| A, n=20, pearson_spearman | 0.009673569 | 0.130892093 | not recorded | 0.031565818 | 0.026941564 |
| A, n=20, rps | 0.009009857 | 0.117191019 | not recorded | 0.235336567 | 0.249247070 |
| A, n=20, spearman | 0.009846186 | 0.132225013 | not recorded | 0.032788408 | 0.015249446 |
| A, n=32, ce | 0.011097712 | 0.134071022 | not recorded | 0.299844644 | 0.320253176 |
| A, n=32, kendall | 0.016131525 | 0.217314784 | not recorded | 0.240049826 | 0.246480855 |
| A, n=32, pearson | 0.009845008 | 0.136078284 | not recorded | 0.141609493 | 0.093277835 |
| A, n=32, pearson_spearman | 0.009813331 | 0.135628557 | not recorded | 0.130983431 | 0.082323061 |
| A, n=32, rps | 0.008650210 | 0.111274795 | not recorded | 0.298788647 | 0.325906588 |
| A, n=32, spearman | 0.010232259 | 0.142828360 | not recorded | 0.134661025 | 0.090815274 |
| A, n=64, ce | 0.008504342 | 0.105963789 | not recorded | 0.379447327 | 0.397561898 |
| A, n=64, kendall | 0.016800125 | 0.244073868 | not recorded | 0.282857990 | 0.293448811 |
| A, n=64, pearson | 0.010230881 | 0.145866339 | not recorded | 0.210489075 | 0.176057883 |
| A, n=64, pearson_spearman | 0.010361303 | 0.148164736 | not recorded | 0.211251146 | 0.176797334 |
| A, n=64, rps | 0.007691175 | 0.097291001 | not recorded | 0.382649738 | 0.408059687 |
| A, n=64, spearman | 0.012385444 | 0.185277289 | not recorded | 0.222468822 | 0.192512953 |
| A, n=128, ce | 0.005499568 | 0.067146902 | not recorded | 0.519843888 | 0.529342941 |
| A, n=128, kendall | 0.018960470 | 0.287637987 | not recorded | 0.394718872 | 0.414060489 |
| A, n=128, pearson | 0.011611950 | 0.169907814 | not recorded | 0.326312365 | 0.311892239 |
| A, n=128, pearson_spearman | 0.011400251 | 0.167757478 | not recorded | 0.321962846 | 0.304136266 |
| A, n=128, rps | 0.006173620 | 0.077364060 | not recorded | 0.527448041 | 0.539685514 |
| A, n=128, spearman | 0.014624901 | 0.241731012 | not recorded | 0.330685661 | 0.325178737 |
| A, n=256, ce | 0.003242722 | 0.039123045 | not recorded | 0.674244280 | 0.676794633 |
| A, n=256, kendall | 0.020942329 | 0.320119688 | not recorded | 0.548376469 | 0.550828902 |
| A, n=256, pearson | 0.012994387 | 0.191637256 | not recorded | 0.485014485 | 0.489068151 |
| A, n=256, pearson_spearman | 0.012146284 | 0.182381321 | not recorded | 0.482968327 | 0.487796058 |
| A, n=256, rps | 0.004215319 | 0.054403607 | not recorded | 0.681983920 | 0.685258164 |
| A, n=256, spearman | 0.018422660 | 0.315094343 | not recorded | 0.501136325 | 0.492545005 |
| A, n=512, ce | 0.001806325 | 0.020872239 | not recorded | 0.774485046 | 0.771371620 |
| A, n=512, kendall | 0.023231514 | 0.366898966 | not recorded | 0.665394691 | 0.665348367 |
| A, n=512, pearson | 0.015478649 | 0.241733397 | not recorded | 0.615082910 | 0.619415411 |
| A, n=512, pearson_spearman | 0.014617582 | 0.232288313 | not recorded | 0.611395818 | 0.616054286 |
| A, n=512, rps | 0.002559563 | 0.035043970 | not recorded | 0.782937080 | 0.777510382 |
| A, n=512, spearman | 0.021239687 | 0.398418351 | not recorded | 0.606912650 | 0.582672400 |
| A, n=1024, ce | 0.001087830 | 0.011452823 | not recorded | 0.855518904 | 0.846873962 |
| A, n=1024, kendall | 0.026579090 | 0.427413745 | not recorded | 0.796385379 | 0.776907780 |
| A, n=1024, pearson | 0.019453508 | 0.291647605 | not recorded | 0.758204531 | 0.750470174 |
| A, n=1024, pearson_spearman | 0.019356339 | 0.299231747 | not recorded | 0.748842704 | 0.745248071 |
| A, n=1024, rps | 0.001471263 | 0.021860356 | not recorded | 0.862706021 | 0.851564655 |
| A, n=1024, spearman | 0.024556696 | 0.495091880 | not recorded | 0.747609147 | 0.681980185 |
| A, n=2048, ce | 0.000683672 | 0.006525403 | not recorded | 0.902050095 | 0.891432707 |
| A, n=2048, kendall | 0.031338895 | 0.496758455 | not recorded | 0.878782799 | 0.845699403 |
| A, n=2048, pearson | 0.025097897 | 0.393749358 | not recorded | 0.855086759 | 0.844677842 |
| A, n=2048, pearson_spearman | 0.024937032 | 0.395374501 | not recorded | 0.851683854 | 0.842348328 |
| A, n=2048, rps | 0.000860059 | 0.013380456 | not recorded | 0.907252737 | 0.897100711 |
| A, n=2048, spearman | 0.028932089 | 0.600485117 | not recorded | 0.849887219 | 0.745853516 |
| B, n=4, ce | 0.073974884 | 0.409761588 | not recorded | 0.279009773 | 0.378408577 |
| B, n=4, kendall | 0.086493817 | 0.464426164 | not recorded | 0.233404346 | 0.288701994 |
| B, n=4, pearson | 0.090112247 | 0.483475426 | not recorded | 0.034202424 | 0.048582574 |
| B, n=4, pearson_spearman | 0.089832232 | 0.482323278 | not recorded | 0.165605341 | 0.205417934 |
| B, n=4, rps | 0.083575702 | 0.450716865 | not recorded | 0.271180876 | 0.364123247 |
| B, n=4, spearman | 0.090081500 | 0.483527241 | not recorded | 0.068878576 | 0.069880539 |
| B, n=8, ce | 0.061559681 | 0.354230944 | not recorded | 0.465322004 | 0.531702145 |
| B, n=8, kendall | 0.081381248 | 0.438513097 | not recorded | 0.376831245 | 0.409850751 |
| B, n=8, pearson | 0.089672546 | 0.482194855 | not recorded | 0.106492611 | 0.095363589 |
| B, n=8, pearson_spearman | 0.088876323 | 0.478284139 | not recorded | 0.199390181 | 0.203237643 |
| B, n=8, rps | 0.076997631 | 0.417892803 | not recorded | 0.446585495 | 0.508555169 |
| B, n=8, spearman | 0.089558046 | 0.481461452 | not recorded | 0.118649499 | 0.117272267 |
| B, n=16, ce | 0.045397582 | 0.280916979 | not recorded | 0.646878508 | 0.687910345 |
| B, n=16, kendall | 0.074231484 | 0.417116233 | not recorded | 0.554700810 | 0.571931132 |
| B, n=16, pearson | 0.089832194 | 0.483846191 | not recorded | 0.169449483 | 0.122505186 |
| B, n=16, pearson_spearman | 0.089564098 | 0.482022372 | not recorded | 0.142997389 | 0.090014334 |
| B, n=16, rps | 0.066549500 | 0.365501085 | not recorded | 0.610886151 | 0.646766035 |
| B, n=16, spearman | 0.089124325 | 0.480172438 | not recorded | 0.163389439 | 0.097215630 |
| B, n=20, ce | 0.039936372 | 0.249906554 | not recorded | 0.704168069 | 0.732532300 |
| B, n=20, kendall | 0.071771249 | 0.416205715 | not recorded | 0.615769347 | 0.628292887 |
| B, n=20, pearson | 0.089698571 | 0.483490643 | not recorded | 0.196701553 | 0.088456691 |
| B, n=20, pearson_spearman | 0.089915001 | 0.484697154 | not recorded | 0.170629113 | 0.068872580 |
| B, n=20, rps | 0.061995492 | 0.341756219 | not recorded | 0.666606468 | 0.690958951 |
| B, n=20, spearman | 0.089365073 | 0.481237437 | not recorded | 0.179911901 | 0.077393948 |
| B, n=32, ce | 0.029178085 | 0.196461850 | not recorded | 0.799804489 | 0.817450680 |
| B, n=32, kendall | 0.068592430 | 0.425423269 | not recorded | 0.732167948 | 0.729857684 |
| B, n=32, pearson | 0.089744608 | 0.483626582 | not recorded | 0.411706588 | 0.267424691 |
| B, n=32, pearson_spearman | 0.089789952 | 0.483755202 | not recorded | 0.425351775 | 0.271377739 |
| B, n=32, rps | 0.051481008 | 0.288509624 | not recorded | 0.763227010 | 0.777761676 |
| B, n=32, spearman | 0.089501435 | 0.481259109 | not recorded | 0.436326584 | 0.288131934 |
| B, n=64, ce | 0.015584350 | 0.123963205 | not recorded | 0.888985621 | 0.904742394 |
| B, n=64, kendall | 0.063026406 | 0.426843820 | not recorded | 0.848999592 | 0.833533321 |
| B, n=64, pearson | 0.090240634 | 0.487137735 | not recorded | 0.777455994 | 0.765263142 |
| B, n=64, pearson_spearman | 0.090548498 | 0.489606015 | not recorded | 0.776070503 | 0.749538944 |
| B, n=64, rps | 0.031972806 | 0.192171756 | not recorded | 0.871808831 | 0.886053720 |
| B, n=64, spearman | 0.089622719 | 0.488566473 | not recorded | 0.770320649 | 0.724211324 |
| B, n=128, ce | 0.010246971 | 0.082826855 | not recorded | 0.910527698 | 0.934123042 |
| B, n=128, kendall | 0.056729267 | 0.388916250 | not recorded | 0.913323745 | 0.887516012 |
| B, n=128, pearson | 0.091539048 | 0.498929469 | not recorded | 0.879237585 | 0.904086353 |
| B, n=128, pearson_spearman | 0.091164530 | 0.499286996 | not recorded | 0.878361130 | 0.891138359 |
| B, n=128, rps | 0.019810325 | 0.131072414 | not recorded | 0.904082083 | 0.924761078 |
| B, n=128, spearman | 0.086906115 | 0.481754041 | not recorded | 0.887081447 | 0.880746614 |
| B, n=256, ce | 0.006830020 | 0.053858242 | not recorded | 0.928902679 | 0.951803024 |
| B, n=256, kendall | 0.049680414 | 0.352770264 | not recorded | 0.948304613 | 0.905211558 |
| B, n=256, pearson | 0.091725146 | 0.511825652 | not recorded | 0.928393980 | 0.948454395 |
| B, n=256, pearson_spearman | 0.090711821 | 0.508901704 | not recorded | 0.930239392 | 0.943528844 |
| B, n=256, rps | 0.011358867 | 0.086186396 | not recorded | 0.928352075 | 0.948980627 |
| B, n=256, spearman | 0.079960400 | 0.459543631 | not recorded | 0.937031502 | 0.900328754 |
| B, n=512, ce | 0.005150268 | 0.037548072 | not recorded | 0.938712672 | 0.961471458 |
| B, n=512, kendall | 0.043207825 | 0.319735864 | not recorded | 0.963550862 | 0.919058527 |
| B, n=512, pearson | 0.091823683 | 0.537080029 | not recorded | 0.948526697 | 0.967204750 |
| B, n=512, pearson_spearman | 0.090111383 | 0.517688913 | not recorded | 0.951501164 | 0.964430629 |
| B, n=512, rps | 0.007101874 | 0.059771133 | not recorded | 0.942725602 | 0.961656338 |
| B, n=512, spearman | 0.074318167 | 0.449770740 | not recorded | 0.958573393 | 0.902414421 |
| B, n=1024, ce | 0.004330628 | 0.027849293 | not recorded | 0.942524743 | 0.965751830 |
| B, n=1024, kendall | 0.036722369 | 0.309483065 | not recorded | 0.976067990 | 0.911278847 |
| B, n=1024, pearson | 0.092750476 | 0.539481547 | not recorded | 0.961710400 | 0.978330574 |
| B, n=1024, pearson_spearman | 0.088029151 | 0.541520935 | not recorded | 0.965967709 | 0.976577610 |
| B, n=1024, rps | 0.005077617 | 0.045870049 | not recorded | 0.948871405 | 0.967849044 |
| B, n=1024, spearman | 0.063181842 | 0.430788329 | not recorded | 0.974230405 | 0.873284357 |
| B, n=2048, ce | 0.003734219 | 0.022190303 | not recorded | 0.948065134 | 0.969617189 |
| B, n=2048, kendall | 0.033040562 | 0.323317575 | not recorded | 0.981440396 | 0.915322494 |
| B, n=2048, pearson | 0.095127192 | 0.595677286 | not recorded | 0.972602847 | 0.985085007 |
| B, n=2048, pearson_spearman | 0.085080339 | 0.591875780 | not recorded | 0.976745315 | 0.983951556 |
| B, n=2048, rps | 0.003816076 | 0.037234322 | not recorded | 0.957120108 | 0.973554119 |
| B, n=2048, spearman | 0.062311097 | 0.498138511 | not recorded | 0.981684451 | 0.851628759 |


Reproduce the evidence with `python3 docs/qa_11-25-2026/phase2/verify_g1_g2_remaining.py 6` from the repository root. HIGH-item checks are separately saved as `verify_g1-H1.py` (experiment 05), `verify_g2-H1.py` / `verify_g2-H2.py` (experiment 11), and `verify_g2-H3.py` (experiment 07), with their printed `.txt` outputs.
