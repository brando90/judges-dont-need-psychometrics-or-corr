# Fresh-family judge comparison checkpoint | 11-21-2026

As of 11-21-2026, Codex coordinates both model runners, owns the results and is the sole notifier. Protocol, sources, model IDs, budgets, failure rule, checks and split are in [PROTOCOL](PROTOCOL_11-21-2026.md). Qwen runs on owned hardware; Claude runs on its native subscription with paid overage disabled. The sister collection owns the prospective test ratings.

11-21-2026 | Selected the data with Claude Fable 5.2 Max. Prepared and hashed both prompt collections before study inference.

11-21-2026 | Protocol published at ff2f07a. Both judges returned valid study responses. The UNT Claude run finished 160/160 on the included subscription. Local Qwen is running. Both prospective request manifests were verified with full context lengths before inference.

11-21-2026 | Closed judge completed 160/160 UNT and 175/175 VeriBench on the included route. The open judge is running locally for UNT and on one H200 at context 65,536 for all VeriBench prompts. Independent token counts match live receipts. Results Doc: https://docs.google.com/document/d/1Gy08bWD394XS6X7FrW5PtKuBeJ9kgiehn61fFNwq6mM/edit

11-21-2026 | Both models completed 160 homework and 175 VeriBench requests apiece. The joint VeriBench forecasts were frozen in b90e7fd and the homework forecasts in 5e4982f. Completed the registered held-out homework analysis. Independent replay verified all 320 source grades, 2,000 training schedules, 312 contrasts, all objective decisions and both rating-depth conditions. Qwen's CE calibration cleared Pearson and uniform on test RPS from four ratings through all seven tested budgets. A training grade histogram achieved lower observed RPS than Qwen's CE fit throughout. Claude's descending test stopped at 256 ratings; six smaller budgets favored CE locally. See [full results](short_answers/RESULTS_11-21-2026.md). [Verification](short_answers/VERIFICATION_11-21-2026.md)

11-21-2026 | Registered native and May-only calibrated forecasts for 100 fresh VeriBench artifacts. No fresh scores were available on the 34 current sheets (0/3,400). Public [forecast commitment](veribench/REGISTRATION_11-21-2026.json), [eligible rating coverage](veribench/COVERAGE_11-21-2026.json), and [private reconstruction archive](https://drive.google.com/file/d/1XOACnps38jXupxJNkf5aQXPOoXe0KHy0/view). No active inference job remains.
