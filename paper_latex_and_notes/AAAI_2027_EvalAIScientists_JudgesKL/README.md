# Paper: AAAI-27 workshop position paper (draft)

Venue: [Effective and Efficient Evaluation for AI Scientists @ AAAI-27](https://eval-ai-scientists.github.io/aaai27/). Position paper, 2–4 pages (references excluded), official AAAI-27 two-column author kit, anonymized PDF, submitted via OpenReview. Deadline **20 Nov 2026, 06:00 UTC** (19 Nov, 10:00 PM PST).

Layout follows agents-config (Rules 35/49): one venue dir under `paper_latex_and_notes/`, main file `00_main_aaai_2027.tex`, numbered section files, `00_main.bib`, `math_commands.tex`, `figures/`, committed PDF, aux files in `_build/`. Style files `aaai2027.sty` and `aaai2027.bst` come from https://aaai.org/authorkit27/.

Build: `make` (tectonic) or `make latexmk`. Note that AAAI asks for a single-file source at camera-ready time, so inline the `\input`s then.
