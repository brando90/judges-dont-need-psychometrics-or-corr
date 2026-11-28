# Mega QA, Fable 5.2 reviewer (28 Nov 2026): NOT RUN

Brando authorized running Claude Code (`claude -p`, Fable 5.2, max effort) on his own Mac with any of his Anthropic CLI accounts (7:27 AM PT, 28 Nov 2026). It could not run:

- default config (`~/.claude`): `claude auth status` reports not logged in;
- `CLAUDE_CONFIG_DIR=~/.claude-vals` and `~/.claude-su`: both report that the login reached its time limit and cannot be renewed, and ask for `claude auth login`.

Logging in needs Brando interactively (browser OAuth), so Grok Bot did not attempt it, and did not read or copy any credential file or fall back to an API key. No Fable output exists; only the Codex review (`qa_codex_vals_28Nov2026.md`) was triaged.

To rerun: after `claude auth login` (any account), from a clone of current main:

    claude -p --model fable --effort max --permission-mode plan < review_prompt.md > qa_fable52_28Nov2026.md

(check with `claude --help` / `/model` that the alias resolves to Fable 5.2; the prompt is the same one given to Codex, kept on the Mac at `~/scratch_grokbot/review_prompt.md`).
