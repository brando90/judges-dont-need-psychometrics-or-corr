# Claude Fable 5.2 Max design consultation | 11-21-2026

As of 11-21-2026, Brando asked Codex to consult Fable before selecting a design. The first-party Claude client selected Fable 5.2 at Max effort and the result identified `claude-fable-5-2`. Codex supplied the live VeriBench count, public corpus design, comparator definitions, and alternatives. The text below is the returned recommendation. The study protocol specifies the implemented decision and its assumptions.

**Decision:** run UNT with leave-one-homework-out cross-fitting now, and register hashed Set N forecasts in parallel.

**Why:** Only UNT yields real held-out human ratings within budget; Set N has none and a new panel is unfunded. Reject the 5-family split: five held-out clusters floor an exact sign-flip p at 1/32 and cap training at 160 labels, so 256 is impossible. Leave-one-out gives ten clusters and 288 training labels.

**VeriBench:** No option establishes a fresh VeriBench advantage. Registering Set N forecasts (historical development only, no solver keys) merely enables a future test; the paper must say so.

**Few families:** Homework is the inference unit: ten paired held-out differences per budget and judge, each averaged over ≥200 seeded, nested label draws. Primary bound: one-sided t(9) with the Nadeau–Bengio fold-overlap correction (variance × (1/10 + 1/9)); exact sign-flip as sensitivity. No cluster-bootstrap primary: anti-conservative at ten clusters. Report label-draw (within-regime) variance separately; it never increases n. Question-level clustering (dependent within homework) is sensitivity only.

**Confirmation:** Hash prompt, grid, tie rule, seeds, and both judges' forecasts before parsing any grade column, then make one confirmatory run. True confirmation is Set N once rated.

**Minimal protocol changes:**
1. Primary arm: two ratings per answer; one-rating arm secondary.
2. Uniform spans the six grades. UNT grades reportedly pile near 5, so uniform is weak here: call a win "informative" only if CE also beats train-histogram by the same bound.
3. Report judges separately; two models cannot identify an openness effect.

**Joint win rule (familywise 0.05):** Per judge, one-sided α = 0.025. Test budgets in fixed descending order, 256→4. A budget passes only if both comparator-minus-CE lower bounds (Pearson, uniform) on held-out normalized RPS exceed zero; stop at the first failure. Failing at 256 means no claim. Report the smallest passing budget; CE and TVD stay descriptive.

**Domain mismatch:** CS short-answer correctness against a reference, two fixed graders rather than a sampled panel, ceiling-skewed integer 0–5, versus theorem-translation artifacts. Public corpus: pretraining exposure may inflate the uniform leg; the Pearson leg shares forecasts and is less exposed.

(Appended Python launcher not run.)
