"""Simulated multi-modal population on a 0–5 rubric.

The population is a testbed. Subpopulation base measures are frozen marginal
counts from VeriBench native_main, DICES-990, and ChaosNLI SNLI+MNLI. Draws
from the tilted laws are simulated. They are not a certification set.

Persona order is (strict, lenient, dices_shape, chaos_shape) only in the
native side table. The Monte Carlo population uses four 0–5 personas:

  strict     reflection of the VeriBench native_main histogram, tilted
  lenient    VeriBench native_main histogram, tilted
  polarized  mixture of the reflected histogram tilted to 0.5 and the
             native_main histogram tilted to 4.5; the bin mean sets the weight
  chaos      ChaosNLI 3-class marginal placed on bins 0, 2, 5 by the
             order entailment < neutral < contradiction, then tilted

The ChaosNLI placement keeps the three native counts and the native order.
It is a transport onto the project rubric, not a claim that those annotators
used a 0–5 score. The native 3-class side table does not use that transport.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import comb
from scipy.stats import rankdata, t as student_t

ROOT = Path(__file__).resolve().parent
SHAPES_PATH = ROOT / "data" / "empirical_shapes.json"

K = 6
Y = np.arange(K, dtype=float)
EPS = 1e-6
SEED = 20261127
N_REP = 400
BUDGETS = (4, 16, 64, 256)
N_BINS = 6
ITEMS_PER_BIN = 8
TRAIN_ITEMS = (0, 1, 2, 3)
VAL_ITEMS = (4, 5)
TEST_ITEMS = (6, 7)
BIN_MEANS = np.array([1.6, 2.0, 2.4, 2.8, 3.2, 3.6], dtype=float)
DELTA = 0.8
PERSONAS = ("strict", "lenient", "polarized", "chaos")
TRAIN_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])
# Deployment mix. Polarized is the multi-modal subpopulation. Means still match the train mix.
SHIFT_WEIGHTS = np.array([0.05, 0.05, 0.80, 0.10])
POLE_LOW = 0.5
POLE_HIGH = 4.5
# Selection order. A tie breaks toward the earlier name, which is a proper-score forecast.
SELECT_POOL = (
    "ce_persona_shift",
    "rps_persona_shift",
    "ce_pooled",
    "pearson_binomial",
    "spearman_point",
    "lenient_point",
)

# Equal-moment witness on {0,1,2,3,4,5}. Means 3/2, 5/2, 7/2 and variances 2, 4, 2.
M1 = np.array([5, 0, 3, 3, 0, 5], dtype=float) / 16
M2 = np.array([7, 9, 0, 0, 9, 7], dtype=float) / 32
LAW_A = np.array([11, 0, 3, 10, 0, 0], dtype=float) / 24  # 11/24, 0, 1/8, 5/12
LAW_B = np.array([10, 3, 0, 11, 0, 0], dtype=float) / 24  # 5/12, 1/8, 0, 11/24


def load_shapes() -> dict:
    return json.loads(SHAPES_PATH.read_text())


def _pmf_from_counts(counts: list[int]) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or int(c.sum()) <= 0:
        raise ValueError("counts must be a nonempty class histogram")
    return c / c.sum()


def tilt_to_mean(template: np.ndarray, target_mean: float) -> np.ndarray:
    """Exponential tilt q(y) ∝ T(y) exp(β y) with mean ``target_mean``.

    Zeros of T stay zero. The mean map is continuous and strictly increasing
    from the lowest supported score to the highest, so every target strictly
    inside that interval is attained. Targets outside are clipped to a point
    mass on the nearest supported endpoint.
    """
    template = np.asarray(template, dtype=float)
    template = np.clip(template, 0.0, None)
    total = float(template.sum())
    if total <= 0:
        raise ValueError("template has no mass")
    template = template / total
    support = np.flatnonzero(template > 0)
    ys = support.astype(float)
    log_t = np.log(template[support])
    lo_m = float(ys[0])
    hi_m = float(ys[-1])
    q = np.zeros(template.shape[0], dtype=float)
    if target_mean <= lo_m + 1e-12:
        q[int(ys[0])] = 1.0
        return q
    if target_mean >= hi_m - 1e-12:
        q[int(ys[-1])] = 1.0
        return q

    def mean_at(beta: float) -> tuple[float, np.ndarray]:
        log_w = log_t + beta * ys
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        return float(np.dot(w, ys)), w

    lo, hi = -80.0, 80.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        m, _ = mean_at(mid)
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    m, w = mean_at(0.5 * (lo + hi))
    if abs(m - target_mean) > 1e-8:
        raise RuntimeError(f"tilt missed target {target_mean} (got {m})")
    q[support] = w
    q /= q.sum()
    return q


def binom_rows(means: np.ndarray) -> np.ndarray:
    """Binomial(5, clip(mean/5, 0.02, 0.98)) for each row mean. Closed form."""
    theta = np.clip(np.asarray(means, dtype=float) / 5.0, 0.02, 0.98)
    ks = np.arange(K)
    coeff = comb(5, ks)
    th = theta[:, None]
    pmf = coeff * th**ks * (1.0 - th) ** (5 - ks)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf


def chaos_transport(counts_3: list[int]) -> np.ndarray:
    """Place a native 3-class histogram on bins 0, 2, and 5, in that order.

    Bins 1, 3, and 4 stay empty. The three counts are not changed.
    """
    if len(counts_3) != 3:
        raise ValueError("expected three native classes")
    out = np.zeros(K, dtype=float)
    out[0] = counts_3[0]
    out[2] = counts_3[1]
    out[5] = counts_3[2]
    return out / out.sum()


def _axis_scores(p: np.ndarray) -> np.ndarray:
    return np.arange(np.asarray(p).shape[-1], dtype=float)


def row_means(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p @ _axis_scores(p)


def row_vars(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    scores = _axis_scores(p)
    m = row_means(p)
    return (p * (scores - m[:, None]) ** 2).sum(axis=1)


def panel_coeffs(pmfs: np.ndarray) -> dict:
    """One-source generalizability and dependability coefficients.

    v_u is the variance of the row means with divisor equal to the number of
    rows. With one source, Φ equals Eρ². A zero denominator is undefined.
    A row-mean variance below 1e-12 is treated as a constant forecast.
    """
    m = row_means(pmfs)
    vu = float(np.mean((m - m.mean()) ** 2))
    ve = float(np.mean(row_vars(pmfs)))
    if vu < 1e-12:
        vu = 0.0
    if ve < 1e-12:
        ve = 0.0
    den = vu + ve
    coef = None if den == 0.0 else vu / den
    return {"vu": vu, "ve": ve, "erho2": coef, "phi_own": coef}


def joint_components(mu_h, var_h, mu_j, var_j) -> dict:
    """Two-source population components. Divisors are the number of items and 2 sources.

    Relative Eρ² excludes the source main effect. Absolute Φ includes it.
    """
    mu_h = np.asarray(mu_h, dtype=float)
    mu_j = np.asarray(mu_j, dtype=float)
    var_h = np.asarray(var_h, dtype=float)
    var_j = np.asarray(var_j, dtype=float)
    grand = 0.5 * (mu_h.mean() + mu_j.mean())
    uni = 0.5 * (mu_h + mu_j)
    v_u = float(np.mean((uni - uni.mean()) ** 2))
    alpha_h = float(mu_h.mean() - grand)
    alpha_j = float(mu_j.mean() - grand)
    v_a = float(0.5 * (alpha_h**2 + alpha_j**2))
    delta = 0.5 * ((mu_h - mu_h.mean()) - (mu_j - mu_j.mean()))
    v_ia = float(np.mean(np.concatenate([delta**2, delta**2])))
    v_e = float(0.5 * (var_h.mean() + var_j.mean()))
    v_delta = v_ia + v_e
    v_Delta = v_delta + v_a
    erho = None if v_u + v_delta == 0 else v_u / (v_u + v_delta)
    phi = None if v_u + v_Delta == 0 else v_u / (v_u + v_Delta)
    return {
        "v_u": v_u,
        "v_source": v_a,
        "v_e": v_e,
        "erho2": erho,
        "phi": phi,
        "source_effect_human": alpha_h,
        "source_effect_judge": alpha_j,
    }


def expected_rps_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Normalized expected RPS, range 0 to 1. Lower is better.

    E[(1/5) Σ_{t=0}^{4} (F_q(t) − 1{y ≤ t})²]
      = (1/5) Σ_t (F_q(t) − F_p(t))² + (1/5) Σ_t F_p(t) (1 − F_p(t)).
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    width = q.shape[1] - 1
    fq = np.cumsum(q, axis=1)[:, :-1]
    fp = np.cumsum(p, axis=1)[:, :-1]
    return ((fq - fp) ** 2).sum(axis=1) / width + (fp * (1.0 - fp)).sum(axis=1) / width


def expected_ce_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    safe = np.where(q > 0, q, 1.0)
    terms = np.where(p > 0, -p * np.log(safe), 0.0)
    out = terms.sum(axis=1)
    bad = ((p > 0) & (q <= 0)).any(axis=1)
    out = out.astype(float)
    out[bad] = np.inf
    return out


def smooth_rows(q: np.ndarray, eps: float = EPS) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    width = q.shape[-1]
    return (1.0 - width * eps) * q + eps


def tvd_rows(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(q, dtype=float) - np.asarray(p, dtype=float)).sum(axis=1)


def rps_of_outcomes(q: np.ndarray) -> np.ndarray:
    """Normalized RPS of forecast rows against each possible score. Shape (bins, classes)."""
    q = np.asarray(q, dtype=float)
    width = q.shape[1]
    f = np.cumsum(q, axis=1)[:, :-1]
    out = np.zeros_like(q, dtype=float)
    ks = np.arange(width - 1)
    for y in range(width):
        ind = (ks >= y).astype(float)
        out[:, y] = ((f - ind) ** 2).sum(axis=1) / (width - 1)
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return pearson(rankdata(a), rankdata(b))


def round_half_up(x) -> np.ndarray:
    return np.floor(np.asarray(x, dtype=float) + 0.5).astype(int)


def mix(weights: np.ndarray, laws: np.ndarray) -> np.ndarray:
    """``laws`` is (persona, bin, class). Returns (bin, class)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.einsum("p,pbc->bc", w, laws)


def build_population(shapes: dict | None = None) -> dict:
    shapes = shapes or load_shapes()
    native = _pmf_from_counts(shapes["veribench_native_main"]["marginal_counts"])
    reflected = native[::-1].copy()
    chaos_base = chaos_transport(shapes["chaosnli_snli_mnli"]["marginal_counts"])
    dices = _pmf_from_counts(shapes["dices990_q_overall"]["marginal_counts"])
    chaos3 = _pmf_from_counts(shapes["chaosnli_snli_mnli"]["marginal_counts"])

    strict = np.stack([tilt_to_mean(reflected, m - DELTA) for m in BIN_MEANS])
    lenient = np.stack([tilt_to_mean(native, m + DELTA) for m in BIN_MEANS])
    # Polarized raters sit on the two empirical poles. The low pole is the
    # reflected native_main histogram tilted to 0.5. The high pole is the
    # native_main histogram tilted to 4.5. The bin mean fixes the mixture weight.
    low = tilt_to_mean(reflected, POLE_LOW)
    high = tilt_to_mean(native, POLE_HIGH)
    weight_high = (BIN_MEANS - float(low @ Y)) / (float(high @ Y) - float(low @ Y))
    if np.any(weight_high < 0) or np.any(weight_high > 1):
        raise RuntimeError("polarized mixture weight left [0, 1]")
    polarized = (1.0 - weight_high)[:, None] * low + weight_high[:, None] * high
    chaos = np.stack([tilt_to_mean(chaos_base, m) for m in BIN_MEANS])
    laws = np.stack([strict, lenient, polarized, chaos])
    return {
        "laws": laws,
        "native_template": native,
        "reflected_template": reflected,
        "chaos_transport_template": chaos_base,
        "dices3": dices,
        "chaos3": chaos3,
        "bin_means": BIN_MEANS.copy(),
    }


def native_side_table(pop: dict) -> dict:
    """Tilt the native 3-class marginals to the same means. No 0–5 embedding."""
    targets = np.array([0.7, 1.0, 1.3])
    dices = np.stack([tilt_to_mean(pop["dices3"], m) for m in targets])
    chaos = np.stack([tilt_to_mean(pop["chaos3"], m) for m in targets])
    # Mean-matched binomial on {0,1,2} is not used. The mean-matched forecast
    # here is the ChaosNLI tilt scored against the DICES tilt: same means.
    return {
        "targets": targets,
        "mean_dices": row_means(dices),
        "mean_chaos": row_means(chaos),
        "pearson_means": pearson(row_means(dices), row_means(chaos)),
        "spearman_means": spearman(row_means(dices), row_means(chaos)),
        "tvd_chaos_vs_dices": float(tvd_rows(chaos, dices).mean()),
        "rps_chaos_on_dices": float(expected_rps_rows(chaos, dices).mean()),
        "rps_dices_on_dices": float(expected_rps_rows(dices, dices).mean()),
        "ce_chaos_on_dices": float(expected_ce_rows(chaos, dices).mean()),
        "ce_dices_on_dices": float(expected_ce_rows(dices, dices).mean()),
        "erho2_dices": panel_coeffs(dices)["erho2"],
        "erho2_chaos": panel_coeffs(chaos)["erho2"],
        "var_dices": row_vars(dices),
        "var_chaos": row_vars(chaos),
    }


def moment_panels() -> dict:
    a_hi = LAW_A[::-1].copy()
    b_hi = LAW_B[::-1].copy()
    panel_u = np.stack([LAW_A, M1, a_hi])
    panel_v = np.stack([LAW_B, M2, b_hi])
    cu = panel_coeffs(panel_u)
    cv = panel_coeffs(panel_v)
    return {
        "U": panel_u,
        "V": panel_v,
        "coeffs_U": cu,
        "coeffs_V": cv,
        "tvd": float(tvd_rows(panel_v, panel_u).mean()),
        "rps_U": float(expected_rps_rows(panel_u, panel_u).mean()),
        "rps_V_on_U": float(expected_rps_rows(panel_v, panel_u).mean()),
        "ce_U": float(expected_ce_rows(panel_u, panel_u).mean()),
        "ce_V_on_U_smooth": float(expected_ce_rows(smooth_rows(panel_v), panel_u).mean()),
        "ce_V_on_U_raw": expected_ce_rows(panel_v, panel_u).tolist(),
        "means_U": row_means(panel_u),
        "vars_U": row_vars(panel_u),
        "means_V": row_means(panel_v),
        "vars_V": row_vars(panel_v),
    }


def point_rows(locations: np.ndarray) -> np.ndarray:
    q = np.zeros((locations.shape[0], K), dtype=float)
    q[np.arange(locations.shape[0]), np.clip(locations, 0, K - 1)] = 1.0
    return q


def spearman_locations(means: np.ndarray) -> np.ndarray:
    """Map midranks of the bin means linearly onto {0,1,2,3,4,5}."""
    ranks = rankdata(np.asarray(means, dtype=float), method="average")
    if means.size == 1:
        scaled = np.zeros(1)
    else:
        scaled = (ranks - 1.0) / (means.size - 1.0) * 5.0
    return round_half_up(scaled)


def lenient_locations(means: np.ndarray) -> np.ndarray:
    return np.clip(round_half_up(means) + 1, 0, K - 1)


def add_one(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    width = c.shape[-1]
    return (c + 1.0) / (c.sum(axis=-1, keepdims=True) + width)


def empirical(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    total = c.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("empirical frequency needs at least one rating in every row")
    return c / total


def score_forecast(q: np.ndarray, truth: np.ndarray) -> dict:
    """Population risk of ``q`` under known ``truth``. Both are (bins, K)."""
    rps = expected_rps_rows(q, truth)
    ce = expected_ce_rows(q, truth)
    ce_s = expected_ce_rows(smooth_rows(q), truth)
    tvd = tvd_rows(q, truth)
    mq, mt = row_means(q), row_means(truth)
    coef = panel_coeffs(q)
    joint = joint_components(mt, row_vars(truth), mq, row_vars(q))
    return {
        "rps": float(rps.mean()),
        "ce": None if np.isinf(ce).any() else float(ce.mean()),
        "ce_smooth": float(ce_s.mean()),
        "tvd": float(tvd.mean()),
        "bias_forecast_minus_true": float((mq - mt).mean()),
        "pearson": pearson(mq, mt),
        "spearman": spearman(mq, mt),
        "erho2": coef["erho2"],
        "phi_own": coef["phi_own"],
        "phi_joint_with_truth": joint["phi"],
        "erho2_joint_with_truth": joint["erho2"],
        "forecast_means": mq,
        "true_means": mt,
    }


def oracle_forecasts(pop: dict) -> dict[str, np.ndarray]:
    laws = pop["laws"]
    strict, lenient, polarized, chaos = laws
    train = mix(TRAIN_WEIGHTS, laws)
    shift = mix(SHIFT_WEIGHTS, laws)
    means = row_means(train)
    return {
        "oracle_shift": shift,
        "oracle_train_mix": train,
        "oracle_strict": strict,
        "oracle_lenient": lenient,
        "oracle_polarized": polarized,
        "oracle_chaos": chaos,
        "ce_persona_shift": shift,  # infinite-data persona reweight equals the shift law
        "rps_persona_shift": shift,
        "ce_pooled": train,
        "pearson_binomial": binom_rows(BIN_MEANS),
        "spearman_point": point_rows(spearman_locations(means)),
        "lenient_point": point_rows(lenient_locations(means)),
    }


def persona_phi_choice(laws_or_fit: np.ndarray, target: np.ndarray) -> dict:
    """Pick the persona with the highest own-source Φ and broadcast its law."""
    best_name = None
    best_phi = -1.0
    table = {}
    for i, name in enumerate(PERSONAS):
        coef = panel_coeffs(laws_or_fit[i])
        phi = coef["phi_own"]
        table[name] = phi
        if phi is not None and phi > best_phi + 1e-15:
            best_phi = phi
            best_name = name
    chosen = laws_or_fit[PERSONAS.index(best_name)]
    return {"selected": best_name, "phi_own": table, "forecast": chosen, "target_score": score_forecast(chosen, target)}


def sample_mean_rps(q: np.ndarray, counts: np.ndarray) -> float:
    """Unbiased average of per-rating normalized RPS. ``counts`` is (bins, K) or (K,)."""
    q = np.asarray(q, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
        counts = counts[None, :]
    total = counts.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("a bin with no ratings has no sample RPS")
    per_y = rps_of_outcomes(q)
    return float((counts * per_y).sum() / total.sum())


def rng_for(*parts: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, *parts])))


def draw_quota(rng: np.random.Generator, laws: np.ndarray, n_per_item: int, items: tuple[int, ...]) -> np.ndarray:
    """Quota sample: n_per_item / 4 ratings from each persona, on each listed item.

    Returns counts of shape (persona, bin, class). Items in one bin share a law,
    and the item index only selects which split the draw belongs to.
    """
    if n_per_item % len(PERSONAS) != 0:
        raise ValueError("quota sample needs a multiple of 4 ratings per item")
    n_each = n_per_item // len(PERSONAS)
    counts = np.zeros((len(PERSONAS), N_BINS, K), dtype=int)
    for p in range(len(PERSONAS)):
        for b in range(N_BINS):
            for _item in items:
                counts[p, b] += rng.multinomial(n_each, laws[p, b])
    return counts


def draw_mixture(rng: np.random.Generator, pmf: np.ndarray, n_per_item: int, n_items: int) -> np.ndarray:
    """Draw ``n_items`` independent items from each bin. Returns (n_items, bins, class) counts."""
    out = np.zeros((n_items, pmf.shape[0], K), dtype=int)
    for b in range(pmf.shape[0]):
        for i in range(n_items):
            out[i, b] = rng.multinomial(n_per_item, pmf[b])
    return out


def fit_from_train(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Fit every forecast from training counts only. ``counts`` is (persona, bin, class)."""
    pooled_counts = counts.sum(axis=0)
    ce_p = add_one(counts)
    rps_p = empirical(counts)
    means = row_means(empirical(pooled_counts))
    return {
        "ce_persona_shift": mix(SHIFT_WEIGHTS, ce_p),
        "rps_persona_shift": mix(SHIFT_WEIGHTS, rps_p),
        "ce_pooled": add_one(pooled_counts),
        "ce_strict": ce_p[0],
        "ce_lenient": ce_p[1],
        "ce_polarized": ce_p[2],
        "ce_chaos": ce_p[3],
        "pearson_binomial": binom_rows(means),
        "spearman_point": point_rows(spearman_locations(means)),
        "lenient_point": point_rows(lenient_locations(means)),
        "persona_fits": ce_p,
    }


def rater_ids(split: str, rep: int, n_per_item: int) -> set[str]:
    """Synthetic rater ids. A rater is one draw and never leaves its split."""
    ids = set()
    n_each = n_per_item // len(PERSONAS)
    items = {"train": TRAIN_ITEMS, "val": VAL_ITEMS, "test": TEST_ITEMS}[split]
    for item in items:
        for b in range(N_BINS):
            for p, name in enumerate(PERSONAS):
                for k in range(n_each):
                    ids.add(f"{split}:rep{rep}:bin{b}:item{item}:persona{name}:draw{k}")
    return ids


def leakage_report() -> dict:
    """The fit sees train counts only. Test ids are disjoint. Overwriting a test array does nothing."""
    pop = build_population()
    train = draw_quota(rng_for(1, 16, 0, 1), pop["laws"], 16, TRAIN_ITEMS)
    test = draw_quota(rng_for(1, 16, 0, 3), pop["laws"], 16, TEST_ITEMS)
    fit_a = fit_from_train(train)
    test_copy = test.copy()
    test_copy[:] = 0
    fit_b = fit_from_train(train)
    keys = [k for k in fit_a if k != "persona_fits"]
    unchanged = all(np.array_equal(fit_a[k], fit_b[k]) for k in keys)
    ids_train = rater_ids("train", 0, 16)
    ids_val = rater_ids("val", 0, 16)
    ids_test = rater_ids("test", 0, 16)
    disjoint = ids_train.isdisjoint(ids_val) and ids_train.isdisjoint(ids_test) and ids_val.isdisjoint(ids_test)
    item_split = set(TRAIN_ITEMS).isdisjoint(VAL_ITEMS) and set(TRAIN_ITEMS).isdisjoint(TEST_ITEMS) and set(VAL_ITEMS).isdisjoint(TEST_ITEMS)
    return {
        "fit_ignores_test_array": bool(unchanged and not np.array_equal(test, test_copy)),
        "rater_ids_disjoint_across_splits": bool(disjoint),
        "item_ids_disjoint_across_splits": bool(item_split),
        "n_train_ids": len(ids_train),
        "n_val_ids": len(ids_val),
        "n_test_ids": len(ids_test),
        "train_counts_not_aliased_to_test": not np.shares_memory(train, test),
    }


def t_interval(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    mean = float(x.mean())
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "p": None, "n": n}
    se = float(x.std(ddof=1) / np.sqrt(n))
    crit = float(student_t.ppf(0.975, n - 1))
    if se == 0.0:
        p = None
    else:
        p = float(2 * student_t.sf(abs(mean / se), n - 1))
    return {"mean": mean, "lo": mean - crit * se, "hi": mean + crit * se, "p": p, "n": n}


def percentile_interval(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lo, hi = np.quantile(x, [0.025, 0.975])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return obj
