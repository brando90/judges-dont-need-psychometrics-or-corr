"""Test-set power and minimum detectable difference on DICES-990.

The two judges are the cross-entropy head and the Pearson head already defined
in fit_eval.py, trained once on the same prompt split (seed 0). The sweep only
changes how many held-out prompts and how many ratings per prompt are used to
score them.

For a proper score, the difference is the Pearson-trained head minus the
cross-entropy-trained head. Positive means the cross-entropy head is better.
For Pearson correlation, the difference is the cross-entropy head's r minus
the Pearson head's r. Positive means the cross-entropy head correlates more
with the held-out ratings.

A design (N_test, k_test) is 200 random subsets. Each subset gets an
item-clustered percentile 95% interval from 200 resamples of its prompts.
Power is the fraction of subsets whose interval excludes 0. The minimum
detectable difference is the median half-width of those intervals: the
smallest absolute difference this design's 95% interval would exclude 0
if the point estimate equalled that difference.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_io import _load_dices
from fit_eval import fit_objective, predict_distributions
from metrics import brier_one, ce_one, pearson_from_sums, rps_one
from run_all import prepare

HERE = Path(__file__).resolve().parent
N_GRID = (25, 50, 100, 150, 198)
K_GRID = (1, 2, 4, 8, 16, 32)
N_OUTER = 200
N_BOOT = 200
SEED = 11


def _item_losses(q, ratings, values):
    ce = np.empty(ratings.size)
    br = np.empty(ratings.size)
    rp = np.empty(ratings.size)
    for j, y in enumerate(ratings):
        ce[j] = ce_one(q, int(y))
        br[j] = brier_one(q, int(y))
        rp[j] = rps_one(q, int(y))
    return ce, br, rp, values[ratings]


def _subset_metrics(ce_a, ce_b, br_a, br_b, rp_a, rp_b, y, s_a, s_b, offsets, k):
    """Prompt-mean losses and correlation stats for one k and one offset per item."""
    n = len(ce_a)
    m_ce = np.empty(n)
    m_br = np.empty(n)
    m_rp = np.empty(n)
    # sufficient stats for each scalar: n, sum y, sum y2, sum s, sum s2, sum sy
    stats_a = np.empty((n, 6))
    stats_b = np.empty((n, 6))
    for i in range(n):
        ni = ce_a[i].size
        idx = (offsets[i] + np.arange(k)) % ni
        ya = y[i][idx]
        m_ce[i] = ce_b[i][idx].mean() - ce_a[i][idx].mean()
        m_br[i] = br_b[i][idx].mean() - br_a[i][idx].mean()
        m_rp[i] = rp_b[i][idx].mean() - rp_a[i][idx].mean()
        sy = float(ya.sum())
        sy2 = float(np.dot(ya, ya))
        for stats, s in ((stats_a, s_a[i]), (stats_b, s_b[i])):
            stats[i] = (k, sy, sy2, s * k, (s * s) * k, s * sy)
    return m_ce, m_br, m_rp, stats_a, stats_b


def _corr_delta(stats_a, stats_b, idx):
    def r_of(stats):
        tot = stats[idx].sum(axis=0)
        return pearson_from_sums(*tot)

    ra, rb = r_of(stats_a), r_of(stats_b)
    if not np.isfinite(ra) or not np.isfinite(rb):
        return np.nan
    return ra - rb


def _power_one(diff_items, stats_a, stats_b, n_take, rng, proper=True):
    n_items = diff_items.shape[0]
    outer = rng.integers(0, n_items, size=(N_OUTER, n_take))
    # inner indexes into the N columns of each outer row
    inner = rng.integers(0, n_take, size=(N_OUTER, N_BOOT, n_take))
    gathered = np.take_along_axis(outer[:, None, :], inner, axis=2)
    if proper:
        # sqrt(RPS) is handled by the caller via raw RPS means, not this branch.
        vals = diff_items[gathered]
        boot = vals.mean(axis=2)
        point = diff_items[outer].mean(axis=1)
    else:
        point = np.empty(N_OUTER)
        boot = np.empty((N_OUTER, N_BOOT))
        for r in range(N_OUTER):
            point[r] = _corr_delta(stats_a, stats_b, outer[r])
            for b in range(N_BOOT):
                boot[r, b] = _corr_delta(stats_a, stats_b, gathered[r, b])
    lo = np.nanquantile(boot, 0.025, axis=1)
    hi = np.nanquantile(boot, 0.975, axis=1)
    finite = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(point)
    separates = finite & ((lo > 0) | (hi < 0))
    power = float(separates.sum() / max(finite.sum(), 1))
    half = 0.5 * (hi - lo)
    half = half[np.isfinite(half)]
    mdd = float(np.median(half)) if half.size else float("nan")
    return {
        "power": power,
        "mdd": mdd,
        "median_difference": float(np.nanmedian(point)),
        "n_finite_subsets": int(finite.sum()),
    }


def _sqrt_rps_power(rp_a, rp_b, offsets, k, n_take, rng):
    """Difference of square roots of mean raw RPS. Positive means the Pearson head is worse."""
    n = len(rp_a)
    mean_a = np.empty(n)
    mean_b = np.empty(n)
    for i in range(n):
        idx = (offsets[i] + np.arange(k)) % rp_a[i].size
        mean_a[i] = rp_a[i][idx].mean()
        mean_b[i] = rp_b[i][idx].mean()
    outer = rng.integers(0, n, size=(N_OUTER, n_take))
    inner = rng.integers(0, n_take, size=(N_OUTER, N_BOOT, n_take))
    gathered = np.take_along_axis(outer[:, None, :], inner, axis=2)

    def delta(idx_rows):
        # idx_rows (R, N) or we call per row
        return np.sqrt(mean_b[idx_rows].mean(axis=1)) - np.sqrt(mean_a[idx_rows].mean(axis=1))

    point = delta(outer)
    boot = np.empty((N_OUTER, N_BOOT))
    for b in range(N_BOOT):
        boot[:, b] = delta(gathered[:, b, :])
    lo = np.quantile(boot, 0.025, axis=1)
    hi = np.quantile(boot, 0.975, axis=1)
    separates = (lo > 0) | (hi < 0)
    half = 0.5 * (hi - lo)
    return {
        "power": float(np.mean(separates)),
        "mdd": float(np.median(half)),
        "median_difference": float(np.median(point)),
        "n_finite_subsets": N_OUTER,
    }


def _style(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)


def _figure(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    colors = {"ce": "#0072B2", "brier": "#E69F00", "sqrt_rps": "#009E73", "pearson_r": "#D55E00"}
    for ax, k, title in (
        (axes[0], 1, "One test rating per prompt"),
        (axes[1], 8, "Eight test ratings per prompt"),
    ):
        for metric, label in (
            ("ce", "Cross-entropy"),
            ("brier", "Brier"),
            ("sqrt_rps", "sqrt(RPS)"),
            ("pearson_r", "Pearson r"),
        ):
            sub = [r for r in rows if r["k_test"] == k and r["metric"] == metric]
            sub = sorted(sub, key=lambda r: r["n_test"])
            ax.plot(
                [r["n_test"] for r in sub],
                [r["power"] for r in sub],
                "o-",
                color=colors[metric],
                label=label,
            )
        ax.axhline(0.80, color="0.4", linestyle=":", linewidth=1, label="Power 0.80")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Held-out test prompts N_test")
        ax.set_ylabel("Power to exclude 0 at 95% (item-clustered)")
        ax.set_title(title)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(
        "DICES-990: power to separate the CE-trained head from the Pearson-trained head",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _smallest(rows, metric, k, threshold=0.80):
    hits = [r for r in rows if r["metric"] == metric and r["k_test"] == k and r["power"] >= threshold]
    if not hits:
        return None
    return min(hits, key=lambda r: (r["n_test"], r["k_test"]))


def main():
    data = _load_dices("990")
    prep = prepare(data)
    packs = {}
    for obj in ("ce", "pearson"):
        packs[obj] = fit_objective(
            prep["x_train"], data.ratings, prep["train"], data.class_values, obj, None, seed=0,
        )
    probs = {}
    mus = {}
    for obj, pack in packs.items():
        prob, mu = predict_distributions(pack, prep["x_test"], obj)
        probs[obj] = prob
        mus[obj] = mu
    test = prep["test"]
    values = data.class_values
    ce_a, br_a, rp_a, y = [], [], [], []
    ce_b, br_b, rp_b = [], [], []
    for row, i in enumerate(test):
        ratings = np.asarray(data.ratings[i], dtype=int)
        a = _item_losses(probs["ce"][row], ratings, values)
        b = _item_losses(probs["pearson"][row], ratings, values)
        ce_a.append(a[0])
        br_a.append(a[1])
        rp_a.append(a[2])
        y.append(a[3])
        ce_b.append(b[0])
        br_b.append(b[1])
        rp_b.append(b[2])
    s_a = mus["ce"]
    s_b = mus["pearson"]
    rng = np.random.default_rng(SEED)
    # One offset per item, reused across N so k-subsamples share a rating draw.
    offsets = np.array([rng.integers(0, ce_a[i].size) for i in range(len(ce_a))])
    rows = []
    for k in K_GRID:
        m_ce, m_br, _m_rp, stats_a, stats_b = _subset_metrics(
            ce_a, ce_b, br_a, br_b, rp_a, rp_b, y, s_a, s_b, offsets, k,
        )
        for n_take in N_GRID:
            if n_take > len(ce_a):
                continue
            for metric, vec, proper in (
                ("ce", m_ce, True),
                ("brier", m_br, True),
            ):
                cell_rng = np.random.default_rng(SEED + 1000 * k + n_take + (0 if metric == "ce" else 17))
                out = _power_one(vec, stats_a, stats_b, n_take, cell_rng, proper=True)
                rows.append({"metric": metric, "k_test": k, "n_test": n_take, **out})
            cell_rng = np.random.default_rng(SEED + 1000 * k + n_take + 29)
            out = _sqrt_rps_power(rp_a, rp_b, offsets, k, n_take, cell_rng)
            rows.append({"metric": "sqrt_rps", "k_test": k, "n_test": n_take, **out})
            cell_rng = np.random.default_rng(SEED + 1000 * k + n_take + 41)
            out = _power_one(m_ce, stats_a, stats_b, n_take, cell_rng, proper=False)
            rows.append({"metric": "pearson_r", "k_test": k, "n_test": n_take, **out})
            print(
                f"k={k} N={n_take} "
                + " ".join(
                    f"{r['metric']} p={r['power']:.2f} mdd={r['mdd']:.4f}"
                    for r in rows[-4:]
                ),
                flush=True,
            )
    summary = {}
    for metric in ("ce", "brier", "sqrt_rps", "pearson_r"):
        summary[metric] = {
            "smallest_n_at_k1_power80": _smallest(rows, metric, 1),
            "smallest_n_at_k8_power80": _smallest(rows, metric, 8),
        }
    payload = {
        "dataset": "dices990",
        "n_test_available": int(len(test)),
        "n_outer_subsets": N_OUTER,
        "n_inner_bootstrap": N_BOOT,
        "interval": "item-clustered percentile 95%",
        "difference_proper": "Pearson-trained minus CE-trained; positive means the CE-trained head is better",
        "difference_correlation": "CE-trained r minus Pearson-trained r; positive means the CE-trained head correlates more",
        "judges": "same split and head as the experiment 36 matrix, refit here",
        "rows": rows,
        "summary": summary,
    }
    out_json = HERE / "results_power_mdd.json"
    out_json.write_text(json.dumps(payload, indent=2))
    _figure(rows, HERE / "figures" / "eval_power_mdd_dices990.png")
    lines = [
        "| k_test | N_test | metric | power | MDD | median difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['k_test']} | {r['n_test']} | {r['metric']} | {r['power']:.3f} | {r['mdd']:.4f} | {r['median_difference']:.4f} |"
        )
    (HERE / "tables" / "eval_power_mdd_dices990.md").write_text("\n".join(lines) + "\n")
    print("wrote", out_json)


if __name__ == "__main__":
    main()
