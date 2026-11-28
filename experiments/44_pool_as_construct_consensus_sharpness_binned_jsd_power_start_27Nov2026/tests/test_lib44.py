"""Deterministic checks for Experiment 44 helpers (fixed seeds, named tolerances)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lib44 as L  # noqa: E402
import run_all as R  # noqa: E402

TOL = 1e-12


def rand_q(rng, n, K):
    return (1 - K * L.FLOOR) * rng.dirichlet(np.ones(K), n) + L.FLOOR


def test_score_values_by_hand():
    q = np.array([0.7, 0.2, 0.1])
    assert L.score_matrix(q, "brier")[0] == pytest.approx(0.09 + 0.04 + 0.01)
    assert L.score_matrix(q, "brier")[2] == pytest.approx(0.49 + 0.04 + 0.81)
    nr = L.score_matrix(q, "nrps")
    assert nr[0] == pytest.approx(((0.7 - 1) ** 2 + (0.9 - 1) ** 2) / 2)
    assert nr[2] == pytest.approx((0.7 ** 2 + 0.9 ** 2) / 2)
    assert L.score_matrix(q, "ce")[0] == pytest.approx(-np.log(0.7))


def test_nrps_for_two_classes_is_half_brier():
    rng = np.random.default_rng(1)
    q = rand_q(rng, 50, 2)
    assert np.allclose(L.score_matrix(q, "nrps") * 2, L.score_matrix(q, "brier"), atol=TOL)


@pytest.mark.parametrize("kind", ["brier", "nrps", "ce"])
def test_strict_propriety_random_forecasts(kind):
    rng = np.random.default_rng(2)
    p = np.array([0.5, 0.3, 0.2])
    best = float((p * L.score_matrix(p, kind)).sum())
    for q in rng.dirichlet(np.ones(3), 2000):
        assert float((p * L.score_matrix(q, kind)).sum()) > best - TOL
    assert min(float((p * L.score_matrix(q, kind)).sum()) for q in rng.dirichlet(np.ones(3), 2000)) > best


def test_one_rating_per_item_estimator_is_exactly_unbiased():
    rng = np.random.default_rng(3)
    K, n = 3, 3
    p = rng.dirichlet(np.ones(K), n)
    q = rand_q(rng, n, K)
    S = L.score_matrix(q, "nrps")
    exact = np.mean([(p[i] * S[i]).sum() for i in range(n)])
    mean_est = 0.0
    for ys in itertools.product(range(K), repeat=n):
        prob = np.prod([p[i, ys[i]] for i in range(n)])
        mean_est += prob * np.mean([S[i, ys[i]] for i in range(n)])
    assert mean_est == pytest.approx(exact, abs=TOL)
    assert L.risk_items(q, p * 1000.0, "nrps").mean() == pytest.approx(exact, abs=1e-9)


def test_temperature_keeps_argmax_and_identity_is_near_floor_only():
    rng = np.random.default_rng(4)
    q = rand_q(rng, 1000, 3)
    for t in (0.5, 0.75, 1.0, 1.5, 2.0):
        assert np.array_equal(L.temper(q, t).argmax(-1), q.argmax(-1))
    assert np.allclose(L.temper(q, 1.0), q, atol=1e-5)
    assert (L.temper(q, 0.5).max(-1) >= q.max(-1) - 1e-5).all()


def test_jsd_basic_values():
    p = np.array([0.2, 0.8, 0.0])
    assert L.jsd2(p, p) == pytest.approx(0, abs=TOL)
    assert L.jsd2(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(1.0, abs=TOL)
    q = np.array([0.5, 0.25, 0.25])
    assert L.jsd2(p, q) == pytest.approx(L.jsd2(q, p), abs=TOL)


def toy(N=60, K=3, n=20, seed=5):
    rng = np.random.default_rng(seed)
    ratings = np.stack([rng.choice(K, size=n, p=rng.dirichlet(np.ones(K))) for _ in range(N)])
    counts = np.stack([np.bincount(r, minlength=K) for r in ratings])
    P = rand_q(rng, N * 5, K).reshape(N, 5, K)
    return dict(K=K, ids=np.arange(N), counts=counts, ratings=ratings, n_avail=np.full(N, n), rater_ids=None), P


def test_jsb_boot_matches_direct_and_is_permutation_invariant():
    d, P = toy()
    counts, q = d["counts"].astype(float), P[:, 0]
    bins = L.human_bins(d["counts"], True)
    direct = L.jsb(counts, q, bins)
    assert L.jsb_boot(counts, q, bins, np.ones((1, len(q))))[0] == pytest.approx(direct, abs=TOL)
    rng = np.random.default_rng(6)
    idx = rng.integers(0, len(q), len(q))
    W = np.bincount(idx, minlength=len(q))[None].astype(float)
    assert L.jsb_boot(counts, q, bins, W)[0] == pytest.approx(L.jsb(counts[idx], q[idx], bins[idx]), abs=TOL)
    q2 = q.copy()
    for b in np.unique(bins):
        ii = np.where(bins == b)[0]
        q2[ii] = q[rng.permutation(ii)]
    assert abs(L.jsb(counts, q2, bins) - direct) < TOL


def test_human_bins_conventions():
    nominal = L.human_bins(np.array([[3, 0, 0], [1, 2, 0], [0, 1, 2], [2, 2, 0]]), False)
    assert nominal.tolist() == [0, 1, 2, 0]  # ties go to the lowest index
    ordinal = L.human_bins(np.array([[3, 0, 0], [1, 2, 0], [0, 1, 2], [2, 2, 0]]), True)
    assert ordinal.tolist() == [0, 1, 2, 0]  # lower median sorted[(n-1)//2]: [0,1,1]->1, [1,2,2]->2, [0,0,1,1]->0


def test_holm_and_boot_p():
    assert np.allclose(L.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert L.boot_p(np.ones(99)) == pytest.approx(2 * 1 / 100)
    assert L.boot_p(np.array([-1.0, 1.0] * 50)) == 1.0


def test_mdd_and_exact_subsampling_variance():
    D = np.array([0.0, 2.0, 0.0, 2.0]); s2 = np.ones(4); n = np.full(4, 2.0)
    sb2, sw2 = L.variance_components(D, s2, n)
    assert sb2 == pytest.approx(4 / 3 - 0.5) and sw2 == pytest.approx(1.0)
    assert L.mdd(sb2, sw2, 100, 1) == pytest.approx(L.Z80 * np.sqrt((sb2 + sw2) / 100))
    # exact enumeration: 4 items x 2 ratings each, draw 2 items and 1 rating per item
    vals = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [0.0, 4.0]])
    Dm = vals.mean(1); s2m = vals.var(1, ddof=1); nn = np.full(4, 2.0)
    ests = [(vals[i, a] + vals[j, b]) / 2 for i, j in itertools.combinations(range(4), 2) for a in range(2) for b in range(2)]
    assert np.var(ests) == pytest.approx(L.v_sub(Dm, s2m, nn, 2, 1), abs=TOL)


def test_pool_draw_risks_matches_manual():
    rng = np.random.default_rng(7)
    Ne, nA, nB, J = 6, 5, 4, 2
    mA = (rng.random((Ne, nA)) < 0.8).astype(float); mB = (rng.random((Ne, nB)) < 0.8).astype(float)
    LA = [sparse.csr_matrix(mA * rng.random((Ne, nA))) for _ in range(J)]
    LB = [sparse.csr_matrix(mB * rng.random((Ne, nB))) for _ in range(J)]
    vA = rng.multinomial(nA, np.full(nA, 1 / nA), size=3).T.astype(float)
    vB = rng.multinomial(nB, np.full(nB, 1 / nB), size=3).T.astype(float)
    W = rng.multinomial(Ne, np.full(Ne, 1 / Ne), size=3).T.astype(float)
    ra, rb = L.pool_draw_risks(LA, sparse.csr_matrix(mA), LB, sparse.csr_matrix(mB), vA, vB, W)
    for c in range(3):
        for j in range(J):
            num_a, num_b, tot = 0.0, 0.0, 0.0
            for i in range(Ne):
                da, db = (mA[i] * vA[:, c]).sum(), (mB[i] * vB[:, c]).sum()
                if da > 0 and db > 0:
                    num_a += W[i, c] * (LA[j].toarray()[i] * vA[:, c]).sum() / da
                    num_b += W[i, c] * (LB[j].toarray()[i] * vB[:, c]).sum() / db
                    tot += W[i, c]
            assert ra[c, j] == pytest.approx(num_a / tot, abs=TOL) and rb[c, j] == pytest.approx(num_b / tot, abs=TOL)


def test_pool_counts_and_tvd():
    item = np.array([0, 0, 0, 0, 1, 1, 1, 1]); y = np.array([0, 0, 1, 1, 2, 2, 2, 1])
    lab = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    cnt = L.pool_item_counts(item, lab, y, 2, 3)
    assert cnt[0, 0].tolist() == [2, 0, 0] and cnt[0, 1].tolist() == [0, 2, 0]
    t, elig = L.mean_tvd(cnt, min_raters=2)
    assert t == pytest.approx((1.0 + 0.5) / 2) and elig.all()


def test_arms_s_j_m_are_deterministic_on_toy_data():
    d, P = toy()
    a1 = R.arm_s("dices350", d, P, 3, 50); a2 = R.arm_s("dices350", d, P, 3, 50)
    assert a1 == a2
    assert all(r["argmax_unchanged"] for r in a1[1])
    j1 = R.arm_j("dices350", d, P, 3, 50, nperm=5); j2 = R.arm_j("dices350", d, P, 3, 50, nperm=5)
    assert j1 == j2 and max(r["max_abs_jsb_change"] for r in j1[1]) < TOL
    m1 = R.arm_m("dices350", d, P, 3, nsub=200); m2 = R.arm_m("dices350", d, P, 3, nsub=200)
    assert m1 == m2


def test_hm2_iso_budget_identity_is_nonnegativity_of_sigma_b2():
    rng = np.random.default_rng(8)
    for _ in range(200):
        sb2, sw2 = rng.random() * 0.1, rng.random() * 0.2
        for B in (120, 400, 1200):
            seq = [(sb2 * k + sw2) / B for k in (1, 2, 5, 10)]
            assert all(a <= b for a, b in zip(seq, seq[1:]))
    # the estimator never returns a negative between-item component
    D = np.array([1.0, 1.0, 1.0, 1.0]); s2 = np.full(4, 4.0); n = np.full(4, 2.0)
    assert L.variance_components(D, s2, n)[0] == 0.0


def test_uniform_forecast_reference_values():
    """Reference values quoted in RESULTS.md: a uniform forecast's score for each rating."""
    q3 = np.full(3, 1 / 3)
    assert np.allclose(L.score_matrix(q3, "brier"), 2 / 3, atol=TOL)
    assert np.allclose(L.score_matrix(q3, "nrps"), [5 / 18, 1 / 9, 5 / 18], atol=TOL)
    assert np.allclose(L.score_matrix(np.full(2, 0.5), "brier"), 0.5, atol=TOL)
