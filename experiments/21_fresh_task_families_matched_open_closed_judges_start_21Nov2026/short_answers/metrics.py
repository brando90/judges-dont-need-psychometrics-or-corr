"""Six-category proper scores, sparse-histogram distance, and the shared calibration grid."""
import itertools
import numpy as np

GRADE = np.arange(6)
COEFFICIENTS = sorted(itertools.product((0.5, 1.0, 2.0), (-1, 0, 1), (-1, 0, 1)),
                      key=lambda v: (np.log2(v[0])**2 + v[1]**2 + v[2]**2, v))
METRICS = ("rps", "ce", "tvd")
TOL = 1e-12


def calibrations(q):
    """With axes [artifact, judge, transform, category], identity is transform zero."""
    p = (1 - 6e-6) * q + 1e-6
    t, b, c = np.asarray(COEFFICIENTS).T
    u = (GRADE - 2.5) / 2.5
    logits = np.log(p)[..., None, :] * t[:, None] + b[:, None] * u + c[:, None] * u**2
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponent / exponent.sum(axis=-1, keepdims=True)


def rating_histograms(ratings):
    return (np.asarray(ratings)[..., None] == GRADE).mean(axis=-2)


def score(q, h):
    """q and h may broadcast. RPS and CE average individual ratings encoded by h."""
    f = q.cumsum(axis=-1)[..., :-1]
    g = h.cumsum(axis=-1)[..., :-1]
    rps = (f*f + g*(1 - 2*f)).mean(axis=-1)
    ce = -(h*np.log(q)).sum(axis=-1)
    tvd = np.abs(h-q).sum(axis=-1)/2
    return np.stack((rps, ce, tvd), axis=-1)


def self_check():
    rng = np.random.default_rng(99)
    p = rng.dirichlet(np.ones(6), size=36).reshape(18, 2, 6)
    q = calibrations(p)
    assert COEFFICIENTS[0] == (1, 0, 0) and q.shape == (18, 2, 27, 6)
    np.testing.assert_allclose(q[:, :, 0], (1 - 6e-6)*p + 1e-6)
    y = rng.integers(0, 6, (18, 2))
    h = rating_histograms(y)
    by_hist = score(p, h[:, None, :])
    for i in range(18):
        for j in range(2):
            observed = y[i]
            f = p[i, j].cumsum()[:-1]
            direct = np.mean((f - (observed[:, None] <= np.arange(5)))**2)
            np.testing.assert_allclose(by_hist[i, j, :2], [direct, -np.log(p[i, j, observed]).mean()])
            assert abs(by_hist[i, j, 2] - np.linalg.norm(h[i]-p[i, j], ord=1)/2) < 1e-12
    print("Direct per-rating scores and identity verified on 36 forecasts.")


if __name__ == "__main__":
    self_check()
