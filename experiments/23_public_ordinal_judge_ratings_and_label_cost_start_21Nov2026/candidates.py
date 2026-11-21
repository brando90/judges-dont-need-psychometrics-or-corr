"""Fixed probability-forecast families and objectives used in the exact search."""
import itertools
import numpy as np

from design import standardize

R = np.arange(1., 6.)
U = (R-3)/2
EPS = 0.000006
TOL = 1e-10


def normalize(logits):
    e = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def build(data, mode):
    if mode == "native":
        p = np.asarray(data.native, dtype=float)
        assert p.shape == data.counts.shape and (p >= 0).all()
        np.testing.assert_allclose(p.sum(axis=-1), 1)
        p = (1-EPS)*p + EPS/5
        params = sorted(itertools.product((.5, 1., 2.), (-1, 0, 1), (-1, 0, 1)),
                        key=lambda t: (np.log2(t[0])**2+t[1]**2+t[2]**2, t))
        forecasts = [normalize(t*np.log(p) + b*U + c*U**2) for t,b,c in params]
        names = [f"native:t={t},b={b},c={c}" for t,b,c in params]
    else:
        x = standardize(data.X, data.train)
        _, _, vt = np.linalg.svd(x[data.train]-x[data.train].mean(axis=0), full_matrices=False)
        loadings = vt[:2].copy()
        for loading in loadings:
            if loading[np.argmax(abs(loading))] < 0:
                loading *= -1
        pc = standardize(x @ loadings.T, data.train)
        z = np.column_stack([x, pc])
        features = list(data.features) + ["PC1","PC2"]
        params = list(itertools.product(range(z.shape[1]), (1,3,5), (-1.5,-.75,.75,1.5), (.65,1.25)))
        reference = (0,3,.75,1.25)
        params.remove(reference)
        params.insert(0, reference)
        forecasts, names = [], []
        for j,a,b,s in params:
            p = normalize(-.5*((R[None,:]-a-b*z[:,j,None])/s)**2)
            forecasts.append((1-EPS)*p + EPS/5)
            names.append(f"{features[j]}:center={a},slope={b},width={s}")
    forecasts.append(np.full((len(data.id),5), .2))
    names.append("uniform")
    return np.stack(forecasts, axis=1), names


def per_item(q, h):
    """q=(item, grade), h=row-normalized observed grade histogram."""
    f, g = np.cumsum(q,axis=-1)[...,:-1], np.cumsum(h,axis=-1)[...,:-1]
    return np.stack(((f*f + g*(1-2*f)).mean(axis=-1), -(h*np.log(q)).sum(axis=-1),
                     abs(h-q).sum(axis=-1)/2, 1+(q*q-2*h*q).sum(axis=-1)),axis=-1)


def loss_by_grade(q):
    """Shape (item, realized grade, candidate). Precompute individual CE and RPS."""
    f = q.cumsum(-1)[...,:-1]
    qlog = -np.log(q).transpose(0,2,1)
    scores = [np.square(f - (grade <= np.arange(1,5))).mean(axis=-1) for grade in R]
    return np.ascontiguousarray(qlog), np.ascontiguousarray(np.stack(scores,axis=1))


def best(values, maximize=False):
    if maximize:
        return int(np.flatnonzero(values >= values.max()-TOL)[0])
    return int(np.flatnonzero(values <= values.min()+TOL)[0])


def pearson_from_totals(m1, m2, my, y1, y2, n):
    vy = y2 - y1*y1/n
    vm = np.maximum(0, m2-m1*m1/n)
    valid = vm > 1e-10*n
    if vy <= 1e-10*n or not valid.any():
        return 0, True
    rho = np.full(len(m1), -np.inf)
    rho[valid] = (my[valid]-m1[valid]*y1/n)/np.sqrt(vm[valid]*vy)
    return best(rho, maximize=True), False


def pearson_for_pair(candidate_means, observed_means):
    """With two observations every eligible Pearson score is exactly +1 or -1."""
    predicted_difference = candidate_means[1] - candidate_means[0]
    observed_difference = observed_means[1] - observed_means[0]
    # For two equally weighted observations the population variance is (difference/2)^2.
    eligible = (predicted_difference / 2)**2 > TOL
    if (observed_difference / 2)**2 <= TOL or not eligible.any():
        return 0, True
    scores = np.full(len(predicted_difference), -np.inf)
    scores[eligible] = np.sign(predicted_difference[eligible]) * np.sign(observed_difference)
    return best(scores, maximize=True), False


def bin_indices(train_means, all_means):
    cuts = np.unique(np.quantile(train_means, (.25,.5,.75)))
    return np.searchsorted(cuts, all_means, side="right")


def bin_distribution(bins, order, picked, prior):
    totals = np.zeros((4,5))
    np.add.at(totals, bins[order], picked)
    return (totals+8*prior)/(totals.sum(axis=1,keepdims=True)+8)
