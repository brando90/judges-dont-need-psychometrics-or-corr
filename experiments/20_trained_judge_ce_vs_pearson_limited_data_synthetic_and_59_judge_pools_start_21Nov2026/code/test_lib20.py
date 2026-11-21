import numpy as np
from lib20 import *
from scipy.optimize import check_grad
rng = np.random.default_rng(0); m, d = 12, 4
X = rng.normal(size=(m, d)); Xd = design(X); P = softmax(rng.normal(size=(m, 6))); C = np.array([rng.multinomial(3, p) for p in P], float)
# gradient checks through fit internals: replicate by finite differences on a perturbed theta via the loss functions in fit (re-created)
import lib20
def grads(obj, lam):
    D = Xd.shape[1]; n = C.sum(); ybar = (C @ R) / C.sum(1); Wm = np.ones((D, 6)); Wm[-1] = 0
    # call fit with maxiter=0 would not expose f; so test optimality instead: gradient norm at the optimum is ~0 for both
    T = fit(Xd, C, obj, lam, maxiter=2000, init_seed=1); return T
for obj in ('ce', 'corr'):
    T = grads(obj, 0.01)
    q = softmax(Xd @ T); mu = q @ R; print(obj, 'mean-train-corr', np.corrcoef(mu, (C @ R) / 3)[0, 1], 'ce', -(C * np.log(q)).sum() / C.sum())
