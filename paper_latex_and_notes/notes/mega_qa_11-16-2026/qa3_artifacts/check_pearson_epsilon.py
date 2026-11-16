"""Bounded check of the derivative of the frozen epsilon-regularized Pearson objective.

This checks the gradient at fitted vectors; it does not refit with a new objective,
change any frozen source, or estimate an effect on held-out experimental results.
"""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
from pathlib import Path
import json
import sys
import numpy as np
from scipy.special import softmax

root = Path.cwd()
out = Path(__file__).resolve().parent
sys.path.insert(0, str(next((root / "experiments").glob("08_*")) / "scripts"))
import complexity as cx

toy = cx.toy
rows = []
for scenario, n, seed in [("A", 4, 11), ("B", 512, 17), ("A", 2048, 17), ("D", 2048, 17)]:
    rng = np.random.default_rng(1000003 * seed + 7919 * n + cx.CODE[scenario])
    x = rng.uniform(-1, 1, n)
    P = np.array([rng.multinomial(3, p) for p in cx.truth(x, scenario)]) / 3
    Phi = toy.phi(x)
    W = toy.fit(Phi, P, "pearson", n, toy.marginal_W0(Phi, P, seed))
    Q = softmax(Phi @ W, 1)
    mu, h = Q @ toy.GRID, P @ toy.GRID
    a, b = mu - mu.mean(), h - h.mean()
    norm_a = np.linalg.norm(a)
    na, nb = norm_a + 1e-12, np.linalg.norm(b) + 1e-12
    r = a @ b / (na * nb)
    code_dmu = -(b / nb - r * a / na) / na
    exact_dmu = -b / (na * nb) + r * a / (na * norm_a)
    def weight_gradient(dmu):
        dZ = Q * (toy.GRID[None] - mu[:, None]) * dmu[:, None]
        return Phi.T @ dZ + (2 / n) * W
    code, exact = weight_gradient(code_dmu), weight_gradient(exact_dmu)
    rows.append({
        "scenario": scenario, "n": n, "seed": seed,
        "centered_mean_norm": float(norm_a),
        "max_abs_weight_gradient_discrepancy": float(np.max(np.abs(code - exact))),
        "code_gradient_norm": float(np.linalg.norm(code)),
        "true_gradient_norm": float(np.linalg.norm(exact)),
    })
receipt = out / "pearson_epsilon_check.json"
previous = json.loads(receipt.read_text())
for before, now in zip(previous, rows):
    for key, value in now.items():
        if isinstance(value, float):
            assert np.isclose(value, before[key], rtol=1e-6, atol=1e-14), (key, before[key], value)
        else:
            assert value == before[key]
receipt.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
