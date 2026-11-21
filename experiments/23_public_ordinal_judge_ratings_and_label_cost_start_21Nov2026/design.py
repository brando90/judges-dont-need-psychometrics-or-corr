"""Load pinned five-grade study inputs and generate nested, matched training labels."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass
class Data:
    name: str
    X: np.ndarray
    counts: np.ndarray
    group: np.ndarray
    id: np.ndarray
    features: np.ndarray
    system: np.ndarray | None
    native: np.ndarray | None
    train: np.ndarray
    test: np.ndarray


def load(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        a = {key: f[key] for key in f.files}
    counts = a["counts"]
    assert counts.ndim == 2 and counts.shape[1] == 5 and np.all(counts >= 0)
    assert np.array_equal(counts, counts.astype(int)) and np.all(counts.sum(1) >= 3)
    assert len(np.unique(a["id"])) == len(counts) == len(a["X"])
    assert np.isfinite(a["X"]).all()
    if "split" in a:
        train = np.where(a["split"] == "train")[0]
        test = np.where(a["split"] == "test")[0]
        assert (len(train), len(test)) == (7058, 2015)
    else:
        groups = sorted(np.unique(a["group"]), key=lambda g: sha256(("corrce23-split|"+str(g)).encode()).digest())
        assert len(groups) == 100
        held = np.isin(a["group"], groups[:40])
        train, test = np.where(~held)[0], np.where(held)[0]
        assert (len(train), len(test)) == (960, 640)
    assert not set(a["group"][train]).intersection(a["group"][test])
    return Data(path.stem, a["X"], counts.astype(int), a["group"], a["id"], a["features"],
                a.get("system"), a.get("native"), train, test)


def draw(data, rep, depth):
    """Select distinct items and individual votes without replacement."""
    assert depth in (1, 3)
    seed = np.random.SeedSequence([20261121, 23, rep])
    order = np.random.default_rng(seed).permutation(data.train)
    rng = np.random.default_rng(np.random.SeedSequence([20261121, 23, rep, 1]))
    left = data.counts[order].copy()
    picked = np.zeros_like(left)
    row = np.arange(len(order))
    for _ in range(depth):
        ticket = rng.integers(left.sum(axis=1))
        col = (left.cumsum(axis=1) <= ticket[:, None]).sum(axis=1)
        picked[row, col] += 1
        left[row, col] -= 1
    assert np.array_equal(picked.sum(axis=1), np.full(len(order), depth))
    return order, picked


def standardize(x, train):
    scale = x[train].std(axis=0)
    return np.clip((x-x[train].mean(axis=0))/np.where(scale > 1e-12, scale, 1), -3, 3)
