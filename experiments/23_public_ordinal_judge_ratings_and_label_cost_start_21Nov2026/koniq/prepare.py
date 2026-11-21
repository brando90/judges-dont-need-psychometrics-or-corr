"""Join the original KonIQ-10k indicator, grade-count and split tables by image ID."""
import argparse
from collections import Counter
import csv
import hashlib
import io
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np


SOURCES = {
    "scores.csv": (
        "https://darus.uni-stuttgart.de/api/access/datafile/72288?format=original",
        "a5173fb66831b2049a581d2396ef6e50d341f2af509e1f9b252d4bd49fa03b5f"),
    "indicators.csv": (
        "https://darus.uni-stuttgart.de/api/access/datafile/72287?format=original",
        "2285fedd5ac5d2b1b24656615237b9ca590fab2e12d5b77312baa0479ce0b04d"),
    "authors_splits.csv": (
        "https://raw.githubusercontent.com/subpic/koniq/1c271570ac83aeb6aa8e7ff25449f3299b0d2e53/"
        "metadata/koniq10k_distributions_sets.csv",
        "d0bd1ad54a60bc36fe172049e46ac76c83554e50ab84acebfd47b82b3e698a0a"),
}
FEATURES = ("brightness", "contrast", "colorfulness", "sharpness", "quality_factor", "bitrate", "hxw")
SPLIT_NAMES = {"training": "train", "validation": "validation", "test": "test"}


def load_csv(cache, name):
    url, expected = SOURCES[name]
    path = cache / name
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Version mismatch for {name} at {url}")
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
            stream.write(content)
        Path(stream.name).replace(path)
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def indexed(rows, key):
    index = {key(row): row for row in rows}
    if len(rows) != len(index):
        raise ValueError("A source has duplicate image IDs")
    return index


def prepare(cache, destination):
    cache.mkdir(parents=True, exist_ok=True)
    scores = indexed(load_csv(cache, "scores.csv"), lambda r: r["image_name"])
    indicators = indexed(load_csv(cache, "indicators.csv"), lambda r: r["image_id"] + ".jpg")
    authors = indexed(load_csv(cache, "authors_splits.csv"), lambda r: r["image_name"])
    if not (scores.keys() == indicators.keys() == authors.keys() and len(scores) == 10073):
        raise ValueError("The vote counts, indicators and official split must cover exactly the same images")
    ids = np.asarray(sorted(scores))
    counts = np.asarray([[int(scores[i][f"c{k}"]) for k in range(1, 6)] for i in ids], dtype=np.int64)
    totals = np.asarray([int(scores[i]["c_total"]) for i in ids])
    assert np.all(counts >= 0) and np.all(totals > 0) and np.array_equal(counts.sum(axis=1), totals)
    assert totals.sum() == 1078154
    # These means and the second file's normalized bin masses validate alignment only.
    recovered_mean = (counts @ np.arange(1, 6)) / totals
    np.testing.assert_allclose(recovered_mean, [float(scores[i]["MOS"]) for i in ids], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(totals, [int(authors[i]["c_total"]) for i in ids])
    probs = np.asarray([[float(authors[i][f"c{k}"]) for k in range(1, 6)] for i in ids])
    np.testing.assert_allclose(probs * totals[:, None], counts, rtol=0, atol=1e-8)

    x = np.asarray([[float(indicators[i][key]) for key in FEATURES] for i in ids], dtype=np.float64)
    assert x.shape == (10073, 7) and np.all(np.isfinite(x))
    split = np.asarray([SPLIT_NAMES[authors[i]["set"]] for i in ids])
    assert Counter(split) == {"train": 7058, "validation": 1000, "test": 2015}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        np.savez_compressed(stream, counts=counts, X=x, id=ids, group=ids, split=split,
                            features=np.asarray(FEATURES), rating_values=np.arange(1, 6))
    Path(stream.name).replace(destination)
    print("All 10,073 image IDs join one-to-one. Feature columns:", ", ".join(FEATURES))
    for name in ("train", "validation", "test", "all"):
        vote_counts = totals if name == "all" else totals[split == name]
        print(f"{name}: {len(vote_counts)} images, {vote_counts.sum()} votes, "
              f"min/median/max {vote_counts.min()}/{np.median(vote_counts):g}/{vote_counts.max()}.")
    print("NPZ SHA-256:", hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="External directory for the three pinned public source CSVs.")
    parser.add_argument("output", type=Path, help="Destination for local numeric NPZ.")
    args = parser.parse_args()
    prepare(args.cache, args.output)
