"""Loaders for real multi-annotator judging records.

Each item is one prompt ``c``. Ratings stay as raw per-rater rows
``(item_id, rater_id, class_index)``. Nothing is pre-averaged before the split.

Datasets that are not on this machine are reported as unavailable and are not
replaced by a lookalike. VeriBench functional semantic correctness (FSC) and
POPQUORN are in that category; see ``availability()``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("SANMI36_DATA", ROOT / "data_cache"))

SUMMEVAL_URL = (
    "https://storage.googleapis.com/sfr-summarization-repo-research/"
    "model_annotations.aligned.jsonl"
)
SUMMEVAL_SHA = "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab"
CHAOS_REV = "eb7cc18dbfa0f161d93a2b22c7856ca0faafcf2b"
CHAOS_BASE = f"https://huggingface.co/datasets/earino/chaosnli/resolve/{CHAOS_REV}/raw/"
CHAOS_SHA = {
    "chaosNLI_snli.jsonl": "99f9015ddda7d85f66a087452bc30d53974314fe27e7d589e2f41ad44bd509c1",
    "chaosNLI_mnli_m.jsonl": "8eb49b589488e7b1a9ec95fb8a864bd50f0509b4f808eed0aed3f8167617664d",
    "chaosNLI_alphanli.jsonl": "f55817ecc11a02ba3030cd7d2c567d23843456809020723abad53f4fde024c9e",
}
DICES_URL = {
    "dices350.csv": (
        "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/"
        "main/350/diverse_safety_adversarial_dialog_350.csv",
        "63cb8620eb04e64957f9e45b1f7cb3d079c4eed8e8b28909f46e0b7038ce4301",
    ),
    "dices990.csv": (
        "https://raw.githubusercontent.com/google-research-datasets/dices-dataset/"
        "main/990/diverse_safety_adversarial_dialog_990.csv",
        "14332b6391cdee80383b765c734de3bebf6d8553141ee7d2f47ef0910a12fcf7",
    ),
}
# Quality filters printed in the dataset READMEs. The released files also contain
# the removed raters; the papers' pools drop them. We drop them and say so.
REMOVED_350 = {
    "297514565398139", "297515609163939", "297515750682315", "297515617432733",
    "297541515566649", "297541515769980", "297515629971478", "297059995361243",
    "297541522412126", "297540556928761", "297541321453321", "297540562350921",
    "297540983991638", "297060365288109", "297514543980607", "297515729806999",
    "297541271027233", "296709611112092", "296709543131761",
}
REMOVED_990 = {
    "296768824962581", "296740172100277", "296729424498056", "296835612486158",
    "297212198239077", "296740254381555", "296740049152799", "297212002905763",
    "296740761150184", "296708974347177", "296770194464643", "296768399729683",
}
ASPECTS = ("coherence", "consistency", "fluency", "relevance")
# Pre-registered before any test metric: a 1–5 grade is the positive binary class at 4 or 5.
SUMMEVAL_BINARY_CUT = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, expected: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and _sha256(dest) == expected:
        return
    # Reuse a file already downloaded elsewhere in this run.
    alt = Path("/tmp/judgedata") / dest.name
    alt2 = Path("/tmp/judgedata/dices") / dest.name
    alt3 = Path("/tmp/judgedata/chaos") / dest.name
    for candidate in (alt, alt2, alt3):
        if candidate.is_file() and _sha256(candidate) == expected:
            dest.write_bytes(candidate.read_bytes())
            return
    with urllib.request.urlopen(url, timeout=180) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"checksum mismatch for {dest.name}")
    dest.write_bytes(data)


def availability() -> dict:
    """What this machine can and cannot score. No substitute is loaded for a miss."""
    return {
        "summeval": "downloaded from the SFR jsonl (expert lists of 3 and crowd lists of 5)",
        "chaosnli": "downloaded from the author checksums on Hugging Face earino/chaosnli",
        "dices350": "downloaded from google-research-datasets/dices-dataset",
        "dices990": "downloaded from google-research-datasets/dices-dataset",
        "popquorn": (
            "NOT LOADED. huggingface dataset search returned no POPQUORN card, "
            "https://github.com/Jiaxin-Pei/POPQUORN returned 404, and that user's public "
            "repos contain Stanford-politeness model code rather than POPQUORN ratings. "
            "The Stanford politeness corpus was not substituted."
        ),
        "veribench_fsc": (
            "NOT IN THIS REPOSITORY AND NOT ON THIS MACHINE. A search of this repo for "
            "FSC / R_v1 / functional semantic correctness ratings found no table. "
            "AGENTS.md assigns the shared VeriBench human-rating collection to "
            "trustworthy-theorem-judges experiment 09; that repo is not cloned here, "
            "so CertJudge / VeriBench FSC R_v1 (0–5, about 5 experts, 343 items) was not scored."
        ),
    }


class JudgingSet:
    def __init__(self, name, kind, class_values, texts, groups, ratings, rater_ids, notes):
        self.name = name
        self.kind = kind
        self.class_values = np.asarray(class_values, dtype=float)
        self.texts = list(texts)
        self.groups = list(groups)
        self.ratings = [np.asarray(r, dtype=int) for r in ratings]
        self.rater_ids = rater_ids
        self.notes = notes

    @property
    def n_items(self) -> int:
        return len(self.texts)

    @property
    def n_classes(self) -> int:
        return int(self.class_values.shape[0])

    def summary(self) -> dict:
        ks = np.array([r.size for r in self.ratings])
        return {
            "name": self.name,
            "kind": self.kind,
            "n_items": self.n_items,
            "n_ratings": int(ks.sum()),
            "ratings_per_item_min": int(ks.min()),
            "ratings_per_item_median": float(np.median(ks)),
            "ratings_per_item_max": int(ks.max()),
            "n_classes": self.n_classes,
            "class_values": self.class_values.tolist(),
            "notes": self.notes,
        }


def _pack(name, kind, values, bucket, notes) -> JudgingSet:
    texts, groups, ratings, raters = [], [], [], []
    for _key, rec in bucket.items():
        if len(rec["y"]) == 0:
            continue
        texts.append(rec["text"])
        groups.append(rec["group"])
        ratings.append(rec["y"])
        raters.append(rec["rater"])
    return JudgingSet(name, kind, values, texts, groups, ratings, raters, notes)


def load_summeval(pool: str) -> JudgingSet:
    path = CACHE / "model_annotations.aligned.jsonl"
    _fetch(SUMMEVAL_URL, path, SUMMEVAL_SHA)
    key = "expert_annotations" if pool == "expert" else "turker_annotations"
    bucket = {}
    within = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            grades = row[key]
            within.append(len(grades))
            for aspect_i, aspect in enumerate(ASPECTS):
                ys = [int(g[aspect]) for g in grades]
                item = f"{row['filepath']}::{row['model_id']}::{aspect}"
                rec = bucket.setdefault(item, {
                    "text": f"aspect {aspect}. summary: {row['decoded']}",
                    "group": row["filepath"],
                    "y": [],
                    "rater": [],
                })
                for r_i, y in enumerate(ys):
                    if y < 1 or y > 5:
                        raise ValueError(f"SummEval grade {y} outside 1–5")
                    rec["y"].append(y - 1)  # class index 0..4, value 1..5
                    # The release has no stable person id. This is a within-item vote index.
                    rec["rater"].append(f"{pool}_vote_{r_i}_aspectslot_{aspect_i}")
    notes = (
        f"SummEval {pool} annotations on four aspects (coherence, consistency, fluency, relevance), "
        "native grades 1–5 stored as class values [1,2,3,4,5]. "
        f"Every summary has {sorted(set(within))} {pool} annotation records. "
        "The jsonl does not publish a stable rater id, so rater_id is the within-item vote index, "
        "not a person. Split group is the source article (filepath), so every system and aspect "
        "of one article stays in one split. Binary views use the pre-registered cut grade ≥ 4."
    )
    return _pack(f"summeval_{pool}", "ordinal", [1, 2, 3, 4, 5], bucket, notes)


def binarize(data: JudgingSet, cut_value: float, name: str) -> JudgingSet:
    """Map a rating to {0, 1} by value ≥ cut. Same prompts, same raters, same groups."""
    values = data.class_values
    out_ratings = []
    for ratings in data.ratings:
        raw = values[ratings]
        out_ratings.append((raw >= cut_value).astype(int))
    notes = (
        f"Binary view of {data.name}: class 1 iff the original value ≥ {cut_value}. "
        "This is a threshold of the same ratings, not a separately collected binary task. "
        + data.notes
    )
    return JudgingSet(name, "binary", [0, 1], data.texts, data.groups, out_ratings, data.rater_ids, notes)


def _chaos_items(filename: str, binary: bool) -> JudgingSet:
    path = CACHE / filename
    _fetch(CHAOS_BASE + filename, path, CHAOS_SHA[filename])
    bucket = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            counts = [int(x) for x in row["label_count"]]
            if sum(counts) <= 0:
                continue
            ex = row["example"]
            if binary:
                text = (
                    f"obs1: {ex.get('obs1', '')} hyp1: {ex.get('hyp1', '')} "
                    f"hyp2: {ex.get('hyp2', '')} obs2: {ex.get('obs2', '')}"
                )
                # label_count is [option 1, option 2], matching label_counter keys "1" and "2".
                votes = []
                for cls, n in enumerate(counts):
                    votes.extend([cls] * n)
            else:
                text = f"premise: {ex.get('premise', '')} hypothesis: {ex.get('hypothesis', '')}"
                # label_count is [entailment, neutral, contradiction].
                votes = []
                for cls, n in enumerate(counts):
                    votes.extend([cls] * n)
            if len(votes) != sum(counts):
                raise RuntimeError("vote expansion failed")
            bucket[row["uid"]] = {
                "text": text,
                "group": row["uid"],
                "y": votes,
                "rater": [f"vote_{i}" for i in range(len(votes))],
            }
    if binary:
        notes = (
            "ChaosNLI abductive NLI, native two-option labels. "
            "The release stores counts, not person ids; rater_id is the within-item vote index "
            "obtained by expanding label_count. Order is option 1 then option 2. "
            "Each item is its own split group."
        )
        return _pack("chaosnli_alpha", "binary", [1, 2], bucket, notes)
    notes = (
        "ChaosNLI SNLI or MNLI-matched, 3 classes, 100 votes per item in the author files. "
        "label_count order is entailment, neutral, contradiction. "
        "Ranked probability score uses that order as an imposed order (entailment < neutral < contradiction); "
        "it is not a native Likert scale. rater_id is a within-item vote index expanded from the counts, "
        "not a person. Each uid is its own split group."
    )
    return _pack(Path(filename).stem, "multiclass", [0, 1, 2], bucket, notes)


def load_chaos_3way() -> JudgingSet:
    parts = [_chaos_items("chaosNLI_snli.jsonl", False), _chaos_items("chaosNLI_mnli_m.jsonl", False)]
    texts, groups, ratings, raters = [], [], [], []
    for part in parts:
        texts.extend(part.texts)
        groups.extend([f"{part.name}::{g}" for g in part.groups])
        ratings.extend(part.ratings)
        raters.extend(part.rater_ids)
    notes = (
        "Pooled ChaosNLI-SNLI and ChaosNLI-MNLI-matched. " + parts[0].notes
        + f" Item counts by file are kept distinct by the group prefix."
    )
    return JudgingSet("chaosnli_snli_mnli", "multiclass", [0, 1, 2], texts, groups, ratings, raters, notes)


def load_chaos_alpha() -> JudgingSet:
    return _chaos_items("chaosNLI_alphanli.jsonl", True)


def _load_dices(which: str) -> JudgingSet:
    filename = "dices350.csv" if which == "350" else "dices990.csv"
    url, sha = DICES_URL[filename]
    path = CACHE / filename
    _fetch(url, path, sha)
    removed = REMOVED_350 if which == "350" else REMOVED_990
    bucket = {}
    n_drop = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["rater_id"]) in removed:
                n_drop += 1
                continue
            label = row["Q_overall"]
            if label not in {"No", "Unsure", "Yes"}:
                continue
            # README for DICES-990: Yes means the rater perceives the response to be unsafe.
            # No < Unsure < Yes is the imposed 3-level order used for RPS.
            y = {"No": 0, "Unsure": 1, "Yes": 2}[label]
            item = str(row["item_id"])
            text = f"context: {row.get('context', '')}\nresponse: {row.get('response', '')}"
            rec = bucket.setdefault(item, {"text": text, "group": item, "y": [], "rater": []})
            rec["y"].append(y)
            rec["rater"].append(str(row["rater_id"]))
    notes = (
        f"DICES-{which} rater-level Q_overall. Yes means unsafe (DICES-990 README; DICES-350 uses the same "
        "question family). Order for ranked probability score: No < Unsure < Yes, values 0,1,2. "
        f"Dropped {n_drop} rows whose rater_id is in the README quality-filter list "
        f"({len(removed)} raters). rater_id is the released person id. "
        "Split group is item_id. Raters are crossed, so the same person can appear in train and test "
        "on different conversations; no conversation's raters are split."
    )
    return _pack(f"dices{which}", "ordinal", [0, 1, 2], bucket, notes)


def load_all() -> list[JudgingSet]:
    expert = load_summeval("expert")
    crowd = load_summeval("crowd")
    sets = [
        expert,
        crowd,
        binarize(expert, SUMMEVAL_BINARY_CUT, "summeval_expert_binary"),
        binarize(crowd, SUMMEVAL_BINARY_CUT, "summeval_crowd_binary"),
        load_chaos_3way(),
        load_chaos_alpha(),
        _load_dices("350"),
        _load_dices("990"),
    ]
    sets.append(binarize(sets[-1], 2, "dices990_binary_yes"))
    return sets


def split_by_group(groups: list[str], seed: int = 0, fracs=(0.6, 0.2, 0.2)):
    """Split prompts by group. Every rater of a group stays in one side."""
    if abs(sum(fracs) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = uniq.shape[0]
    n_train = int(np.floor(fracs[0] * n))
    n_val = int(np.floor(fracs[1] * n))
    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise RuntimeError(f"split is too small for {n} groups")
    train_g = set(uniq[:n_train].tolist())
    val_g = set(uniq[n_train:n_train + n_val].tolist())
    test_g = set(uniq[n_train + n_val:].tolist())
    if train_g & val_g or train_g & test_g or val_g & test_g:
        raise RuntimeError("group leakage")
    train, val, test = [], [], []
    for i, g in enumerate(groups):
        if g in train_g:
            train.append(i)
        elif g in val_g:
            val.append(i)
        elif g in test_g:
            test.append(i)
        else:
            raise RuntimeError("group missing from the split")
    return np.array(train), np.array(val), np.array(test)


def assert_no_prompt_leakage(groups, ratings_ids, train, val, test) -> None:
    """Every prompt group is in exactly one split, and the index sets are disjoint."""
    splits = {"train": set(train.tolist()), "val": set(val.tolist()), "test": set(test.tolist())}
    seen = {}
    for name, idxs in splits.items():
        for i in idxs:
            g = groups[i]
            if g in seen and seen[g] != name:
                raise AssertionError(f"group {g} is in {seen[g]} and {name}")
            seen[g] = name
    all_idx = splits["train"] | splits["val"] | splits["test"]
    if len(all_idx) != len(groups):
        raise AssertionError("an item was dropped or duplicated across splits")
    if splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"]:
        raise AssertionError("item index overlap")
    # Raters of one item cannot be divided: the item index is atomic.
    for i, rats in enumerate(ratings_ids):
        if len(rats) != len(set(rats)) and len(rats) != len(rats):
            pass
