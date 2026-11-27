"""Loaders for real multi-annotator judging records.

Each item is one prompt ``c``. Ratings stay as raw per-rater rows
``(item_id, rater_id, class_index)``. Nothing is pre-averaged before the split.

Covers SummEval (expert and crowd), ChaosNLI (SNLI+MNLI and alphaNLI),
DICES-350 / DICES-990, POPQUORN (offensiveness and politeness), VeriBench
functional semantic correctness (FSC R_v1, live-synced native_main, and pooled),
and read-only CertJudge LLM judge panel verification from cert-judge.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
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
VERIBENCH_BINARY_CUT = 4

POPQUORN_REV = "dbb118c29b3d9bcce5aa172d8249ded9ece4df84"
POPQUORN_SPECS = {
    "offensiveness": (
        f"https://raw.githubusercontent.com/Jiaxin-Pei/Potato-Prolific-Dataset/{POPQUORN_REV}/dataset/offensiveness/raw_data.csv",
        "fb0589093d45da76bf4ef4d0abff679b540bdd5b17af5764913b3ccd0a069e1e",
        "offensiveness",
    ),
    "politeness": (
        f"https://raw.githubusercontent.com/Jiaxin-Pei/Potato-Prolific-Dataset/{POPQUORN_REV}/dataset/politeness_rating/raw_data.csv",
        "beb7c68c6366ab27819cdf40fec7eb76676f5903c99dad95c13d4bd2bc28365f",
        "politeness",
    ),
}

VERIBENCH_REV = "d1c473559d3a7d5f7e7b4f708a9d7b9469afd779"
VERIBENCH_ITEMS_SHA = "ad9ed552fc2ffac6328e6010bf6be82215dd74f1eddd791622d433126877a96c"
VERIBENCH_EXP100_REL = "experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs"

TTJ_REV = "460466dd10b735c5ff25f9e9cb21f0441fcf960b"
TTJ_EXP09_REL = "experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026"
TTJ_MANIFEST_SHA = "3db24dd8aa0c764454fc05f17c866550a6ccd37e47646fee0f7a5d61289ee33b"
TTJ_PACKET_TREE_SHA = "2a4fed2cbc08d563351698d36097e92d4b547f98c210a0f619f46add8a43274e"

CERTJUDGE_REV = "f6047849853e65fd533a6e5df2416d6a9d9447d6"
CERTJUDGE_SHAS = {
    "study_a_e63_proper_scores.json": "24dcaaebd280d1d2c6e356be2ebadee8c28d376721350758bed70e56e7ec1655",
    "study_b_e39_proper_scores.json": "32056ae51d646f2232923c7c5fb12015d7bcdf5cd2dd3faecf880bb0bcac46fb",
    "judged_B3_calibrated.csv": "fac4f0c4cc1ca7459a75b4cb2362b13fafc30828e7e63c9133cd7853ed73b30f",
    "judged_summary.json": "5e4a75b6e1b56e2adb86ae59ac67ec24bbe7f8e61633b2807e99e0f2b774839b",
}
REPO_ROOT = ROOT.parents[1]


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


def _fetch_text_cached(url: str, dest: Path, expected_sha: str | None = None) -> str:
    """Download a pinned public file once into data_cache/, verifying expected_sha when given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        raw = dest.read_bytes()
        if expected_sha is None or hashlib.sha256(raw).hexdigest() == expected_sha:
            return raw.decode("utf-8")
    with urllib.request.urlopen(url, timeout=180) as response:
        raw = response.read()
    if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError(f"checksum mismatch for {url}")
    dest.write_bytes(raw)
    return raw.decode("utf-8")


def _resolve_sibling_text(
    env_var: str,
    sibling_name: str,
    github_repo: str,
    rev: str,
    rel_path: str,
    expected_sha: str | None = None,
) -> str:
    """Read ``rel_path`` at the pinned ``rev``, verifying ``expected_sha`` when provided.

    Order: ``git show rev:rel_path`` from the env/sibling checkout (pinned bytes, independent of
    that checkout's working tree), then a matching working-tree file (or unverified local file only
    when no ``.git`` repo is present and no ``expected_sha`` is given), then the pinned
    raw.githubusercontent.com URL cached under ``data_cache/<sibling>/<rev>/``.
    """
    base = Path(os.environ[env_var]) if env_var in os.environ else (REPO_ROOT.parent / sibling_name)
    is_git = (base / ".git").exists()
    if is_git:
        proc = subprocess.run(
            ["git", "-C", str(base), "show", f"{rev}:{rel_path}"],
            capture_output=True, check=False,
        )
        if proc.returncode == 0:
            raw = proc.stdout
            if expected_sha is not None and hashlib.sha256(raw).hexdigest() != expected_sha:
                raise RuntimeError(f"SHA-256 mismatch at {sibling_name}@{rev[:8]}:{rel_path}")
            return raw.decode("utf-8")
    local = base / rel_path
    if local.is_file():
        raw = local.read_bytes()
        if expected_sha is not None:
            if hashlib.sha256(raw).hexdigest() == expected_sha:
                return raw.decode("utf-8")
        elif not is_git:
            return raw.decode("utf-8")
    url = f"https://raw.githubusercontent.com/{github_repo}/{rev}/{rel_path}"
    return _fetch_text_cached(url, CACHE / sibling_name / rev[:12] / rel_path, expected_sha=expected_sha)


def availability() -> dict:
    """What this machine scored and from which verified sources."""
    return {
        "summeval": "downloaded from the SFR jsonl (expert lists of 3 and crowd lists of 5)",
        "chaosnli": "downloaded from the author checksums on Hugging Face earino/chaosnli",
        "dices350": "downloaded from google-research-datasets/dices-dataset",
        "dices990": "downloaded from google-research-datasets/dices-dataset",
        "popquorn": (
            f"downloaded from Jiaxin-Pei/Potato-Prolific-Dataset@{POPQUORN_REV[:8]} "
            "(offensiveness 1,500 items / 13,036 ratings; politeness_rating 3,718 items / 25,042 ratings; 1-5 scale)"
        ),
        "veribench_fsc_rv1": (
            f"loaded from brando90/veribench@{VERIBENCH_REV[:8]} Exp 100 items.jsonl "
            "(75 items, 25 tasks, 0-5 scale; all5 345 ratings, expert3 195 ratings)"
        ),
        "veribench_fsc_native_main": (
            f"loaded from brando90/trustworthy-theorem-judges@{TTJ_REV[:8]} Exp 09 live-synced 11-26-2026 "
            "(100 items, 25 tasks, 0-5 scale; 250 non-coordinator numeric ratings across 6 active raters; T30 HTTP 401 unverified)"
        ),
        "certjudge_real_panels": (
            f"loaded from brando90/cert-judge@{CERTJUDGE_REV[:8]} Exp 68 Study A (E63 6 Qwen3-8B arms) "
            "& Study B (E39 56 LLM configs) + Exp 67 interim (3 judges on 100 native files)"
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


def load_popquorn(task: str) -> JudgingSet:
    """Load POPQUORN ordinal 1-5 ratings (offensiveness or politeness)."""
    if task not in POPQUORN_SPECS:
        raise ValueError(f"Unknown POPQUORN ordinal task: {task}")
    url, expected_sha, score_col = POPQUORN_SPECS[task]
    path = CACHE / f"popquorn_{task}.csv"
    _fetch(url, path, expected_sha)
    bucket = {}
    dropped = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                val = float(row[score_col])
            except (TypeError, ValueError):
                dropped += 1
                continue
            y_int = int(round(val))
            if abs(val - y_int) > 1e-9 or y_int < 1 or y_int > 5:
                dropped += 1
                continue
            item = str(row["instance_id"])
            rec = bucket.setdefault(item, {
                "text": f"task {task}. text: {row.get('text', '')}",
                "group": item,
                "y": [],
                "rater": [],
            })
            rec["y"].append(y_int - 1)  # class index 0..4, value 1..5
            rec["rater"].append(str(row["user_id"]))
    notes = (
        f"POPQUORN {task} (Jiaxin-Pei/Potato-Prolific-Dataset@{POPQUORN_REV[:8]}), "
        f"native grades 1–5 stored as class values [1,2,3,4,5]. Dropped {dropped} invalid rows. "
        "Split group is instance_id; raters are crossed across instances. "
        "Binary view uses cut grade >= 4."
    )
    return _pack(f"popquorn_{task}", "ordinal", [1, 2, 3, 4, 5], bucket, notes)


def _format_veribench_text(task_label: str, domain: str, py_src: str, gold_lean: str, cand_lean: str) -> str:
    return (
        f"task: {task_label} domain: {domain}\n"
        f"python_source:\n{py_src[:4000]}\n"
        f"gold_lean:\n{gold_lean[:6000]}\n"
        f"candidate_lean:\n{cand_lean[:8000]}"
    )


def load_veribench_fsc_rv1(pool: str = "all5") -> JudgingSet:
    """Load VeriBench FSC R_v1 (75 items, 25 tasks, 0-5 scale) from veribench Exp 100."""
    if pool not in ("all5", "expert3"):
        raise ValueError(f"Unknown VeriBench R_v1 pool: {pool}")
    items_raw = _resolve_sibling_text(
        "VERIBENCH_ROOT", "veribench", "brando90/veribench", VERIBENCH_REV,
        f"{VERIBENCH_EXP100_REL}/items.jsonl",
        expected_sha=VERIBENCH_ITEMS_SHA,
    )
    cols = ("p1", "p2", "p3", "p4", "p5") if pool == "all5" else ("p1", "p2", "p3")
    bucket = {}

    def _pinned(row, path_key, sha_key):
        # Read one artifact at VERIBENCH_REV and check it against the SHA-256 recorded in items.jsonl.
        return _resolve_sibling_text(
            "VERIBENCH_ROOT", "veribench", "brando90/veribench", VERIBENCH_REV,
            row[path_key].replace("inputs/", f"{VERIBENCH_EXP100_REL}/", 1),
            expected_sha=row[sha_key],
        )

    for line in items_raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        py_src = _pinned(row, "py_ann_path", "py_ann_sha256")
        gold_lean = _pinned(row, "ref_ann_path", "ref_ann_sha256")
        cand_lean = _pinned(row, "candidate_path", "candidate_sha256")
        ys, rats = [], []
        for col in cols:
            v = row.get(col)
            if v is None:
                continue
            val = float(v)
            y_int = int(round(val))
            if abs(val - y_int) > 1e-9 or y_int < 0 or y_int > 5:
                raise ValueError(f"VeriBench score {v} is not an integer in 0..5")
            ys.append(y_int)
            rats.append(col)
        if not ys:
            continue
        bucket[row["item"]] = {
            "text": _format_veribench_text(row["task"], row.get("domain", ""), py_src, gold_lean, cand_lean),
            "group": row["task"],
            "y": ys,
            "rater": rats,
        }
    name = "veribench_fsc_rv1" if pool == "all5" else "veribench_fsc_rv1_expert3"
    notes = (
        f"VeriBench functional semantic correctness (FSC) R_v1 ({pool}) from brando90/veribench@{VERIBENCH_REV[:8]} "
        f"Exp 100 (75 artifacts, 25 tasks, 0–5 scale). Split group is task so all 3 samples per task stay together. "
        "Binary view uses cut score >= 4."
    )
    return _pack(name, "ordinal", [0, 1, 2, 3, 4, 5], bucket, notes)


def load_veribench_fsc_native_main() -> JudgingSet:
    """Load live-synced VeriBench FSC native_main ratings from trustworthy-theorem-judges Exp 09."""
    manifest_raw = _resolve_sibling_text(
        "TTJ_ROOT", "trustworthy-theorem-judges", "brando90/trustworthy-theorem-judges",
        TTJ_REV, f"{TTJ_EXP09_REL}/ratings/RAW_MANIFEST.json",
        expected_sha=TTJ_MANIFEST_SHA,
    )
    manifest = json.loads(manifest_raw)
    native_entries = [e for e in manifest["files"] if e.get("packet") == "native_main"]
    # Resolve canonical item -> (task_folder, candidate_file) from A1 sheet.
    a1_entry = next(e for e in native_entries if e["rater_id"] == "A1")
    a1_csv = _resolve_sibling_text(
        "TTJ_ROOT", "trustworthy-theorem-judges", "brando90/trustworthy-theorem-judges",
        TTJ_REV, f"{TTJ_EXP09_REL}/{a1_entry['path']}",
        expected_sha=a1_entry["sha256"],
    )
    canonical = {}
    for row in csv.DictReader(io.StringIO(a1_csv)):
        canonical[row["item_id"]] = (row["task_folder"], row["candidate_file"])

    ratings_by_item = defaultdict(list)
    seen_keys = set()
    n_blank_or_na = 0
    for entry in native_entries:
        rid = entry["rater_id"]
        if rid == "A1":
            # Exclude coordinator practice column from independent team ratings (A1 has 0 main ratings).
            continue
        csv_text = _resolve_sibling_text(
            "TTJ_ROOT", "trustworthy-theorem-judges", "brando90/trustworthy-theorem-judges",
            TTJ_REV, f"{TTJ_EXP09_REL}/{entry['path']}",
            expected_sha=entry["sha256"],
        )
        for row in csv.DictReader(io.StringIO(csv_text)):
            s = row["final_score_0_5"].strip()
            if s not in {"0", "1", "2", "3", "4", "5"}:
                n_blank_or_na += 1
                continue
            key = (rid, row["item_id"])
            if key in seen_keys:
                raise RuntimeError(f"Duplicate (rater_id, item_id) in native_main: {key}")
            seen_keys.add(key)
            ratings_by_item[row["item_id"]].append((rid, int(s)))

    task_cache = {}
    bucket = {}
    packet_digest = hashlib.sha256()
    for item_id, pairs in sorted(ratings_by_item.items()):
        task_folder, cand_rel = canonical[item_id]
        if task_folder not in task_cache:
            py_rel = f"{TTJ_EXP09_REL}/rating_task_packet/main_rating_task/{task_folder}/python_source/source.py"
            py_src = _resolve_sibling_text(
                "TTJ_ROOT", "trustworthy-theorem-judges", "brando90/trustworthy-theorem-judges",
                TTJ_REV, py_rel,
            )
            gold_rel = f"{TTJ_EXP09_REL}/rating_task_packet/main_rating_task/{task_folder}/gold_lean/reference.lean"
            gold_lean = _resolve_sibling_text(
                "TTJ_ROOT", "trustworthy-theorem-judges", "brando90/trustworthy-theorem-judges",
                TTJ_REV, gold_rel,
            )
            for rel, txt in ((py_rel, py_src), (gold_rel, gold_lean)):
                packet_digest.update(f"{rel}:{hashlib.sha256(txt.encode('utf-8')).hexdigest()}\n".encode("utf-8"))
            task_cache[task_folder] = (py_src, gold_lean)
        py_src, gold_lean = task_cache[task_folder]
        cand_path = f"{TTJ_EXP09_REL}/rating_task_packet/main_rating_task/{task_folder}/{cand_rel.split('/', 1)[1]}"
        cand_lean = _resolve_sibling_text(
            "TTJ_ROOT", "trustworthy-theorem-judges", "brando90/trustworthy-theorem-judges",
            TTJ_REV, cand_path,
        )
        packet_digest.update(f"{cand_path}:{hashlib.sha256(cand_lean.encode('utf-8')).hexdigest()}\n".encode("utf-8"))
        bucket[item_id] = {
            "text": _format_veribench_text(task_folder, "native_main", py_src, gold_lean, cand_lean),
            "group": task_folder,
            "y": [s for _, s in pairs],
            "rater": [rid for rid, _ in pairs],
        }
    if packet_digest.hexdigest() != TTJ_PACKET_TREE_SHA:
        raise RuntimeError(
            f"TTJ native_main packet tree SHA-256 mismatch: {packet_digest.hexdigest()} != {TTJ_PACKET_TREE_SHA}"
        )
    n_active = len({rid for v in bucket.values() for rid in v["rater"]})
    notes = (
        f"VeriBench FSC native_main (native-heavy-11-11-2026) synced 11-26-2026 from "
        f"trustworthy-theorem-judges@{TTJ_REV[:8]} Exp 09 (100 artifacts, 25 tasks, 0–5 scale, "
        f"{sum(len(v['y']) for v in bucket.values())} non-coordinator numeric ratings across {n_active} active raters; "
        f"{n_blank_or_na} blank/N/A rows excluded and never treated as 0; T30 HTTP 401 unverified). "
        "Split group is task_folder."
    )
    return _pack("veribench_fsc_native_main", "ordinal", [0, 1, 2, 3, 4, 5], bucket, notes)


def load_veribench_fsc_pooled(rv1: JudgingSet | None = None, nat: JudgingSet | None = None) -> JudgingSet:
    """Pool VeriBench FSC R_v1 (all5, 75 items, 25 tasks) and native_main (100 items, 25 disjoint tasks)."""
    rv1 = rv1 if rv1 is not None else load_veribench_fsc_rv1("all5")
    nat = nat if nat is not None else load_veribench_fsc_native_main()
    # Check the raw task names (before prefixing) and the artifact texts: a shared task would put
    # one task's Python source and gold Lean on both sides of the split.
    if set(rv1.groups) & set(nat.groups):
        raise RuntimeError("Task collision when pooling R_v1 and native_main")
    if set(rv1.texts) & set(nat.texts):
        raise RuntimeError("Artifact text collision when pooling R_v1 and native_main")
    texts = rv1.texts + nat.texts
    groups = [f"rv1::{g}" for g in rv1.groups] + [f"native::{g}" for g in nat.groups]
    ratings = rv1.ratings + nat.ratings
    raters = rv1.rater_ids + nat.rater_ids
    rv1_all = np.concatenate(rv1.ratings)
    nat_all = np.concatenate(nat.ratings)
    notes = (
        f"Pooled VeriBench FSC R_v1 ({rv1.n_items} items, {len(set(rv1.groups))} tasks, {rv1_all.size} ratings) "
        f"and live-synced native_main ({nat.n_items} items, {len(set(nat.groups))} disjoint tasks, {nat_all.size} ratings) "
        f"= {len(texts)} items, {len(set(groups))} tasks, {rv1_all.size + nat_all.size} ratings on 0–5 scale. "
        f"Caveat: native_main has {100 * np.mean(nat_all == 5):.1f}% ratings at 5 (mean {nat_all.mean():.2f}) "
        f"vs R_v1 mean {rv1_all.mean():.2f}, so pooled results mix two rating regimes."
    )
    return JudgingSet("veribench_fsc_pooled", "ordinal", [0, 1, 2, 3, 4, 5], texts, groups, ratings, raters, notes)


def load_certjudge_real_panels() -> dict:
    """Load read-only verified proper-scoring evaluations of real LLM judges from cert-judge Exp 67 & 68."""
    e63_raw = _resolve_sibling_text(
        "CERT_JUDGE_ROOT", "cert-judge", "brando90/cert-judge", CERTJUDGE_REV,
        "experiments/68_proper_scoring_expansion_e39_e63_and_vb_ratings_start_24Nov2026/results/study_a_e63_proper_scores.json",
        expected_sha=CERTJUDGE_SHAS["study_a_e63_proper_scores.json"],
    )
    e39_raw = _resolve_sibling_text(
        "CERT_JUDGE_ROOT", "cert-judge", "brando90/cert-judge", CERTJUDGE_REV,
        "experiments/68_proper_scoring_expansion_e39_e63_and_vb_ratings_start_24Nov2026/results/study_b_e39_proper_scores.json",
        expected_sha=CERTJUDGE_SHAS["study_b_e39_proper_scores.json"],
    )
    exp67_csv_raw = _resolve_sibling_text(
        "CERT_JUDGE_ROOT", "cert-judge", "brando90/cert-judge", CERTJUDGE_REV,
        "experiments/67_new_vb_team_ratings_rps_ce_tvd_rerun_of_the_certjudge_panel/results/interim_11-24-2026/judged_B3_calibrated.csv",
        expected_sha=CERTJUDGE_SHAS["judged_B3_calibrated.csv"],
    )
    exp67_json_raw = _resolve_sibling_text(
        "CERT_JUDGE_ROOT", "cert-judge", "brando90/cert-judge", CERTJUDGE_REV,
        "experiments/67_new_vb_team_ratings_rps_ce_tvd_rerun_of_the_certjudge_panel/results/interim_11-24-2026/judged_summary.json",
        expected_sha=CERTJUDGE_SHAS["judged_summary.json"],
    )
    e63 = json.loads(e63_raw)
    e39 = json.loads(e39_raw)
    exp67_sum = json.loads(exp67_json_raw)
    exp67_rows = list(csv.DictReader(io.StringIO(exp67_csv_raw)))

    e63_summary = {}
    for target in ("all5", "expert3"):
        tblock = e63["targets"][target]
        arms = {}
        for arm_name, a in tblock["arms"].items():
            m = a["ensemble"]
            arms[arm_name] = {
                "n_seeds": a.get("n_seeds"),
                "rps": m["rps"]["val"], "rps_ci": [m["rps"]["ci_lo"], m["rps"]["ci_hi"]],
                "ce": m["ce"]["val"], "ce_ci": [m["ce"]["ci_lo"], m["ce"]["ci_hi"]],
                "tvd": m["tvd"]["val"], "brier": m["brier"]["val"],
                "bias_05": m["bias_05"]["val"], "spearman": m["spearman"]["val"],
                "pearson": m["pearson"]["val"], "icc_a1": m["icc_a1"]["val"],
            }
        e63_summary[target] = {
            "const_trainval": tblock["const_trainval_baseline"],
            "arms": arms,
        }

    llms_all = [u for u in e39["units"] if u["kind"] == "llm"]
    llms_test = [u for u in llms_all if u["split"] == "test"]
    if len(llms_all) != e39["n_all_llm"] or len(llms_test) != e39["n_test_llm"]:
        raise RuntimeError("E39 unit counts disagree with the published n_all_llm / n_test_llm")
    def _agg_e39(units):
        return {
            "n_configs": len(units),
            "raw_rps_mean": float(np.mean([u["metrics"]["raw_all5_rps"] for u in units])),
            "ce_cal_rps_mean": float(np.mean([u["metrics"]["ce_cal_all5_rps"] for u in units])),
            "pe_cal_rps_mean": float(np.mean([u["metrics"]["pe_cal_all5_rps"] for u in units])),
            "raw_ce_mean": float(np.mean([u["metrics"]["raw_all5_ce"] for u in units])),
            "ce_cal_ce_mean": float(np.mean([u["metrics"]["ce_cal_all5_ce"] for u in units])),
            "pe_cal_ce_mean": float(np.mean([u["metrics"]["pe_cal_all5_ce"] for u in units])),
            "ce_cal_beats_pe_cal_rps": int(sum(u["metrics"]["ce_cal_all5_rps"] < u["metrics"]["pe_cal_all5_rps"] for u in units)),
            "ce_cal_beats_pe_cal_ce": int(sum(u["metrics"]["ce_cal_all5_ce"] < u["metrics"]["pe_cal_all5_ce"] for u in units)),
            "raw_spearman_mean": float(np.mean([u["metrics"]["raw_all5_spearman"] for u in units])),
            "ce_cal_spearman_mean": float(np.mean([u["metrics"]["ce_cal_all5_spearman"] for u in units])),
            "pe_cal_spearman_mean": float(np.mean([u["metrics"]["pe_cal_all5_spearman"] for u in units])),
        }

    exp67_new = []
    for r in exp67_rows:
        if r["set"] == "new" and r["comparison"] == "CE-fit minus Pearson-fit":
            exp67_new.append({
                "family": r["family"],
                "n_tasks": int(r["n_tasks"]),
                "dRPS": float(r["dRPS"]), "dRPS_ci": [float(r["dRPS_lo"]), float(r["dRPS_hi"])],
                "dCE": float(r["dCE"]), "dCE_ci": [float(r["dCE_lo"]), float(r["dCE_hi"])],
                "dTVD": float(r["dTVD"]), "dTVD_ci": [float(r["dTVD_lo"]), float(r["dTVD_hi"])],
            })
        elif r["set"] == "new" and r["comparison"] == "levels":
            exp67_new.append({
                "family": r["family"],
                "levels": {
                    "CONST_RPS": float(r["CONST_RPS"]), "CONST_CE": float(r["CONST_CE"]), "CONST_TVD": float(r["CONST_TVD"]),
                    "CE_fit_RPS": float(r["CE-fit_RPS"]), "CE_fit_CE": float(r["CE-fit_CE"]), "CE_fit_TVD": float(r["CE-fit_TVD"]),
                    "Pearson_fit_RPS": float(r["Pearson-fit_RPS"]), "Pearson_fit_CE": float(r["Pearson-fit_CE"]), "Pearson_fit_TVD": float(r["Pearson-fit_TVD"]),
                },
            })

    return {
        "source_commit": CERTJUDGE_REV,
        "e63_study_a": e63_summary,
        "e39_study_b": {
            "n_heldout_items": e39["human_loro_heldout_all5"]["n_items"],
            "n_heldout_tasks": e39["human_loro_heldout_all5"]["smoothed_ci"]["n_tasks"],
            "const_train_all5": e39["const_train_baseline_heldout_all5"],
            "human_loro_all5": e39["human_loro_heldout_all5"],
            "all_llm": _agg_e39(llms_all),
            "test_llm": _agg_e39(llms_test),
        },
        "exp67_native_interim": {
            "n_files": exp67_sum["n_files"],
            "n_tasks": exp67_sum["n_tasks"],
            "n_ratings": exp67_sum["n_ratings"],
            "per_rater": exp67_sum["per_rater"],
            "calibrated_new": exp67_new,
        },
    }


def load_all() -> list[JudgingSet]:
    expert = load_summeval("expert")
    crowd = load_summeval("crowd")
    pop_off = load_popquorn("offensiveness")
    pop_pol = load_popquorn("politeness")
    vb_rv1 = load_veribench_fsc_rv1("all5")
    vb_rv1_exp3 = load_veribench_fsc_rv1("expert3")
    vb_native = load_veribench_fsc_native_main()
    vb_pooled = load_veribench_fsc_pooled(vb_rv1, vb_native)
    d990 = _load_dices("990")
    sets = [
        expert,
        crowd,
        binarize(expert, SUMMEVAL_BINARY_CUT, "summeval_expert_binary"),
        binarize(crowd, SUMMEVAL_BINARY_CUT, "summeval_crowd_binary"),
        load_chaos_3way(),
        load_chaos_alpha(),
        _load_dices("350"),
        d990,
        binarize(d990, 2, "dices990_binary_yes"),
        pop_off,
        binarize(pop_off, SUMMEVAL_BINARY_CUT, "popquorn_offensiveness_binary"),
        pop_pol,
        binarize(pop_pol, SUMMEVAL_BINARY_CUT, "popquorn_politeness_binary"),
        vb_rv1,
        vb_rv1_exp3,
        binarize(vb_rv1, VERIBENCH_BINARY_CUT, "veribench_fsc_rv1_binary"),
        vb_native,
        vb_pooled,
        binarize(vb_pooled, VERIBENCH_BINARY_CUT, "veribench_fsc_pooled_binary"),
    ]
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
    # Raters of one item cannot be divided: the item index is atomic, so the checks above suffice.
    if len(ratings_ids) != len(groups):
        raise AssertionError("rater-id lists and groups have different lengths")
