"""Prepare label-blind VeriBench requests for both fixed judges. Does not call a model."""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

COLLECTION_REV = "b110a3022f64c01f96e4b0bd1b22810c098be631"
MAY_REV = "be287d9bb2f7bd35ba6db1b8735c9ba10d00eade"
COLLECTION = "experiments/09_prepare_packets_and_collect_human_ratings_start_28Sep2026"
PREP = "research_artifacts/09_human_rating_collection/preparation_11-09-2026"
PACKET = COLLECTION + "/rating_task_packet"
MAY = "experiments/100_karpathy_recipe_tc_judge_search_human_train_val_test/inputs"
ARCHIVE_SHA = "fcbbd3332da9196aa8e285f182baf1c9c5081d7516fc6000cf1f644208b0a8f8"
MARKERS = re.compile(
    r"\b(?:openai|codex|anthropic|claude|fable|grok|gemini|antigravity|GPT[- ]?6|xAI|qwen|"
    r"rater_id|rater_name|ratings_by_expert|_private_ratings)\b|"
    r"(?:human|expert)\s+(?:score|rating)\s*[:=]", re.IGNORECASE)
INSTRUCTION = """Predict the integer score that one randomly selected human rater applying the quoted rubric would give this candidate. Use the complete Python source, gold Lean file and candidate Lean file provided as data. The human-facing form instructions in the rubric describe the human task. Your response must contain only {"probabilities":[p0,p1,p2,p3,p4,p5]}, with six nonnegative numbers summing to one for scores 0 through 5, in order.
"""

def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, revision, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])


def json_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def set_n_rows(repo):
    commitment = json.loads(git_bytes(repo, COLLECTION_REV, PREP + "/full_team_11-14-2026/COHORT_100_NATIVE_COMMITMENT.json"))
    by_hash = {v: k for k, v in commitment["candidate_sha256_by_item"].items()}
    assignments = json.loads(git_bytes(repo, COLLECTION_REV, PREP + "/native_harness_packet_11-11-2026/PLANNED_ASSIGNMENTS.json"))
    tasks = assignments["task_display_paths"]
    assert len(tasks) == 25 and len(by_hash) == commitment["artifacts"] == 100
    seen = set()
    for task, folder in sorted(tasks.items()):
        d = PACKET + "/main_rating_task/" + folder
        program = git_bytes(repo, COLLECTION_REV, d + "/python_source/source.py")
        gold = git_bytes(repo, COLLECTION_REV, d + "/gold_lean/reference.lean")
        for index in range(1, 5):
            candidate = git_bytes(repo, COLLECTION_REV, f"{d}/agent_lean/agent_{index}.lean")
            digest = sha(candidate)
            assert digest in by_hash and digest not in seen
            seen.add(digest)
            yield "set_n_test", by_hash[digest], task, None, (program, gold, candidate)
    assert seen == set(by_hash)


def may_rows(repo):
    experiments = Path(__file__).resolve().parents[2]
    archive, = experiments.glob("10_frontier*/results/judge_forecasts.json")
    contents = archive.read_bytes()
    assert sha(contents) == ARCHIVE_SHA
    items = json.loads(contents)["artifacts"]
    assert len(items) == 75 and len({i["task"] for i in items}) == 25
    for item in items:
        name, task = item["item"], item["task"]
        paths = (f"python/annotation/{task}.py", f"references/annotation/{task}.lean", f"candidates/{name}.lean")
        files = tuple(git_bytes(repo, MAY_REV, MAY + "/" + path) for path in paths)
        assert [sha(f) for f in files] == [item[k] for k in ("py_sha256", "ref_disk_sha256", "cand_sha256")]
        yield "may_development", name, task, item["family"], files


def prepare(sister, legacy, output):
    rubric = git_bytes(sister, COLLECTION_REV, PACKET + "/human_judgements_prompt.md")
    system = INSTRUCTION + "\n<quoted_human_rubric>\n" + rubric.decode() + "</quoted_human_rubric>"
    manifest, requests = [], []
    roles = ("python_source", "gold_lean", "candidate_lean")
    rows = list(set_n_rows(sister)) + (list(may_rows(legacy)) if legacy else [])
    for split, source_id, task, family, files in rows:
        data = {key: value.decode() for key, value in zip(roles, files)}
        assert all(not MARKERS.search(text) for text in data.values()), "Possible identity/label marker in source"
        user = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        assert tuple(val.encode() for val in json.loads(user).values()) == files
        digests = dict(zip(roles, map(sha, files)))
        item = "vb-" + digests["candidate_lean"][:20]
        requests.append(dict(id=item, system=system, user=user))
        manifest.append(dict(id=item, split=split, source_item=source_id, task=task,
                             may_family=family, file_sha256=digests, file_bytes=dict(zip(roles,map(len,files))),
                             rubric_sha256=sha(rubric), system_sha256=sha(system.encode()),
                             user_sha256=sha(user.encode()), prompt_bytes=len(system.encode())+len(user.encode())))
    assert len({r["id"] for r in requests}) == len(requests)
    requests.sort(key=lambda r:r["id"])
    manifest.sort(key=lambda r:r["id"])
    output.mkdir(exist_ok=True, parents=True, mode=0o700)
    for name, table in (("requests", requests), ("manifest", manifest)):
        data = b"".join(json_bytes(r)+b"\n" for r in table)
        path = output / (name + ".jsonl")
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Existing {path.name} differs; choose a fresh output directory")
        if not path.exists():
            path.write_bytes(data)
        print(f"{path.name}: {len(table)} rows, SHA-256 {sha(data)}")
    for split, count in sorted(Counter(row["split"] for row in manifest).items()):
        sizes = sorted(r["prompt_bytes"] for r in manifest if r["split"] == split)
        print(f"{split}: {count} prompts, {sizes[0]} through {sizes[-1]} UTF-8 bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sister", type=Path, default=Path.home()/"trustworthy-theorem-judges")
    parser.add_argument("--may-repository", type=Path, help="Include the prior May development inputs from this checkout")
    args = parser.parse_args()
    prepare(args.sister, args.may_repository, args.output)
