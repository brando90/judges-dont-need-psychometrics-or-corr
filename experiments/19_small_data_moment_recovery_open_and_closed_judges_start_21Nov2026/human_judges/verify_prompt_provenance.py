"""Reconstruct archived P0 prompts offline. Print aggregate checks without source or rating text."""

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from unittest.mock import patch

EXPERIMENTS = Path(__file__).resolve().parents[2]
OPEN_STUDY, = EXPERIMENTS.glob("09_*")
CLOSED_STUDY, = EXPERIMENTS.glob("10_*")
CLOSED_JUDGE_IDS = {"gpt54_mini__P0", "gpt61_sol__P0"}
RATING_FIELDS = re.compile(
    r"ratings_by_expert|_private_ratings|rater_(?:name|id)|human_(?:ratings?|score)|"
    r"human\s+(?:rating|score)\s*[:=]|expert\s+(?:rating|score)\s*[:=]",
    re.IGNORECASE,
)


def local_module(name, source_path):
    spec = importlib.util.spec_from_file_location(name, source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verify_open_aliases():
    """Match the forecast keys to the runner's publisher repository for each alias."""
    runner = OPEN_STUDY / "scripts/run_mercury2_open_judges.py"
    assignments = (
        node for node in ast.parse(runner.read_text()).body if isinstance(node, ast.Assign)
    )
    settings, = (
        node.value for node in assignments
        if any(isinstance(target, ast.Name) and target.id == "MODELS" for target in node.targets)
    )
    repositories = {
        ast.literal_eval(alias): ast.literal_eval(config.elts[1]).replace("--", "/", 1)
        for alias, config in zip(settings.keys, settings.values)
    }
    manifest = json.loads(Path(__file__).with_name("models.json").read_text())
    data = json.loads((OPEN_STUDY / "results/open_weight_forecasts.json").read_text())
    open_models = [model for model in manifest["models"] if model["weight_access"] == "open"]
    assert len(open_models) == 2
    for model in open_models:
        forecast_key = model["forecast_key"]
        alias, prompt = forecast_key.split("__")
        assert prompt == "P0_human_mirror" and repositories[alias] == model["model"]
        forecasts = data["verbal_forecasts"][forecast_key]
        assert len(forecasts) == len(data["artifact_order"]) == 75
        assert all(len(row) == 6 for row in forecasts)
        assert sum(len(row) for row in forecasts) == model["source_probability_entries"]
        assert sum(p == 0 for row in forecasts for p in row) == model["source_zero_entries"]
        print(f"{forecast_key} matches runner repository {repositories[alias]} and 75 forecasts.")


def verify_cached_forecast(model, expected, system, user, archive, prompts):
    """Return the matched repeat slot: 1 for the original request or 99 for its repair."""
    original_cap = model["max_tokens"]
    for slot, token_cap in ((1, original_cap), (99, min(2200, 2 * original_cap))):
        # This dictionary and its serialization reproduce the archived request fingerprint.
        request = {
            "provider": archive.MODEL_CATALOG[model["model"]]["provider"],
            "model": model["model"],
            "effort": model["effort"],
            "max_output_tokens": token_cap,
            "temperature": None,
            "system": system,
            "user": user,
            "repeat_slot": slot,
            "mode": "text",
        }
        fingerprint = digest(json.dumps(request, sort_keys=True))
        cache_file = archive.API_CACHE_DIR / f"{fingerprint}.json"
        if not cache_file.exists():
            continue
        response = json.loads(cache_file.read_text())
        parsed = prompts.parse_probability_response(response.get("text", ""))
        if response.get("status") == "OK" and parsed["ok"]:
            probabilities = [round(float(value), 8) for value in parsed["probabilities"]]
            assert probabilities == expected["probabilities"]
            return slot

    raise ValueError(f"Could not reproduce a cached forecast for {model['judge_id']}")


def main():
    verify_open_aliases()
    open_prompts = local_module("open_judge_prompts", OPEN_STUDY / "scripts/build_prompts.py")
    archive = local_module("closed_judge_archive", CLOSED_STUDY / "scripts/common.py")
    # The historical prompts module imports its sibling as "common".
    with patch.dict("sys.modules", {"common": archive}):
        closed_prompts = local_module("closed_judge_prompts", CLOSED_STUDY / "scripts/prompts.py")

    open_manifest = json.loads((OPEN_STUDY / "results/input_manifest.json").read_text())
    closed_forecasts = json.loads((CLOSED_STUDY / "results/judge_forecasts.json").read_text())
    open_artifacts = {row["item"]: row for row in open_manifest["artifacts"]}
    closed_artifacts = {row["item"]: row for row in closed_forecasts["artifacts"]}
    assert len(open_artifacts) == 75 and open_artifacts.keys() == closed_artifacts.keys()
    for open_field, closed_field in (
        ("py_ann_sha256", "py_sha256"),
        ("candidate_sha256", "cand_sha256"),
        ("ref_ann_disk_sha256", "ref_disk_sha256"),
    ):
        assert all(
            artifact[open_field] == closed_artifacts[item_id][closed_field]
            for item_id, artifact in open_artifacts.items()
        )
    annotation_matches = sum(
        artifact["ref_ann_items_jsonl_sha256"] == closed_artifacts[item_id]["ref_disk_sha256"]
        for item_id, artifact in open_artifacts.items()
    )
    assert annotation_matches == 15
    rubric = open_prompts.load_rater_instructions()
    assert rubric == closed_prompts.rater_instructions_verbatim()
    open_system = open_prompts.build_system_prompt("P0_human_mirror", rubric)
    assert digest(open_system) == open_manifest["prompt_system_sha256"]["P0_human_mirror"]

    source_dir = archive.VB_EXP100_DIR
    manifest_lines = (source_dir / "inputs/items.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 75
    models = [
        model for model in closed_forecasts["arm_a_specs"]
        if model["judge_id"] in CLOSED_JUDGE_IDS
    ]
    assert len(models) == 2
    forecasts_by_judge = {
        judge_id: {row["item"]: row for row in closed_forecasts["native_prob_forecasts"][judge_id]}
        for judge_id in CLOSED_JUDGE_IDS
    }

    matched = repairs = marked_files = 0
    for line in manifest_lines:
        item = json.loads(line)
        item_id = item["item"]
        texts = []
        for path_field, hash_field in (
            ("py_ann_path", "py_ann_sha256"),
            ("ref_ann_path", "ref_ann_disk_sha256"),
            ("candidate_path", "candidate_sha256"),
        ):
            file_bytes = (source_dir / item[path_field]).read_bytes()
            assert hashlib.sha256(file_bytes).hexdigest() == open_artifacts[item_id][hash_field]
            texts.append(file_bytes.decode("utf-8", errors="replace"))
        python_text, reference_text, candidate_text = texts
        marked_files += sum(bool(RATING_FIELDS.search(text)) for text in texts)

        system, user = closed_prompts.render_single_call_prompt(
            py_name=Path(item["py_ann_path"]).name,
            py_text=python_text,
            ref_name=item["gold_file_name"],
            ref_text=reference_text,
            cand_name=item["candidate_file"],
            cand_text=candidate_text,
            rubric_text=closed_prompts.get_base_rubric_text("P0_human_mirror"),
        )
        for model in models:
            expected = forecasts_by_judge[model["judge_id"]][item_id]
            slot = verify_cached_forecast(model, expected, system, user, archive, closed_prompts)
            matched += 1
            repairs += int(slot == 99)

    assert marked_files == 0
    print(
        "All 75 three-file inputs hash-match both studies. "
        f"Annotation-era reference hashes match {annotation_matches}/75."
    )
    print("Open P0 system fingerprint and common human rubric verified.")
    print(
        f"Rebuilt and matched {matched}/150 closed P0 input fingerprints "
        f"and parsed outputs ({repairs} repairs)."
    )
    print(f"Named rating-field markers found in {marked_files}/225 artifact-file occurrences.")


if __name__ == "__main__":
    main()
