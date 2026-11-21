"""Acquire pinned USR public files, verify joins, and run the preregistered dialogue extension."""
import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze
from candidates import R, bin_distribution, bin_indices, build, per_item
from design import Data, draw

RELEASE = "https://raw.githubusercontent.com/Shikib/usr/b1e50bb05f11d193cf9fde96e94d189ad84fc57d/"
SOURCES = {
    "pc.json": ("https://shikib.com/pc_usr_data.json", "36f4cb78b1381f4ae3325f068f68a07d45ec6e944281c4535f121359faca2676"),
    "tc.json": ("https://shikib.com/tc_usr_data.json", "e3263bf756e99a44d839815054aab83f6dc765bad6a6b2a55f154b2d07759366"),
    "examples_both_dev.tsv": (RELEASE + "examples/both/dev.tsv", "b047b3387f7e030e3cadfbaf12acfa84ea29544c6bc0b010e5052c622c851481"),
    "examples_fct_dev.tsv": (RELEASE + "examples/fct/dev.tsv", "4b77188719b4ce1137ecfd09fb06f22b64a56298d1bd1c471ae2d59a238e418b"),
    "examples_undr_test.lm": (RELEASE + "examples/undr/test.lm", "da12c47b2e2ace74f21c34bf5c070b2bda2b4d0dfbfc07b03fefe7ad2143bd52"),
    "examples_both_dr.scores": (RELEASE + "examples/both/dr.scores", "0f5e7f3fc2f5a0b19d11338e7c8248e3009e6ce0da174dc8a263de71d330261b"),
    "examples_fct_dr.scores": (RELEASE + "examples/fct/dr.scores", "4363b7aa0623410b8a83c3e909d7c9e7fbc6a2de08c6f59144054c801d7515ff"),
    "examples_undr_mlm_roberta.scores": (RELEASE + "examples/undr/mlm_roberta.scores", "248a6d1259834a0bff91d3c428e61e50a71907b8fbe9dd10d09d8f368dd9fd57"),
}
TRACKS = ("topical_usr", "topical_lexical", "persona_lexical")
LEXICAL = ("log_words", "distinct_fraction", "context_overlap", "grounding_overlap")
COMPONENTS = ("context_and_fact_retrieval", "fact_retrieval", "masked_language_model")


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical(response, context, fact):
    words = tokens(response)
    denominator = max(len(words), 1)
    context_words, fact_words = set(tokens(context)), set(tokens(fact))
    return (np.log1p(len(words)), len(set(words))/denominator,
            sum(w in context_words for w in words)/denominator,
            sum(w in fact_words for w in words)/denominator)


def fetch(cache):
    cache.mkdir(exist_ok=True, parents=True)
    for fname, (url, digest) in SOURCES.items():
        path = cache / fname
        if not path.exists():
            payload = urlopen(url, timeout=40).read()
            assert hashlib.sha256(payload).hexdigest() == digest, url
            path.write_bytes(payload)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, fname


def components(cache, topical):
    """Join by response text on the five archived systems; verify all three sets of model inputs."""
    def rows(name):
        return (cache / name).read_text().splitlines()

    reference = {}
    for context_index, group in enumerate(topical):
        for response in group["responses"]:
            if response["model"] == "Original Ground Truth":
                continue
            key = tuple(tokens(response["response"]))
            assert key not in reference
            reference[key] = (context_index, response["model"])
    combined = rows("examples_both_dev.tsv")
    facts = rows("examples_fct_dev.tsv")
    assert combined[0] == combined[1] and facts[0] == facts[1]
    mlm = rows("examples_undr_test.lm")
    scores = np.column_stack([ast.literal_eval((cache / file).read_text())
                              for file in ("examples_both_dr.scores", "examples_fct_dr.scores",
                                           "examples_undr_mlm_roberta.scores")])
    assert scores.shape == (300, 3) and np.isfinite(scores).all()
    assert len(mlm) == len(facts)-1 == len(combined)-1 == len(reference) == len(scores)
    joined = {}
    for row, (both, fact, masked) in enumerate(zip(combined[1:], facts[1:], mlm)):
        both, fact = both.split("\t"), fact.split("\t")
        assert both[4] == fact[4]
        wrapped = tokens(both[4])
        assert wrapped[0] == "go" and wrapped[-1] == "eos"
        key = reference[tuple(wrapped[1:-1])]
        assert key not in joined
        source = topical[key[0]]
        context_fragment = tokens(both[3].split("_eos")[0])
        fact_fragment = tokens(fact[3].split("_eos")[0])
        assert context_fragment and fact_fragment
        assert " ".join(context_fragment) in " ".join(tokens(source["context"]))
        assert " ".join(fact_fragment) in " ".join(tokens(source["fact"]))
        assert tokens(masked)[-len(wrapped):] == wrapped
        joined[key] = scores[row]
    assert len(joined) == 300
    return joined


def prepare(cache, track):
    corpus = "pc" if track == "persona_lexical" else "tc"
    contexts = json.loads((cache / (corpus + ".json")).read_bytes())
    archived = components(cache, contexts) if track == "topical_usr" else {}
    keys = [hashed("corrce23-usr|" + row["context"]) for row in contexts]
    assert len(set(keys)) == len(contexts) == 60
    features, counts, groups, systems, ids, normalized_replies = [], [], [], [], [], []
    for i, group in enumerate(contexts):
        assert len(set(group["annotators"])) == len(group["annotators"]) == 3
        for response in group["responses"]:
            system = response["model"]
            grades = response["Overall"]
            assert len(grades) == 3 and all(type(g) is int and 1 <= g <= 5 for g in grades)
            if track == "topical_usr" and system == "Original Ground Truth":
                continue
            features.append(archived[i, system] if archived else
                            lexical(response["response"], group["context"], group["fact"]))
            normalized_replies.append(" ".join(tokens(response["response"])))
            counts.append(np.bincount(grades, minlength=6)[1:])
            groups.append(keys[i])
            systems.append(system)
            ids.append(hashed(keys[i] + "|" + system))
    group_array = np.array(groups)
    heldout = np.isin(group_array, sorted(keys)[:24])
    assert len(set(ids)) == len(ids) and len(set(Counter(groups).values())) == 1
    assert len(set(group_array[heldout])) == 24 and len(set(group_array[~heldout])) == 36
    assert set(np.array(systems)[heldout]) == set(np.array(systems)[~heldout])
    shared = set(np.array(normalized_replies)[heldout]) & set(np.array(normalized_replies)[~heldout])
    info = dict(track=track, responses=len(groups), contexts=60, individual_grades=3*len(groups),
                systems=len(set(systems)), raters=len({r for row in contexts for r in row["annotators"]}),
                test_contexts=24, test_items=int(heldout.sum()), train_contexts=36,
                train_items=int((~heldout).sum()), shared_normalized_reply_strings=len(shared))
    data = Data(track, np.array(features), np.array(counts), group_array, np.array(ids),
                np.array(COMPONENTS if archived else LEXICAL), np.array(systems), None,
                np.flatnonzero(~heldout), np.flatnonzero(heldout))
    return data, info


def symmetric_calibration(data, output, budgets):
    with (output / "selections.csv").open() as f:
        selections = {(int(r["draw"]), int(r["budget"])): r for r in csv.DictReader(f) if r["depth"] == "1"}
    with (output / "scores.csv").open() as f:
        baseline = {int(r["budget"]): r for r in csv.DictReader(f)
                    if r["method"] == "calibrated_pearson" and r["depth"] == "1"}
    q, names = build(data, "features")
    means = q @ R
    bins = {name: bin_indices(means[data.train, j], means[:, j]) for j, name in enumerate(names)}
    methods = ("ce", "pearson")
    metrics = ("rps", "ce", "tvd")
    observed = data.counts[data.test] / 3
    averages = np.zeros((len(budgets), len(methods), len(data.test), len(metrics)))
    for rep in range(64):
        order, picked = draw(data, rep, 1)
        for i, budget in enumerate(budgets):
            observed_train = picked[:budget]
            prior = (observed_train.sum(axis=0) + .5) / (budget + 2.5)
            for j, method in enumerate(methods):
                partition = bins[selections[rep, budget][method]]
                posterior = bin_distribution(partition, order[:budget], observed_train, prior)
                losses = per_item(posterior[partition[data.test]], observed)
                averages[i, j] += losses[:, :3] / 64
    groups = data.group[data.test]
    deltas = averages[:, 1] - averages[:, 0]
    by_group = np.stack([deltas[:, groups == g].mean(axis=1) for g in sorted(set(groups))])
    rng = np.random.default_rng(20261124)
    group_count = len(by_group)
    weights = rng.multinomial(group_count, np.full(group_count, 1/group_count), size=4000) / group_count
    intervals = np.quantile(np.einsum("rg,gbm->rbm", weights, by_group), [.025, .975], axis=0)
    rows = []
    for i,budget in enumerate(budgets):
        for j, metric in enumerate(metrics):
            ce, pearson = averages[i, :, :, j].mean(axis=1)
            np.testing.assert_allclose(pearson, float(baseline[budget][metric]), rtol=0, atol=1e-12)
            rows.append(dict(budget=budget, metric=metric, test_items=len(groups), test_contexts=group_count,
                             calibrated_ce=ce, calibrated_pearson=pearson, pearson_minus_ce=pearson-ce,
                             paired_context_lower95=intervals[0,i,j], paired_context_upper95=intervals[1,i,j]))
    analyze.save(output / "symmetric_calibration.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/corr-ce-public23/usr"))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--track", choices=TRACKS, nargs="+", default=TRACKS)
    args = parser.parse_args()
    fetch(args.cache)
    for track in args.track:
        data, provenance = prepare(args.cache, track)
        print(provenance, flush=True)
        if args.prepare_only:
            continue
        budgets = sorted({6, 12, 24, 48, 96, 180, len(data.train)})
        analyze.BUDGETS = budgets
        output = args.out / track
        analyze.run(data, "features", output)
        symmetric_calibration(data, output, budgets)
        analyze.save(output / "coverage.csv", [provenance])


if __name__ == "__main__":
    main()
