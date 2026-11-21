"""Join released expert annotations, automatic metric scores, and archived G-Eval responses."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import re
import tempfile
from urllib.request import urlopen

import numpy as np

G_EVAL = "https://raw.githubusercontent.com/nlpyang/geval/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/"
BART = "https://raw.githubusercontent.com/neulab/BARTScore/248f511cb34ae3753fc81f7d7a945de5bfe33458/"
FILES = {
    "model_annotations.aligned.jsonl": (
        "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl",
        "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab"),
    "bart_metrics.pkl": (BART + "SUM/SummEval/final_p.pkl",
                         "b5a19921636e95719b076bd067543409256b7fffd426e9b16fd4db4d0a4f632a"),
    "gpt4_con_detailed.json": (G_EVAL + "results/gpt4_con_detailed.json",
                               "e09609ab249fc6aa2e4f11b7c16d7531b7036b163c8251e54a04ce4c3efa76ed"),
    "gpt4_rel_detailed.json": (G_EVAL + "results/gpt4_rel_detailed.json",
                               "c222933ff5ba552bb9ec3d2b495022a89b8019cf246710988a4e49804d47585a"),
    "con_detailed.txt": (G_EVAL + "prompts/summeval/con_detailed.txt",
                         "ce74a0de8f19f2f0d31c77d1df76eece9b29fd14fee1f2bb82b462846eac62fa"),
    "rel_detailed.txt": (G_EVAL + "prompts/summeval/rel_detailed.txt",
                         "2e836f9d9aece1832065b1a166a14d4ec0c050085adcee396af8d365b13daf1d"),
}
FEATURES = ("rouge2_f", "rougel_f", "bert_score_f", "bart_score_src_hypo")
GRADE_LINE = re.compile(r"\A\s*([1-5])(?:[ \t]*/[ \t]*5)?[ \t]*(?:\r?\n|\Z)")


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = directory / name
        if not path.exists():
            with urlopen(url, timeout=90) as source:
                data = source.read()
                assert hashlib.sha256(data).hexdigest() == expected, name
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
                tmp.write(data)
            Path(tmp.name).replace(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # The hash-checked release contains only Python containers and these NumPy scalar types.
        from numpy.core.multiarray import scalar
        return {("numpy.core.multiarray", "scalar"): scalar, ("numpy", "dtype"): np.dtype}[module, name]


def candidate_key(text):
    without_brackets = re.sub(r"-(?:lrb|rrb|lsb|rsb|lcb|rcb|lrc|rrc)-", " ", text.lower())
    return "".join(re.findall(r"\w+", without_brackets))


def agrees_with_archive(text, expected):
    if text == expected:
        return True
    return text.encode("cp1252").decode("utf8") == expected


def duplicate_report(rows, key):
    groups = defaultdict(list)
    for (article, _), row in rows.items():
        groups[key(row["decoded"])].append(article)
    repeated = [articles for articles in groups.values() if len(articles) > 1]
    return dict(unique_candidates=len(groups), repeated_clusters=len(repeated),
                rows_in_repeated_clusters=sum(map(len, repeated)),
                clusters_crossing_articles=sum(len(set(articles)) > 1 for articles in repeated))


def prepare(cache, output):
    fetch(cache)
    annotations = [json.loads(line) for line in (cache / "model_annotations.aligned.jsonl").read_bytes().splitlines()]
    raw = {(row["id"], row["model_id"]): row for row in annotations}
    keys = sorted(raw)
    assert len(keys) == len(annotations) == 1600
    assert len({a for a, _ in keys}) == 100 and len({s for _, s in keys}) == 16
    with (cache / "bart_metrics.pkl").open("rb") as stream:
        archives = NumericUnpickler(stream).load()
    assert set(keys) == {(article, system) for article, doc in archives.items() for system in doc["sys_summs"]}
    assert len({doc["src"] for doc in archives.values()}) == 100
    score_rows = []
    for article, system in keys:
        item = archives[article]["sys_summs"][system]
        assert candidate_key(item["sys_summ"]) == candidate_key(raw[article, system]["decoded"])
        score_rows.append([float(item["scores"][feature]) for feature in FEATURES])
    features = np.asarray(score_rows)
    assert features.shape == (1600, 4) and np.isfinite(features).all()
    assert all(len(row["expert_annotations"]) == 3 and len(row["turker_annotations"]) == 5
               for row in raw.values())
    print("Exact-text duplicate audit", duplicate_report(raw, lambda text: text))
    print("Normalized-text duplicate audit", duplicate_report(raw, candidate_key))
    output.mkdir(parents=True, exist_ok=True)
    for dimension, short in (("consistency", "con"), ("relevance", "rel")):
        judgments = json.loads((cache / f"gpt4_{short}_detailed.json").read_bytes())
        by_key = {(r["doc_id"], r["system_id"]): r for r in judgments}
        assert len(by_key) == len(judgments) and set(by_key) == set(keys)
        template = (cache / f"{short}_detailed.txt").read_text()
        votes = np.zeros((len(keys), 5), dtype=np.int64)
        counts = np.zeros_like(votes)
        changed_source = changed_candidate = 0
        for row_index, key in enumerate(keys):
            article, system = key
            response = by_key[key]
            expected_source = archives[article]["src"]
            expected_candidate = archives[article]["sys_summs"][system]["sys_summ"]
            assert agrees_with_archive(response["source"], expected_source)
            assert agrees_with_archive(response["system_output"], expected_candidate)
            changed_source += response["source"] != expected_source
            changed_candidate += response["system_output"] != expected_candidate
            rebuilt = template.replace("{{Document}}", response["source"]).replace("{{Summary}}", response["system_output"])
            assert rebuilt == response["prompt"]
            for annotation in raw[key]["expert_annotations"]:
                grade = annotation[dimension]
                assert type(grade) is int and 1 <= grade <= 5
                counts[row_index, grade - 1] += 1
            assert len(response["all_responses"]) == 20
            for completion in response["all_responses"]:
                match = GRADE_LINE.match(completion)
                if match:
                    votes[row_index, int(match.group(1)) - 1] += 1
        valid = votes.sum(axis=1)
        native = np.divide(votes, valid[:, None], out=np.full(votes.shape, 0.2), where=valid[:, None] > 0)
        np.testing.assert_allclose(native.sum(axis=1), 1)
        assert (counts.sum(axis=1) == 3).all()
        path = output / f"summeval_{dimension}.npz"
        np.savez_compressed(path, X=features, counts=counts, group=np.array([a for a, _ in keys]),
                            id=np.array([a + "::" + s for a, s in keys]), features=np.array(FEATURES),
                            system=np.array([s for _, s in keys]), native=native, native_valid_votes=valid)
        print(dimension, "valid votes", int(valid.sum()), "of", len(keys)*20,
              "per-item valid-vote counts", sorted(Counter(map(int, valid)).items()), "fallbacks", int((valid == 0).sum()),
              "source rows requiring encoding repair", changed_source, "candidate rows", changed_candidate)
        print(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.cache, args.output)
