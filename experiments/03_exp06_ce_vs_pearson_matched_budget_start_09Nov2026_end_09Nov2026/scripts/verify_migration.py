"""TLDR: Verify preserved experiment files and private relocation without fitting any models."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, with_private=False):
    manifest = json.loads((root / "MIGRATION_MANIFEST_11-09-2026.json").read_text())
    for record in manifest["files"]:
        path = root / record["destination"]
        assert path.stat().st_size == record["bytes"], path
        assert digest(path) == record["sha256"], path
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
    assert len(manifest["files"]) == manifest["public_file_count"]
    freeze = json.loads((root / "FREEZE.json").read_text())
    for relative, expected in freeze["files"].items():
        assert digest(root / relative) == expected, relative
    imported = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    # The original six source imports remain governed by their original manifest.
    for record in imported["files"].values():
        assert digest(root / record["snapshot"]) == record["sha256"]
    private_count = 0
    if with_private:
        receipt_path = root / manifest["private_receipt"]
        assert digest(receipt_path) == manifest["private_receipt_sha256"]
        receipt = json.loads(receipt_path.read_text())
        private = root / "private/historical_exp06"
        old_path = Path(receipt["source"])
        assert old_path.is_symlink() and old_path.resolve() == private.resolve()
        for record in receipt["files"]:
            path = private / record["path"]
            assert path.stat().st_size == record["bytes"], record["path"]
            assert digest(path) == record["sha256"], record["path"]
        private_count = len(receipt["files"])
        assert private_count == manifest["private_file_count"]
        assert len(list((private / "inputs/notes_11-07-2026/assets").glob("*.png"))) == 4
        protocol = json.loads((root / "protocol.json").read_text())
        assert digest(private / "cached_v1/dataset.json") == protocol["dataset_sha256"]
    return {
        "status": "PASS", "public_files_verified": len(manifest["files"]),
        "original_frozen_files_verified": len(freeze["files"]),
        "original_source_imports_verified": len(imported["files"]),
        "private_files_verified": private_count, "private_checked": with_private,
        "scientific_runs_started": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-private", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = verify(Path(__file__).resolve().parents[1], args.with_private)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
