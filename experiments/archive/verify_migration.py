"""Verify the preserved objective-study migration without running an experiment."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# Folder lifecycle dates were added on 11-13-2026 (agents-config Rule 88). The manifest
# keeps the destination paths recorded at migration time; read files from the current folders.
RENAMED_11_13_2026 = {
    'experiments/03_exp06_ce_vs_pearson_matched_budget/':
        'experiments/03_exp06_ce_vs_pearson_matched_budget_start_09Nov2026_end_09Nov2026/',
    'experiments/archive/11_formulation4_train_only_ce_mixture_cv/':
        'experiments/archive/11_formulation4_train_only_ce_mixture_cv_start_10Nov2026_end_10Nov2026/',
    'experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits/':
        'experiments/archive/12_formulation2_vs_4_corr_vs_ce_80_20_splits_start_10Nov2026_end_10Nov2026/',
    'experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings/':
        'experiments/archive/13_toy_pearson_vs_ce_nonlinear_ratings_start_10Nov2026_end_10Nov2026/',
    'experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20/':
        'experiments/archive/14_open_weight_judge_pool_corr_vs_ce_80_20_start_10Nov2026_end_10Nov2026/',
}


def current_path(recorded):
    text = recorded.as_posix()
    for old, new in RENAMED_11_13_2026.items():
        if text.startswith(old):
            return Path(new + text[len(old):])
    return recorded


def verify(source_repo=None, check_source_removed=False):
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'experiments/archive/MIGRATION_11-11-2026.json').read_text())
    python_files = json_files = identical = notice_only = 0
    seen = set()
    for record in manifest['files']:
        relative = Path(record['destination_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert str(relative) not in seen, relative
        seen.add(str(relative))
        data = (root / current_path(relative)).read_bytes()
        assert sha256(data) == record['destination_sha256'], relative
        if record['preservation'] == 'byte_identical':
            assert sha256(data) == record['source_sha256'], relative
            assert len(data) == record['bytes'], relative
            identical += 1
        else:
            assert relative == Path('experiments/03_exp06_ce_vs_pearson_matched_budget/historical_exp06/README.md')
            assert record['preservation'] == 'original_readme_plus_relocation_notice_in_source'
            notice_only += 1
        if relative.suffix == '.py':
            ast.parse(data.decode(), filename=str(relative))
            python_files += 1
        elif relative.suffix == '.json':
            json.loads(data)
            json_files += 1
        if source_repo:
            recorded = subprocess.check_output(
                ['git', 'show', manifest['source_commit'] + ':' + record['source_path']], cwd=source_repo)
            assert sha256(recorded) == record['source_sha256'], record['source_path']
            blob = subprocess.check_output(
                ['git', 'rev-parse', manifest['source_commit'] + ':' + record['source_path']],
                cwd=source_repo, text=True).strip()
            assert blob == record['source_git_blob'], record['source_path']
            if record['preservation'] != 'byte_identical':
                expected = data.decode().replace(
                    '**TLDR:**', manifest['source06_navigation_notice'] + '\n\n**TLDR:**', 1)
                assert recorded.decode() == expected, 'Unexpected scientific README difference'
    assert len(seen) == sum(manifest['file_counts'].values()) == 195
    assert identical == 194 and notice_only == 1
    for name, count in manifest['file_counts'].items():
        prefix = 'experiments/archive/' + name + '/'
        assert sum(r['source_path'].startswith(prefix) for r in manifest['files']) == count
    if check_source_removed:
        assert source_repo, '--check-source-removed requires --source-repo'
        for name in manifest['file_counts']:
            assert not (source_repo / 'experiments/archive' / name).exists(), name
    return {
        'status': 'PASS', 'preserved_files': len(seen), 'byte_identical_source_files': identical,
        'navigation_notice_only': notice_only, 'newly_transferred_files': manifest['new_files_copied'],
        'python_syntax_files': python_files, 'parsed_json_files': json_files,
        'source_git_verified': bool(source_repo), 'source_trees_removed': check_source_removed,
        'scientific_runs_started': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path)
    parser.add_argument('--check-source-removed', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.source_repo, args.check_source_removed), indent=2))


if __name__ == '__main__':
    main()
