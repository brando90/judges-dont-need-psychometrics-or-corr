#!/usr/bin/env python3
"""Sync Experiment 25 report, figures, CSVs, and reproducibility zip to the Google Drive Desktop folder."""
from __future__ import annotations

import html
import json
import re
import shutil
import zipfile
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
LOCAL_DRIVE_DIR = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-brandojazz@gmail.com/My Drive/Agent Sync (human-readable)/judges-dont-need-psychometrics-or-corr/experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026"
)


def inline_md_to_html(text: str) -> str:
    esc = html.escape(text)
    esc = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
    esc = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc)
    esc = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", esc)
    esc = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', esc)
    return esc


def _open_ul(out: list[str], in_ul: bool) -> bool:
    if not in_ul:
        out.append("<ul>")
    return True


def _flush_ul(out: list[str], in_ul: bool) -> bool:
    if in_ul:
        out.append("</ul>")
    return False


def _render_table(lines: list[str], start_idx: int) -> tuple[list[str], int]:
    headers = [c.strip() for c in lines[start_idx].strip().strip("|").split("|")]
    out = [
        '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; font-size: 9.5pt;">',
        "<tr>" + "".join(f"<th>{inline_md_to_html(h)}</th>" for h in headers) + "</tr>",
    ]
    idx = start_idx + 2
    while idx < len(lines) and lines[idx].startswith("|"):
        cols = [c.strip() for c in lines[idx].strip().strip("|").split("|")]
        out.append("<tr>" + "".join(f"<td>{inline_md_to_html(c)}</td>" for c in cols) + "</tr>")
        idx += 1
    out.append("</table>")
    return out, idx


def _render_block_line(line: str, out: list[str]) -> None:
    for prefix, tag in (("# ", "h1"), ("## ", "h2"), ("### ", "h3")):
        if line.startswith(prefix):
            out.append(f"<{tag}>{inline_md_to_html(line[len(prefix):].strip())}</{tag}>")
            return
    if line.startswith("!["):
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if m:
            out.append(f"<p><em>[Figure: {html.escape(m.group(2))} — {html.escape(m.group(1))}]</em></p>")
        return
    out.append(f"<p>{inline_md_to_html(line)}</p>")


def md_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    out = ['<html><body style="font-family: Arial, sans-serif; font-size: 11pt;">']
    in_ul = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            in_ul = _flush_ul(out, in_ul)
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|---"):
            in_ul = _flush_ul(out, in_ul)
            tbl_lines, i = _render_table(lines, i)
            out.extend(tbl_lines)
            continue
        if line.startswith("- "):
            in_ul = _open_ul(out, in_ul)
            out.append(f"<li>{inline_md_to_html(line[2:].strip())}</li>")
        else:
            in_ul = _flush_ul(out, in_ul)
            _render_block_line(line, out)
        i += 1
    _flush_ul(out, in_ul)
    out.append("</body></html>")
    return "\n".join(out)


def _write_repro_zip(zip_path: Path) -> None:
    entries: list[tuple[Path, str]] = [
        (EXP_DIR / rel, rel)
        for rel in (
            "README.md",
            "PROTOCOL_11-21-2026.md",
            "HOW_TO_READ_RESULTS.md",
            "RESULTS_11-21-2026.md",
            "BUDGET_TABLES_11-21-2026.md",
            "CKPT_exp25_scale_and_llama.md",
        )
    ]
    entries.extend((p, f"scripts/{p.name}") for p in sorted((EXP_DIR / "scripts").glob("*.py")))
    entries.extend((p, f"results/{p.name}") for p in sorted((EXP_DIR / "results").iterdir()) if p.name != "email_receipts.json")
    entries.extend((p, f"figures/{p.name}") for p in sorted((EXP_DIR / "figures").glob("*")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arcname in entries:
            zf.write(src_path, arcname=arcname)


def main() -> None:
    results_md = (EXP_DIR / "RESULTS_11-21-2026.md").read_text(encoding="utf-8")
    tables_md = (EXP_DIR / "BUDGET_TABLES_11-21-2026.md").read_text(encoding="utf-8")
    html_bytes = md_to_html(results_md + "\n\n---\n\n" + tables_md).encode("utf-8")

    LOCAL_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    html_local_path = LOCAL_DRIVE_DIR / "Experiment_25_RESULTS_11-21-2026.html"
    html_local_path.write_bytes(html_bytes)

    copies = {
        EXP_DIR / "RESULTS_11-21-2026.md": LOCAL_DRIVE_DIR / "Experiment_25_RESULTS_11-21-2026.md",
        EXP_DIR / "BUDGET_TABLES_11-21-2026.md": LOCAL_DRIVE_DIR / "Experiment_25_BUDGET_TABLES_11-21-2026.md",
        EXP_DIR / "HOW_TO_READ_RESULTS.md": LOCAL_DRIVE_DIR / "Experiment_25_HOW_TO_READ_RESULTS.md",
        EXP_DIR / "figures/exp25_qwen3_4b_scale_curves.png": LOCAL_DRIVE_DIR / "Experiment_25_qwen3_4b_scale_curves_11-21-2026.png",
        EXP_DIR / "figures/exp25_llama31_8b_and_cross_family.png": LOCAL_DRIVE_DIR / "Experiment_25_llama31_8b_and_cross_family_11-21-2026.png",
        EXP_DIR / "results/scores.csv": LOCAL_DRIVE_DIR / "Experiment_25_scores_11-21-2026.csv",
        EXP_DIR / "results/contrasts.csv": LOCAL_DRIVE_DIR / "Experiment_25_contrasts_11-21-2026.csv",
    }
    for src, dst in copies.items():
        shutil.copyfile(src, dst)

    zip_path = LOCAL_DRIVE_DIR / "Experiment_25_reproducibility_11-21-2026.zip"
    _write_repro_zip(zip_path)

    receipt = {
        "local_drive_dir": str(LOCAL_DRIVE_DIR),
        "native_gdoc_status": "skipped_no_collaborators_desktop_mirror_synced",
        "local_drive_copies": [p.name for p in [*copies.values(), html_local_path, zip_path]],
    }
    (EXP_DIR / "results" / "drive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
