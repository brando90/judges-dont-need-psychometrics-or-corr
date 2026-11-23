#!/usr/bin/env python3
"""Sync Experiment 24 report, figures, CSVs, zip archive, and native Google Doc to Google Drive."""
from __future__ import annotations

import html
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

EXP_DIR = Path(__file__).resolve().parent.parent
DRIVE_FOLDER_ID = "1QKHpBRLFOaYIw5IHwv2yAqZaHvBq-moG"
LOCAL_DRIVE_DIR = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-brandojazz@gmail.com/My Drive/Agent Sync (human-readable)/judges-dont-need-psychometrics-or-corr"
)
CREDS_FILE = Path.home() / "keys/gspread_authorized_user.json"
GDOC_TITLE = "When does cross-entropy weight training beat Pearson for open-weight LLM judges? Experiment 24 (11-21-2026)"
HEADING_TAGS = (("# ", "h1"), ("## ", "h2"), ("### ", "h3"))


def inline_md_to_html(text: str) -> str:
    esc = html.escape(text)
    esc = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
    esc = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc)
    esc = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", esc)
    esc = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', esc)
    return esc


def _consume_table(lines: list[str], start: int) -> tuple[list[str], int]:
    headers = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    out = [
        '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; font-size: 9.5pt;">',
        "<tr>" + "".join(f"<th>{inline_md_to_html(h)}</th>" for h in headers) + "</tr>",
    ]
    idx = start + 2
    while idx < len(lines) and lines[idx].startswith("|"):
        vals = [c.strip() for c in lines[idx].strip().strip("|").split("|")]
        out.append("<tr>" + "".join(f"<td>{inline_md_to_html(c)}</td>" for c in vals) + "</tr>")
        idx += 1
    out.append("</table>")
    return out, idx


def _render_non_list_line(line: str) -> str | None:
    for prefix, tag in HEADING_TAGS:
        if line.startswith(prefix):
            return f"<{tag}>{inline_md_to_html(line[len(prefix):].strip())}</{tag}>"
    if line.startswith("!["):
        match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if not match:
            return None
        return f"<p><em>[Figure in repository/Drive: {html.escape(match.group(2))} — {html.escape(match.group(1))}]</em></p>"
    return f"<p>{inline_md_to_html(line)}</p>"


def md_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    out = ['<html><body style="font-family: Arial, sans-serif; font-size: 11pt;">']
    in_ul = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            i += 1
            continue
        if line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline_md_to_html(line[2:].strip())}</li>")
            i += 1
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|---"):
            table_html, i = _consume_table(lines, i)
            out.extend(table_html)
            continue
        rendered = _render_non_list_line(line)
        if rendered is not None:
            out.append(rendered)
        i += 1
    if in_ul:
        out.append("</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def _write_repro_zip(zip_path: Path) -> None:
    doc_names = (
        "README.md",
        "PROTOCOL_11-21-2026.md",
        "HOW_TO_READ_RESULTS.md",
        "RESULTS_11-21-2026.md",
        "BUDGET_TABLES_11-21-2026.md",
        "CKPT_helpsteer2_lora.md",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in doc_names:
            zf.write(EXP_DIR / rel, arcname=rel)
        for p in sorted((EXP_DIR / "scripts").glob("*.py")):
            zf.write(p, arcname=f"scripts/{p.name}")
        for p in sorted((EXP_DIR / "results").iterdir()):
            if p.name != "email_receipts.json":
                zf.write(p, arcname=f"results/{p.name}")
        for p in sorted((EXP_DIR / "figures").glob("*")):
            zf.write(p, arcname=f"figures/{p.name}")


def _sync_local_drive_files(html_bytes: bytes) -> list[str]:
    LOCAL_DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    html_local_path = LOCAL_DRIVE_DIR / "Experiment_24_RESULTS_11-21-2026.html"
    html_local_path.write_bytes(html_bytes)
    copies = {
        EXP_DIR / "RESULTS_11-21-2026.md": LOCAL_DRIVE_DIR / "Experiment_24_RESULTS_11-21-2026.md",
        EXP_DIR / "figures/helpsteer2_lora_learning_curves.png": LOCAL_DRIVE_DIR / "Experiment_24_helpsteer2_lora_learning_curves_11-21-2026.png",
        EXP_DIR / "figures/helpsteer2_calibration_and_depth.png": LOCAL_DRIVE_DIR / "Experiment_24_helpsteer2_calibration_and_depth_11-21-2026.png",
        EXP_DIR / "results/scores.csv": LOCAL_DRIVE_DIR / "Experiment_24_scores_11-21-2026.csv",
        EXP_DIR / "results/contrasts.csv": LOCAL_DRIVE_DIR / "Experiment_24_contrasts_11-21-2026.csv",
    }
    for src, dst in copies.items():
        shutil.copyfile(src, dst)
    zip_path = LOCAL_DRIVE_DIR / "Experiment_24_reproducibility_11-21-2026.zip"
    _write_repro_zip(zip_path)
    return [str(p) for p in [*copies.values(), html_local_path, zip_path]]


def _upload_native_gdoc(html_bytes: bytes, synced_files: list[str]) -> dict:
    drive_folder_url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"
    try:
        creds = Credentials.from_authorized_user_info(json.loads(CREDS_FILE.read_text(encoding="utf-8")))
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        q = f"trashed = false and '{DRIVE_FOLDER_ID}' in parents and name = '{GDOC_TITLE}'"
        found = drive.files().list(q=q, fields="files(id, name, webViewLink)").execute().get("files", [])
        media = MediaIoBaseUpload(io.BytesIO(html_bytes), mimetype="text/html", resumable=False)
        if found:
            file_id = found[0]["id"]
            drive.files().update(fileId=file_id, media_body=media, fields="id, name, webViewLink").execute()
        else:
            meta = {"name": GDOC_TITLE, "mimeType": "application/vnd.google-apps.document", "parents": [DRIVE_FOLDER_ID]}
            file_id = drive.files().create(body=meta, media_body=media, fields="id, name, webViewLink").execute()["id"]
        drive.permissions().create(fileId=file_id, body={"type": "anyone", "role": "reader"}, fields="id, type, role").execute()
        meta_check = drive.files().get(fileId=file_id, fields="id, name, webViewLink, shared, permissions(type, role)").execute()
        return {
            "drive_folder_id": DRIVE_FOLDER_ID,
            "drive_folder_url": drive_folder_url,
            "native_gdoc_status": "synced",
            "google_doc_id": meta_check["id"],
            "google_doc_url": meta_check["webViewLink"],
            "shared": meta_check["shared"],
            "permissions": meta_check["permissions"],
            "local_drive_copies": synced_files,
        }
    except Exception as exc:
        return {
            "drive_folder_id": DRIVE_FOLDER_ID,
            "drive_folder_url": drive_folder_url,
            "native_gdoc_status": "blocked_oauth_expired",
            "native_gdoc_error_type": type(exc).__name__,
            "native_gdoc_error_message": str(exc),
            "google_doc_id": None,
            "google_doc_url": None,
            "shared": None,
            "permissions": [],
            "local_drive_copies": synced_files,
        }


def main() -> None:
    results_md = (EXP_DIR / "RESULTS_11-21-2026.md").read_text(encoding="utf-8")
    tables_md = (EXP_DIR / "BUDGET_TABLES_11-21-2026.md").read_text(encoding="utf-8")
    html_bytes = md_to_html(results_md + "\n\n---\n\n" + tables_md).encode("utf-8")
    synced_files = _sync_local_drive_files(html_bytes)
    receipt = _upload_native_gdoc(html_bytes, synced_files)
    (EXP_DIR / "results" / "drive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
