#!/usr/bin/env python3
"""Send [RUN START] / [RUN DONE] notification email via Brando's Gmail SMTP app password."""
from __future__ import annotations

import argparse
import json
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
RECEIPTS_PATH = EXP_DIR / "results" / "email_receipts.json"
PASS_FILE = Path.home() / "keys" / "gmail_app_password.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    password = PASS_FILE.read_text(encoding="utf-8").strip()
    assert password, f"Empty password in {PASS_FILE}"

    msg = EmailMessage()
    msg["From"] = "brandojazz@gmail.com"
    msg["To"] = "brando.science@gmail.com"
    msg["Subject"] = args.subject
    msg.set_content(args.body)

    recipients = ["brando.science@gmail.com"]
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login("brandojazz@gmail.com", password)
        refused = smtp.send_message(msg, from_addr="brandojazz@gmail.com", to_addrs=recipients)

    receipt = {
        "tag": args.tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "from": "brandojazz@gmail.com",
        "to": ["brando.science@gmail.com"],
        "bcc": ["brando.science@gmail.com"],
        "subject": args.subject,
        "refused": refused,
        "status": "sent" if not refused else "partial_refusal",
    }
    existing = json.loads(RECEIPTS_PATH.read_text(encoding="utf-8")) if RECEIPTS_PATH.exists() else []
    existing.append(receipt)
    RECEIPTS_PATH.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"Email {args.tag} sent: subject={args.subject!r}, refused={refused}")


if __name__ == "__main__":
    main()
