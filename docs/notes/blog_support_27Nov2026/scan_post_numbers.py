#!/usr/bin/env python3
"""Scan a blog post and compare every number in its prose with the registry printed by blog_numbers_check.py.

Usage:
    python3 -I scan_post_numbers.py POST.md [CHECK_OUTPUT.txt]

What it does (deterministic, no network):
  1. Reads the registry of unsigned numeric tokens that blog_numbers_check.py printed after its PASS lines
     (default file: blog_numbers_check_output_run1_27Nov2026.txt next to this script).
  2. Strips front matter, fenced code blocks, link targets, heading numbers and list numbers from the post.
  3. Extracts every number token left (signs and percent signs dropped; ranges such as 80-82% give two tokens).
  4. Classifies each token: REGISTERED (exact match), ROUNDED (a decimal token that equals a registered token
     rounded to fewer decimals; reported so a reviewer can see it), YEAR (whitelisted citation years),
     SMALL (an integer 0-9, listed for manual review because words such as "one" or "two" carry the same meaning),
     or UNREGISTERED (exits 1).
  5. Prints banned strings found, the word count and the read time the Jekyll layout will show.

A token being registered means some PASSing claim in blog_numbers_check.py printed it; it does not prove the post
uses it for that claim. A reviewer still reads each sentence against the claim text (the check output lists both).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
post_path = Path(sys.argv[1])
check_out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "blog_numbers_check_output_run1_27Nov2026.txt"

# ---------------------------------------------------------------- registry
registry = {}
in_reg = False
for line in check_out.read_text().splitlines():
    if line.startswith("REGISTRY of numeric tokens"):
        in_reg = True
        continue
    if in_reg:
        m = re.match(r"^\s+(\S+)\s+(\S+)\s*$", line)
        if m:
            registry[m.group(1)] = m.group(2)
if not registry:
    sys.exit("no registry found in " + str(check_out))

# ---------------------------------------------------------------- strip the post
raw = post_path.read_text()
text = raw
if text.startswith("---"):
    text = text.split("---", 2)[2]  # drop front matter
text = re.sub(r"```.*?```", " ", text, flags=re.S)  # fenced code (bibtex)
text = re.sub(r"\]\([^)]*\)", "]", text)  # markdown link and image targets
text = re.sub(r"\{\{[^}]*\}\}", " ", text)  # liquid
text = re.sub(r"https?://\S+", " ", text)
text = re.sub(r"(?m)^#{1,6}\s*\d+\.\s*", "", text)  # heading numbers "## 3."
text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)  # list numbers "1. "
text = re.sub(r"(?m)^---\s*$", " ", text)
text = re.sub(r"(?m)^\*Brando Miranda[^\n]*\*\s*$", " ", text)  # byline carries the hand-written read time
MODEL_NAMES = [r"GPT-6 Astra", r"GPT-4", r"Claude Opus 5\.5", r"Opus 5\.5", r"Kimi K3", r"GLM 5\.3", r"DeepSeek V4\.1 Flash", r"MNLI", r"SNLI", r"DICES-350", r"DICES-990", r"DICES", r"\bSection 3\b", r"\bsection 3\b"]
for nm in MODEL_NAMES:
    text = re.sub(nm, " ", text)

YEARS = {"1988", "2000", "2019", "2022", "2024", "2025", "2026", "2027"}
LEVELS = {"95"}  # confidence level of intervals, a convention not a result
NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")

rows = {}
for m in NUM.finditer(text):
    tok = m.group(1).rstrip(",")
    if tok.endswith("."):
        tok = tok[:-1]
    rows.setdefault(tok, 0)
    rows[tok] += 1


def rounded_match(tok):
    """True if tok (a decimal) equals some registered decimal rounded to len(tok decimals)."""
    if "." not in tok:
        return None
    d = len(tok.split(".")[1])
    for r in registry:
        if "." in r and len(r.split(".")[1]) > d:
            try:
                if f"{float(r):.{d}f}" == tok:
                    return r
            except ValueError:
                pass
    return None


bad = []
print(f"{'token':>9}  {'n':>3}  status")
for tok in sorted(rows, key=lambda s: (float(s.replace(",", "")), s)):
    n = rows[tok]
    if tok in registry:
        if re.fullmatch(r"\d", tok):
            status = f"SMALL (manual review; registered for {registry[tok]})"
        else:
            status = f"REGISTERED ({registry[tok]})"
    elif tok in YEARS:
        status = "YEAR"
    elif tok in LEVELS:
        status = "LEVEL (interval confidence level)"
    elif rounded_match(tok):
        status = f"ROUNDED from {rounded_match(tok)} ({registry[rounded_match(tok)]})"
    elif re.fullmatch(r"\d", tok):
        status = "SMALL (manual review; not registered)"
    else:
        status = "UNREGISTERED"
        bad.append(tok)
    print(f"{tok:>9}  {n:>3}  {status}")

# ---------------------------------------------------------------- banned strings and layout facts
banned = {
    "gemini / share link": r"gemini|share\.google|g\.co/gemini",
    "advisor name": r"koyejo|sanmi",
    "appendix": r"appendix",
    "sources or materials list": r"(?im)^\s*(#+\s*)?(\*\*)?(sources|materials|references)\b",
    "provenance or AI note": r"claude\.ai|chatgpt|ai-generated|written with|generated (on|by)|conversation with",
    "internal working folder link": r"/experiments/|experiments/\d|\(experiments",
    "defensive hedging": r"fair use|paraphrased,|crops only|not quoted",
    "placeholder": r"todo|tbd|xxx|lorem",
}
raw_nocode = re.sub(r"```.*?```", " ", raw, flags=re.S)
hits = {}
for name, pat in banned.items():
    found = re.findall(pat, raw_nocode, flags=re.I)
    if found:
        hits[name] = len(found)
print("\nbanned patterns found:", hits or "none")
# Mimic _layouts/post.html: rendered HTML, code blocks dropped, tags stripped (image alt text lives in a tag, so it is not
# counted; rules, heading marks and list numerals are markup, not words), whitespace-separated words, (words + 100) // 200.
body = re.sub(r"```.*?```", " ", raw.split("---", 2)[2], flags=re.S)
body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)  # image with alt text
body = re.sub(r"\[\s*\]\([^)]*\)", " ", body)  # link left empty around an image
body = re.sub(r"\]\([^)]*\)", "]", body)
body = re.sub(r"\{\{[^}]*\}\}", " ", body)
body = re.sub(r"(?m)^---\s*$", " ", body)
body = re.sub(r"(?m)^#{1,6}\s+", "", body)
body = re.sub(r"(?m)^\s*\d+\.\s+", "", body)
words = len(body.split())
print(f"words (layout-style count, code blocks and image alt text excluded): {words}; read time shown: ~{(words + 100) // 200} min; stays under 15 min while words <= 2899")
if (words + 100) // 200 >= 15:
    bad.append("read time not under 15 min")
print(f"UNREGISTERED tokens: {bad or 'none'}")
sys.exit(1 if bad or hits else 0)
