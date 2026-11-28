"""Word count of the paper abstract, as quoted in docs/notes/abstract_rewrite_28Nov2026.md.

Rule: drop LaTeX comment lines, \\begin{abstract}, \\end{abstract} and \\noindent;
count each inline math span as one token; drop remaining LaTeX commands and braces;
count whitespace-separated tokens that contain a letter or a digit.

Usage: python3 -I -B docs/notes/abstract_wordcount_28Nov2026.py [path-to-1-abstract.tex]
Prints the count; exit code 0. Deterministic, no randomness.
"""
import re
import sys

DEFAULT = "paper_latex_and_notes/AAAI_2027_EvalAIScientists_JudgesKL/1-abstract.tex"


def count(tex: str) -> int:
    t = re.sub(r"(?m)^\s*%.*$", "", tex)
    t = re.sub(r"\\begin\{abstract\}|\\end\{abstract\}|\\noindent", "", t)
    t = re.sub(r"\$[^$]*\$", " M ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", "", t)
    t = t.replace("{", " ").replace("}", " ").replace("~", " ").replace("--", "-")
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(path, encoding="utf-8") as fh:
        print(count(fh.read()))
