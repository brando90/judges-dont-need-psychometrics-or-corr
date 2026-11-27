"""Audit helper (Claude, 11-26-2026).
Usage: python3 print_all_axioms.py <lean source> <output lean file> [<committed lean_check.txt>]
Writes a copy of the source with `#print axioms <name>` appended for every theorem or lemma that the source
does not already print, and reports how many names the committed record lists. Compile the output with the
pinned toolchain, e.g.  cd <lean project>; lake env lean <output lean file>."""
import re, sys
src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
printed = set(re.findall(r'^#print axioms\s+(\S+)', src, flags=re.M))
thms = re.findall(r'^\s*(?:theorem|lemma)\s+([A-Za-z0-9_\.\']+)', src, flags=re.M)
extra = [t for t in thms if t not in printed]
open(out_path, "w").write(src.rstrip("\n") + "\n\n-- audit: print axioms for every theorem not already printed\n"
                          + "\n".join(f"#print axioms {t}" for t in extra) + "\n")
print(f"theorems/lemmas in source: {len(thms)} | #print axioms already in source: {len(printed)} | appended: {len(extra)}")
if len(sys.argv) > 3:
    lc = open(sys.argv[3]).read()
    listed = [l.strip() for l in lc.split("\n") if re.fullmatch(r"[A-Za-z0-9_']+", l.strip())]
    infos = set(re.findall(r"'([A-Za-z0-9_\.']+)' depends on axioms", lc))
    print(f"committed record: bare names listed = {len(listed)} | names with a printed axiom line = {len(infos)}")
    print(f"theorems in source with no mention in the committed record: {len(set(thms) - set(listed) - infos)}")
