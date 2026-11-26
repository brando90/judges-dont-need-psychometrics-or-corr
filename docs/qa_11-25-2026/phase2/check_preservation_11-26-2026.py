"""Verify protected evidence, dated notes and local links (11-26-2026)."""
from pathlib import Path
import subprocess
import re
import gzip
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
BASE = '4ce401c4711cc5eb944c86ea0e5cac862ed02356'
def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])
baseline=set(git('ls-tree','-r','--name-only',BASE).decode().splitlines())
all_changed=git('diff','--name-only',BASE).decode().splitlines()
changed=[name for name in all_changed if name in baseline]
new=git('ls-files','--others','--exclude-standard').decode().splitlines()+[name for name in all_changed if name not in baseline]
tracked=set(git('ls-files').decode().splitlines())
bad=[]
table_count=0
for name in changed:
    p=ROOT/name
    if not p.exists():
        bad.append(('deleted file',name));continue
    if p.suffix.lower() in {'.csv','.json','.jsonl','.gz','.npz','.npy','.zip'} or re.search(r'(protocol|prereg|freeze)',p.name,re.I):
        bad.append(('protected file changed',name))
    if p.suffix=='.md':
        old=git('show',f'{BASE}:{name}').decode()
        text=p.read_text()
        for block in re.findall(r'(?:^\|.*\n)+',old,re.M):
            table_count+=1
            if block not in text:bad.append(('original table changed',name))
for name in sorted(set(changed+new)):
    p=ROOT/name
    if p.suffix!='.md':continue
    text=p.read_text()
    title=next((line for line in text.splitlines() if line.startswith('# ')),'')
    if '11-26-2026' not in title or '11-26-2026' not in '\n'.join(text.splitlines()[:8]):
        bad.append(('undated note',name))
    for label,url in re.findall(r'\[([^\]]*)\]\(([^)]+)\)',text):
        if re.match(r'^(https?://|mailto:|#|/)',url):continue
        target=unquote(url.split('#')[0].split(' "')[0].strip('<>'))
        # Only enforce newly introduced links; preserved historical errors have companion corrections.
        old=git('show',f'{BASE}:{name}').decode() if name in changed else ''
        if f']({url})' in old:continue
        if target and not (p.parent/target).exists():bad.append(('new broken link',name,target))
secret_candidates=[]
patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'\b(?:sk-proj-|sk-ant-api\d*-)[A-Za-z0-9_-]{20,}',rb'\bgh[pousr]_[A-Za-z0-9]{30,}',rb'\bAKIA[A-Z0-9]{16}\b']
scanned=0
for name in sorted(tracked|set(new)):
    p=ROOT/name
    if not p.is_file() or p.stat().st_size>30_000_000:continue
    data=p.read_bytes()
    if p.suffix=='.gz':
        try:data=gzip.decompress(data)
        except OSError:continue
    scanned+=1
    for pattern in patterns:
        for m in re.finditer(pattern,data):
            secret_candidates.append((name,data[:m.start()].count(b'\n')+1))
print('Preservation and documentation check (11-26-2026)')
print('Baseline:',BASE)
print('Changed existing files:',len(changed),'original Markdown table blocks retained:',table_count)
print('Known credential-pattern scan files:',scanned,'candidate locations:',secret_candidates)
print('Problems:',bad)
print('No secret values are printed. Pattern scanning does not prove absence of every possible secret.')
raise SystemExit(bool(bad or secret_candidates))
