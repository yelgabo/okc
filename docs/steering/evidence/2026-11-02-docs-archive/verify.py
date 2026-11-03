from pathlib import Path
import subprocess,json,hashlib,re,os
root=Path.cwd();m=json.loads(Path('archive/manifest-2026-11-02.json').read_text());base=m['base'];mapping={r['old']:r['new'] for r in m['records']}
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base],text=True).splitlines();known=set(paths);dirs={str(parent) for f in paths for parent in Path(f).parents};bad=[];checked=0
for row in m['records']:
 assert not Path(row['old']).exists() or Path(row['old']).is_dir()
 b=Path(row['new']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['archived_sha256']
 if Path(row['old']).suffix.lower() not in ('.md','.html','.htm'):assert row['bytes_unchanged'],row['old']
for old in paths:
 final=mapping.get(old,old);p=Path(final)
 if not p.exists() or p.suffix.lower() not in ('.md','.html','.htm'):continue
 before=subprocess.check_output(['git','show',base+':'+old]);after=p.read_bytes()
 def decode(b):
  try:return b.decode('utf-8')
  except UnicodeDecodeError:return b.decode('cp1252')
 def targets(text,source):
  urls=re.findall(r'\]\(([^)\n]+)\)',text)+re.findall(r'(?:href|src)\s*=\s*["\x27]([^"\x27]+)',text,re.I);found=set()
  for url in urls:
   url=url.strip('<>')
   if re.match(r'(?:[a-zA-Z][\w+.-]*:|/|#)',url):continue
   path=url.split('#')[0].split('?')[0]
   found.add(os.path.normpath(os.path.join(os.path.dirname(source),path)))
  return found
 actual=targets(decode(after),final)
 for t in targets(decode(before),old):
  if t not in known and t not in dirs:continue
  expected=mapping.get(t,t)
  if t in dirs and any(x.startswith(t+'/') for x in mapping) and all(x in mapping for x in known if x.startswith(t+'/')):expected='archive/'+t
  checked+=1
  if expected not in actual or not Path(expected).exists():bad.append([old,t,expected])
for f in ['muse_decode_layer/cand/g1b4_stage_release.py','muse_decode_layer/submission.py','muse_attention_decode/submission.py','muse_moe_w4a16/submission.py','muse_bf16_proj/submission.py','muse_prefill_layer/submission.py']:
 assert Path(f).read_bytes()==subprocess.check_output(['git','show',base+':'+f])
print('Existing local link targets checked',checked,'regressions',len(bad))
print(json.dumps(bad[:25],indent=2));Path('.build/docs-archive').mkdir(parents=True,exist_ok=True);Path('.build/docs-archive/link-check.json').write_text(json.dumps({'checked':checked,'regressions':bad},indent=2)+'\n')
assert not bad
print('PASS: archive hashes, raw-byte preservation, existing link targets and six current source identities')
