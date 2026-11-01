from pathlib import Path
import gzip,re,json,hashlib,difflib
p=Path('../okc-today-proj/docs/steering/evidence/2026-11-01-g-short-final-rmw-compile');out=Path('docs/steering/evidence/2026-11-01-g-ordering-restoration');rs=[]
def clean(s):
 lines=[]
 for x in s.splitlines():
  if re.match(r'\s*\.(?:loc|file)\b',x) or x.lstrip().startswith('//') or not x.strip():continue
  if re.match(r'\s*\.section\s+\.debug',x):break
  lines.append(x.strip())
 return '\n'.join(lines)
for tag in ['tiny','tiny2']:
 texts=[gzip.decompress((p/(tag+'-'+a)/'device.ptx.gz').read_bytes()).decode() for a in ['control','relaxed']];a,b=map(clean,texts)
 assert a.count('atom.acq_rel.gpu.global.add.s32')==1 and b.count('atom.relaxed.gpu.global.add.u32')==1
 assert a.replace('atom.acq_rel.gpu.global.add.s32','atom.relaxed.gpu.global.add.u32')==b
 lines=texts[1].splitlines();i=next(i for i,x in enumerate(lines) if 'atom.relaxed.gpu.global.add.u32' in x);tail='\n'.join(lines[i:]);assert not re.search(r'\bld\.global\.nc\b',tail)
 assert 'membar.gl;' in '\n'.join(lines[i-8:i]);dest,addr=re.search(r'atom.relaxed.gpu.global.add.u32 (\S+), \[(\S+)\]',lines[i]).groups();assert re.search(r'setp.ne.s32[^\n]*'+re.escape(dest),tail);assert 'ld.acquire.gpu.global.b32' in tail
 loc='';loads=[]
 for x in lines[i:]:
  if x.strip().startswith('.loc'):loc=x.strip()
  if re.search(r'\bld\.global',x):loads.append(dict(loc=loc,instruction=x.strip()))
 rs.append(dict(build=tag,normalized_ptx_one_atomic_delta=True,control_ptx_sha256=hashlib.sha256(texts[0].encode()).hexdigest(),treatment_ptx_sha256=hashlib.sha256(texts[1].encode()).hexdigest(),returned_old_compared=True,preceding_membar_gl=True,post_counter_global_nc=0,post_counter_loads=loads))
(out/'compiled-proof-census.json').write_text(json.dumps(rs,indent=2),encoding='utf8');print('TWO_SELECTED_PTX_PROOFS_PASS')
