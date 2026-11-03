from pathlib import Path
import json,re,difflib,collections,hashlib
import argparse
p=argparse.ArgumentParser(description='Existing-output PTX and normalized SASS comparison; no compilation or GPU');p.add_argument('cache',type=Path);a=p.parse_args();r=a.cache;d=json.loads((r/'comparison.json').read_text())
for k in ('tiny','tiny2'):
 def norm(arm):
  s=(r/(k+'-'+arm)/'device.ptx').read_text()
  return '\n'.join(x for x in s.splitlines() if not x.lstrip().startswith(('.loc ','.loc\t','.file ')))
 a,b=norm('control'),norm('relaxed');z=b.replace('atom.relaxed.gpu.global.add.u32','atom.acq_rel.gpu.global.add.s32')
 assert a==z,(k,'Unexpected PTX delta')
 print(k,'PTX_EXACT_EXCEPT_ONE_ATOM',a==z)
 if a!=z: print(''.join(difflib.unified_diff(a.splitlines(True),z.splitlines(True)))[:1500])
 row=next(x for x in d['records'] if x['build']==k)
 c,t=[[re.sub(r'AT_INSTRUCTION_[0-9]+','RELOCATED_TARGET',y) for y in x['ops']] for x in row['arms']]
 oldblock=['@P0 MEMBAR.ALL.CTA ;','@P0 MEMBAR.ALL.GPU ;','@P0 ERRBAR;','@P0 CGAERRBAR ;','@P0 ATOMG.E.ADD.S32.STRONG.GPU PT, R25, desc[UR10][R16.64], R25 ;','@P0 CCTL.IVALL ;']
 newblock=['@P0 ATOMG.E.ADD.STRONG.GPU PT, R25, desc[UR10][R16.64], R25 ;']
 text='\n'.join(x for x in c if x!='NOP;')
 assert text.count('\n'.join(oldblock))==1,(k,'Atomic lowering scope drift')
 assert text.replace('\n'.join(oldblock),'\n'.join(newblock),1)=='\n'.join(x for x in t if x!='NOP;'),(k,'Nonlocal SASS delta')
 row['sass_only_one_rmw_lowering_block_plus_relocations_and_nops']=True
 sm=list(difflib.SequenceMatcher(None,c,t).get_opcodes());hunks=[]
 for tag,i,j,u,v in sm:
  if tag!='equal':hunks.append(dict(tag=tag,control_range=[i,j],treatment_range=[u,v],control=c[max(0,i-3):j+3],treatment=t[max(0,u-3):v+3]))
 row['ptx_exact_except_atomic_and_locations']=a==z
 row['normalized_control_ptx_sha256']=hashlib.sha256(a.encode()).hexdigest()
 row['normalized_relaxed_ptx_sha256']=hashlib.sha256(b.encode()).hexdigest()
 row['sass_hunks']=hunks
 assert row['resources_equal'],(k,'Resource drift')
 print(json.dumps(hunks,indent=2))
(r/'comparison.json').write_text(json.dumps(d,indent=2)+'\n')
