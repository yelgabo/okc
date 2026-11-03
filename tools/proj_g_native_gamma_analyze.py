from pathlib import Path
import json,re,collections,hashlib,difflib
from proj_g_native_gamma_compile import norm_ptx
import argparse
p=argparse.ArgumentParser(description='Existing gamma compile-output census only; no compile/GPU');p.add_argument('cache',type=Path);a=p.parse_args();r=a.cache;d=json.loads((r/'comparison.json').read_text());assert d['status'] in ('compiled_all_fixed_keys','blocked'),d['status']
def census(s):
 c=collections.Counter()
 for l in norm_ptx(s).splitlines():
  for op in re.findall(r'(?<![A-Za-z0-9_.])([a-z][a-z0-9_.:]+)\s+[^;{}]*;',l):c[op]+=1
 return dict(c)
def sassc(ops):return dict(collections.Counter(re.sub(r'^@!?U?P[0-9]+ +','',x).split()[0].rstrip(';')for x in ops))
for row in d['records']:
 for arm,a in row['arms'].items():
  pt=(r/(row['build']+'-'+arm)/'device.ptx').read_text();a['ptx_counts']=census(pt)
  a['sass_counts']={name:sassc(ops)for name,ops in a['sass'].items()}
 base,on=row['arms']['baseline'],row['arms']['on'];a,b=base['ptx_counts'],on['ptx_counts']
 row['ptx_count_delta']={k:b.get(k,0)-a.get(k,0)for k in sorted(set(a)|set(b))if b.get(k,0)!=a.get(k,0)}
 critical=lambda c:{k:v for k,v in c.items()if re.match(r'(?:fence|membar|bar\.|mbarrier|wgmma|mma\.|cp\.)',k)}
 row['explicit_ptx_sync_mma_counts_equal']=critical(a)==critical(b)
 # Census only after terminal resource blocker; do not broaden compilation/tuning.
 row['entries']=[]
 for name,ops in base['sass'].items():
  other=on['sass'][name];ca,cb=base['sass_counts'][name],on['sass_counts'][name]
  row['entries'].append(dict(symbol=name,baseline_instructions=len(ops),on_instructions=len(other),opcode_delta={k:cb.get(k,0)-ca.get(k,0)for k in sorted(set(ca)|set(cb))if cb.get(k,0)!=ca.get(k,0)}))
 print(row['build'],'PTX_DELTA',row['ptx_count_delta'],'ENTRIES',[(x['baseline_instructions'],x['on_instructions'])for x in row['entries']])
(r/'analysis.json').write_text(json.dumps(d,indent=2)+'\n')
