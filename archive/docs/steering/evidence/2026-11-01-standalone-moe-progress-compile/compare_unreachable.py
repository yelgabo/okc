import re,json,hashlib,collections
from pathlib import Path
root=Path('/root/.cache/okc-standalone-progress')
def load(kind):
 p=root/kind
 m=json.loads((p/'manifest.json').read_text())
 s=(p/'device.sass').read_text()
 out={}
 for name in m['lowered']:
  start=s.index('.section\t.text.'+name+',')
  end=s.find('\n\t.section\t.text.',start+1)
  body=s[start:end if end>=0 else len(s)]
  ops=[]
  for line in body.splitlines():
   q=re.match(r'\s*/\*[0-9a-f]+\*/\s*(.*?)\s*;.*',line)
   if q: ops.append(q[1])
  key='skinny1' if 'sk_8_3_0' in name else 'main47'
  census=collections.Counter(re.sub(r'^@!?U?P\d+\s+','',x).split()[0] for x in ops)
  out[key]=dict(lowered=name,instructions=len(ops),opcode_census=dict(census),critical=[x for x in ops if re.search(r'(^|\s)(?:H(?:G)?MMA|HMUL2|FENCE|MEMBAR|WARPGROUP|BAR|SYNCS)[. ]',x)])
 return m,out
c,co=load('control');t,to=load('unreachable')
r={'control_manifest':c,'repair_manifest':t,'entries':{}}
for k in co:
 a,b=co[k],to[k]
 ac,bc=a['opcode_census'],b['opcode_census']
 r['entries'][k]={'control_instructions':a['instructions'],'repair_instructions':b['instructions'],'opcode_delta':{x:bc.get(x,0)-ac.get(x,0) for x in sorted(set(ac)|set(bc)) if bc.get(x,0)!=ac.get(x,0)},'critical_control':a['critical'],'critical_repair':b['critical'],'critical_exact':a['critical']==b['critical']}
(root/'unreachable-comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:{a:v for a,v in z.items() if not a.startswith('critical_') or a=='critical_exact'} for k,z in r['entries'].items()},indent=2))
