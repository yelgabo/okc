"""Recompute archived selected sm90a static counts. No GPU and no performance inference."""
import argparse,collections,gzip,json,re
from pathlib import Path

def counts(path):
 s=gzip.decompress(path.read_bytes()).decode();ins=[q[1] for x in s.splitlines() if(q:=re.match(r'\s*/\*[0-9a-f]+\*/\s*(.*?)\s*$',x)) and not q[1].startswith('.')]
 special={k:sum(k in x for x in ins) for k in ('HGMMA','MMA','BAR','MEMBAR','FENCE','LDL','STL','BPT.TRAP','WARPGROUP.ARRIVE')}
 waits=collections.Counter(re.search(r'WARPGROUP\.DEPBAR\.LE\s+gsb0,\s*(0x[0-9a-f]+)',x)[1] for x in ins if 'WARPGROUP.DEPBAR.LE' in x)
 return dict(instructions=len(ins),counts=special,wait_immediates=dict(waits))
def compare(root):
 a=counts(root/'control-compile-main51.sass.gz');b=counts(root/'compile-main51.sass.gz')
 for k in a['counts']:
  if k!='BPT.TRAP':assert a['counts'][k]==b['counts'][k],k
 assert a['counts']['BPT.TRAP']==0 and b['counts']['BPT.TRAP']>0
 assert a['wait_immediates']==b['wait_immediates']
 return dict(control=a,repair=b,scope='Static selected instruction counts; same math/wait/fence counts, not dynamic timing or performance.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=compare(a.evidence);s=json.dumps(r,indent=2);print(s)
 if a.output:a.output.write_text(s+'\n')
