"""Reproduce selected opcode/operand comparison from archived nvdisasm text.
Run: python tools/dl_owned_verify_sass_compare.py
Offsets, whitespace, and compiler-generated local-label names are normalized;
all opcodes, predicates, registers, operands, and instruction order remain.
"""
import gzip,hashlib,json,re
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]/'docs/steering/evidence/2026-11-01-owned-verify-production'
SELECT={'wd':'wd_kernelILi1ELi128ELi3','boundary':'attn_boundary_geo_kernelILi1ELi512','route':'route_kernelIfLb1'}
def instructions(path,needle):
 s=gzip.decompress(path.read_bytes()).decode()
 chunks=re.split(r'(?m)^\s*\.section\s+\.text\.',s)[1:]
 selected=[c for c in chunks if needle in c.splitlines()[0]]
 assert len(selected)==1,(path,len(selected))
 entry=re.search(r'(?m)^\s*\.type\s+(\S+),@function',selected[0]).group(1)
 labels={}
 def label(m):
  k=m.group();return labels.setdefault(k,'LABEL'+str(len(labels)))
 out=[]
 for line in selected[0].splitlines():
  m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(.*;)',line)
  if m:
   x=re.sub(r'\.L_[A-Za-z0-9_]+',label,m.group(1).replace(entry,'ENTRY'))
   out.append(' '.join(x.split()))
 assert out
 return out
if __name__=='__main__':
 result={}
 for kind,needle in SELECT.items():
  variants=[];seq=[]
  for tag in ('orig','prod'):
   a=instructions(BASE/(tag+'-'+kind)/'device.sass.gz',needle);seq.append(a)
   variants.append({'tag':tag+'-'+kind,'instructions':len(a),'opcode_operand_sha256':hashlib.sha256('\n'.join(a).encode()).hexdigest()})
  assert seq[0]==seq[1],kind
  result[kind]={'selected':needle,'SASS_equal':True,'variants':variants}
 print(json.dumps(result,indent=2))
