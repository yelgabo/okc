"""Archive one closed paired-RNE compile screen; no compile/GPU or variant search."""
import argparse,collections,gzip,hashlib,json,re,shutil
from pathlib import Path
from proj_g_native_gamma_compile import norm_ptx

def run(cache,out):
 d=json.loads((cache/'comparison.json').read_text());out.mkdir(parents=True,exist_ok=True)
 d['status']='CLOSED_DEPENDENCY_BUDGET';d['GPU']=False;d['actual_exposed_us']=None
 ledger=[]
 for row in d['records']:
  for label,a in zip(('control','pair'),row['arms']):
   folder=cache/(row['build']+'-'+label);dest=out/folder.name;dest.mkdir(exist_ok=True)
   ops=a.pop('ops');a['normalized_opcode_operand_sha256']=hashlib.sha256('\n'.join(ops).encode()).hexdigest()
   a['non_nop_instructions']=sum(not re.match(r'NOP(?: |;)',x)for x in ops)
   pt=norm_ptx((folder/'device.ptx').read_text())
   a['ptx_instruction_counts']=dict(collections.Counter(re.findall(r'(?<![A-Za-z0-9_.])([a-z][a-z0-9_.:]+)\s+[^;{}]*;',pt)))
   for name in ('manifest.json','nvrtc.log','ptxas.log'):shutil.copyfile(folder/name,dest/name)
   for name in ('input.cu','device.ptx','device.cubin','device.sass'):
    raw=(folder/name).read_bytes()
    with gzip.GzipFile(filename=str(dest/(name+'.gz')),mode='wb',mtime=0)as f:f.write(raw)
    ledger.append(dict(path=str((dest/(name+'.gz')).relative_to(out)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
  a,b=row['arms'];aa,bb=a['ptx_instruction_counts'],b['ptx_instruction_counts']
  row['ptx_count_delta']={k:bb.get(k,0)-aa.get(k,0)for k in sorted(set(aa)|set(bb))if bb.get(k,0)!=aa.get(k,0)}
  critical=lambda c:{k:v for k,v in c.items()if re.match(r'(?:fence|membar|bar\.|mbarrier|wgmma|mma\.|cp\.)',k)}
  row['sync_mma_counts_equal']=critical(aa)==critical(bb);assert row['sync_mma_counts_equal']
  print(row['build'],'instructions',a['instructions'],b['instructions'],'NONNOP',a['non_nop_instructions'],b['non_nop_instructions'],'PTXDELTA',row['ptx_count_delta'])
 (out/'comparison.json').write_text(json.dumps(d,indent=2)+'\n')
 (out/'archive.json').write_text(json.dumps(dict(files=ledger,builds=4,GPU=False,scope='Compressed compiler evidence only, not runtime/source packaging'),indent=2)+'\n')
 header=Path('/root/.cache/okc-cuda128/packages/nvidia/cuda_runtime/include')
 (out/'header-pins.json').write_text(json.dumps([dict(path=str(header/n),sha256=hashlib.sha256((header/n).read_bytes()).hexdigest())for n in ('cuda_bf16.h','cuda_bf16.hpp')],indent=2)+'\n')
 for k in ('tiny-control','tiny-pair'):
  p=cache/k/'device-line.sass'
  if p.exists():
   with gzip.GzipFile(filename=str(out/k/'device-line.sass.gz'),mode='wb',mtime=0)as f:f.write(p.read_bytes())
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.cache,a.output)
