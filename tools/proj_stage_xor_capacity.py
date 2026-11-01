"""CPU-only natural-weight capacity census, NOT the H200 owning weight bank."""
import ast,argparse,json,math
from pathlib import Path
import torch

def census(artifact):
 tree=ast.parse(artifact.read_text(encoding='utf8'));ns={'torch':torch,'BALANCE_GROUP':4}
 names={'bf16_bits','tile_values','window_base','code_bits','balance_perm','xor_expected_layout','sxs_expected_layout'}
 for n in tree.body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_XOR_CONFIG':ns['_XOR_CONFIG']=ast.literal_eval(n.value)
 exec(compile(ast.Module([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],[]),'capacity-functions','exec'),ns)
 torch.set_num_threads(4);rng=torch.Generator().manual_seed(5151);rows=[]
 for layer,n,k in [('qkv_gate',2304,8192),('out',8192,1024),('pooler',4096,8192),('unpooler',8192,4096),('router',256,8192)]:
  w=(torch.randn(n,k,generator=rng)/math.sqrt(k)).to(torch.float32 if layer=='router' else torch.bfloat16)
  if layer=='router':continue
  bits=ns['bf16_bits'](w);base=ns['window_base'](bits);_,bad,_=ns['code_bits'](bits,base)
  targets={'qkv_gate':['qkv8F','qkv16F','qkv32'],'out':['o16F','o32F'],'pooler':['pool16F','pool32F'],'unpooler':['unp16F','unp32F','slice8F']}[layer]
  cache={}
  for name in targets:
   g=ns['_XOR_CONFIG'][name]['geom'];R=g['R'];nr=1024 if name=='slice8F' else n
   key=(R,nr,layer=='qkv_gate',g['NW'],g['UPW'],g['NT'],g['WSK'])
   if key not in cache:
    perm=ns['balance_perm'](bad,R) if layer=='qkv_gate' and int(bad.sum()) else torch.arange(n)
    tv=ns['tile_values'](bad[perm][:nr].contiguous().to(torch.int32));count=tv.sum((-1,-2))
    st=torch.arange(k//64//g['UPS']).view(-1,1,1,1,1);c=torch.arange(nr//R).view(1,-1,1,1,1)
    warp=torch.arange(g['NW']).view(1,1,-1,1,1);u=torch.arange(g['UPW']).view(1,1,1,-1,1);nt=torch.arange(g['NT']).view(1,1,1,1,-1)
    counts=count[c*(R//8)+(warp//g['WSK'])*g['NT']+nt,st*g['UPS']+(warp%g['WSK'])*g['UPW']+u]
    totals=counts.sum((2,3,4));capacity=int(totals.max());cache[key]=dict(capacity=capacity,min_stage=int(totals.min()),mean_stage=float(totals.double().mean()),stages=totals.numel(),records=int(totals.sum()),max_unit=int(counts.max()))
   stats=cache[key];layout=ns['sxs_expected_layout'](name,2,stats['capacity']);old=ns['xor_expected_layout'](name,1,0)
   row=dict(name=name,**stats,layout=layout,global_sx_dynamic=old['dynamic'],extra_dynamic=layout['dynamic']-old['dynamic'],payload_bytes=layout['weight_payload'],padded_records_per_stage=layout['record_bytes'],record_padding_fraction=layout['record_bytes']*stats['stages']/(stats['records']*2) if stats['records'] else None)
   rows.append(row);print(name,stats['capacity'],layout['dynamic'],row['extra_dynamic'],flush=True)
 return dict(scope='CPU Torch RNG independent natural population; NOT bit-identical H200 judge weights or admission measurement',seed=5151,rows=rows)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.write_text(json.dumps(census(a.artifact),indent=2)+'\n',encoding='utf8')
