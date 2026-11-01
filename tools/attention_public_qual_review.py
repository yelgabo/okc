"""CPU facade for the actual public hybrid metadata selector. No Torch/GPU imports."""
import argparse,ast,hashlib,json
from pathlib import Path
class Tensor:
 def __init__(self,dtype,shape):self.dtype=dtype;self.shape=shape;self.device='cuda:0';self.contiguous=True
 def is_contiguous(self):return self.contiguous

def run(path):
 source=path.read_text(encoding='utf8');tree=ast.parse(source)
 names={'_verify_owned_metadata','_verify_override_bit'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==2
 ns={'_VERIFY_OVERRIDE_MASK':15};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-selector-review>','exec'),ns)
 def data(batch,length,window):
  cfg=dict(batch=batch,q_len=length,tokens=batch*length,window=window,page_size=1,hidden=8192,num_q_heads=16,num_kv_heads=2,head_dim=64,n_experts=256,local_experts=32,top_k=8,kv_scale=1.0,rope_theta=500000. if window else 0.)
  specs={'attn_input':('torch.bfloat16',(batch*length,8192)),'stream':('torch.float32',(batch*length,8192)),'page_table':('torch.int32',(batch,86008)),'seq_lens':('torch.int32',(batch,)),'cache_loc':('torch.int64',(batch*length,)),'local_start':('torch.int32',(1,)),'k_scale':('torch.float32',()),'v_scale':('torch.float32',()),'k_cache':('torch.float8_e4m3fn',(batch*86008,1,2,64)),'v_cache':('torch.float8_e4m3fn',(batch*86008,1,2,64))}
  return dict(config=cfg,**{k:Tensor(*v) for k,v in specs.items()})
 rows=[]
 for b in (1,16,32):
  for q in (1,8):
   for w in (2048,0):
    d=data(b,q,w);expected=0 if q!=8 or b==1 else 1<<((0 if b==16 else 2)+(w==0));actual=ns['_verify_override_bit'](d);assert actual==expected;(rows.append(dict(batch=b,q_len=q,window=w,bit=actual)))
 rejects=0
 for change in ('scale','dtype','stride','cache','device','rope'):
  d=data(16,8,2048)
  if change=='scale':d['config']['kv_scale']=2.
  elif change=='dtype':d['attn_input'].dtype='torch.float32'
  elif change=='stride':d['page_table'].contiguous=False
  elif change=='cache':d['k_cache'].shape=(2,2,2,64)
  elif change=='device':d['stream'].device='cuda:1'
  else:d['config']['rope_theta']=1.
  assert ns['_verify_override_bit'](d)==0;rejects+=1
 # Nonselected shapes may omit native-only metadata, since guard must return early.
 assert ns['_verify_override_bit']({'config':{'batch':1,'q_len':1,'window':0}})==0
 return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),cells=rows,negative_metadata=rejects,nonselected_early_fallback=True,gpu=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=run(a.source);text=json.dumps(r,indent=2);print(text)
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n',encoding='utf8')
