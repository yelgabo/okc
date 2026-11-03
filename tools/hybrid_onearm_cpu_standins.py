"""Explicit CPU arithmetic/CUDA stand-ins for one-arm qualification orchestration tests."""
import contextlib,types
import torch

class Stream:
 def wait_stream(self,other):pass
class Graph:
 def __init__(self,cuda):self.cuda=cuda;self.ops=[]
 def replay(self):
  if self.cuda.stale:return
  for op in self.ops:op()
class Cuda:
 def __init__(self):self.active=None;self.stale=False;self.sync_fail=False
 def Stream(self,*a,**kw):return Stream()
 def current_stream(self,*a,**kw):return Stream()
 @contextlib.contextmanager
 def stream(self,stream):yield
 def CUDAGraph(self,*a,**kw):return Graph(self)
 @contextlib.contextmanager
 def graph(self,graph,*a,**kw):
  assert self.active is None;self.active=graph
  try:yield
  finally:self.active=None
 def synchronize(self,*a,**kw):
  if self.sync_fail:raise RuntimeError('injected CUDA sync failure')
 def mem_get_info(self,*a,**kw):return 120<<30,140<<30
 def empty_cache(self):pass
 def current_device(self):return 0
 def get_device_capability(self,*a,**kw):return (9,0)
 def get_device_properties(self,*a,**kw):return types.SimpleNamespace(major=9,minor=0,multi_processor_count=132,total_memory=140<<30)
 @contextlib.contextmanager
 def device(self,*a,**kw):yield

def inputs(batch,length,context,seed):
 """Correct metadata/layout with short CPU cache; not a natural-input distribution oracle."""
 t=batch*length;g=torch.Generator().manual_seed(seed);h=8192
 stream=torch.randn(t,h,generator=g);x=torch.randn(t,h,generator=g).bfloat16()
 table=torch.zeros((batch,86008),dtype=torch.int32)
 table[:,:length]=torch.arange(t,dtype=torch.int32).reshape(batch,length)
 return dict(config=dict(batch=batch,q_len=length,tokens=t,window=2048 if context=='sliding' else 0,page_size=1,hidden=h,num_q_heads=16,num_kv_heads=2,head_dim=64,n_experts=256,local_experts=32,top_k=8,kv_scale=1.,rope_theta=500000. if context=='sliding' else 0.,qk_scale_factor=5.4730064863875,sm_scale=64**-.5),stream=stream,attn_input=x,page_table=table,seq_lens=torch.zeros(batch,dtype=torch.int32),cache_loc=torch.arange(t,dtype=torch.int64),local_start=torch.zeros(1,dtype=torch.int32),k_scale=torch.tensor(1.),v_scale=torch.tensor(1.),k_cache=torch.zeros((t,1,2,64),dtype=torch.float8_e4m3fn),v_cache=torch.zeros((t,1,2,64),dtype=torch.float8_e4m3fn),positions=torch.arange(t),cache_seqlens=torch.full((batch,),length,dtype=torch.int32),cu_seqlens_q=torch.arange(0,t+1,length,dtype=torch.int32),scheduler_metadata=None)

def front(data,weights,*a):
 t=data['config']['tokens'];start=(abs(int(data['stream'][0,0]*100))%8)*32
 biased=torch.zeros(t,256);biased[:,start:start+8]=torch.arange(100.,108.)
 value=float(abs(int(data['stream'][0,1]*10))%3+1)
 return dict(biased=biased,scores=biased.softmax(-1),h=data['attn_input'].float(),stream1=data['stream']+data['attn_input'].float(),local_start=int(data['local_start'][0]),k_new=torch.full((data['config']['batch'],data['config']['q_len'],2,64),value),v_new=torch.full((data['config']['batch'],data['config']['q_len'],2,64),value+1))

def back(front,weights,ids):return front['h'].bfloat16(),front['stream1']

def kernel(data,state,cuda,flags):
 f=front(data,None);ids=f['biased'].topk(8,dim=-1).indices.to(torch.int32)
 out=(torch.empty_like(data['attn_input']),torch.empty_like(data['stream']),ids.clone())
 def op():
  current=front(data,None);h,s=back(current,None,ids)
  out[0].copy_(h);out[1].copy_(s);out[2].copy_(current['biased'].topk(8,dim=-1).indices)
  data['k_cache'].view(-1,2,64).copy_(current['k_new'].reshape(-1,2,64).to(torch.float8_e4m3fn));data['v_cache'].view(-1,2,64).copy_(current['v_new'].reshape(-1,2,64).to(torch.float8_e4m3fn))
  if flags.get('output_nan'):out[0].fill_(float('nan'))
  if flags.get('route_bad'):out[2].fill_(0)
  if flags.get('cache_bad'):data['k_cache'].zero_()
 op()
 if cuda.active is not None:cuda.active.ops.append(op)
 return out
