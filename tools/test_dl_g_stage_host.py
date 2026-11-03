"""Execute actual diagnostic prepare/formatter using TorchCPU and explicit CUDA facade.
This tests host flow/transport/failure handling, not native CUDA math or scheduling.
"""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import types
import unittest
import torch
from dl_g_stage_analyze import expected_jobs,stage_order,analyze_trace,distortion,compact

def make_trace(ids,grid,mode):
 w=[0]*475;jobs,counts=expected_jobs(ids.tolist());w[207]=len(jobs)
 for j,job in enumerate(jobs):w[208+3*j:211+3*j]=job
 for c,cta in enumerate((0,grid//2,grid-1)):
  base=1_700_000_000_000_000_000+c*10000;w[472+c]=base;w[204+c]=base+100000
  qa=0
  for pb in (0,1):
   total=len(jobs)<<(9-pb);g0,g1=total*cta//grid,total*(cta+1)//grid;count=g1-g0
   q0=qa if pb else 0;t0=base+1000+pb*40000;t1=t0+30000
   w[168+(c*2+pb)*6:174+(c*2+pb)*6]=[g0,g1,q0,q0+count,t0,t1]
   if pb==0:qa=count
   for k,ord in enumerate((0,7,15,31)):
    if mode==3 or ord>=count:continue
    a=t0+(ord+1)*100
    row=[a,a+20,a+40,0,0,0,3] if mode==1 else [0,0,0,a,a+20,a+40,0]
    w[((c*2+pb)*4+k)*7:((c*2+pb)*4+k+1)*7]=row
 return w

class Graph:
 def __init__(self,cuda):self.cuda=cuda;self.calls=[]
 def replay(self):
  for fn,args in self.calls:fn(*args)
class Cuda:
 def __init__(self):self.active=None;self.last=0
 def CUDAGraph(self):return Graph(self)
 @contextlib.contextmanager
 def graph(self,g):
  self.active=g
  try:yield
  finally:self.active=None
 def synchronize(self):pass
 def Event(self,**kw):
  cuda=self
  return types.SimpleNamespace(record=lambda:None,synchronize=lambda:None,elapsed_time=lambda e:.1*(1.002 if cuda.last in (1,2) else 1.001 if cuda.last==3 else 1))
class Torch:
 def __init__(self):self.cuda=Cuda();self.backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)))
 def __getattr__(self,n):return getattr(torch,n)
 def empty(self,*args,**kw):
  if args and args[0]==64*1024*1024:args=(1,)
  return torch.empty(*args,**kw)
class Module:
 def __init__(self,mode,t):self.mode=mode;self.t=t
 def sc6_resources(self):return [96,0,0,231168,1,576]*2
 def okc_run(self,*args):
  if self.t.cuda.active is not None:self.t.cuda.active.calls.append((self.run,args))
  self.run(*args)
 def run(self,w,ws,x,ids,tw,out,lo,nloc,grid):
  valid=(ids>=0)&(ids<32)
  out.copy_((x.float()*((tw*valid).sum(-1)[:,None])).bfloat16())
  if self.mode:ws[4].copy_(torch.tensor(make_trace(ids,grid,self.mode),dtype=torch.int64))
  self.t.cuda.last=self.mode

def namespace():
 t=Torch();mods={i:Module(i,t) for i in range(4)}
 def ext(flags):
  mode=next((int(x.rsplit("=",1)[1]) for x in flags if x.startswith("-DG_STAGE_PROBE=")),0);return mods[mode]
 def gen(m,seed,dev):
  g=torch.Generator().manual_seed(seed)
  from dl_g_next_expert_traffic import decode
  logpath=Path(__file__).resolve().parents[2]/"okc/muse_moe_w4a16/profiles/frontier-normalized.1c6ebc76b91d4b7f.log"
  masks=decode(logpath.read_text(encoding="utf8"))["masks"]
  rows=[]
  for i in range(m):
   local=[e for e in range(32) if masks[(i+seed)%256]&(1<<e)]
   rows.append(local+list(range(32,32+8-len(local))))
  ids=torch.tensor(rows,dtype=torch.int32)
  return {"hidden_states":torch.randn((m,4096),generator=g).bfloat16(),"topk_ids":ids,"topk_weights":torch.full((m,8),.125)}
 baseline="""def prepare(weights): return {}
def kernel_fn(data,state):
 x=data['hidden_states'];ids=data['topk_ids'];tw=data['topk_weights']
 valid=(ids>=0)&(ids<32)
 return (x.float()*((tw*valid).sum(-1)[:,None])).bfloat16()
"""
 ns=dict(torch=t,math=math,prepare=lambda w:{"lo":0,"cnt":32,"ps":{},"sfx":("-DOKC_PRMTSC=1",),"sfxm":("-DOKC_PRMTSC=1","-DOKC_SC6W=1",),"w":[],"grid":132},
  PDL=False,EMAX=3072,WGS_FIXW=4096,WGS_XROWS=4160,_main_defines=lambda s:("-DOKC_WGS=1",),_ext=ext,_SC_BASELINE=baseline,
  _gen_input=gen,_make_scratch=lambda d:[torch.zeros(8) for _ in range(3)],analyze_trace=analyze_trace,distortion=distortion,compact=compact)
 exec(Path(__file__).with_name("moe_g_stage_body.py").read_text(),ns)
 return ns,t

class HostTests(unittest.TestCase):
 def test_actual_whole_prepare_and_transport(self):
  torch.set_num_threads(1);ns,t=namespace();weights={k:torch.zeros((256,1)) for k in ("w13","w2")};weights["post_norm_w"]=torch.zeros(4096)
  log=io.StringIO()
  with contextlib.redirect_stdout(log):state=ns['prepare'](weights)
  lines=log.getvalue().splitlines();report=json.loads(next(v.split(' ',1)[1] for v in lines if v.startswith('GSP_SUMMARY ')))
  self.assertEqual(len(report['cases']),6);self.assertEqual(report['tallies']['fresh'],36)
  self.assertTrue(all(g[-1] for r in report['cases'] for g in r[4]))
  capsule=next(v.split(' ',1)[1] for v in lines if v.startswith('GSP_CAPSULE '))
  self.assertLess(sum(len(v) for v in lines),11000)
  import base64,zlib,struct
  raw=zlib.decompress(base64.b64decode(capsule));self.assertEqual(struct.unpack_from('<4sHH',raw),(b'GSP1',132,256))
  self.assertEqual(len(raw),8+1900*8+256*4)
  self.assertTrue(t.backends.cuda.matmul.allow_tf32)
 def test_inner_budget_failure_is_fatal(self):
  from unittest.mock import patch
  ns,t=namespace();weights={k:torch.zeros((256,1)) for k in ("w13","w2")};weights["post_norm_w"]=torch.zeros(4096)
  calls=[0]
  def clock():
   calls[0]+=1
   return 280. if calls[0]>5 else 0.
  log=io.StringIO()
  with patch("time.monotonic",side_effect=clock),contextlib.redirect_stdout(log):
   with self.assertRaises(RuntimeError):ns['prepare'](weights)
  self.assertNotIn("GSP_DONE",log.getvalue());self.assertTrue(t.backends.cuda.matmul.allow_tf32)
 def test_fresh_stale_output_is_fatal(self):
  ns,t=namespace();mod=ns["_ext"](("-DG_STAGE_PROBE=1",));normal=mod.run;cache=[]
  def stale(*args):
   normal(*args)
   out=args[5]
   if not cache:cache.append(out.clone())
   else:out.copy_(cache[0])
  mod.run=stale
  weights={k:torch.zeros((256,1)) for k in ("w13","w2")};weights["post_norm_w"]=torch.zeros(4096)
  with contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(RuntimeError):ns['prepare'](weights)
  self.assertTrue(t.backends.cuda.matmul.allow_tf32)
 def test_nan_reference_fails(self):
  ns,t=namespace()
  with self.assertRaises(RuntimeError):ns['_gsp_errors'](torch.zeros(2,4),torch.full((2,4),float('nan')))
 def test_no_false_gate(self):
  phases=[[30.]*6 for _ in range(6)];med=[100,100,100,100,104,100]
  self.assertFalse(distortion(med,phases)['detail_gates'][0]['pass'])
  phases[5][2]=32
  self.assertFalse(distortion([100]*6,phases)['detail_gates'][1]['pass'])

if __name__=="__main__":unittest.main()
