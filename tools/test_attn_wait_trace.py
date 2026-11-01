"""CPU-only gates; does not claim CUDA host/runtime or H200 validation."""
import ast,hashlib,unittest
from pathlib import Path
from attn_wait_trace_patch import patch

ROOT=Path(__file__).resolve().parents[1]
class WaitTraceTests(unittest.TestCase):
 def test_generated_and_compiled_pin(self):
  import json
  cuda=(ROOT/'.build/today-attn/wait-trace.cu').read_bytes()
  for name in ('wait-trace-v8','wait-trace-d1','wait-control-v8','wait-control-d1'):
   m=json.loads((ROOT/'profiles/attention-wait-trace-local'/name/'manifest.json').read_text())
   self.assertEqual(hashlib.sha256(cuda).hexdigest(),m['source_sha256'])
   self.assertEqual(m['exitcode'],0)
   for r in m['resources'].values():self.assertEqual((r['stack_bytes'],r['spill_store_bytes'],r['spill_load_bytes']),(0,0,0))
 def test_fences_and_copy_counts(self):
  s=(ROOT/'.build/today-attn/wait-trace.cu').read_text()
  self.assertIn('detail(d_cons,8);++d_cons;',s)
  self.assertIn('mbar_arrive(empty0 + 8 * st);\n                if constexpr(DETAILS)',s)
  self.assertIn('mbar_wait(full0 + 8 * st, cph);\n                if constexpr(DETAILS)',s)
  # Applying twice must fail instead of silently instrumenting an unknown source.
  with self.assertRaises(AssertionError):patch(s)
 def test_signed_wrap_and_presence(self):
  src=(ROOT/'tools/attn_wait_trace_body.py').read_text();tree=ast.parse(src)
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_wait_q','_wait_metrics')]
  env={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'metrics','exec'),env)
  r=[0]*56;r[4]=r[5]=2
  r[16:26]=[1,1,0xfffffff0,0x10,0x30,0x40,0x20,0x28,0x50,0x70]
  v=env['_wait_metrics']([r])
  self.assertEqual(v['region_us']['empty_wait'][1],.032)
  self.assertEqual(v['region_us']['copy_publish_to_ready'][1],-.008)
  r[5]=3
  with self.assertRaises(AssertionError):env['_wait_metrics']([r])
 def test_cross_segment_ordinals(self):
  # Producer and consumer walk the same concatenation, including empty segments.
  for lengths in ((0,1,3,0,129),(1,0,1),(32,33,64,1)):
   producer=[i for i in range(sum(lengths)) if i in (0,16,64,128)]
   consumer=[];ordinal=0
   for n in lengths:
    for _ in range(n):
     if ordinal in (0,16,64,128):consumer.append(ordinal)
     ordinal+=1
   self.assertEqual(producer,consumer)
 def test_private_graph_and_coverage(self):
  s=(ROOT/'.build/today-attn/wait-trace.py').read_text()
  compile(s,'generated','exec')
  self.assertIn('graph._wait_inputs=data',s)
  self.assertIn('v.clone() if isinstance(v,torch.Tensor)',s)
  self.assertIn("len(_WAIT_ROWS)==24 and len(_AB_INMED)==12",s)
  self.assertIn("perturb<=limit and null<=limit",s)
  self.assertIn("limit=max(.5,.01*min",s)
  self.assertIn('511019 + k',s)
 def test_full_report_codec(self):
  import random
  rng=random.Random(18);src=(ROOT/'tools/attn_wait_trace_body.py').read_text();tree=ast.parse(src)
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wait_final')
  regions=('empty_wait','issue_after_empty','refetch_issue_region','consumer_ready_wait','ready_to_release','release_to_next','copy_publish_to_ready','tile_cadence')
  rows=[]
  for b in (16,32):
   for l in (1,8):
    for seed in range(511019,511022):
     for arm in ('detailA','detailB'):
      rows.append(dict(batch=b,length=l,seed=seed,arm=arm,repetitions=[dict(sample_counts={str(x):rng.randrange(128,265) for x in (0,16,64,128)},region_us={k:[round(rng.uniform(0,25),3) for _ in range(4)] for k in regions},tiles=[128,149.5,176,190]) for _ in range(2)]))
  env={'_WAIT_ROWS':rows,'_WAIT_RES':{},'_AB_INMED':{(f'{b}/{l}',s):dict(ctrl=100.,ctrl2=100.1,detailA=100.2,detailB=100.3) for b in (16,32) for l in (1,8) for s in range(3)},'print':lambda *a,**k:None}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'codec','exec'),env)
  env['_wait_final']()
if __name__=='__main__':unittest.main()
