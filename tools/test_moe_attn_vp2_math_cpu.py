"""Real CPU tensor tests of actual private references/checker; not GPU kernel proof."""
import ast
from pathlib import Path
import unittest
import math
import torch

ROOT=Path(__file__).resolve().parents[1]

def namespace():
 tree=ast.parse((ROOT/'.build/today-moe/vp2_bf16_private.py').read_text(encoding='utf8'))
 names={'_ob_rms','_ob_rope','_ob_reference','_vp_bf_reference','_ob_rel','_vp_refs','_vp_check'}
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 ns={'torch':torch,'math':math,'FP8':torch.float8_e4m3fn,'RMS_EPS':1e-5,'_OB_THETA':500000.}
 exec(compile(ast.Module(nodes,[]),'<actual-vp2-refs>','exec'),ns)
 return ns

def inputs(seed):
 gen=torch.Generator().manual_seed(seed)
 return {'qkv':torch.randn((1,8,2304),generator=gen).bfloat16(),
  'seq_lens':torch.zeros(1,dtype=torch.int32),'page_table':torch.arange(8,dtype=torch.int32)[None,:],
  'k_cache':torch.zeros((8,1,2,64)),'v_cache':torch.zeros((8,1,2,64)),
  'k_scale':torch.tensor(1.),'v_scale':torch.tensor(1.),
  'config':{'batch':1,'q_len':8,'num_kv_heads':2,'num_q_heads':16,'head_dim':64,'page_size':1,
   'window':2048,'rope_theta':500000.,'qk_scale_factor':5.4730064863875,'sm_scale':.125,'kv_scale':1.}}

class Math(unittest.TestCase):
 def test_actual_dual_oracle_and_cache(self):
  ns=namespace();d=inputs(824731);refs=ns['_vp_refs'](d)
  d['k_cache'].view(1,8,2,64).copy_(refs[2].float());d['v_cache'].view(1,8,2,64).copy_(refs[3].float())
  errors=ns['_vp_check'](d,refs[0].bfloat16(),refs)
  self.assertEqual(errors[-2:],[0.,0.]);self.assertTrue(all(math.isfinite(v) for v in errors))
  self.assertTrue(bool((refs[0]!=refs[1]).any()),'separate output-boundary oracle must be exercised')

 def test_reference_nan_and_stale_and_cache_negatives(self):
  ns=namespace();d=inputs(824737);old=ns['_vp_refs'](d);d2=inputs(824743);refs=ns['_vp_refs'](d2)
  d2['k_cache'].view(1,8,2,64).copy_(refs[2].float());d2['v_cache'].view(1,8,2,64).copy_(refs[3].float())
  with self.assertRaisesRegex(AssertionError,'VP2_MATH'):ns['_vp_check'](d2,old[0].bfloat16(),refs)
  bad=list(refs);bad[0]=torch.full_like(refs[0],float('nan'))
  with self.assertRaisesRegex(AssertionError,'REFERENCE_NONFINITE'):ns['_vp_check'](d2,refs[0].bfloat16(),bad)
  with self.assertRaisesRegex(AssertionError,'OUTPUT_NONFINITE'):ns['_vp_check'](d2,torch.full_like(refs[0],float('nan')),refs)
  d2['k_cache'].fill_(100)
  with self.assertRaisesRegex(AssertionError,'VP2_CACHE'):ns['_vp_check'](d2,refs[0].bfloat16(),refs)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
