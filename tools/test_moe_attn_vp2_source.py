import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from moe_attn_vp2_make import make,literal,OWN,PUBLIC

ROOT=Path('C:/Users/gretc/Documents/GitHub/okc')
if not ROOT.exists():ROOT=Path('/mnt/c/Users/gretc/Documents/GitHub/okc')

class Source(unittest.TestCase):
 def test_actual_builder_scope_and_precision(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'vp.py';make(ROOT,out);s=out.read_text(encoding='utf8');t=ast.parse(s)
   self.assertLess(len(s.encode()),1_000_000)
   owned=literal(t,'_VP_OWNED');ot=ast.parse(owned)
   self.assertFalse(any(isinstance(n,ast.Try) for n in ot.body))
   self.assertEqual(literal(ot,'_ONE_WS_PICK'),0)
   before=literal(ast.parse((ROOT/'muse_attention_decode/submission.py').read_text(encoding='utf8')),'_ONE_SRC')
   after=literal(ot,'_ONE_SRC');prefix=after[:after.index('std::vector<int64_t> vp_owned_resources(){')]
   self.assertEqual(prefix.replace('acc[e]=__bfloat162float(__float2bfloat16_rn(acc[e]*inv));','acc[e] *= inv;'),before+'\n')
   raw=ast.parse((ROOT/'.build/today/kel/kernels/muse_attention_decode.py').read_text(encoding='utf8'))
   vp=literal(t,'_VP_CUDA');original=literal(raw,'ZH_VFG_CUDA')
   for primitive in ('wg_wait0();','wg_commit();','v_fence_async();','barrier.cluster','mbarrier.try_wait','st.async.shared::cluster'):
    self.assertEqual(original.count(primitive),vp.count(primitive),primitive)
   self.assertEqual(vp.count('__float2bfloat16_rn(b0*inv)'),1)
   self.assertIn('clusters>=2',vp)
   self.assertIn('_VP_LICENSE',s);self.assertIn('Apache License',literal(t,'_VP_LICENSE'))
   self.assertEqual(literal(t,'_VP_MANIFEST')['sch'],0)
   self.assertEqual(len(literal(t,'_VP_OFFSETS')),12)

 def test_score_gate_actual_function(self):
  import math,statistics,json
  body=Path(__file__).with_name('moe_attn_vp2_body.py').read_text(encoding='utf8');tree=ast.parse(body)
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_vp_score')
  ns={'math':math,'statistics':statistics,'_VP_MANIFEST':{'seeds':[824731,824737,824743]}}
  exec(compile(ast.Module([fn],[]),'<actual-score>','exec'),ns)
  def rows(delta):return [(b,l,w,s,[100,100,100-delta if (b,l,w)==(1,8,1) else 100,100-delta if (b,l,w)==(1,8,1) else 100]) for b in (1,16,32) for l in (1,8) for w in (0,1) for s in ns['_VP_MANIFEST']['seeds']]
  self.assertTrue(ns['_vp_score'](rows(12))['keep'])
  self.assertFalse(ns['_vp_score'](rows(0))['keep'])
  self.assertFalse(ns['_vp_score'](rows(-12))['keep'])
  self.assertEqual(ns['_vp_score'](rows(12)),ns['_vp_score'](json.loads(json.dumps(rows(12)))))
  bad=rows(12);bad[0][4][0]=float('nan')
  with self.assertRaisesRegex(AssertionError,'BAD_TIMING'):ns['_vp_score'](bad)

if __name__=='__main__':unittest.main()
