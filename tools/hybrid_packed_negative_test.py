"""Runtime source-assembly negatives; explicit paths to frozen generated backends."""
import os,unittest
from pathlib import Path
from hybrid_packed_source_calls import capture,canonical
ROOT=Path(os.environ['OKC_PACKED_DIR'])
class Negative(unittest.TestCase):
 def test_removed_moe_dispatch_rejects(self):
  source=(ROOT/'owned-15-comment.py').read_text()
  def mutate(ns):ns['_MOE1_HOST_SRC']=ns['_MOE1_HOST_SRC'].replace('//@MOE1_DISPATCH@','',1)
  with self.assertRaisesRegex(AssertionError,'missing live MOE51'):capture(source,'owned',mutate)
 def test_removed_named_gle_anchor_rejects(self):
  source=(ROOT/'G-comment.py').read_text()
  def mutate(ns):
   old=ns['bfp_SK2_SRC'];assert '// every thread' in old;ns['bfp_SK2_SRC']=old.replace('// every thread','',1)
  with self.assertRaises(AssertionError):capture(source,'G',mutate)
if __name__=='__main__':unittest.main()
