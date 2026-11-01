"""CPU-only proof of bounded G slice checkpoint; no GPU/qualification claim."""
import ast,unittest
from pathlib import Path
from dl_g_slice_checkpoint import build,literals
ROOT=Path(__file__).resolve().parents[1].parent/'okc'
class Slice(unittest.TestCase):
    def test_dead_branch_only_and_budget_failure(self):
        source=(ROOT/'.build/today/g1b4-stage-retire.py').read_text(encoding='utf8')
        out,r=build(source)
        self.assertFalse(r['fits']);self.assertEqual(r['four_literal_bytes'],520047)
        self.assertEqual(r['prototype_bytes'],857866)
        self.assertEqual(literals(ast.parse(source)),literals(ast.parse(out)))
        self.assertIn("raise RuntimeError('G slice supports only verify32 sliding/page1')",out)
        original=ast.parse(source);revised=ast.parse(out)
        for kind in ('prepare','_launch7','_decode_attention7','_workspace7','_attn_selftest','_fork_native_moe'):
            a=[ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,ast.FunctionDef) and n.name==kind]
            b=[ast.dump(n,include_attributes=False) for n in revised.body if isinstance(n,ast.FunctionDef) and n.name==kind]
            self.assertEqual(a,b)
    def test_unreviewed_source_rejected(self):
        source=(ROOT/'.build/today/g1b4-stage-retire.py').read_text(encoding='utf8')
        with self.assertRaises(AssertionError):build(source.replace('PDL = False','PDL = True',1))

if __name__=='__main__':unittest.main()
