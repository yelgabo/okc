import ast
import contextlib
import hashlib
import io
import os
from pathlib import Path
import unittest

from moe_make_sc6_compare import make,LF_PIN

ROOT=Path(__file__).resolve().parents[1]
DONOR=Path(os.environ.get('OKC_SC6_SOURCE',ROOT.parent/'okc/.build/today/kel/kernels/muse_moe_w4a16.py'))


class SourceTests(unittest.TestCase):
    def test_transform(self):
        src=DONOR.read_text(encoding='utf-8')
        body=(ROOT/'tools/moe_sc6_compare_body.py').read_text(encoding='utf-8')
        out=make(src,(ROOT/'muse_moe_w4a16/baseline.py').read_text(encoding='utf-8'),body)
        self.assertEqual(out,make(src,(ROOT/'muse_moe_w4a16/baseline.py').read_text(encoding='utf-8'),body))
        tree=ast.parse(out)
        d=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='CUDA_SRC')
        self.assertEqual(d.count('fence.proxy.async.global;'),3)
        self.assertIn('if (nj == 0) return 0;',d)
        self.assertIn('cudaLaunchCooperativeKernel',d)
        self.assertIn('#define OKC_YBF 0',d)
        self.assertIn('#undef OKC_MFIN\n#define OKC_MFIN 0',d)
        self.assertIn('k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT',d)
        self.assertIn('MID_DEFINES += ("-DOKC_DF=0"',out)
        self.assertIn('PS_SHAPES = {}',out)
        self.assertIn('PS_PATCHES = {}',out)
        self.assertNotIn('reference.py',out)
        with self.assertRaises(AssertionError): make(src+'\n','',body)

    def test_actual_summary_gate(self):
        body=(ROOT/'tools/moe_sc6_compare_body.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='_sc_summary')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-summary>','exec'),ns)
        def rows(a,b,c,d): return [(m,seed,[a,b,c,d]) for m in (1,8,16,32,128,256) for seed in (901,902)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(ns['_sc_summary'](rows(100,100,98,98))[2])
            self.assertFalse(ns['_sc_summary'](rows(100,100,100,100))[2])
            self.assertFalse(ns['_sc_summary'](rows(100,100,102,102))[2])
            self.assertFalse(ns['_sc_summary'](rows(100,100,98,96))[2]) # compressed null too large


if __name__=='__main__':unittest.main()
