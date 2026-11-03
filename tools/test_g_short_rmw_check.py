import ast
import copy
import math
import os
from pathlib import Path
import unittest
from g_short_rmw_check import build
from g_short_final_rmw import helper, NEW, OLD

ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1]))


class ShortHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.manifest=build(ROOT)
        cls.original=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')
        cls.tree=ast.parse(cls.source)

    def test_native_bodies_and_fixed_policy(self):
        def literal(tree,name):
            return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])==name))
        before=ast.parse(self.original)
        for name in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
            control=literal(before,name);candidate=literal(self.tree,name)
            resource=candidate[len(control):] if name=='MOE_CUDA_SRC' else None
            if name!='MOE_CUDA_SRC':
                candidate=candidate.replace(NEW,OLD)
            self.assertTrue(candidate.startswith(control))
            self.assertIn('std::vector<int64_t> wl_resources()',candidate[len(control):])
        selected=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_moe')
        # First definition is actual source policy; final one is observer wrapper.
        text=ast.get_source_segment(self.source,selected)
        self.assertIn('key in ("tiny","tiny2")',text)
        self.assertNotIn('OKC_STAGE_RETIRE=0',text)
        self.assertLess(self.manifest['bytes'],1_000_000)
        self.assertEqual(build(ROOT),(self.source,self.manifest))

    def summary(self):
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wl_summary')
        ns={'_WL_LOADS':(1,8,16,32,128,256),'_WL_SEEDS':(993811,993817,993821),'_WL_AFFECTED':(1,8)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-summary>','exec'),ns)
        return ns['_wl_summary']

    def rows(self,treatment=95.):
        return [dict(tokens=t,context=c,seed=s,times_us=[100.,100.,treatment if t in (1,8) else 100.,treatment if t in (1,8) else 100.])
                for t in (1,8,16,32,128,256) for c in ('sliding','full') for s in (993811,993817,993821)]

    def test_actual_fullscore_gate(self):
        summary=self.summary()
        result=summary(self.rows(),'all')
        self.assertTrue(result['gate'])
        expected=math.log(100/95)/3
        for seed in result['seeds']:
            self.assertTrue(all(abs(x-expected)<1e-12 for x in seed['raw_allrow_log_contribution']))
        self.assertFalse(summary(self.rows(100),'all')['gate'])
        bad=self.rows();bad[0]['times_us'][2]=102
        self.assertFalse(summary(bad,'all')['gate'])
        bad=self.rows();bad[0]['times_us'][0]=80
        self.assertFalse(summary(bad,'all')['gate'])
        for corrupt in (float('nan'),0.,float('inf')):
            bad=self.rows();bad[0]['times_us'][1]=corrupt
            with self.assertRaisesRegex(AssertionError,'BAD_SUMMARY_TIMING'):summary(bad,'all')
        with self.assertRaises(AssertionError):summary(self.rows()[:-1],'all')


if __name__=='__main__':unittest.main()
