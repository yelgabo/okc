"""RNE representation CPU/source proof; not an executed CUDA kernel."""
import ast,os,unittest
from pathlib import Path
from proj_g_short_pair_rne import build,PAIR,GATE,GATE_NEW,DOWN,DOWN_NEW
ROOT=Path(os.environ.get('OKC_REPO_ROOT',Path(__file__).resolve().parents[1].parent/'okc'))
SOURCE=(ROOT/'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8')

class Tests(unittest.TestCase):
    def test_source_scope_default_and_exact_products(self):
        self.assertEqual(build(SOURCE)[0],SOURCE)
        s,m=build(SOURCE,True);old=ast.parse(SOURCE);new=ast.parse(s)
        self.assertEqual(len(new.body),len(old.body)+2)
        for a,b in zip(old.body,new.body):
            if isinstance(a,ast.Assign) and ast.unparse(a.targets[0])=="_MA_ALT_SRCS['moe_helper0']":
                before,after=ast.literal_eval(a.value),ast.literal_eval(b.value)
                self.assertIn('__floats2bfloat162_rn(a, b)',after)
                for original,replacement,count in ((GATE,GATE_NEW,1),(DOWN,DOWN_NEW,2)):
                    self.assertEqual(after.count(original),count) # exact original remains flag0 branch
                    self.assertEqual(after.count(replacement),count)
                self.assertIn('gsc,',after.replace(' * gsc,',' gsc,'))
                for token in ('fence.proxy','__threadfence','bar.sync','mbar_arrive','fork_bf16(1.f +'):
                    self.assertEqual(before.count(token),after.count(token))
            else:self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))
        self.assertTrue(m['gamma_unchanged']);self.assertEqual(m['h_unrolled_pair_sites_per_entry'],6)
    def test_independent_all_rounding_classes_and_lane_order(self):
        import torch
        # Every binary32 high-half x eight residual representatives covers all
        # rounding classes: below/at/above half, sign, exponent, overflow, NaN.
        hi=torch.arange(65536,dtype=torch.int64)[:,None]
        lo=torch.tensor([0,1,0x7ffe,0x7fff,0x8000,0x8001,0xfffe,0xffff],dtype=torch.int64)[None,:]
        a=((hi<<16)|lo).flatten();b=a.roll(123457)
        def scalar(x):
            h=x>>16;l=x&65535
            rounded=(h+((l>0x8000)|((l==0x8000)&((h&1)!=0))).long())&65535
            return torch.where((x&0x7fffffff)>0x7f800000,0x7fff,rounded)
        def packed_lane(x):
            out=((x+0x7fff+((x>>16)&1))>>16)&65535
            return torch.where((x&0x7fffffff)>0x7f800000,0x7fff,out)
        bits=packed_lane(a)|(packed_lane(b)<<16)
        x=(bits&65535)<<16;y=(bits>>16)<<16
        self.assertTrue(torch.equal(x,scalar(a)<<16));self.assertTrue(torch.equal(y,scalar(b)<<16))
        self.assertTrue(torch.equal((x>>16)&65535,scalar(a)))
        print('RNE_BINARY32_ROUNDING_CLASS_PAIRS',len(a))

if __name__=='__main__':unittest.main()
