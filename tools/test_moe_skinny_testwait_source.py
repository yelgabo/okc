"""Source equivalence gate: only two skinny full-slot sites may change policy."""
import ast
from pathlib import Path
import re
import unittest
from moe_skinny_testwait_check import build,HOOK,RESOURCE,QUERY

ROOT=Path(__file__).resolve().parent.parent


def constants(src):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(src).body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('_CUDA_SRC','_CPP_SRC','_SKINNY_BLOCK','SKINNY_VARIANTS')}


class SkinnyWaitSource(unittest.TestCase):
    def test_exact_scope(self):
        src=(ROOT/'muse_moe_w4a16/submission.py').read_text()
        old=constants(src); new=constants(build(src))
        resource=RESOURCE.replace('@@QUERY@@','\n'.join(QUERY.replace('@MODE@',str(m)) for m in (0,256,128,384)))
        self.assertEqual(new['_CUDA_SRC'].replace(resource+'\n',''),old['_CUDA_SRC'])
        self.assertEqual(new['_CPP_SRC'].replace('\nstd::vector<int64_t> moe_wait_resources();',''),old['_CPP_SRC'])
        restored=new['_SKINNY_BLOCK'].replace(HOOK+'\n','').replace(
            'full_slot_ready(&S.full[slot], (q / kNS) & 1)',
            'mbar_try(&S.full[slot], (q / kNS) & 1)')
        self.assertEqual(restored,old['_SKINNY_BLOCK'])
        self.assertEqual(new['_SKINNY_BLOCK'].count('full_slot_ready(&S.full[slot]'),2)
        self.assertEqual(new['SKINNY_VARIANTS'][:4],old['SKINNY_VARIANTS'])
        self.assertEqual(set(map(int,re.findall(r'kMode\s*&\s*(\d+)',new['_SKINNY_BLOCK']))),{15,16,32,64,128})
        for mode in (0,256,128,384):
            self.assertEqual(mode&15,0)
            self.assertEqual(mode&(16|32|64),0)
            self.assertNotIn(mode,(3,5,6,7))
        # New modes never become production tuner candidates implicitly.
        self.assertFalse(any(m in (0,3,5,6,7) for _,_,m in new['SKINNY_VARIANTS'][4:]))


if __name__=='__main__': unittest.main()
