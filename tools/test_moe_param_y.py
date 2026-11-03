"""Check the prepared-parameter contract and preservation of the active parent source."""
from pathlib import Path
import unittest


class ParamY(unittest.TestCase):
    def test_prepared_parameter_preserves_original_product(self):
        self.assertTrue(Path(__file__).with_name('moe_param_y.py').is_file(),'prepared-gamma transform required')
        from moe_param_y import gamma,reconstruct
        from moe_factor_y import reconstruct as original
        for raw in range(65536):
            if raw&0x7f80==0x7f80:continue
            for down in (0,0x8000,1,0x8001,0x3f81,0xbf81,0x7f7f,0xff7f):
                self.assertEqual(reconstruct(down,gamma(raw)),original(down,raw))

    def test_native_and_active_parent_source_binding(self):
        self.assertTrue(Path(__file__).with_name('moe_param_y.py').is_file(),'prepared-gamma native required')
        from moe_param_y import transform
        from moe_factor_y_harness import build
        from cuda_issue_groups import BASE,NATIVE_PIN,sha
        from prod_size_compile_g import bindings
        from expert_top100_pregamma import capture
        base=BASE.read_text();native=bindings(base)['MOE_CUDA_SRC'];changed,edits=transform(native);restored=changed
        for old,new,count in reversed(edits):
            self.assertEqual(restored.count(new),count);restored=restored.replace(new,old)
        self.assertEqual(restored,native)
        self.assertEqual(sha(native),NATIVE_PIN)
        self.assertEqual(build()[1]['source_sha256'],'78506b743ace84e1fc0b088fe7fe3b23009f3de0bf5136e20413504026e07492')
        self.assertEqual(native.count('fence.proxy.async'),changed.count('fence.proxy.async'))
        self.assertTrue('__fmul_rn(down.x,raw.x)' in changed)
        old,routes=capture(base);new,newroutes=capture(base.replace(native,changed));self.assertEqual(routes,newroutes)
        for a,b in zip(old,new):
            self.assertEqual(a['extra_cuda_cflags'],b['extra_cuda_cflags'])
            self.assertEqual(a['cuda_sources']!=b['cuda_sources'],'-DOKC_WGS=1' in a['extra_cuda_cflags'])


if __name__=='__main__':unittest.main()
