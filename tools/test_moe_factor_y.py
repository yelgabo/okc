"""Check exact factorized storage and selected native read/write coverage."""
from pathlib import Path
import struct
import unittest


def f32(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def bits(value):
    try:return struct.unpack('<I',struct.pack('<f',value))[0]
    except OverflowError:return 0xff800000 if value<0 else 0x7f800000

def bf16(value):
    raw=bits(value)
    return ((raw+0x7fff+((raw>>16)&1))>>16)&0xffff


class FactorY(unittest.TestCase):
    def test_every_finite_bf16_roundtrips_and_products_match(self):
        path=Path(__file__).with_name('moe_factor_y.py')
        self.assertTrue(path.is_file(),'factorized representation implementation required')
        from moe_factor_y import reconstruct
        gammas=(0x0000,0x3c00,0x3f00,0x3f80,0x3f81,0xbf80,0xbf81,0x7f7f,0xff7f)
        for raw in range(65536):
            if raw&0x7f80==0x7f80:continue
            down=f32(raw<<16)
            for w in gammas:
                gamma=f32(bf16(1.+f32(w<<16))<<16)
                expected=bits(down*gamma)
                self.assertEqual(reconstruct(raw,w),expected)
        raw=0x3f81;w=0x3c00
        exact=reconstruct(raw,w)
        self.assertEqual(exact,0x3f820200)
        self.assertNotEqual(exact,bf16(f32(exact))<<16)

    def test_selected_source_preserves_math_and_publication(self):
        path=Path(__file__).with_name('moe_factor_y.py')
        self.assertTrue(path.is_file(),'factorized native transform required')
        from moe_factor_y import transform
        from cuda_issue_groups import BASE,NATIVE_PIN,sha
        from prod_size_compile_g import bindings
        from expert_top100_pregamma import capture
        base=BASE.read_text();native=bindings(base)['MOE_CUDA_SRC'];self.assertEqual(sha(native),NATIVE_PIN)
        candidate,edits=transform(native);restored=candidate
        for old,new,count in reversed(edits):
            self.assertEqual(restored.count(new),count);restored=restored.replace(new,old)
        self.assertEqual(restored,native)
        for name in ('wgs_seg','wgs_phase','wgs_flush','wgs_gather','wgs_wproducer','wgs_xproducer','wgs_fix','wgs_cnt'):
            begin=native.index(name+'(');begin=native.rfind('__device__',0,begin);end=native.index('\n}',begin)+2
            self.assertTrue(native[begin:end] in candidate,name)
        self.assertEqual(native.count('fence.proxy.async'),candidate.count('fence.proxy.async'))
        self.assertTrue('#define OKC_YBF 0' in candidate)
        self.assertTrue('#define OKC_YFACTOR 1' in candidate)
        self.assertEqual(candidate.count('wgs_factor_pair(yr.'),8)
        self.assertEqual(candidate.count('wgs_factor_y2(p,q,col)'),2)
        old,routes=capture(base);new,newroutes=capture(base.replace(native,candidate));self.assertEqual(routes,newroutes)
        for a,b in zip(old,new):
            self.assertEqual(a['extra_cuda_cflags'],b['extra_cuda_cflags'])
            self.assertEqual(a['cuda_sources']!=b['cuda_sources'],'-DOKC_WGS=1' in a['extra_cuda_cflags'])


if __name__=='__main__':unittest.main()
