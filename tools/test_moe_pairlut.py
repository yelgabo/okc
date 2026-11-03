"""Check the full code/scale domain against the incumbent's bit decoder."""
import struct
import unittest

from moe_pairlut import table,pack

GRID=(0.,.5,1.,1.5,2.,3.,4.,6.)


def fp32(bits):
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def bf16(value):
    raw=struct.unpack('<I',struct.pack('<f',value))[0]
    return ((raw+0x7fff+((raw>>16)&1))>>16)&0xffff


def rotate(q,n):
    return ((q<<n)|(q>>(32-n)))&0xffffffff


def incumbent(q):
    return [q&0x81c081c0,(q*16)&0x81c081c0,rotate(q,10)&0x81c081c0,
            ((q*64)&0x80c080c0)|(rotate(q,14)&0x01000100)]


class Codec(unittest.TestCase):
    def test_selected_source_and_all_ordering_operations_are_preserved(self):
        from cuda_issue_groups import BASE,NATIVE_PIN,sha
        from prod_size_compile_g import bindings
        from expert_top100_pregamma import capture
        from moe_pairlut import transform
        base=BASE.read_text();native=bindings(base)['MOE_CUDA_SRC'];self.assertEqual(sha(native),NATIVE_PIN)
        changed,edits=transform(native);restored=changed
        for old,new,count in reversed(edits):
            self.assertEqual(restored.count(new),count);restored=restored.replace(new,old)
        self.assertEqual(restored,native)
        for name in ('wgs_epi_tile','wgs_flush','wgs_phase','wgs_wproducer','wgs_xproducer','wgs_gather'):
            start=native.index('void '+name+'(');end=native.index('\n}',start)+2
            self.assertIn(native[start:end],changed)
        for op in ('wg_fence();','wg_commit();','wg_wait<1>();','wg_wait<0>();','fence.proxy.async','mbar_arrive1('):
            self.assertEqual(native.count(op),changed.count(op))
        before,routes=capture(base);after,newroutes=capture(base.replace(native,changed))
        self.assertEqual(routes,newroutes)
        for a,b in zip(before,after):
            self.assertEqual(a['extra_cuda_cflags'],b['extra_cuda_cflags'])
            self.assertEqual(a['cuda_sources']!=b['cuda_sources'],'-DOKC_WGS=1' in a['extra_cuda_cflags'])

    def test_complete_gate_rejects_unchanged_path_drift(self):
        from expert_top100_moe_harness_contract import decision
        from moe_pairlut_contract import pairlut_decision
        import copy
        rows=[[s,t,d,[100.,100.,97.,97.] if t>=32 else [100.]*4,[8,1,1] if t==1 else [t,1,t],
               [100/97]*4 if t>=32 else [1.]*4,[0.,0.]]
              for s in (1102031,1102037,1102049) for t in (1,8,16,32,128,256) for d in ('normal','skewed')]
        self.assertTrue(pairlut_decision(rows,decision(rows))['passed'])
        bad=copy.deepcopy(rows);bad[0][5]=[1.04]*4
        self.assertTrue(decision(bad)['component_pass'])
        self.assertFalse(pairlut_decision(bad,decision(bad))['passed'])

    def test_all_eligible_pairs_and_signed_zero(self):
        data=table();self.assertEqual(len(data),16384)
        for sf in range(63):
            scale=fp32((0x6000+sf*16)<<16)
            for pair in range(256):
                expected=[]
                for code in (pair&15,pair>>4):
                    value=GRID[code&7]
                    if code&8:value=-value
                    raw=bf16(value*2.**-126)
                    expected.append(bf16(fp32(raw<<16)*scale))
                self.assertEqual(data[sf*256+pair]&0xffffffff,expected[0]|expected[1]<<16,(sf,pair))
        self.assertEqual(data[0x88]&0xffffffff,0x80008000)

    def test_four_register_layout_matches_original_packing(self):
        import sys
        sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
        from prod_size_compile_g import bindings
        from cuda_issue_groups import BASE
        import ast
        source=BASE.read_text();ns=bindings(source)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_stored_pos')
        exec(ast.get_source_segment(source,node),ns);pos=ns['_stored_pos']()
        for offset in range(8):
            for code in range(16):
                codes=[0]*8;codes[offset]=code
                original=sum(((c>>b)&1)<<pos[(i//2,i%2,b)] for i,c in enumerate(codes) for b in range(4))
                rs=incumbent(original);packed=pack(codes)
                for i in range(4):
                    pair=(packed>>(8*i))&255
                    for half,c in enumerate((pair&15,pair>>4)):
                        value=GRID[c&7]
                        if c&8:value=-value
                        self.assertEqual((rs[i]>>(16*half))&0xffff,bf16(value*2.**-126))


if __name__=='__main__':unittest.main()
