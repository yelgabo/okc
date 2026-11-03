"""Check direct shared staging against the incumbent global gather layout."""
import importlib.util
from pathlib import Path
import unittest


class DirectInput(unittest.TestCase):
    def test_all_valid_and_padded_rows_match_original_stage_layout(self):
        path=Path(__file__).with_name('moe_direct_input.py')
        self.assertTrue(path.is_file(),'direct-input mapping implementation required')
        import moe_direct_input as model
        for count in range(1,65):
            tokens=[(r*13+r//3)%257 for r in range(count)]
            padded=(count+7)//8*8
            for stage in range(16):
                seen={}
                for lane in range(32):
                    for rb in range(0,padded,8):
                        r=rb+(lane&7)
                        for z in range(8):
                            g=4*z+(lane>>3)
                            destination,source=model.addresses(tokens,stage,lane,rb,z)
                            self.assertNotIn(destination,seen)
                            self.assertEqual(destination,((r//8)*32+g)*8+r%8)
                            if r<count:
                                base=tokens[r]*4096+stage*256
                                expected=tuple(base//2+8*((g>>1)&3)+2*(g>>3)+(g&1)+32*h for h in range(4))
                                self.assertEqual(source,expected)
                            else:self.assertIsNone(source)
                            seen[destination]=source
                self.assertEqual(set(seen),set(range(padded*32)))
        self.assertEqual(model.layout([],0),{})

    def test_source_reverses_and_keeps_consumer_and_phase_b_protocol(self):
        path=Path(__file__).with_name('moe_direct_input.py')
        self.assertTrue(path.is_file(),'direct-input native transform required')
        from moe_direct_input import transform
        from cuda_issue_groups import BASE,NATIVE_PIN,sha
        from prod_size_compile_g import bindings
        from expert_top100_pregamma import capture
        base=BASE.read_text();native=bindings(base)['MOE_CUDA_SRC'];self.assertEqual(sha(native),NATIVE_PIN)
        candidate,edits=transform(native);restored=candidate
        for old,new,count in reversed(edits):
            self.assertEqual(restored.count(new),count);restored=restored.replace(new,old)
        self.assertEqual(restored,native)
        self.assertNotIn('  wgs_gather(p, S);',candidate)
        self.assertIn('if (pb && lane != 0) return;',candidate)
        self.assertIn('if (lane == 0) mbar_arrive1(bar);',candidate)
        self.assertTrue('bar.sync 8, 32;' in candidate,'explicit producer publication barrier')
        for name in ('wgs_seg','wgs_phase','wgs_wproducer','wgs_flush','wgs_epi_tile'):
            begin=native.index('void '+name+'(');end=native.index('\n}',begin)+2
            self.assertIn(native[begin:end],candidate)
        old,routes=capture(base);new,newroutes=capture(base.replace(native,candidate))
        self.assertEqual(routes,newroutes)
        for a,b in zip(old,new):
            self.assertEqual(a['extra_cuda_cflags'],b['extra_cuda_cflags'])
            self.assertEqual(a['cuda_sources']!=b['cuda_sources'],'-DOKC_WGS=1' in a['extra_cuda_cflags'])


if __name__=='__main__':unittest.main()
