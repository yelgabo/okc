import ast
import base64
import struct
import unittest
import zlib
from pathlib import Path

from dl_g1_wgs_frontier import EXPERTS, WARPS, analyze, encode_trace, patch_device, trace_words

ROOT = Path(__file__).resolve().parents[1]


class FinalizerModel(unittest.TestCase):
    def setUp(self):
        self.ids = [[64]+[255]*7, [65]+[255]*7, [64,65]+[255]*6]
        self.marks = [0]*trace_words(2,3)
        # CTA 0 wrote expert 0. CTA 1 wrote expert 1. Both timestamps
        # precede the respective CTA's joined phase-B endpoint.
        self.marks[0] = 1020
        self.marks[WARPS*EXPERTS+1] = 1080
        self.marks[2*WARPS*EXPERTS:] = [1000,1002,1050,1090,1095,1098,
                                          1100,1103,1111,1110,1123,1116]

    def test_fixed_owner_and_nonlocal_routes(self):
        result = analyze(self.marks,self.ids,64,2)
        # CTA0 ideal ends at max(1050+10,1080)+5. CTA1 ends at1090+20.
        # Overall finish can advance by 13 ns, not the sum of everyone's wait.
        self.assertEqual(result['ideal_saving_us'], .013)
        self.assertEqual(result['unique_experts'],2)
        self.assertEqual(result['max_join_wait_us'],.045)
        with self.assertRaises(ValueError):
            analyze(self.marks,self.ids,0,2)

    def test_invalid_and_empty_samples(self):
        for index,value in ((0,1099),(-1,0),(-4,1000)):
            damaged = self.marks.copy()
            damaged[index] = value
            with self.assertRaises(ValueError):
                analyze(damaged,self.ids,64,2)
        self.marks[0] = self.marks[WARPS*EXPERTS+1] = 0
        empty = analyze(self.marks,[[255]*8]*3,64,2)
        self.assertEqual(empty['unique_experts'],0)
        self.assertIsNone(empty['last_writer_issue_us'])

    def test_capsule_preserves_ns_and_route_masks(self):
        raw = zlib.decompress(base64.b64decode(encode_trace(self.marks,self.ids,64,2)))
        self.assertEqual(struct.unpack_from('<4sHH',raw), (b'WGF1',2,3))
        fields = struct.unpack_from('<47I',raw,8)
        self.assertEqual(fields[:3], (21,81,0))
        self.assertEqual(fields[-3:],(1,2,3))
        self.assertEqual(fields[32:38],(0,2,50,90,95,98))

    def test_instrumentation_is_conditional_and_device_is_pinned(self):
        tree=ast.parse((ROOT/'muse_decode_layer/cand/g1b3_native_experts.py').read_text(encoding='utf-8'))
        device=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and getattr(n.targets[0],'id','')=='MOE_CUDA_SRC')
        patched=patch_device(device)
        # Exercise the inserted guards as a preprocessor would with FRONTIER off.
        kept=[];depth=0
        for line in patched.splitlines(keepends=True):
            key=line.lstrip()
            if depth:
                if key.startswith(('#if ', '#ifdef ', '#ifndef ')): depth+=1
                elif key.startswith('#endif'):depth-=1
            elif key.startswith('#if OKC_FRONTIER'):depth=1
            else:kept.append(line)
        self.assertEqual(depth,0)
        self.assertEqual(''.join(kept),device)
        with self.assertRaises(ValueError):
            patch_device(device+'\n')


if __name__ == '__main__':
    unittest.main()
