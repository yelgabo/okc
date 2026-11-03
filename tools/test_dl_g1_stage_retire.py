import ast
from pathlib import Path
import unittest

from dl_build_g1_qualification import build
from dl_g1_stage_retire import build_layer, patch_device, SIGNATURE


class StageRetirement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        base = (root / 'muse_decode_layer/cand/g1b2_native_experts.py').read_text('utf8')
        cls.layer = build(base, bf16_attention=True).decode()
        cls.cuda = cls.literal(cls.layer)

    @staticmethod
    def literal(source):
        return next(ast.literal_eval(n.value) for n in ast.parse(source).body
                    if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'MOE_CUDA_SRC')

    def test_pure_matches_diagnostic_path_with_exact_boundary(self):
        on = patch_device(self.cuda)
        off = self.cuda
        switch = patch_device(self.cuda, switch=True)
        before, branches = switch.split('#if defined(OKC_STAGE_RETIRE) && OKC_STAGE_RETIRE\n')
        # The inserted switch surrounds the whole segment; the literal's own
        # conditional directives are retained. Use its exact two function texts.
        def function(text):
            start = text.index(SIGNATURE)
            return text[start:text.index('\n}\n', start) + 2]

        old, new = function(off), function(on)
        self.assertTrue(branches.startswith(new + '\n#else\n' + old + '\n#endif'))
        self.assertEqual(switch.replace(new + '\n#else\n' + old, old).count(SIGNATURE), 1)
        self.assertEqual(before + branches[len(new + '\n#else\n' + old + '\n#endif'):], off.replace(old, ''))
        self.assertEqual(self.literal(build_layer(self.layer).decode()), on)
        self.assertEqual(old.count('mbar_arrive1(S.bars + BW_E + ws)'), new.count('mbar_arrive1(S.bars + BW_E + ws)'))
        self.assertIn('wg_wait<0>();\n    asm volatile("fence.proxy.async.shared::cta;', new)
        self.assertIn('mbar_arrive1(S.bars + BX_E + xs);\n    ++seq;', new)
        self.assertNotIn('BX_E + prev', new)
        self.assertIn('stage(std::false_type{});\n  wg_wait<0>();', new)
        with self.assertRaises(ValueError):
            patch_device(on)

    def test_same_releases_for_all_nonempty_partitions(self):
        # Each phase owns a positive contiguous subset of the 16 K stages.
        # Test single-stage segments and every wrap of the NX=3/4 ring too.
        for size in (3, 4):
            for initial in range(2 * size):
                for begin in range(16):
                    for end in range(begin + 1, 17):
                        inputs = [(initial + relative) % size for relative in range(end - begin)]
                        previous = None
                        legacy = []
                        for slot in inputs:
                            if previous is not None:
                                legacy.append(previous)
                            previous = slot
                        legacy.append(previous)
                        self.assertEqual(legacy, inputs)


if __name__ == '__main__':
    unittest.main()
