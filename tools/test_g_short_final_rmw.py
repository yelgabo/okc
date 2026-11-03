import ast
import itertools
import os
from pathlib import Path
import unittest

import g_short_final_rmw as patch


ROOT = Path(os.environ.get('OKC_REPO_ROOT', Path(__file__).resolve().parents[1]))


class ShortFinalRmw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf-8')

    def test_exact_scope_and_off_default(self):
        for enabled in (False, True):
            source, manifest = patch.build(self.source, enabled)
            tree, original = ast.parse(source), ast.parse(self.source)
            patched = next(n for n in tree.body if isinstance(n, ast.Assign)
                           and ast.unparse(n.targets[0]) == "_MA_ALT_SRCS['moe_helper0']")
            value = ast.literal_eval(patched.value)
            self.assertEqual(value.replace(patch.NEW, patch.OLD), ast.literal_eval(patch.helper(self.source).value))
            # Normalize ONLY the changed CUDA literal, then compare every AST node.
            patched.value = patch.helper(self.source).value
            if enabled:
                tail = tree.body[-2:]
                self.assertEqual([ast.unparse(n.target) for n in tail], ['MOE_TINY_DEFINES', 'MOE_TINY2_DEFINES'])
                self.assertEqual([ast.literal_eval(n.value) for n in tail], [(f'-D{patch.FLAG}=1',)] * 2)
                tree.body = tree.body[:-2]
            self.assertEqual(ast.dump(tree), ast.dump(original))
            self.assertEqual(manifest['enabled'], enabled)
            self.assertEqual(value.count('atom.relaxed.gpu.global.add.u32'), 1)
            self.assertIn('!OKC_TINY || !OKC_WS || OKC_SPLIT || OKC_WGS', value)

    def test_all_fences_generations_and_waits_unchanged(self):
        source, _ = patch.build(self.source, True)
        before = ast.literal_eval(patch.helper(self.source).value)
        after = ast.literal_eval(patch.helper(source).value)
        for token in ('__threadfence()', 'WS_JOIN()', 'CTA_SYNC()', 'fence.proxy.',
                      'while (ld_acquire(bc)', 'atomicAdd(p.ctr + 3, 1)',
                      'p.ctr + 4 + (S.gen & 1)', '"l"(fc), "r"(pd.nst)'):
            self.assertEqual(before.count(token), after.count(token), token)
        with self.assertRaises(AssertionError):
            patch.build(self.source + '\n')

    def test_counter_epoch_arithmetic(self):
        # Arithmetic/identity fixture only, NOT a PTX memory-model simulator.
        for size in range(1, 8):
            for order in itertools.permutations(range(size)):
                counters = [0, 0]
                for generation in range(4):
                    current = generation & 1
                    counters[current ^ 1] = 0
                    seen = []
                    for _ in order:
                        old = counters[current]
                        counters[current] += 1
                        seen.append(old)
                    self.assertEqual(seen, list(range(size)))
                    self.assertEqual(counters[current], size)
                    self.assertEqual(seen.count(size - 1), 1)


if __name__ == '__main__':
    unittest.main()
