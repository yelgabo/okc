"""Check queue ownership, replay epochs and source selection without a GPU."""
import random
import unittest

import moe_bqueue as queue
from cuda_issue_groups import BASE, NATIVE_PIN, sha
from prod_size_compile_g import bindings
from expert_top100_pregamma import capture


class BQueue(unittest.TestCase):
    def test_selected_source_and_reverse(self):
        base = BASE.read_text()
        native = bindings(base)['MOE_CUDA_SRC']
        self.assertEqual(sha(native), NATIVE_PIN)
        candidate, edits = queue.transform(native)
        restored = candidate
        for old, new, count in reversed(edits):
            self.assertEqual(restored.count(new), count)
            restored = restored.replace(new, old)
        self.assertEqual(restored, native)
        self.assertEqual(native[native.index('#include <torch/extension.h>'):],
                         candidate[candidate.index('#include <torch/extension.h>'):])
        before, routes = capture(base)
        after, selected = capture(base.replace(native, candidate))
        self.assertEqual(routes, selected)
        for control, treatment in zip(before, after, strict=True):
            self.assertEqual(control['extra_cuda_cflags'], treatment['extra_cuda_cflags'])
            self.assertEqual(control['cuda_sources'] != treatment['cuda_sources'],
                             '-DOKC_WGS=1' in control['extra_cuda_cflags'])
        for name in ('wgs_seg', 'wgs_flush', 'wgs_epi_tile', 'wgs_gather'):
            start = native.index('void ' + name + '(')
            end = native.index('\n}', start) + 2
            self.assertIn(native[start:end], candidate)

    def test_unit_ownership_and_retirement(self):
        rng = random.Random(20261102)
        for jobs in (0, 1, 7, 32, 64, 96):
            for initial in (0, 1, 2, 15, 31):
                seq = [initial] * 132
                seen = []
                for unit in range(jobs * 16):
                    cta = rng.randrange(132)
                    expected = list(range(seq[cta], seq[cta] + 16))
                    roles = [expected[:] for _ in range(3)]
                    self.assertEqual(roles[0], roles[1])
                    self.assertEqual(roles[1], roles[2])
                    for n in expected:
                        self.assertEqual(n % 3, roles[2][n - seq[cta]] % 3)
                    seen.extend((unit, stage) for stage in range(16))
                    seq[cta] += 16
                self.assertEqual(set(seen), {(u, s) for u in range(jobs * 16) for s in range(16)})
                self.assertEqual(len(seen), len(set(seen)))

    def test_replay_counter_ownership(self):
        native = bindings(BASE.read_text())['MOE_CUDA_SRC']
        self.assertIn('#if OKC_WGS\n#undef OKC_MFIN\n#define OKC_MFIN 0', native)
        self.assertIn('if (i0 < 2) p.ctr[i0] = 0;', native)
        source, _ = queue.transform(native)
        self.assertIn('atomicAdd(p.ctr + 1, 1)', source)
        for total in (0, 16, 512, 1536):
            counter = 0
            for replay in range(16):
                self.assertEqual(counter, 0)
                claims = list(range(counter, counter + total + 132))
                self.assertEqual([x for x in claims if x < total], list(range(total)))
                counter += total + 132
                counter = 0


if __name__ == '__main__':
    unittest.main()
