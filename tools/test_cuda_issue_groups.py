import ast
from pathlib import Path
import re
import unittest

from tools.cuda_issue_groups import BASE, BASE_PIN, NATIVE_PIN, sha, transform
from tools.prod_size_compile_g import bindings
from tools.cuda_scale_constant import transform as scale_transform


class IssueGroups(unittest.TestCase):
    def test_exact_source_and_reversible_edits(self):
        base = BASE.read_text()
        self.assertEqual(sha(base), BASE_PIN)
        native = bindings(base)['MOE_CUDA_SRC']
        self.assertEqual(sha(native), NATIVE_PIN)
        result, edits = transform(native)
        for old, new, count in reversed(edits):
            self.assertEqual(result.count(new), count)
            result = result.replace(new, old)
        self.assertEqual(result, native)

    def test_group_banks_and_deferred_boundary(self):
        native = bindings(BASE.read_text())['MOE_CUDA_SRC']
        fci = int(re.search(r'^#define OKC_WGSFCI (\d+)', native, re.M)[1])
        original_size = int(re.search(r'^#define OKC_WGSGS (\d+)', native, re.M)[1])
        self.assertEqual((fci, original_size), (4, 2))
        target = (fci + 1) * original_size
        for ntt in range(1, 9):
            size = 4 if ntt <= 4 else 2
            flush_group = (target + size - 1) // size - 1
            issued, retired, banks, flush = [], set(), {}, []
            for group in range(16 // size):
                bank = group & 1
                if bank in banks:
                    self.assertIn(banks[bank], retired)
                banks[bank] = group
                issued.extend(range(group * size, (group + 1) * size))
                retired.update(range(group))
                if group == flush_group:
                    flush.append(len(issued))
            self.assertEqual(issued, list(range(16)))
            self.assertEqual(flush, [12 if ntt <= 4 else 10])
            self.assertGreaterEqual(flush[0], target)
            self.assertLess(flush[0] - target, size)

    def test_deferred_flush_reachable_at_all_control_groups(self):
        for original_group in range(8):
            target = (original_group + 1) * 2
            for size in (2, 4):
                group = (target + size - 1) // size - 1
                self.assertIn(group, range(16 // size))
                self.assertLess((group + 1) * size - target, size)

    def test_rejects_changed_source(self):
        with self.assertRaises(ValueError):
            transform('constexpr int GS = 7;')

    def test_host_and_publication_text_unchanged(self):
        native = bindings(BASE.read_text())['MOE_CUDA_SRC']
        result, edits = transform(native)
        marker = '#include <torch/extension.h>'
        self.assertEqual(native[native.index(marker):], result[result.index(marker):])
        for token in ('fence.proxy.async', 'wg_wait<0>()', 'mbar_arrive1(', 'griddepcontrol',
                      'red.release', 'wg_fence();', 'wg_commit();', 'wg_wait<1>();'):
            self.assertEqual(native.count(token), result.count(token), token)

    def test_scale_change_preserves_host_and_only_removes_zero_dependency(self):
        native = bindings(BASE.read_text())['MOE_CUDA_SRC']
        result, edits = scale_transform(native)
        self.assertEqual(len(edits), 1)
        old, new, count = edits[0]
        self.assertEqual(result.replace(new, old), native)
        marker = '#include <torch/extension.h>'
        self.assertEqual(native[native.index(marker):], result[result.index(marker):])
        self.assertEqual(result.count('  p.zero = 0;'), 1)

    def test_scale_requires_verified_host_zero_assignment(self):
        native = bindings(BASE.read_text())['MOE_CUDA_SRC']
        with self.assertRaises(ValueError):
            scale_transform(native.replace('  p.zero = 0;', '  p.zero = 1;'))


if __name__ == '__main__':
    unittest.main()
