"""Check standalone factor-y ownership, dispatch, fallback selection and graph-safe forwarding."""
import unittest

import torch

from cuda_issue_groups import NATIVE_PIN, sha
from expert_top100_moe_test import environment, inputs, prepare_small
from moe_factor_y_release import build


class Release(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest = build()

    def test_exact_candidate_and_owned_forwarding(self):
        ns, calls, launches = environment(self.source)
        one = prepare_small(ns)[0]
        two = prepare_small(ns)[0]
        self.assertTrue(one['_expert_pair_arm'])
        self.assertEqual(len(calls), 5)
        self.assertFalse(self.manifest['public_submission_approved'])
        for call in calls:
            wgs = '-DOKC_WGS=1' in call['extra_cuda_cflags']
            self.assertEqual(sha(call['cuda_sources']), self.manifest['native_sha256' if wgs else 'helper_sha256'])
            self.assertNotEqual(sha(call['cuda_sources']), NATIVE_PIN)
        self.assertFalse({x.data_ptr() for x in one['ma_scratch']} & {x.data_ptr() for x in two['ma_scratch']})
        for tokens in (1, 4, 8, 16, 32, 33, 128, 256, 257):
            data = inputs(tokens, tokens)
            result = ns['kernel_fn'](data, one)
            self.assertTrue(torch.equal(result, data['hidden_states']))
            for actual, key in zip(launches[-1][1][2:5], ('hidden_states', 'topk_ids', 'topk_weights')):
                self.assertIs(actual, data[key])
        one['_expert_pair_arm'] = False
        with self.assertRaisesRegex(RuntimeError, 'STATE_IDENTITY_OR_ARM'):
            ns['kernel_fn'](inputs(32), one)
        with self.assertRaisesRegex(RuntimeError, 'INVALID_REBIND'):
            ns['_moe_bind_factor_y'](two, True)
        with self.assertRaisesRegex(RuntimeError, 'INVALID_REBIND'):
            ns['_moe_bind_factor_y'](dict(ma_ext={}, ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True), False)

    def test_original_scale_fallback_selection(self):
        for prmt, sc6 in ((False, False), (True, False), (True, True)):
            with self.subTest(prmt=prmt, sc6=sc6):
                ns, calls, launches = environment(self.source)
                ns['_ma_scales_prmt_exact'] = lambda *args: prmt
                ns['_sc6_ok'] = lambda *args: sc6
                state = prepare_small(ns)[0]
                self.assertEqual(bool(state['ma_sfx']), prmt)
                self.assertEqual(state['ma_sc6'], sc6)
                self.assertEqual(set(state['ma_ext']), {'tiny', 'tiny2', 'mid', 'main'} | ({'b4'} if prmt else set()))
                self.assertEqual(len(calls), 5 if prmt else 4)
                for call in calls:
                    flags = call['extra_cuda_cflags']
                    self.assertEqual('-DOKC_PRMTSC=1' in flags, prmt)
                    self.assertEqual('-DOKC_SC6W=1' in flags, sc6 and '-DOKC_WGS=1' in flags)
                    self.assertIn(sha(call['cuda_sources']), {self.manifest['native_sha256'], self.manifest['helper_sha256']})
                for tokens in (1, 4, 8, 16, 32, 33, 128, 256, 257):
                    self.assertTrue(torch.equal(ns['kernel_fn'](inputs(tokens), state), inputs(tokens)['hidden_states']))


if __name__ == '__main__':
    unittest.main()
