"""Check candidate-only preparation and call forwarding with a CPU facade."""
import unittest

import torch

from expert_top100_moe_test import environment, inputs, prepare_small
from moe_bqueue_standalone import build


class Standalone(unittest.TestCase):
    def test_private_candidate_entry_and_owned_state(self):
        source, manifest = build()
        ns, calls, launches = environment(source)
        result = prepare_small(ns)
        state = result[0] if isinstance(result, tuple) else result
        self.assertTrue(state['_expert_pair_arm'])
        self.assertEqual(len(calls), 5)
        self.assertFalse(manifest['public_submission'])
        for tokens in (1, 8, 16, 32, 33, 128, 256, 257):
            data = inputs(tokens, tokens)
            output = ns['kernel_fn'](data, state)
            self.assertTrue(torch.equal(output, data['hidden_states']))
            self.assertIs(launches[-1][1][2], data['hidden_states'])
            self.assertIs(launches[-1][1][3], data['topk_ids'])
            self.assertIs(launches[-1][1][4], data['topk_weights'])
        state['_expert_pair_arm'] = False
        with self.assertRaisesRegex(RuntimeError, 'STATE_IDENTITY_OR_ARM'):
            ns['kernel_fn'](inputs(32), state)


if __name__ == '__main__':
    unittest.main()
