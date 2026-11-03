"""Exercise matched orchestration with real CPU Torch and simulated device calls."""
import base64
import contextlib
import io
import json
import types
import unittest
import zlib

import torch
import d32_attn_worker_harness as builder
from test_expert_top100_harness_torch import Host


class AttentionHost(Host):
    def __init__(self, source, manifest, failure=None):
        super().__init__(source + '\n_expert_pair_bind = _d32_pair_bind\n', manifest, failure)
        self.namespace['_wl_weight_signature'] = lambda weights: 'cpu-fixture'

    def prepare_device_state(self, weights):
        state = super().prepare_device_state(weights)
        state['d32_attention_eligible'] = True
        state['d32_attention_module'] = types.ModuleType('attention_treatment')
        return state

    def make_inputs(self, batch, length, context, seed):
        data = super().make_inputs(batch, length, context, seed)
        data['config'].update(batch=batch, window=0 if context == 'full' else 2048,
                              num_q_heads=16, num_kv_heads=2, head_dim=64,
                              kv_scale=1., rope_theta=0., sm_scale=.125, qk_scale_factor=1.)
        return data

    def layer(self, data, state):
        self.namespace['_d32_attention_module'](data['config'], state)
        return super().layer(data, state)

    def front(self, data, weights, *args):
        result = super().front(data, weights, *args)
        result['fp8_finite_probe'] = result['source'].to(torch.float8_e4m3fn)
        if self.failure == 'fp8_nonfinite':
            result['fp8_finite_probe'][0, 0] = float('nan')
        return result

    def run(self):
        self.namespace['_EH_OFFLINE_HOLD'] = False
        self.log = io.StringIO()
        with contextlib.redirect_stdout(self.log):
            self.state = self.namespace['prepare'](self.weights())
        encoded = self.log.getvalue().split('D32_ATTENTION_PAIR_REPORT ')[1].strip()
        return json.loads(zlib.decompress(base64.b64decode(encoded)))


class GeneratedHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source, cls.manifest = builder.build()

    def test_four_arms_full_generated_prepare(self):
        host = AttentionHost(self.source, self.manifest)
        report = host.run()
        self.assertEqual((len(report['rows']), report['math_checks'], report['fresh_checks'],
                          report['restores'], report['transitions']), (36, 144, 1152, 144, 96))
        self.assertFalse(report['decision']['passed'])
        self.assertEqual(sorted(s['_d32_pair_arm'] for s in host.owners.values()),
                         [False, False, True, True])
        self.assertEqual(len({s['ma_scratch'][0].untyped_storage().data_ptr()
                              for s in host.owners.values()}), 4)
        self.assertEqual(report['official_arm'], 'attention_treatment')
        self.assertTrue(report['cleanup_restored'])
        for row in report['rows']:
            target = row['tokens'] == 32 and row['context'] == 'full'
            self.assertEqual(row['selected'], [[False], [False], [target], [target]])

    def test_injected_failures_do_not_publish_a_report(self):
        for failure in ('prepare', 'unknown_owner', 'output_alias', 'replay_output', 'fp8_nonfinite'):
            with self.subTest(failure=failure):
                host = AttentionHost(self.source, self.manifest, failure)
                with self.assertRaises((RuntimeError, AssertionError)):
                    host.run()
                self.assertNotIn('D32_ATTENTION_PAIR_REPORT ', host.log.getvalue())
                self.assertIsNone(host.namespace['_EH_CACHE'])
                self.assertFalse(host.namespace['_EH_BUILDS'])
                self.assertIs(host.namespace['_ma_ext'], host.original['_EH_BUILD'])
                self.assertIs(host.namespace['_fork_native_moe'], host.original['_EH_NATIVE'])
                self.assertTrue(host.torch.backends.cuda.matmul.allow_tf32)

    def test_route_diagnostic_keeps_strict_gate(self):
        host = AttentionHost(self.source, self.manifest)
        check = host.namespace['_d32_require_same_routes']
        ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        outputs = [(None, None, ids.clone()) for _ in range(4)]
        arms = [dict(_d32_pair_arm=i >= 2, _d32_probe_seen=[i >= 2]) for i in range(4)]
        check(outputs, arms, 32, 'full', 1102031, 'initial')
        outputs[2][2][1, 0] = 7
        log = io.StringIO()
        with contextlib.redirect_stdout(log), self.assertRaisesRegex(AssertionError, 'ROUTING_MISMATCH'):
            check(outputs, arms, 32, 'full', 1102031, 'fresh-0')
        event = json.loads(log.getvalue().split('D32_ROUTE_MISMATCH ')[1])
        self.assertEqual((event['tokens'], event['context'], event['stage']), (32, 'full', 'fresh-0'))
        self.assertEqual(event['mismatches'][0]['placement'], 2)
        self.assertEqual(event['mismatches'][0]['rows'], [1])


if __name__ == '__main__':
    unittest.main()
