import base64
import contextlib
import copy
import io
import json
import unittest
import zlib

import torch
import d32_route_observer as builder
from d32_attn_worker_harness_torch_test import AttentionHost
from test_expert_top100_harness_torch import Host


class ObserverHost(AttentionHost):
    def __init__(self, source, manifest):
        super().__init__(source, manifest)
        self.namespace.update(_D32_ROUTE_DECODE=self.decode, _D32_ROUTE_POOL=self.pool)

    def decode(self, data, state, qkv):
        return data['stream'] + qkv

    def pool(self, h, hr, tokens, state, pick):
        return (h[:, :1].expand(tokens, 4096).clone(),
                h[:, :1].expand(tokens, 256).clone()), -1

    def layer(self, data, state):
        ns = self.namespace
        ns['_d32_attention_module'](data['config'], state)
        h = ns['_decode_attention7'](data, state, data['attn_input'])
        ns['_pool_router_fp32'](h, None, data['config']['tokens'], state, 'bfp', _d32_callsite='cpu-layer')
        return Host.layer(self, data, state)


class RouteObserver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source, cls.manifest = builder.build()

    def test_same_set_swap_preserves_gate_and_records_boundaries(self):
        host = AttentionHost(self.source, self.manifest)
        ns = host.namespace
        ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        outputs = [(None, None, ids.clone()) for _ in range(4)]
        arms = []
        for i in range(4):
            data = {k: torch.zeros(2, 4) for k in ('stream', 'attn_input', 'k_cache', 'v_cache')}
            probe = dict(qkv=torch.zeros(2, 4), attention=torch.zeros(2, 4), normalized=torch.zeros(2, 4),
                         pooled=torch.zeros(2, 4096), logits=torch.zeros(2, 256),
                         weights=torch.full((2, 2), .5), local_ids=ids.clone())
            arms.append(dict(_d32_pair_arm=i >= 2, _d32_probe_seen=[False],
                _d32_route_probe=probe, _d32_route_data=data,
                _d32_route_before={k:v.clone() for k,v in data.items()},
                _d32_route_native_branch=dict(flags=['-DOKC_B4FIN=1'],sha256='cpu-fixture'),
                _d32_route_branch=dict(callsite='fixture', pick='bfp', layout='split')))
        outputs[3][2][1] = torch.tensor([4, 3])
        arms[3]['_d32_route_probe']['normalized'][1, 0] = .125
        log = io.StringIO()
        with contextlib.redirect_stdout(log), self.assertRaisesRegex(AssertionError, 'ROUTING_MISMATCH'):
            ns['_d32_require_same_routes'](outputs, arms, 32, 'sliding', 1102031, 'fresh-7')
        self.assertIn('D32_ROUTE_OBSERVER ', log.getvalue())
        encoded = next(l.split(' ', 1)[1] for l in log.getvalue().splitlines()
                       if l.startswith('D32_ROUTE_OBSERVER '))
        report = json.loads(zlib.decompress(base64.b64decode(encoded)))
        event = report['events'][-1]
        self.assertEqual(event['comparisons'][3]['cross']['normalized']['max_abs'], .125)
        self.assertTrue(event['mismatches'][0]['same_expert_sets'])
        self.assertEqual(event['mismatches'][0]['placement'], 3)
        self.assertNotIn('D32_ATTENTION_PAIR_REPORT ', log.getvalue())
        arms[3]['_d32_route_probe']['attention'] = arms[0]['_d32_route_probe']['attention']
        with self.assertRaisesRegex(AssertionError, 'OWNER_ALIAS'):
            ns['_d32_require_same_routes'](outputs, arms, 32, 'sliding', 1102031, 'fresh-7')

    def test_complete_generated_prepare_and_replay(self):
        host = ObserverHost(self.source, self.manifest)
        with self.assertRaisesRegex(RuntimeError, 'D32_OBSERVER_COMPLETE_NO_TIMING'):
            host.run()
        encoded = next(l.split(' ', 1)[1] for l in host.log.getvalue().splitlines()
                       if l.startswith('D32_ROUTE_OBSERVER '))
        report = json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(builder.read_report(host.log.getvalue(), self.manifest['pins']), report)
        self.assertEqual(report['reason'], 'EIGHT_FRESH_REPLAYS_COMPLETE')
        self.assertEqual(len(report['events']), 9)
        self.assertFalse(report['timing_eligible'])
        self.assertNotIn('D32_ATTENTION_PAIR_REPORT ', host.log.getvalue())
        for event in report['events']:
            self.assertFalse(event['mismatches'])
            for comparison in event['comparisons']:
                self.assertEqual(comparison['selected'], [False])
                self.assertEqual(comparison['branch']['layout'], 'split')
                self.assertTrue(all(v['equal'] for v in comparison['cross'].values()))
                if event['stage'] not in ('initial', 'fresh-0'):
                    changed = {k for k,v in comparison['previous'].items() if not v['equal']}
                    self.assertEqual(changed, {'before_k_cache', 'before_v_cache'}
                                     if event['stage']=='fresh-1' else set())
        owners = list(host.owners.values())
        self.assertEqual(len(owners), 4)
        self.assertEqual(len({a['_d32_route_probe']['attention'].data_ptr() for a in owners}), 4)
        self.assertIsNone(host.namespace['_EH_CACHE'])
        self.assertFalse(host.namespace['_EH_BUILDS'])
        for kind in ('pin', 'selection', 'coverage', 'timing'):
            bad = copy.deepcopy(report)
            if kind == 'pin':
                bad['pins']['parent'] = 'wrong'
            elif kind == 'selection':
                bad['events'][0]['comparisons'][0]['selected'] = [True]
            elif kind == 'coverage':
                del bad['events'][0]['comparisons'][0]['cross']['attention']
            else:
                bad['timing_eligible'] = True
            blob = base64.b64encode(zlib.compress(json.dumps(bad).encode())).decode()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                builder.read_report('D32_ROUTE_OBSERVER '+blob, self.manifest['pins'])

    def test_forwarding_and_unknown_layout(self):
        host = ObserverHost(self.source, self.manifest)
        ns = host.namespace
        h = torch.ones(2, 4)
        state = dict(_wl_probe=True, _d32_route_probe={})
        expected = ((torch.zeros(2, 4096), torch.zeros(2, 256)), -1)
        seen = []
        def pool(*args):
            seen.append(args)
            return expected
        ns['_D32_ROUTE_POOL'] = pool
        self.assertIs(ns['_pool_router_fp32'](h, None, 2, state, 'bfp', _d32_callsite='cpu-pool'), expected)
        self.assertIs(seen[0][0], h)
        self.assertIs(seen[0][3], state)
        self.assertIs(state['_d32_route_probe']['pooled'], expected[0][0])
        self.assertTrue(torch.equal(h, torch.ones_like(h)))
        ns['_D32_ROUTE_POOL'] = lambda *args: (torch.zeros(2, 3), 1)
        with self.assertRaisesRegex(AssertionError, 'COMBINED_LAYOUT'):
            ns['_pool_router_fp32'](h, None, 2, state, 'bfp', _d32_callsite='cpu-pool')

    def test_combined_layout_and_native_forwarding(self):
        host = ObserverHost(self.source, self.manifest)
        ns = host.namespace
        state = dict(_wl_probe=True, _d32_route_probe={'qkv': torch.zeros(2, 4), 'attention': torch.zeros(2, 4)})
        data = {k:torch.zeros(2, 4) for k in ('stream', 'attn_input', 'k_cache', 'v_cache')}
        ns['_d32_observer_before'](state, data)
        raw = torch.zeros(2, 2, 4608)
        raw[:, :, :4096] = 2
        raw[:, :, 4096:4352] = 3
        raw[:, :, 4352:] = 5
        expected = raw, 2
        ns['_D32_ROUTE_POOL'] = lambda *args: expected
        self.assertIs(ns['_pool_router_fp32'](torch.zeros(2, 4), None, 2, state, True,
                                             _d32_callsite='cpu-combined'), expected)
        seen = []
        class Native:
            _module = object()
            def okc_run(self, *args):
                seen.append(args)
                return 123
        ns['_D32_ROUTE_NATIVE'] = lambda *args: Native()
        ns['_EH_BUILDS'][id(Native._module)] = (('-DOKC_B4FIN=1',),
            'c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71')
        ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        weights = torch.full((2, 2), .5)
        args = (None, None, torch.ones(2, 4), ids, weights, torch.zeros(2, 4), 0, 32, 132)
        self.assertEqual(ns['_eh_native'](32, state).okc_run(*args), 123)
        self.assertEqual(len(seen[0]), len(args))
        self.assertTrue(all(a is b for a,b in zip(seen[0],args)))
        self.assertIs(state['_d32_route_probe']['weights'], weights)
        snapshot = ns['_d32_route_snapshot'](state, ids)
        self.assertTrue(torch.equal(snapshot['pooled'], torch.full((2, 4096), 4, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(snapshot['logits'], torch.full((2, 256), 16.)))
        del state['_d32_route_probe']['attention']
        with self.assertRaisesRegex(AssertionError, 'MISSING_BOUNDARY'):
            ns['_d32_route_snapshot'](state, ids)

    def test_fp8_diagnostic_values(self):
        host = ObserverHost(self.source, self.manifest)
        snapshot = host.namespace['_d32_route_cpu']
        value = torch.ones(2, dtype=torch.float8_e4m3fn)
        self.assertEqual(snapshot(value).dtype, value.dtype)
        value[1] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'NONFINITE'):
            snapshot(value)

    def test_explicit_callsite_tags(self):
        self.assertNotIn('_getframe', self.source)
        self.assertEqual(self.source.count('_d32_callsite='), 5)
        for name in ('_after_attention@', '_graft_after_attention@', '_sl_tile_after_attention@'):
            self.assertIn(name, self.source)


if __name__ == '__main__':
    unittest.main()
