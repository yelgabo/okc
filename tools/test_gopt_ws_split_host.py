"""Test generated host control flow with synthetic native outputs and metadata."""
import base64
import contextlib
import io
import json
import unittest
import zlib

import torch

import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small
from gopt_ws_split import build


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest, _, _ = build()

    def environment(self, fault=None):
        ns, runtime = old.environment(self.source, fault)
        original_launch = runtime.launch
        def launch(candidate, args, replay=False):
            scratch, hidden, ids = args[1:4]
            gen = int(scratch[0][3])
            original_launch(candidate, args, replay)
            scratch[0].zero_()
            scratch[0][3] = gen+1
            scratch[1].zero_()
            if candidate and hidden.shape[0] <= 16:
                meta = scratch[3]
                pos = ns['_ws_positions'](ids.flatten().tolist())
                meta.view(torch.int16)[:len(pos)].copy_(torch.tensor(pos, dtype=torch.int16))
                meta[64] = gen
                if fault == 'metadata': meta[64] = gen+2
        runtime.launch = launch
        def load(**kwargs):
            runtime.builds.append(kwargs)
            candidate = '-DOKC_WS_SPLIT=1' in kwargs['extra_cuda_cflags']
            class Extension:
                def okc_set_pdl(self, value):
                    assert value == 0
                def okc_ws_finalize_probe(self, args):
                    meta, ss, y, tw, output, ctr, done = args
                    gen = int(meta[64]); tokens = output.shape[0]
                    pos = meta.view(torch.int16)[:tokens*8].tolist()
                    result = torch.zeros_like(output,dtype=torch.float32)
                    for slot in range(8):
                        q = torch.tensor(pos[slot::8],dtype=torch.int64)
                        idx = q.clamp_min(0)
                        scale = tw[:,slot]*torch.rsqrt(ss[gen,idx]*(1./4096)+1e-8)
                        result.add_(torch.where((q>=0)[:,None],scale[:,None]*y[idx],0.))
                    output.copy_(result.to(torch.bfloat16))
                    ctr[:2].zero_();ctr[3]=gen+1;done.zero_()
                def okc_run(self, *args):
                    if runtime.active is not None:
                        runtime.active.calls.append((candidate, args))
                    runtime.launch(candidate, args)
            return Extension()
        ns.update(load_inline=load, _H_NATIVE_LOADER=load)
        entry = ns['prepare']
        def prepared(weights):
            scratch = ns['_ma_make_scratch']
            ns['_ma_make_scratch'] = lambda device: [torch.zeros(4104, dtype=torch.int32),
                torch.zeros(3072, dtype=torch.float32), torch.zeros(80, dtype=torch.bfloat16),
                torch.zeros(128, dtype=torch.int32)]
            try:
                return entry(weights)
            finally:
                ns['_ma_make_scratch'] = scratch
        ns['_ws_host_entry'] = prepared
        return ns, runtime

    def capsule(self, log, marker):
        blobs = [line.split(' ',1)[1] for line in log.splitlines() if line.startswith(marker+' ')]
        self.assertEqual(len(blobs),1)
        self.assertLess(len(blobs[0]),12000)
        return json.loads(zlib.decompress(base64.b64decode(blobs[0])))

    def test_complete_ownership_replay_metadata_and_official_arm(self):
        ns, runtime = self.environment()
        log = io.StringIO()
        with contextlib.redirect_stdout(log): state, _ = prepare_small(ns, '_ws_host_entry')
        report = self.capsule(log.getvalue(), 'GOPT_WS_SPLIT_REPORT')
        from gopt_ws_split_read import capsule
        checked, verdict = capsule(log.getvalue(), self.manifest)
        self.assertEqual(checked, report)
        broken = dict(report, counters={**report['counters'], 'finalizer_reference_cases':5})
        with self.assertRaisesRegex(RuntimeError,'reference coverage'):
            capsule('GOPT_WS_SPLIT_REPORT '+ns['_ws_encode'](broken),self.manifest)
        self.assertIs(state, runtime.states[2])
        self.assertEqual(len(report['rows']),36)
        self.assertEqual(report['counters']['finalizer_reference_cases'],6)
        self.assertEqual(report['counters']['captures'],216)
        self.assertEqual(report['counters']['observations'],378)
        self.assertEqual(report['counters']['timed_replays'],1152)
        self.assertEqual(report['counters']['restorations'],54)
        self.assertEqual(len(runtime.builds),8)
        self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
        self.assertFalse(report['component_pass'])
        self.assertTrue(all(torch.count_nonzero(t)==0 for s in runtime.states for t in s['ma_scratch']))

    def test_metadata_guard_and_replay_failures_preserve_error(self):
        for fault in ('metadata','guard','stale','corrupt'):
            with self.subTest(fault=fault):
                ns, runtime = self.environment(fault)
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    with self.assertRaises(RuntimeError): prepare_small(ns, '_ws_host_entry')
                report = self.capsule(log.getvalue(), 'GOPT_WS_SPLIT_PARTIAL')
                self.assertEqual(report['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
                self.assertTrue(all(graph.reset_done for graph in runtime.graphs))

    def test_completed_rows_survive_final_failure_exits(self):
        for fault in ('cleanup','post-cleanup-budget','serialization','final-budget'):
            with self.subTest(fault=fault):
                ns, runtime = self.environment()
                if fault == 'cleanup':
                    original = ns['_h_reset']
                    def reset(states):
                        original(states)
                        raise RuntimeError('injected cleanup')
                    ns['_h_reset'] = reset
                elif fault == 'serialization':
                    original_encode = ns['_ws_encode']
                    def encode(report):
                        if 'component_pass' in report: raise RuntimeError('injected serialization')
                        return original_encode(report)
                    ns['_ws_encode'] = encode
                else:
                    budget = ns['_expert_moe_budget']
                    def fail(deadline):
                        if ns['_WS_LIVE']['stage'] == fault: raise RuntimeError('injected '+fault)
                        budget(deadline)
                    ns['_expert_moe_budget'] = fail
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    with self.assertRaisesRegex(RuntimeError, 'injected '+fault): prepare_small(ns, '_ws_host_entry')
                report = self.capsule(log.getvalue(),'GOPT_WS_SPLIT_PARTIAL')
                self.assertEqual(len(report['rows']),36)
                self.assertEqual(report['counters']['timed_replays'],1152)
                self.assertNotIn('GOPT_WS_SPLIT_REPORT ',log.getvalue())

    def test_primary_error_survives_cleanup_error(self):
        ns, runtime = self.environment('metadata')
        def reset(states): raise RuntimeError('injected cleanup')
        ns['_h_reset'] = reset
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'latched generation'): prepare_small(ns,'_ws_host_entry')
        report = self.capsule(log.getvalue(),'GOPT_WS_SPLIT_PARTIAL')
        self.assertIn('latched generation',report['error'])
        self.assertIn('injected cleanup',report['cleanup_error'])


if __name__ == '__main__': unittest.main()
