"""Check actual prepared gamma values, ownership and every captured argument binding."""
import base64
import contextlib
import io
import json
import unittest
import zlib

import torch

import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small
from moe_param_y_harness import build
from moe_param_y import gamma


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source,cls.manifest=build()

    def environment(self,fault=None):
        ns,runtime=old.environment(self.source,fault);native=runtime.launch
        def launch(candidate,args,replay=False):
            candidate=any(args[0][7].data_ptr()==s['ma_w'][14].data_ptr() for s in runtime.states)
            native(candidate,args,replay)
        runtime.launch=launch
        return ns,runtime

    def test_torch_gamma_over_finite_bf16_domain(self):
        raw=torch.arange(65536,dtype=torch.int32)
        raw=raw[(raw&0x7f80)!=0x7f80]
        values=raw.to(torch.int16).view(torch.bfloat16)
        prepared=(values.float()+1.0).to(torch.bfloat16).view(torch.int16).to(torch.int32)&0xffff
        expected=torch.tensor([gamma(v) for v in raw.tolist()],dtype=torch.int32)
        self.assertTrue(torch.equal(prepared,expected))

    def test_full_ownership_and_all_captured_argument_vectors(self):
        ns,runtime=self.environment();log=io.StringIO()
        with contextlib.redirect_stdout(log):state,_=prepare_small(ns)
        report=json.loads(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('EXPERT_MOE_CAPSULE ')))
        self.assertEqual(report['gamma_binding_checks'],36)
        self.assertEqual(report['loaded_builds'],list(range(7)))
        self.assertEqual(report['counters'],dict(observations=378,restorations=54,captures=216,timed_replays=1152,worst=[0.,0.]))
        self.assertIs(state,runtime.states[2]);self.assertEqual(len(runtime.graphs),216)
        for i,graph in enumerate(runtime.graphs):
            owner=runtime.states[i%4];args=graph.calls[0][1];weights=args[0];tokens=args[2].shape[0]
            self.assertEqual(len(weights),14)
            expected=list(owner['ma_w'][:14])
            if owner['_expert_pair_arm'] and tokens>16:expected[7]=owner['ma_w'][14]
            self.assertEqual([v.data_ptr() for v in weights],[v.data_ptr() for v in expected])
            self.assertTrue(graph.reset_done)
        for owner in runtime.states:
            self.assertEqual(len(owner['ma_w']),15)
            self.assertTrue(torch.equal(owner['ma_w'][14],(owner['ma_w'][7].float()+1).to(torch.bfloat16)))
            for value,control in zip(owner['ma_w'][:14],runtime.states[0]['ma_w'][:14]):self.assertTrue(torch.equal(value,control))

    def test_bad_argument_and_derived_alias_fail_before_timing(self):
        for fault in ('binding','alias'):
            ns,runtime=self.environment();log=io.StringIO()
            if fault=='binding':
                def wrong(state,tokens,arm):return state['ma_w'][:7]+[state['ma_w'][14]]+state['ma_w'][8:14]
                ns['_param_y_weights']=wrong
            else:
                original=ns['_h_ownership']
                def alias(states):
                    states[1]['ma_w'][14]=states[0]['ma_w'][14]
                    return original(states)
                ns['_h_ownership']=alias
            with self.subTest(fault=fault),contextlib.redirect_stdout(log):
                with self.assertRaisesRegex(RuntimeError,'argument binding|private weight/scratch storage'):prepare_small(ns)
            self.assertFalse(runtime.graphs)
            partial=json.loads(zlib.decompress(base64.b64decode(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PARAM_Y_PARTIAL ')))))
            self.assertEqual(partial['status'],'FAILED_PARTIAL_NOT_QUALIFIED')

    def test_corruption_and_final_failure_stay_failed(self):
        for fault in ('corrupt','serialization'):
            ns,runtime=self.environment('corrupt' if fault=='corrupt' else None);log=io.StringIO()
            if fault=='serialization':
                def fail(report):raise RuntimeError('injected serialization')
                ns['_pg_encode']=fail
            with self.subTest(fault=fault),contextlib.redirect_stdout(log):
                with self.assertRaisesRegex(RuntimeError,'nonfinite output|injected serialization'):prepare_small(ns)
            partial=json.loads(zlib.decompress(base64.b64decode(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PARAM_Y_PARTIAL ')))))
            self.assertEqual(partial['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
            if fault=='serialization':self.assertEqual(len(partial['rows']),36)
            self.assertTrue(all(g.reset_done for g in runtime.graphs))


if __name__=='__main__':unittest.main()
