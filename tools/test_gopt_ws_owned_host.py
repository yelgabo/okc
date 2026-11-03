"""Exercise generated host orchestration with CPU storage and synthetic CUDA output."""
import base64
import ast
import contextlib
import io
import json
import unittest
from unittest.mock import patch
import zlib

import torch

import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small
from gopt_ws_owned import build, PACKING, FP32_PROBE


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest, _ = build()

    def environment(self, fault=None):
        ns, runtime = old.environment(self.source, fault)
        def load(**kwargs):
            runtime.builds.append(kwargs)
            candidate = 'OWN shape/grid' in kwargs['cuda_sources']
            class Extension:
                def okc_set_pdl(self,value):assert value==0
                def okc_run(self,*args):
                    if runtime.active is not None:runtime.active.calls.append((candidate,args))
                    runtime.launch(candidate,args)
            return Extension()
        ns.update(load_inline=load,_H_NATIVE_LOADER=load,_owned_pack=lambda t:t.clone(),
                  _owned_fp32_probes=lambda weights,states,deadline,counters:None)
        return ns,runtime

    def run_entry(self,ns):
        log=io.StringIO()
        with contextlib.redirect_stdout(log):state,_=prepare_small(ns)
        return state,log.getvalue()

    def test_actual_packing_on_cpu_and_corruption_detection(self):
        def require(c,msg):
            if not c:raise RuntimeError(msg)
        ns=dict(torch=torch,require=require)
        exec(PACKING,ns)
        raw=(torch.arange(8*36864,dtype=torch.int64)*73+19).to(torch.uint8).reshape(8,36864)
        result=ns['_owned_pack'](raw).reshape(2,4,36864)
        restored=torch.empty_like(raw).reshape(2,4,36864)
        for st in range(4):
            for w in range(4):
                for h in range(2):
                    restored[:,w,(2*st+h)*4096:(2*st+h+1)*4096]=result[:,st,w*8192+h*4096:w*8192+(h+1)*4096]
                    restored[:,w,32768+(2*st+h)*512:32768+(2*st+h+1)*512]=result[:,st,32768+w*1024+h*512:32768+w*1024+(h+1)*512]
        self.assertTrue(torch.equal(restored.reshape(raw.shape),raw))
        class CorruptTorch:
            def __getattr__(self,key):return getattr(torch,key)
            def equal(self,a,b):return False
        ns['torch']=CorruptTorch()
        with self.assertRaisesRegex(RuntimeError,'round trip'):ns['_owned_pack'](raw)

    def test_probe_decoder_matches_problem_nvfp4(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'muse_moe_w4a16'))
        from nvfp4 import dequantize
        tree=ast.parse(FP32_PROBE)
        decode=tree.body[0].body[0]
        module=ast.fix_missing_locations(ast.Module(body=[decode],type_ignores=[]))
        ns=dict(torch=torch)
        exec(compile(module,'<actual embedded FP32 decoder>','exec'),ns)
        packed=torch.arange(256,dtype=torch.uint8).reshape(16,16)
        scales=torch.tensor([0.,2.**-9,.5,1.,2.,16.,240.,448.]*4,dtype=torch.float8_e4m3fn).reshape(16,2)
        actual=ns['decode'](packed.repeat(1,128),scales.repeat(1,128),torch.tensor(.125))
        expected=dequantize(packed.repeat(1,128),scales.repeat(1,128),torch.tensor(.125))
        self.assertTrue(torch.equal(actual,expected))

    def test_complete_release_dispatch_ownership_and_replay(self):
        ns,rt=self.environment()
        state,log=self.run_entry(ns)
        self.assertIs(state,rt.states[2])
        self.assertEqual(len(rt.builds),6)
        self.assertEqual([len(s['ma_w']) for s in rt.states],[14,14,16,16])
        self.assertTrue(all(g.reset_done for g in rt.graphs))
        marker='GOPT_WS_OWN_REPORT '
        encoded=next(l[len(marker):] for l in log.splitlines() if l.startswith(marker))
        report=json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(len(report['rows']),36)
        self.assertEqual(report['counters']['captures'],216)
        self.assertFalse(report['component_pass'])
        progress=[l.split(' ',1)[1] for l in log.splitlines() if l.startswith('GOPT_WS_OWN_PROGRESS ')]
        self.assertEqual(len(progress),36)
        self.assertEqual(len(json.loads(zlib.decompress(base64.b64decode(progress[-1])))['rows']),36)
        self.assertEqual(set(rt.states[2]['ma_ext']),{'tiny','owned','b4','main'})

    def test_failure_rows_and_primary_error_survive_cleanup(self):
        for fault in ('cleanup','post-cleanup-budget','serialization','final-budget'):
            with self.subTest(fault=fault):
                ns,rt=self.environment()
                if fault=='cleanup':
                    reset=ns['_h_reset']
                    def fail(states):reset(states);raise RuntimeError('injected cleanup')
                    ns['_h_reset']=fail
                elif fault=='serialization':
                    encode=ns['_ws_encode']
                    def fail(report):
                        if 'component_pass' in report:raise RuntimeError('injected serialization')
                        return encode(report)
                    ns['_ws_encode']=fail
                else:
                    budget=ns['_expert_moe_budget']
                    def fail(deadline):
                        if ns['_WS_LIVE']['stage']==fault:raise RuntimeError('injected '+fault)
                        budget(deadline)
                    ns['_expert_moe_budget']=fail
                log=io.StringIO()
                with contextlib.redirect_stdout(log):
                    with self.assertRaisesRegex(RuntimeError,'injected '+fault):prepare_small(ns)
                blob=next(l.split(' ',1)[1] for l in log.getvalue().splitlines() if l.startswith('GOPT_WS_OWN_PARTIAL '))
                report=json.loads(zlib.decompress(base64.b64decode(blob)))
                self.assertEqual(len(report['rows']),36)
                self.assertEqual(report['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
        ns,rt=self.environment('wrong')
        def reset(states):raise RuntimeError('secondary cleanup')
        ns['_h_reset']=reset
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'paired output error'):prepare_small(ns)


if __name__=='__main__':unittest.main()
