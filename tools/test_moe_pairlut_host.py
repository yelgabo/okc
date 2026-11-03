"""Check actual Torch packing and CPU orchestration, with CUDA calls explicitly facaded."""
import base64
import contextlib
import io
import json
import unittest
import zlib

import torch

import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small
from moe_pairlut_harness import build
from moe_pairlut import table


class Host(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source,cls.manifest=build()

    def environment(self,fault=None):
        ns,runtime=old.environment(self.source,fault)
        native=runtime.launch
        def launch(candidate,args,replay=False):
            candidate=args[0][6].numel()>256
            native(candidate,args,replay)
        runtime.launch=launch
        ns['_wgs_copy_pairlut']=lambda codes,scales,rows,sc6:codes.clone()
        return ns,runtime

    def test_actual_codepacking_is_bijective_and_scale_bytes_are_unchanged(self):
        ns={};exec(self.source,ns);ns['torch']=torch
        generator=torch.Generator().manual_seed(8117)
        codes=torch.randint(0,256,(256,2048),dtype=torch.uint8,generator=generator)
        scales=torch.randint(0x40,0x7f,(256,256),dtype=torch.uint8,generator=generator).view(torch.float8_e4m3fn)
        rows=torch.randperm(256,generator=generator)
        oldwords=ns['_pack_codes_nat'](codes,rows).to(torch.int64)&0xffffffff
        newwords=ns['_pack_codes_nat_pairlut'](codes,rows).to(torch.int64)&0xffffffff
        for r in range(4):
            for half in range(2):
                for bit in range(4):
                    self.assertTrue(torch.equal((oldwords>>ns['_POS'][(r,half,bit)])&1,(newwords>>(r*8+half*4+bit))&1))
        a=ns['_wgs_copy'](codes,scales,rows,True).view(-1,8960)
        b=ns['_wgs_copy_pairlut'](codes,scales,rows,True).view(-1,8960)
        self.assertTrue(torch.equal(a[:,8192:],b[:,8192:]))
        self.assertFalse(torch.equal(a[:,:8192],b[:,:8192]))

    def test_complete_ownership_and_original_skinny_state(self):
        ns,runtime=self.environment();log=io.StringIO()
        with contextlib.redirect_stdout(log):state,_=prepare_small(ns)
        report=json.loads(next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('EXPERT_MOE_CAPSULE ')))
        self.assertEqual(report['runtime_id'],self.manifest['runtime_id'])
        self.assertEqual(len(report['rows']),36)
        self.assertEqual(report['counters'],dict(observations=378,restorations=54,captures=216,timed_replays=1152,worst=[0.,0.]))
        self.assertIs(state,runtime.states[2])
        for i,owner in enumerate(runtime.states):
            lut=owner['ma_w'][6]
            self.assertEqual(lut.numel(),16640 if i>=2 else 256)
            self.assertTrue(torch.equal(lut[:256],runtime.states[0]['ma_w'][6]))
            if i>=2:self.assertEqual(lut[256:].tolist(),table())
        self.assertTrue(all(g.reset_done for g in runtime.graphs))
        self.assertEqual(len(runtime.builds),7)
        self.assertLess(len(log.getvalue()),16000)
        self.__class__.complete_log=log.getvalue()

    def test_failed_math_stays_failed_and_retains_partial(self):
        ns,runtime=self.environment('corrupt');log=io.StringIO()
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'nonfinite output'):prepare_small(ns)
        capsule=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PAIRLUT_PARTIAL '))
        partial=json.loads(zlib.decompress(base64.b64decode(capsule)))
        self.assertEqual(partial['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
        self.assertIn('nonfinite output',partial['error'])
        self.assertTrue(all(g.reset_done for g in runtime.graphs))

    def test_all_final_failures_preserve_all_rows(self):
        for fault in ('cleanup','post-cleanup-budget','serialization','final-budget'):
            with self.subTest(fault=fault):
                ns,runtime=self.environment();log=io.StringIO()
                if fault=='cleanup':
                    reset=ns['_h_reset']
                    def failed_reset(states):
                        reset(states)
                        raise RuntimeError('injected cleanup')
                    ns['_h_reset']=failed_reset
                elif fault=='serialization':
                    def failed_encode(report):raise RuntimeError('injected serialization')
                    ns['_pl_encode']=failed_encode
                else:
                    original=ns['_expert_moe_budget']
                    def budget(deadline):
                        if ns['_PL_LIVE']['stage']==fault:raise RuntimeError('injected '+fault)
                        original(deadline)
                    ns['_expert_moe_budget']=budget
                with contextlib.redirect_stdout(log):
                    with self.assertRaisesRegex(RuntimeError,'injected '+fault):prepare_small(ns)
                encoded=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PAIRLUT_PARTIAL '))
                partial=json.loads(zlib.decompress(base64.b64decode(encoded)))
                self.assertEqual(len(partial['rows']),36)
                self.assertEqual(partial['counters']['timed_replays'],1152)
                self.assertLess(len(encoded),12000)

    def test_repeated_prepare_uses_verified_compiled_cache(self):
        ns,runtime=self.environment();log=io.StringIO()
        with contextlib.redirect_stdout(log):
            first,_=prepare_small(ns)
            second,_=prepare_small(ns)
        self.assertEqual(len(runtime.builds),7)
        self.assertIsNot(first['ma_ext'],second['ma_ext'])
        self.assertFalse({x.data_ptr() for x in first['ma_w']} & {x.data_ptr() for x in second['ma_w']})

    def test_byte_scale_fallback_keeps_original_packing(self):
        ns,runtime=self.environment();ns['_sc6_ok']=lambda *args:False
        def wrong_pack(*args):raise AssertionError('lookup packing in byte fallback')
        ns['_wgs_copy_pairlut']=wrong_pack
        ns['_fallback_entry']=lambda w:ns['_expert_moe_prepare'](w,True)
        state,_=prepare_small(ns,'_fallback_entry')
        self.assertFalse(state['ma_sc6'])
        self.assertEqual(state['ma_w'][6].numel(),256)

    def test_cleanup_cannot_mask_primary_error(self):
        ns,runtime=self.environment('corrupt');log=io.StringIO()
        def cleanup(states):raise RuntimeError('secondary cleanup')
        ns['_h_reset']=cleanup
        with contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(RuntimeError,'nonfinite output'):prepare_small(ns)
        encoded=next(x.split(' ',1)[1] for x in log.getvalue().splitlines() if x.startswith('MOE_PAIRLUT_PARTIAL '))
        partial=json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertIn('secondary cleanup',partial['cleanup_error'])


if __name__=='__main__':unittest.main()
