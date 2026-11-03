"""Actual all12 orchestration plus exact-ID oracle-cache negatives, CPU only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch

import test_g_common_release_host as old
from g_common_release_budget import BACK,SUPPORT,build


class All12(old.Startup):
    @classmethod
    def setUpClass(cls):
        cls.sources={arm:build(old.ROOT,arm)[0] for arm in ('common','short-rmw')}

    def test_actual_reference_reuse_and_complete_counts(self):
        for arm,source in self.sources.items():
            ns,stats,prod,audit=old.fixture(source,arm)
            original=ns['_GQ_REF_BACK'];calls=[]
            def counted(front,weights,ids):
                calls.append((id(front),ids.clone()));return original(front,weights,ids)
            ns['_GQ_REF_BACK']=counted
            state,report=self.call(ns)
            self.assertIs(state,prod[0]);self.assertEqual(stats['validators'],72)
            self.assertEqual((report['checks'],report['captures'],report['fresh_checked']),(72,12,36))
            self.assertEqual((report['empty_checks'],report['empty_captures'],report['restored']),(60,12,48))
            self.assertEqual(report['reference_cache'],{'hit':96,'miss':48,'peak_entries':4})
            self.assertEqual(len(calls),48);self.assertIsNone(ns['_GQB_CACHE'])

    def test_cache_cleanup_after_validation_exception(self):
        ns,_,_,_=old.fixture(self.sources['common'],'common')
        validate=ns['_qual_validate'];calls=[0]
        def bad(*args):
            validate(*args);calls[0]+=1
            if calls[0]==2:raise RuntimeError('after cache hit')
        ns['_qual_validate']=bad
        with self.assertRaisesRegex(RuntimeError,'after cache hit'):self.call(ns)
        self.assertIsNone(ns['_GQB_CACHE']);self.assertEqual(ns['_GQB_COUNTS']['hit'],2)

    def test_full_prepare_inference_and_gradient_contexts_both_arms(self):
        for arm,source in self.sources.items():
            for inference in (False,True):
                for gradient in (False,True):
                    with self.subTest(arm=arm,inference=inference,gradient=gradient):
                        ns,stats,production,audit=old.fixture(source,arm)
                        prepare=ns['_GQ_NORMAL_PREPARE'];private=ns['_gq_private']
                        observed=[]
                        def normal(weights):
                            observed.append((torch.is_inference_mode_enabled(),torch.is_grad_enabled()))
                            return prepare(weights)
                        def checked_private(*args):
                            self.assertFalse(torch.is_inference_mode_enabled())
                            self.assertFalse(torch.is_grad_enabled())
                            return private(*args)
                        ns['_GQ_NORMAL_PREPARE']=normal;ns['_gq_private']=checked_private
                        with torch.inference_mode(inference),torch.set_grad_enabled(gradient):
                            state,report=self.call(ns)
                            self.assertEqual((torch.is_inference_mode_enabled(),torch.is_grad_enabled()),(inference,gradient))
                            self.assertEqual(torch.is_inference(state['qkv']),inference)
                        self.assertEqual(observed,[(inference,gradient)])
                        self.assertEqual(report['reference_cache'],dict(hit=96,miss=48,peak_entries=4))
                        phases=report['prepare_phases']
                        self.assertEqual(phases['normal_prepare_begin'],1)
                        self.assertEqual(phases['normal_prepare_end'],1)
                        for phase in ('input_allocate_begin','front_bf16_begin','front_fp8_begin','front_end'):
                            self.assertEqual(phases[phase],24)
                        self.assertEqual(phases['cell_begin'],12);self.assertEqual(phases['cell_end'],12)
                        self.assertEqual(phases['validate_begin'],72);self.assertEqual(phases['validate_end'],72)
                        self.assertEqual(phases['reference_back_miss'],48)
                        self.assertLessEqual(sum(phases.values()),500)
                        self.assertEqual(stats['validators'],72);self.assertIsNone(ns['_GQB_CACHE'])

    def test_context_and_cache_restore_on_private_and_normal_failure(self):
        for arm,source in self.sources.items():
            for stage in ('_GQ_NORMAL_PREPARE','_qual_validate'):
                for inference in (False,True):
                    with self.subTest(arm=arm,stage=stage,inference=inference):
                        ns,_,_,_=old.fixture(source,arm)
                        original=ns[stage]
                        def fail(*args):
                            original(*args)
                            raise RuntimeError('context restoration probe')
                        ns[stage]=fail
                        with torch.inference_mode(inference),torch.enable_grad():
                            with self.assertRaisesRegex(RuntimeError,'context restoration probe'):
                                self.call(ns)
                            self.assertEqual(torch.is_inference_mode_enabled(),inference)
                            self.assertTrue(torch.is_grad_enabled())
                        self.assertIsNone(ns['_GQB_CACHE'])

    def test_failure_context_and_phase_log_bound(self):
        ns,_,_,_=old.fixture(self.sources['short-rmw'],'short-rmw')
        def fail(*args):raise RuntimeError('phase probe')
        ns['_qual_validate']=fail;captured=io.StringIO()
        with torch.inference_mode(),contextlib.redirect_stdout(captured):
            with self.assertRaisesRegex(RuntimeError,'phase probe'):
                ns['prepare']({})
        self.assertIn("GQUAL_FAILURE_CONTEXT validate_begin (1, 'sliding')",captured.getvalue())
        self.assertIsNone(ns['_GQB_CACHE'])

    def test_actual_report_matches_pinned_reader(self):
        from g_common_rmw_receipt import source_contract,qualification
        for arm,source in self.sources.items():
            ns,_,_,_=old.fixture(source,arm)
            with torch.inference_mode():state,report=self.call(ns)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'repaired.py';path.write_bytes(source.encode())
                contract=source_contract(path)
                observed=qualification('G_COMMON_QUAL_DONE '+json.dumps(report)+'\n',contract)
            self.assertEqual(observed['status'],'complete_native')
            self.assertEqual(observed['treatment_exercised'],arm=='short-rmw')
            self.assertEqual(report['prepare_phases']['cleanup_end'],1)
            self.assertGreater(report['prepare_phase_elapsed_seconds'],0)
            self.assertAlmostEqual(sum(report['prepare_phase_seconds'].values()),report['prepare_phase_elapsed_seconds'])
            self.assertGreaterEqual(report['prepare_phase_seconds']['cleanup_begin'],0)

    def test_finite_wrong_output_and_cache_rejected_after_memo_hits(self):
        for arm,source in self.sources.items():
            for target in ('output','cache'):
                for inference in (False,True):
                    with self.subTest(arm=arm,target=target,inference=inference):
                        ns,_,_,_=old.fixture(source,arm)
                        validate=ns['_qual_validate'];calls=[]
                        def corrupt(base,data,output,refs,weights):
                            calls.append(None)
                            if len(calls)==2:
                                (output[0] if target=='output' else data['k_cache']).add_(1000)
                            validate(base,data,output,refs,weights)
                        ns['_qual_validate']=corrupt
                        with torch.inference_mode(inference):
                            with self.assertRaisesRegex(AssertionError,'MODEL_ROUTER_CACHE'):self.call(ns)
                        self.assertEqual(ns['_GQB_COUNTS']['hit'],2)
                        self.assertIsNone(ns['_GQB_CACHE'])

    def test_repeated_prepare_has_independent_cache_epoch(self):
        for arm,source in self.sources.items():
            ns,stats,production,_=old.fixture(source,arm)
            for inference in (True,False):
                with torch.inference_mode(inference):state,report=self.call(ns)
                self.assertIs(state,production[-1]);self.assertIsNone(ns['_GQB_CACHE'])
                self.assertEqual(report['reference_cache'],dict(hit=96,miss=48,peak_entries=4))
                self.assertEqual(report['prepare_phases']['normal_prepare_begin'],1)
            self.assertEqual(stats['prepares'],2)


class ExactCache(unittest.TestCase):
    def fixture(self):
        calls=[]
        def back(front,weights,ids):
            calls.append(ids.clone())
            result=front['h']+ids.sum().float()
            return result,result*2
        def finite(value,label):
            assert bool(torch.isfinite(value).all()),'NONFINITE'
        ns=dict(torch=torch,time=time,_GQ_REF_BACK=back,_qual_finite=finite,
            _gq_budget=lambda s:None)
        exec(SUPPORT+'\n'+BACK,ns)
        ns['_GQB_START']=time.monotonic();ns['_GQB_CACHE']={}
        return ns,calls,{'h':torch.arange(8.).reshape(2,4),'local_start':0},{},torch.arange(16).reshape(2,8)

    def test_exact_ids_distinct_bank_and_mutation(self):
        ns,calls,front,weights,ids=self.fixture();run=ns['_gq_ref_back']
        with contextlib.redirect_stdout(io.StringIO()):
            a=run(front,weights,ids);b=run(front,weights,ids.clone())
            self.assertIs(a,b);self.assertEqual(len(calls),1)
            ids[0,0]=100;changed=run(front,weights,ids)
            self.assertEqual(len(calls),2);self.assertFalse(torch.equal(a[0],changed[0]))
            # Changing the live graph IDs never changes the saved exact key.
            self.assertEqual(ns['_GQB_CACHE'][id(front)]['ids'][0,0],100)
            ids[0,0]=0;run(front,weights,ids);self.assertEqual(len(calls),3)
            second={'h':front['h'].clone(),'local_start':0}
            run(second,weights,ids);self.assertEqual(len(calls),4)
            front['h'].add_(1);run(front,weights,ids);self.assertEqual(len(calls),5)

    def test_result_mutation_and_nonfinite_hit_are_fatal(self):
        for bypass_version in (False,True):
            ns,calls,front,weights,ids=self.fixture()
            with contextlib.redirect_stdout(io.StringIO()):
                out=ns['_gq_ref_back'](front,weights,ids)
                (out[0].data if bypass_version else out[0]).fill_(float('nan'))
                with self.assertRaisesRegex(AssertionError,'NONFINITE|RESULT_MUTATED'):
                    ns['_gq_ref_back'](front,weights,ids)
                self.assertEqual(len(calls),1)

    def test_snapshot_mutation_owner_change_and_nonfinite_miss(self):
        for change,expected in (
            (lambda ns,front,weights,ids:ns['_GQB_CACHE'][id(front)]['ids'].add_(1),'IDS_MUTATED'),
            (lambda ns,front,weights,ids:ns['_GQB_CACHE'][id(front)].__setitem__('weights',{}),'REFERENCE_OWNER'),
        ):
            ns,calls,front,weights,ids=self.fixture()
            with contextlib.redirect_stdout(io.StringIO()):
                ns['_gq_ref_back'](front,weights,ids);change(ns,front,weights,ids)
                with self.assertRaisesRegex(AssertionError,expected):ns['_gq_ref_back'](front,weights,ids)
                self.assertEqual(len(calls),1)
        ns,calls,front,weights,ids=self.fixture();front['h'].fill_(float('nan'))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(AssertionError,'NONFINITE'):ns['_gq_ref_back'](front,weights,ids)


if __name__=='__main__':unittest.main()
