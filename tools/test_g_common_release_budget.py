"""Actual all12 orchestration plus exact-ID oracle-cache negatives, CPU only."""
import contextlib
import io
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


if __name__=='__main__':unittest.main()
