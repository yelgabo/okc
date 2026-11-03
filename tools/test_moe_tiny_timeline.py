"""Validate helper selection, timestamp bounds and failure retention on CPU."""
import base64
import contextlib
import copy
import io
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib
import moe_tiny_timeline_result as result_reader

import torch
from cuda_issue_groups import BASE, sha
from prod_size_compile_g import bindings
from moe_tiny_timeline import transform, trace, STRIDE, HELPER_PIN
from moe_tiny_timeline_harness import build
import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small


def synthetic(jobs=32):
    values=[0]*(136*STRIDE)
    rng=random.Random(44)
    for c in range(132):
        r=[0]*STRIDE
        entry=1000000+rng.randrange(1000)
        a0,a1=jobs*512*c//132,jobs*512*(c+1)//132
        b0,b1=jobs*256*c//132,jobs*256*(c+1)//132
        nA,nB=a1-a0,b1-b0
        r[0]=entry;r[1]=entry+450;r[2]=entry+500
        r[3]=r[2]+nA*100+100;r[4]=r[3]+10
        r[5]=r[4]+nB*100+100;r[6]=r[5]+10;r[7]=r[6]+100;r[8]=r[7]+100
        r[16]=r[5]-20;r[17:23]=a0,a1,b0,b1,2,1
        for q in range(nA+nB):
            pb=q>=nA
            start=(r[4]+(q-nA)*100) if pb else r[2]+q*100
            d0,d1=(start,start+5) if q==nA else (0,0)
            end=0 if q>=nA+nB-min(2,nB) else start+75
            r[32+q*6:38+q*6]=d0,d1,start+10,start+20,start+60,end
        values[c*STRIDE:(c+1)*STRIDE]=r
    return values


def decode(log):
    lines=[x.split(' ',1)[1] for x in log.splitlines() if x.startswith('MOE_TINY_TIMELINE ')]
    assert len(lines)==1 and len(lines[0])<=12000
    return json.loads(zlib.decompress(base64.b85decode(lines[0])))


class Timeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source,cls.manifest=build()

    def test_exact_selected_source_and_reversible_instrument(self):
        helper=bindings(BASE.read_text())['_MA_ALT_SRCS']['moe_helper0']
        self.assertEqual(sha(helper),HELPER_PIN)
        observed,edits=transform(helper)
        for before,after,count in reversed(edits):
            self.assertEqual(observed.count(after),count)
            observed=observed.replace(after,before)
        self.assertEqual(observed,helper)
        calls=self.manifest['compiler_calls']
        self.assertEqual(len(calls),7)
        for call in calls[5:]:
            self.assertEqual(call['cuda_sources'],self.manifest['observer_helper_sha256'])
            self.assertIn('-DOKC_WS=1',call['extra_cuda_cflags'])
            self.assertNotIn('-DOKC_WGS=1',call['extra_cuda_cflags'])
        self.assertEqual(self.manifest['official_arm'],'control0')

    def test_trace_capacity_orders_partitions_and_corruption(self):
        for jobs in (0,1,2,8,32):
            r=trace(synthetic(jobs),132)
            self.assertEqual(len(r['ctas']),132)
            self.assertLessEqual(max(c[9]+c[10] for c in r['ctas']),192)
        for kind in ('stale','unit','partition','helper'):
            x=synthetic()
            if kind=='stale':x[8]=0
            elif kind=='unit':x[35]=x[36]+100
            elif kind=='partition':x[17]=1
            else:x[37]=0
            with self.assertRaises(ValueError):trace(x,132)

    def test_observer_cost_and_noise_cannot_qualify_a_gain(self):
        ns={}
        exec(self.source,ns)
        for sample in ([26.,26.,26.5,26.5],[26.,26.,16.,16.],[26.,26.,26.,90.]):
            cost=ns['_tt_costs']([[1102031,1,'normal',sample]])
            self.assertEqual(cost[0][-1],sample[2]==26.5)

    def env(self,fault=None):
        ns,rt=old.environment(self.source,fault)
        original=rt.launch
        def launch(candidate,args,replay=False):
            original(candidate,args,replay)
            if candidate and len(args[1])==4:
                args[1][3].copy_(torch.tensor(synthetic(),dtype=torch.int64))
        rt.launch=launch
        entry=ns['prepare']
        def prepare(weights):
            scratch=ns['_ma_make_scratch']
            ns['_ma_make_scratch']=lambda dev:scratch(dev)+[torch.zeros(136*STRIDE,dtype=torch.int64)]
            try:return entry(weights)
            finally:ns['_ma_make_scratch']=scratch
        ns['host_entry']=prepare
        return ns,rt

    def test_complete_host_forwarding_coverage_capsule_and_restore(self):
        ns,rt=self.env();log=io.StringIO()
        with patch.object(old.harness.adapter,'CANDIDATE_NATIVE',self.manifest['observer_helper_sha256']):
            with contextlib.redirect_stdout(log):state,_=prepare_small(ns,'host_entry')
        report=decode(log.getvalue())
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['loaded_builds'],list(range(7)))
        self.assertEqual(report['counters'],dict(observations=42,restorations=6,captures=24,timed_replays=768,worst=[0.,0.]))
        self.assertEqual([x[0] for x in report['traces']],[1,8])
        self.assertIs(state,rt.states[0]);self.assertTrue(report['observer_valid'])
        self.assertTrue(all(g.reset_done for g in rt.graphs))
        self.assertTrue(all(torch.count_nonzero(t)==0 for s in rt.states for t in s['ma_scratch']))
        self.__class__.complete=report

    def test_partial_preserves_primary_and_post_cleanup_failures(self):
        for fault in ('wrong','cleanup','post-cleanup-budget','serialization','final-budget'):
            ns,rt=self.env('wrong' if fault=='wrong' else None);log=io.StringIO()
            if fault=='cleanup':
                reset=ns['_h_reset']
                def failed(states):reset(states);raise RuntimeError('injected cleanup')
                ns['_h_reset']=failed
            elif fault in ('post-cleanup-budget','final-budget'):
                budget=ns['_expert_moe_budget']
                def failed(deadline):
                    if ns['_TT_LIVE'].get('stage')==fault:raise RuntimeError('injected '+fault)
                    budget(deadline)
                ns['_expert_moe_budget']=failed
            elif fault=='serialization':
                encode=ns['_tt_encode']
                def failed(report):
                    if report['status']=='complete':raise RuntimeError('injected serialization')
                    return encode(report)
                ns['_tt_encode']=failed
            with patch.object(old.harness.adapter,'CANDIDATE_NATIVE',self.manifest['observer_helper_sha256']):
                with contextlib.redirect_stdout(log):
                    with self.assertRaises(RuntimeError):prepare_small(ns,'host_entry')
            report=decode(log.getvalue());self.assertEqual(report['status'],'failed')
            if fault!='wrong':self.assertEqual(len(report['rows']),6)
            self.assertTrue(all(g.reset_done for g in rt.graphs))

    def test_receipt_association_and_incomplete_evidence_rejection(self):
        report=copy.deepcopy(self.__class__.complete)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'profiles').mkdir()
            (folder/'paired.py').write_text(self.source)
            (folder/'paired.manifest.json').write_text(json.dumps(self.manifest))
            (folder/'accepted.json').write_text(json.dumps(dict(job_id='synthetic',source_sha256=self.manifest['source_sha256'])))
            def files():
                encoded=base64.b85encode(zlib.compress(json.dumps(report).encode(),9)).decode()
                log='MOE_TINY_TIMELINE '+encoded+'\n'
                (folder/'profiles'/'synthetic.log').write_text(log)
                receipt=dict(id='synthetic',team='yelgabo',board='muse_moe_w4a16',mode='check',status='done',
                             output=dict(log='synthetic.log'),result=dict(passed=True,returncode=0,log_tail=log,score_us=6000.))
                (folder/'status-wait.raw.json').write_text(json.dumps(receipt))
            files()
            with patch.object(result_reader,'build',return_value=(self.source,self.manifest)):
                result=result_reader.read(folder,'synthetic')
                self.assertTrue(result['observer_usable']);self.assertFalse(result['algorithm_gain'])
                for fault in ('job','coverage','counter'):
                    bad=copy.deepcopy(report)
                    if fault=='coverage':report['traces']=report['traces'][:1]
                    elif fault=='counter':report['counters']['captures']=23
                    files()
                    with self.assertRaises(Exception):result_reader.read(folder,'wrong' if fault=='job' else 'synthetic')
                    report=bad


if __name__=='__main__':unittest.main()
