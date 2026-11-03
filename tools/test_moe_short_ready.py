"""Check the new dependency representation, selected source and replay reset."""
import contextlib
import base64
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import zlib
from unittest.mock import patch

import torch
from prod_size_compile_g import bindings
from cuda_issue_groups import BASE,sha
from moe_short_ready import transform,contributors,HELPER_PIN
from moe_short_ready_harness import build
from moe_short_gate import short_decision
import moe_short_ready_read as owning_reader
import expert_top100_moe_harness_test as old
from expert_top100_moe_test import prepare_small


class ShortReady(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source,cls.manifest=build()

    def test_all_contributor_intervals_and_adversarial_publication(self):
        rng=random.Random(13)
        for grid in range(1,137):
            for jobs in range(1,33):
                total=jobs*512
                ranges=[(total*c//grid,total*(c+1)//grid) for c in range(grid)]
                ready={c for c in range(grid) if rng.randrange(2)}
                for job in range(jobs):
                    lo,hi=job*512,(job+1)*512
                    pieces={c:max(0,min(b,hi)-max(a,lo)) for c,(a,b) in enumerate(ranges)}
                    expected={c for c,n in pieces.items() if n}
                    actual=set(contributors(jobs,grid,job))
                    self.assertEqual(actual,expected)
                    self.assertEqual(sum(pieces.values()),512)
                    self.assertEqual(actual<=ready,sum(pieces[c] for c in ready)==512)
                    self.assertFalse(actual<=actual-{max(actual)})

    def test_exact_selected_source_reverse_fences_and_reserved_layout(self):
        control=bindings(BASE.read_text())['_MA_ALT_SRCS']['moe_helper0']
        self.assertEqual(sha(control),HELPER_PIN)
        revised,edits=transform(control)
        for before,after,count in reversed(edits):
            self.assertEqual(revised.count(after),count);revised=revised.replace(after,before)
        self.assertEqual(revised,control)
        revised=transform(control)[0]
        for token in ('__threadfence();','fence.proxy.async','WS_JOIN();','CTA_SYNC();','atom.acq_rel.gpu.global.add.s32'):
            self.assertEqual(revised.count(token),control.count(token))
        for call in self.manifest['compiler_calls'][5:]:
            self.assertEqual(call['cuda_sources'],self.manifest['candidate_native_sha256'])
            self.assertIn('-DOKC_WS=1',call['extra_cuda_cflags'])
            self.assertNotIn('-DOKC_SPLIT=1',call['extra_cuda_cflags'])
        self.assertEqual(bindings(self.source)['MOE_CUDA_SRC'],bindings(BASE.read_text())['MOE_CUDA_SRC'])
        self.assertIn('JMAX <= 128 && 128 + 136 <= 512',revised)

    def env(self,fault=None):
        ns,rt=old.environment(self.source,fault)
        launch=rt.launch
        def native(candidate,args,replay=False):
            launch(candidate,args,replay)
            args[1][0][136:272].zero_()
            if candidate and replay and fault=='flag':args[1][0][136]=1
        rt.launch=native
        entry=ns['prepare']
        def prepare(weights):
            scratch=ns['_ma_make_scratch']
            def resized(device):
                result=scratch(device);result[0]=torch.zeros(512,dtype=torch.int32);return result
            ns['_ma_make_scratch']=resized
            try:return entry(weights)
            finally:ns['_ma_make_scratch']=scratch
        ns['host_entry']=prepare
        return ns,rt

    def test_complete_host_selected_calls_graph_updates_and_reset(self):
        ns,rt=self.env();out=io.StringIO()
        with patch.object(old.harness.adapter,'CANDIDATE_NATIVE',self.manifest['candidate_native_sha256']):
            with contextlib.redirect_stdout(out):state,_=prepare_small(ns,'host_entry')
        report=json.loads(next(x.split(' ',1)[1] for x in out.getvalue().splitlines() if x.startswith('EXPERT_MOE_CAPSULE ')))
        self.assertIs(state,rt.states[2]);self.assertTrue(state['_expert_pair_arm'])
        self.assertEqual(len(report['rows']),36);self.assertEqual(report['loaded_builds'],list(range(7)))
        self.assertEqual(report['counters'],dict(observations=378,restorations=54,captures=216,
                                                timed_replays=1152,worst=[0.,0.],flag_zero_checks=1512))
        self.assertFalse(report['component_pass']);self.assertFalse(short_decision(report['rows'])['passed'])
        self.assertTrue(all(g.reset_done for g in rt.graphs))
        self.assertTrue(all(torch.count_nonzero(t)==0 for s in rt.states for t in s['ma_scratch']))
        self.__class__.complete_log=out.getvalue()

    def test_wrong_output_stale_flags_and_cleanup_remain_failed(self):
        for fault in ('wrong','flag','cleanup','post-cleanup-budget','serialization','final-budget'):
            ns,rt=self.env(fault if fault in ('wrong','flag') else None);out=io.StringIO()
            if fault=='cleanup':
                reset=ns['_h_reset']
                def fail(states):reset(states);raise RuntimeError('injected cleanup')
                ns['_h_reset']=fail
            elif fault in ('post-cleanup-budget','final-budget'):
                budget=ns['_expert_moe_budget']
                def fail(deadline):
                    if ns['_SR_LIVE'].get('stage')==fault:raise RuntimeError('injected '+fault)
                    budget(deadline)
                ns['_expert_moe_budget']=fail
            elif fault=='serialization':
                def fail(report):raise RuntimeError('injected serialization')
                ns['_sr_encode']=fail
            with patch.object(old.harness.adapter,'CANDIDATE_NATIVE',self.manifest['candidate_native_sha256']):
                with contextlib.redirect_stdout(out):
                    with self.assertRaises(RuntimeError):prepare_small(ns,'host_entry')
            self.assertIn('MOE_SHORT_READY_PARTIAL ',out.getvalue())
            self.assertNotIn('EXPERT_MOE_CAPSULE ',out.getvalue())
            self.assertTrue(all(g.reset_done for g in rt.graphs))

    def test_reader_requires_own_source_and_complete_frozen_short_gate(self):
        job='0123456789abcdef'
        root=Path(__file__).resolve().parents[1]
        raw=json.loads((root/'docs/steering/evidence/2026-11-02-moe-tiny-timeline/status-wait.raw.json').read_text())
        raw.update(id=job,content_hash='2'*64,payload_ref='muse_moe_w4a16/'+'2'*64+'.py',
                   output=dict(log='synthetic.'+job+'.log'))
        raw['result']['log_tail']=self.__class__.complete_log
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'profiles').mkdir()
            (folder/'paired.py').write_text(self.source)
            (folder/'paired.manifest.json').write_text(json.dumps(self.manifest))
            (folder/'accepted.json').write_text(json.dumps(dict(job_id=job,source_sha256=self.manifest['source_sha256'])))
            (folder/'status-wait.raw.json').write_text(json.dumps(raw))
            (folder/'profiles'/('synthetic.'+job+'.log')).write_text(self.__class__.complete_log)
            with patch.object(owning_reader,'build',return_value=(self.source,self.manifest)):
                result=owning_reader.read(folder,job)
                self.assertFalse(result['target_qualified']);self.assertEqual(len(result['report']['rows']),36)
                with self.assertRaises(RuntimeError):owning_reader.read(folder,'fedcba9876543210')
                bad=self.__class__.complete_log.replace('"flag_zero_checks":1512','"flag_zero_checks":1511')
                (folder/'profiles'/('synthetic.'+job+'.log')).write_text(bad)
                with self.assertRaises(RuntimeError):owning_reader.read(folder,job)
                report=json.loads(next(x.split(' ',1)[1] for x in self.__class__.complete_log.splitlines()
                                       if x.startswith('EXPERT_MOE_CAPSULE ')))
                for row in report['rows']:
                    if row[1]==1:row[3][2:]=[990.,990.];row[5]=[1000/990]*4
                self.assertTrue(short_decision(report['rows'])['passed'])
                report['status']='FAILED_PARTIAL_NOT_QUALIFIED'
                encoded=base64.b64encode(zlib.compress(json.dumps(report).encode(),9)).decode()
                log='MOE_SHORT_READY_PARTIAL '+encoded+'\n'
                raw['result'].update(passed=False,returncode=1,status='error',score_us=None,log_tail=log)
                (folder/'status-wait.raw.json').write_text(json.dumps(raw))
                (folder/'profiles'/('synthetic.'+job+'.log')).write_text(log)
                self.assertFalse(owning_reader.read(folder,job)['target_qualified'])


if __name__=='__main__':unittest.main()
