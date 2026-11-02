"""Stdlib CPU tests. Synthetic receipts/CUDA facades are NOT target evidence."""
import ast
import base64
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import time
import types
import unittest
import zlib

import expert_top100_harness as builder
from expert_top100_harness_contract import BANKS, LOADS, decision, route_key, validate_rows
from expert_top100_harness_read import OFFICIAL, capsule, read
from expert_top100_harness_runtime import BODY


def rows():
    return [dict(tokens=t, context=c, seed=s, times_us=[100.,100.,97.,97.], component_us=[50.,50.,48.,48.],
        actual_routes=[[[t,route_key(t),i>=2]] for i in range(4)],local_start=0,
        reference_group_counts=[t*8]+[0]*7) for t in LOADS for c in ('sliding','full') for s in BANKS]


def report(manifest):
    rs=rows()
    return dict(schema=1,pins=manifest['pins'],official_arm='exact607',cleanup_restored=True,
        math_checks=144,restores=144,fresh_checks=1152,transitions=96,prepare_s=123.,
        owners_bytes=[100]*4,weight_signature='a'*64,
        readonly_inventory=[dict(device='cuda:0',bytes=100,paths=[['weights/w',[10],'torch.bfloat16']])],
        rows=rs,decision=decision(rs))


def encode(r):
    return 'EXPERT_TOP100_REPORT '+base64.b64encode(zlib.compress(json.dumps(r,separators=(',',':')).encode(),9)).decode()+'\n'


def fixture(manifest,source):
    job='0123456789abcdef';content='b'*64;payload_ref='muse_decode_layer/'+content+'.py'
    records=[dict(event='test',status='pass',index=i) for i in range(13)]
    records.append(dict(event='setup',status='pass',error=None,prepare_s=123.))
    for load,layer in sorted(OFFICIAL):
        case=dict(load=load,layer=layer,weight=46 if layer=='sliding' else 16)
        records.append(dict(event='benchmark',case=case,status='pass',flags=[],median_ns=1.,median_raw_ns=1.,baseline_ns=1.,baseline_raw_ns=1.))
        records.append(dict(event='profile',case=case,report='synthetic CPU fixture; no CUDA'))
    receipt=dict(id=job,board='muse_decode_layer',mode='check',team='yelgabo',status='done',content_hash=content,
        payload_ref=payload_ref,output=dict(log='/archive/'+job+'.log'),
        result=dict(status='ok',passed=True,returncode=0,checks=dict(ok=True),flags=[],error=None,records=records))
    upload=dict(job=job,team='yelgabo',mode='check',board='muse_decode_layer',content_hash=content,
        payload_ref=payload_ref,source_sha256=manifest['source_sha256'],source_bytes=manifest['bytes'])
    log=encode(report(manifest))
    att=dict(job=job,owning_log=receipt['output']['log'],sha256=hashlib.sha256(log.encode()).hexdigest(),source_sha256=manifest['source_sha256'])
    return [receipt,job,log,manifest,source.encode(),upload,att]


class Contract(unittest.TestCase):
    def test_all_crosses_all_banks_and_weighting(self):
        r=rows();d=validate_rows(r)
        self.assertTrue(d['gate'])
        self.assertEqual(len(d['banks']),3)
        self.assertAlmostEqual(d['banks'][0]['cross_log_gains'][0],math.log(100/97))
        for row in r:
            if row['context']=='full':row['times_us']=[200.,200.,194.,194.]
        self.assertAlmostEqual(decision(r)['banks'][0]['weighted'][0]['us'][0],(46*100+16*200)/62)
        for row in r:
            if row['seed']==BANKS[-1]:row['times_us'][3]=row['times_us'][0]
        self.assertFalse(decision(r)['gate'])

    def test_missing_duplicate_invalid_timing_and_actual_routes(self):
        for bad in (rows()[:-1],rows()[:-1]+[rows()[0]]):
            with self.assertRaisesRegex(AssertionError,'coverage'):validate_rows(bad)
        for value in (0.,-1.,float('nan'),float('inf'),True):
            r=rows();r[0]['times_us'][2]=value
            with self.assertRaisesRegex(AssertionError,'timing'):validate_rows(r)
        r=rows();r[0]['actual_routes'][2][0][1]='main'
        with self.assertRaisesRegex(AssertionError,'actual route'):validate_rows(r)
        r=rows();r[0]['local_start']=1
        with self.assertRaisesRegex(AssertionError,'local start'):validate_rows(r)

    def test_frozen_one_percent_and_twice_null(self):
        r=rows()
        for row in r:row['times_us']=[100.,100.,99.1,99.1]
        self.assertFalse(decision(r)['gate'])
        for row in r:row['times_us']=[100.,100.,99.,99.]
        self.assertTrue(decision(r)['gate'])
        for row in r:row['times_us']=[100.,98.,96.,96.]
        self.assertFalse(decision(r)['gate'])  # one cross < twice independent-control null

    def test_regression_equality_stops_and_unchanged_drift_is_disclosed(self):
        r=rows()
        affected=next(x for x in r if x['tokens']==32)
        affected['times_us'][2]=101.
        self.assertFalse(decision(r)['gate'])
        self.assertTrue(decision(r)['banks'][0]['affected_regressions'])
        r=rows();r[0]['times_us'][2]=101.
        self.assertFalse(decision(r)['banks'][0]['affected_regressions'])
        self.assertTrue(any(v[-1]==1.01 for v in decision(r)['banks'][0]['unchanged_drift']))


class Generated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.manifest=builder.build()

    def test_deterministic_exact_package_prefix_and_no_native_repair(self):
        self.assertEqual((self.source,self.manifest),builder.build())
        package,_=builder.pair.build()
        self.assertTrue(self.source.startswith(package+'\n_EH_OFFLINE_HOLD=True\n_EH_PINS='))
        self.assertLess(self.manifest['bytes'],1_000_000)
        self.assertEqual(self.manifest['target_fit'],'UNMEASURED')
        self.assertFalse(self.manifest['dispatch_ready'])
        suffix=self.source[len(package):]
        self.assertNotIn('qual_att7',suffix)
        self.assertNotIn('SHORT_FINAL_RELAXED',suffix)
        self.assertNotIn('common_control',suffix)
        self.assertEqual(self.manifest['pins']['candidate'],builder.CANDIDATE)
        self.assertIn("_wl_check(fresh,data[i],outputs[i],weights,freshrefs)",suffix)

    def test_strict_reader_positive_fixture_has_no_release_entitlement(self):
        result=read(*fixture(self.manifest,self.source))
        self.assertTrue(result['decision']['gate'])
        self.assertFalse(result['dispatch_ready'])
        self.assertFalse(result['release_eligible'])

    def test_wrong_job_team_mode_payload_source_and_log(self):
        mutations=[lambda a:a[0].update(id='wrong'),lambda a:a[0].update(team='other'),
            lambda a:a[0].update(mode='run'),lambda a:a[0].update(payload_ref='wrong'),
            lambda a:a.__setitem__(4,b'wrong'),lambda a:a[5].update(source_sha256='0'*64),
            lambda a:a[6].update(sha256='0'*64)]
        for mutate in mutations:
            args=fixture(self.manifest,self.source);mutate(args)
            with self.assertRaises(AssertionError):read(*args)

    def test_missing_official_records_invalid_times_and_report(self):
        for event in ('test','benchmark','profile'):
            args=fixture(self.manifest,self.source)
            records=args[0]['result']['records'];records.remove(next(r for r in records if r['event']==event))
            with self.assertRaises(AssertionError):read(*args)
        args=fixture(self.manifest,self.source)
        args[0]['result']['records'][1]['index']=0
        with self.assertRaisesRegex(AssertionError,'13 tests'):read(*args)
        for mutation in (lambda r:r.update(fresh_checks=144),lambda r:r['pins'].update(candidate='0'*64),
                         lambda r:r.update(transitions=16),lambda r:r.update(prepare_s=295.),
                         lambda r:r.update(cleanup_restored=False),lambda r:r['decision'].update(gate=False)):
            r=report(copy.deepcopy(self.manifest));mutation(r)
            with self.assertRaises(AssertionError):capsule(encode(r),self.manifest)
        with self.assertRaises(AssertionError):capsule(encode(report(self.manifest))*2,self.manifest)

    def test_owning_archived_job_and_recipe_are_reference_only(self):
        path=builder.ROOT/'docs/steering/evidence/2026-11-01-rmw-common-pair-v4/own-terminal-receipt.json'
        old=json.loads(path.read_text())
        self.assertEqual(old['id'],'c5e4f89672f743bb')
        self.assertTrue(old['result']['passed'])
        self.assertNotEqual(builder.pair.PIN,'4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280')
        with self.assertRaises(AssertionError):read(old,old['id'],*fixture(self.manifest,self.source)[2:])

    def test_actual_entry_restores_on_success_failure_and_budget(self):
        # Execute generated prepare itself. Private numerical loop/CUDA are
        # deliberately stand-ins; this proves finally/official-owner semantics only.
        for failure in ('none','normal','private','budget'):
            original=(object(),object(),object());state={};guard=[1]
            backend=types.SimpleNamespace(allow_tf32=True)
            torch=types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=backend)),
                inference_mode=lambda value:contextlib.nullcontext(),no_grad=contextlib.nullcontext)
            ns=dict(torch=torch,time=time,_EH_OFFLINE_HOLD=False,_ma_ext=original[0],_fork_native_moe=original[1],_ab_ref_back=original[2],
                _EH_BUILDS={},_AB_DEQ={},_GUARDS=guard,_GUARDED_IDS={3},_eh_build=object(),_eh_native=object(),_eh_back=object())
            def normal(w):
                if failure=='normal':raise RuntimeError('normal failure')
                return state
            def private(w,s):
                self.assertIs(s,state);guard.append(2);ns['_GUARDED_IDS'].add(4)
                if failure=='private':raise RuntimeError('private failure')
                return report(self.manifest)
            ns.update(_EH_PREPARE=normal,_eh_private=private)
            exec(builder.function(BODY,'_eh_budget'),ns)
            if failure=='budget':ns['time']=types.SimpleNamespace(monotonic=iter([0.,296.]).__next__)
            exec(builder.function(BODY,'prepare'),ns)
            with contextlib.redirect_stdout(io.StringIO()) as log:
                if failure=='none':self.assertIs(ns['prepare']({}),state)
                else:
                    with self.assertRaises((RuntimeError,AssertionError)):ns['prepare']({})
            self.assertEqual((ns['_ma_ext'],ns['_fork_native_moe'],ns['_ab_ref_back']),original)
            self.assertTrue(backend.allow_tf32);self.assertEqual(guard,[1]);self.assertEqual(ns['_GUARDED_IDS'],{3})
            self.assertIsNone(ns['_EH_CACHE'])
            self.assertEqual('EXPERT_TOP100_REPORT' in log.getvalue(),failure=='none')

    def test_delivered_prepare_is_stopped_before_any_build_or_gpu_call(self):
        ns={'_EH_OFFLINE_HOLD':True}
        exec(builder.function(BODY,'prepare'),ns)
        with self.assertRaisesRegex(RuntimeError,'STOP_CPU_HOST_GATE'):ns['prepare']({})

    def test_overlap_rejection_uses_storage_intervals(self):
        text=(builder.ROOT/'tools/dl_g1_layer_owners.py').read_text()
        ns={};exec(builder.function(text,'_wl_overlap'),ns)
        ns['_wl_intervals']=lambda arm,readonly:arm['intervals']
        exec(builder.function(text,'_wl_check_owners'),ns)
        arms=[dict(attention_scratch={},intervals=[('cpu',i*100,100)]) for i in range(4)]
        self.assertEqual(ns['_wl_check_owners'](arms,set()),[100]*4)
        arms[3]['intervals']=[('cpu',299,100)]
        with self.assertRaisesRegex(AssertionError,'overlap'):ns['_wl_check_owners'](arms,set())

    def test_actual_call_is_recorded_after_success_only(self):
        ns={'route_key':route_key,'_pdl':lambda t:False,'_EH_SELECTED':self.manifest['selected']}
        module=object();state=dict(_fork_native_eligible=True,ma_ok=True,_wl_probe=True,_wl_hits=[],_expert_pair_arm=True)
        ext=types.SimpleNamespace(_module=module,okc_run=lambda *a:None)
        state['ma_ext']={'b4':ext};ns['_EH_NATIVE']=lambda t,s:ext
        expected=self.manifest['selected']['1:b4'];ns['_EH_BUILDS']={id(module):(expected['flags'],expected['sha256'])}
        tree=ast.parse(BODY);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_ExpertCall')
        exec(ast.get_source_segment(BODY,node),ns);exec(builder.function(BODY,'_eh_native'),ns)
        observed=ns['_eh_native'](32,state)
        self.assertFalse(state['_wl_hits'])
        observed.okc_run(1,2,3);self.assertEqual(state['_wl_hits'],[(32,'b4',True)])
        state['_wl_hits']=[]
        def fail(*a):raise RuntimeError('native failure')
        ext.okc_run=fail
        with self.assertRaisesRegex(RuntimeError,'native failure'):observed.okc_run()
        self.assertFalse(state['_wl_hits'])
        ns['_EH_BUILDS'][id(module)]=(expected['flags'],'0'*64)
        with self.assertRaisesRegex(AssertionError,'SELECTED_SOURCE'):ns['_eh_native'](32,state)

    def test_reference_cache_stale_versions_and_front_change(self):
        # Minimal versioned-value facade exercises the actual cache predicates,
        # not Torch's allocator, inference tensors or numerical operations.
        class Tensor:
            shape=(2,);dtype='f32';device='cpu'
            def __init__(self,value):self.value=value;self._version=0
            def clone(self):return Tensor(self.value)
            def mutate(self,value):self.value=value;self._version+=1
        torch=types.SimpleNamespace(Tensor=Tensor,equal=lambda a,b:a.value==b.value,
            isfinite=lambda t:types.SimpleNamespace(all=lambda:True))
        ns=dict(torch=torch,_EH_CACHE={},_eh_budget=lambda:None)
        from g_common_release_budget import SUPPORT
        exec(builder.function(SUPPORT,'_gqb_signature'),ns)
        misses=[]
        def back(*args):misses.append(1);return (Tensor(3),Tensor(4))
        ns['_EH_BACK']=back;exec(builder.function(BODY,'_eh_back'),ns)
        front={'h':Tensor(1)};weights={};ids=Tensor(2)
        ns['_eh_back'](front,weights,ids);ns['_eh_back'](front,weights,ids)
        self.assertEqual(len(misses),1)
        front['h'].mutate(2);ns['_eh_back'](front,weights,ids)
        self.assertEqual(len(misses),2)
        with self.assertRaisesRegex(AssertionError,'OWNER'):ns['_eh_back'](front,{},ids)
        entry=ns['_EH_CACHE'][id(front)];entry['ids'].mutate(4)
        with self.assertRaisesRegex(AssertionError,'IDS_STALE'):ns['_eh_back'](front,weights,ids)
        ns['_EH_CACHE']={};ns['_eh_back'](front,weights,ids)
        ns['_EH_CACHE'][id(front)]['result'][0].mutate(7)
        with self.assertRaisesRegex(AssertionError,'RESULT_STALE'):ns['_eh_back'](front,weights,ids)

    def test_capture_restores_inputs_and_replay_writes_actual_output(self):
        class Tensor:
            def __init__(self,value):self.value=value
            def clone(self):return Tensor(self.value)
            def copy_(self,other):self.value=other.value
        class Stream:
            def wait_stream(self,other):pass
        class Graph:
            current=None
            def __init__(self):self.ops=[]
            def replay(self):
                for op in self.ops:op()
        @contextlib.contextmanager
        def capture(graph):
            Graph.current=graph
            try:yield
            finally:Graph.current=None
        cuda=types.SimpleNamespace(Stream=Stream,current_stream=Stream,stream=lambda s:contextlib.nullcontext(),
            synchronize=lambda:None,CUDAGraph=Graph,graph=capture)
        def run(data,state):
            out=Tensor(0)
            def op():
                out.value=data['stream'].value+data['attn_input'].value
                data['stream'].value*=2;data['attn_input'].value*=2
            if Graph.current is not None:Graph.current.ops.append(op)
            op();return out
        ns=dict(torch=types.SimpleNamespace(cuda=cuda),_wl_run=run)
        # Use the generated corrected function, not a reimplementation.
        nodes=[n for n in ast.parse(self.source).body if isinstance(n,ast.FunctionDef) and n.name=='_wl_capture']
        self.assertEqual(len(nodes),1);exec(ast.get_source_segment(self.source,nodes[0]),ns)
        data=dict(stream=Tensor(3),attn_input=Tensor(5))
        graph,out=ns['_wl_capture'](data,{})
        self.assertEqual(out.value,8)  # Without post-capture restoration this is 16.
        data['stream'].value=7;data['attn_input'].value=11;graph.replay()
        self.assertEqual(out.value,18)


if __name__=='__main__':
    unittest.main(verbosity=2)
