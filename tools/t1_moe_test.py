"""Real Torch CPU host/storage tests; all CUDA/native math and times are explicit facades."""
import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import t1_moe_harness as harness
import t1_moe_read as reader
from t1_moe_contract import BANKS, DISTRIBUTIONS, LOADS, AFFECTED, decision
from expert_top100_moe_harness_test import HostRuntime
from expert_top100_moe_test import prepare_small, inputs, local_weights


class Runtime(HostRuntime):
    def load(self, **kwargs):
        self.builds.append(kwargs)
        runtime = self
        candidate = harness.sha(kwargs['cuda_sources']) == harness.NATIVE_PIN

        class Extension:
            descriptor = kwargs

            def okc_set_pdl(self, value):
                if value != 0:
                    raise AssertionError('PDL changed')

            def okc_run(self, *arguments):
                if candidate:
                    raise AssertionError('T1 wrong host binding')
                self.record(False, arguments)

            def t1_three_run(self, *arguments):
                if not candidate or arguments[2].shape != (1, 4096):
                    raise AssertionError('T1 wrong size/factory')
                self.record(True, (*arguments, 0, 32, 132))

            def record(self, selected, arguments):
                if runtime.active is not None:
                    runtime.active.calls.append((selected, arguments))
                runtime.launch(selected, arguments)
        return Extension()

    def launch(self, candidate, arguments, replay=False):
        if candidate:
            if arguments[2].shape[0] != 1:
                raise AssertionError('treatment escaped T1')
            for tensor in arguments[1][1:]:
                tensor.add_(1)
            if replay and self.fault == 't1guard':
                self.states[2]['t1_guards'][0][0][-1] = 0
        super().launch(candidate, arguments, replay)


def environment(source, fault=None):
    namespace = {}
    exec(compile(source, '<actual-held-T1-moe>', 'exec'), namespace)
    runtime = Runtime(namespace, fault)
    namespace.update(torch=runtime, load_inline=runtime.load, _H_NATIVE_LOADER=runtime.load, H_HOLD=False)
    original = namespace['_h_ownership']

    def ownership(states):
        runtime.states = states
        return original(states)
    namespace['_h_ownership'] = ownership
    return namespace, runtime


class T1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest = harness.build()
        namespace, runtime = environment(cls.source)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            state, packed = prepare_small(namespace)
        cls.full = namespace, runtime, state, output.getvalue()

    def test_deterministic_held_source_native_flags_and_no_pregamma(self):
        self.assertEqual(harness.build(), (self.source, self.manifest))
        self.assertTrue(self.manifest['hold'])
        self.assertFalse(self.manifest['dispatch_ready'])
        self.assertEqual(self.manifest['affected_tokens'], [1])
        self.assertEqual(AFFECTED, (1,))
        namespace, runtime = environment(self.source)
        self.assertEqual(harness.sha(namespace['_T1_THREE_CUDA']), harness.NATIVE_PIN)
        self.assertEqual(namespace['_T1_THREE_FLAGS'], harness.t1.FLAGS)
        self.assertNotIn('expert_top100_pregamma', self.source)
        self.assertNotIn('_expert_pair_bind', self.source)
        self.assertEqual(set(namespace['_MA_ALT_SRCS']), {'moe_helper0', 'moe_helper0df'})
        namespace['H_HOLD'] = True
        with self.assertRaisesRegex(RuntimeError, 'HOLD'):
            namespace['prepare']({})
        self.assertEqual(runtime.builds, [])
        self.assertFalse(any('reference' in ast.unparse(node) for node in ast.walk(ast.parse(self.source))
                             if isinstance(node, (ast.Import, ast.ImportFrom))))

    def test_actual_final_prepare_all54_cases_candidate_entry_and_cleanup(self):
        namespace, runtime, state, log = self.full
        capsule, verdict = reader.capsule(log, self.manifest)
        self.assertIs(state, runtime.states[2])
        self.assertTrue(state['_t1_arm'])
        self.assertFalse(verdict['component_pass'])
        self.assertEqual(capsule['affected_tokens'], [1])
        self.assertEqual(capsule['counters'], dict(observations=378, restorations=54,
            captures=216, timed_replays=1152, worst=[0., 0.]))
        self.assertEqual(len(runtime.graphs), 216)
        self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
        self.assertEqual(len({tensor.data_ptr() for graph in runtime.graphs[:4]
            for tensor in graph.calls[0][1][2:5]}), 12)
        self.assertIs(namespace['load_inline'].__self__, runtime)
        for owner in runtime.states:
            self.assertNotIn('_expert_pair_arm', owner)
            self.assertEqual(len(owner['t1_scratch']), 3 if owner['_t1_arm'] else 0)
            self.assertTrue(all(torch.count_nonzero(tensor) == 0 for tensor in
                owner['ma_scratch'] + owner['t1_scratch']))
        for index, graph in enumerate(runtime.graphs):
            candidate, arguments = graph.calls[0]
            self.assertEqual(candidate, index % 4 >= 2 and arguments[2].shape[0] == 1)
        data = namespace['generate_input'](1, 'ep8', 1459, device='cpu')
        self.assertTrue(torch.isfinite(namespace['kernel_fn'](data, state)).all())

    def test_actual_nine_sizes_both_arms_factories_flags_and_independent_states(self):
        namespace, runtime, state, log = self.full
        expected = {1:'tiny', 4:'tiny', 8:'tiny2', 16:'mid', 32:'b4',
                    33:'main', 128:'main', 256:'main', 257:'main'}
        original_calls, unused = harness.capture(harness.t1.BASE.read_text())
        control_descriptors = {call['name']:call for call in original_calls}
        actual_descriptors = {call['name']:call for call in runtime.builds}
        self.assertEqual(len(actual_descriptors), 6)
        for name, call in control_descriptors.items():
            self.assertEqual(actual_descriptors[name], call)
        selected = []
        for index, owner in enumerate(runtime.states):
            self.assertEqual(set(owner['ma_ext']), {'tiny','tiny2','mid','b4','main'})
            for tokens, family in expected.items():
                module = namespace['_t1_selected'](tokens, owner)
                treatment = tokens == 1 and index >= 2
                self.assertIs(module, owner['t1_module'] if treatment else owner['ma_ext'][family])
                data = inputs(tokens)
                output = namespace['_expert_moe_run'](data, owner, index >= 2)
                self.assertFalse(output.data_ptr() == data['hidden_states'].data_ptr())
                descriptor = dict(module._module.descriptor)
                for key in ('cpp_sources', 'cuda_sources'):
                    descriptor[key] = harness.sha(descriptor[key])
                selected.append(dict(placement=index,tokens=tokens,
                    family='T1-three' if treatment else family, descriptor=descriptor))
        self.__class__.selection = selected
        native = actual_descriptors['t1_three_launch_v1']
        self.assertEqual(native['extra_cflags'], ['-O3','-std=c++20'])
        self.assertEqual(native['extra_cuda_cflags'], harness.t1.FLAGS)
        self.assertEqual(harness.sha(native['cuda_sources']), harness.NATIVE_PIN)
        candidates = runtime.states[2:]
        self.assertTrue(all(left.data_ptr() != right.data_ptr() for left,right in
            zip(candidates[0]['t1_scratch'],candidates[1]['t1_scratch'])))
        copied = copy.deepcopy(candidates[0])
        self.assertTrue(all(left.data_ptr() != right.data_ptr() for left,right in
            zip(candidates[0]['t1_scratch'],copied['t1_scratch'])))
        namespace['_expert_moe_run'](inputs(1),copied,True)

    def test_actual_packing_row_order_unit_bytes_scales_and_gamma(self):
        namespace, runtime = environment(self.source)
        local = local_weights()
        local['w2'] = local['w2'] + 64
        with patch.dict(prepare_small.__globals__, local_weights=lambda: local):
            state, rows = prepare_small(namespace, 'expert_moe_prepare_candidate')
        self.assertEqual(len(rows),64)
        self.assertEqual(rows[0][1][:16].tolist(),list(range(8))+list(range(4096,4104)))
        for index,name in ((2,'w13_gs'),(5,'w2_gs'),(7,'post_expert_norm')):
            self.assertTrue(torch.equal(state['ma_w'][index],local[name]))
        expected = torch.arange(32, dtype=torch.int32).view(torch.uint8)
        self.assertTrue(torch.equal(state['ma_w'][10], expected))
        self.assertTrue(torch.equal(state['ma_w'][11], (torch.arange(32,dtype=torch.int32)+64).view(torch.uint8)))
        self.assertFalse(torch.equal(state['ma_w'][10],state['ma_w'][11]))
        real = {}
        exec(harness.adapter_source()[0],real)
        codes=torch.arange(16*2048,dtype=torch.int64).to(torch.uint8).reshape(16,2048)
        scales=torch.arange(16*256,dtype=torch.int64).to(torch.uint8).reshape(16,256)
        packed=real['_pack_codes'](codes,torch.arange(16))
        scale_bytes=real['_pack_scales'](scales.view(torch.float8_e4m3fn),torch.arange(16))
        unit=real['_unit_order'](packed,scale_bytes)
        self.assertEqual(unit.numel(),36864)
        self.assertTrue(torch.equal(unit[:32768],packed.view(torch.uint8)))
        self.assertTrue(torch.equal(unit[32768:],scale_bytes))

    def test_actual_full_inference_entry_and_exact_scratch_metadata(self):
        with torch.inference_mode():
            namespace, runtime = environment(self.source)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                state, unused = prepare_small(namespace)
            self.assertTrue(torch.is_inference(state['ma_w'][0]))
            expected = [((8,4096),torch.bfloat16),((8,4096),torch.float32),((8,256),torch.float32)]
            self.assertEqual([(tuple(value.shape),value.dtype) for value in state['t1_scratch']],expected)
            self.assertTrue(all(torch.is_inference(value) for value in state['t1_scratch']))
            for tokens in (1,8,256):
                data = namespace['generate_input'](tokens,'ep8',1289,device='cpu')
                self.assertTrue(torch.isfinite(namespace['kernel_fn'](data,state)).all())
            reader.capsule(output.getvalue(),self.manifest)

    def test_original_complete_notices_retained(self):
        base = harness.t1.BASE.read_text(encoding='utf8')
        start = base.index('#                                  Apache License')
        end = base.index('# Added by yelgabo, November 1, 2026:',start)
        self.assertIn(base[start:end],self.source)
        self.assertIn('\n'.join(base.splitlines()[1:17]),self.source)

    def test_exact_original_function_reversal_outside_guard_binding(self):
        base = harness.t1.BASE.read_text(encoding='utf8')
        before = {node.name:ast.get_source_segment(base,node) for node in ast.parse(base).body
                  if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        after = {node.name:ast.get_source_segment(self.source,node) for node in ast.parse(self.source).body
                 if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for name in self.manifest['closure']:
            text = after[name]
            if name == '_fork_native_ma_prepare':
                text = harness.replace_once(text,'[_expert_moe_guarded(t, st) for t in _ma_make_scratch(dev)]',
                                            '[_guarded(t) for t in _ma_make_scratch(dev)]')
            self.assertEqual(text,before[name],name)

    def test_no_stale_crossed_arm_or_unsupported_shard(self):
        namespace, runtime, state, log = self.full
        for key in ('_expert_pair_arm','_expert_pair_arm_bound'):
            wrong=dict(state);wrong[key]=False
            with self.assertRaisesRegex(RuntimeError,'STALE'):
                namespace['_expert_moe_run'](inputs(1),wrong,True)
        wrong=dict(state);wrong['_t1_arm_bound']=False
        with self.assertRaisesRegex(RuntimeError,'IDENTITY'):
            namespace['_expert_moe_run'](inputs(1),wrong,True)
        data=inputs(1);data['config']['local_start']=32
        with self.assertRaisesRegex(RuntimeError,'CONFIG'):
            namespace['_expert_moe_run'](data,state,True)
        data=inputs(1);data['config']['sharding']='full'
        with self.assertRaisesRegex(RuntimeError,'SHARD'):
            namespace['_expert_moe_run'](data,state,True)
        with self.assertRaisesRegex(RuntimeError,'IDENTITY'):
            namespace['_expert_moe_run'](inputs(1),runtime.states[0],True)

    def test_frozen_native_precision_packing_flags_and_source_negatives(self):
        namespace, runtime = environment(self.source)
        native = namespace['_T1_THREE_CUDA']
        for old, new in (('weights[10].data_ptr()', 'weights[11].data_ptr()'),
                         ('weights[7].data_ptr()', 'weights[5].data_ptr()'),
                         ('__float2bfloat16_rn(combined)', '__float2half_rn(combined)'),
                         ('SILU_DIV(gate, 1.f + __expf(-gate)) * up', 'gate * up')):
            self.assertIn(old, native)
            with patch.object(harness.t1, 'native_source', return_value=native.replace(old,new)):
                with self.assertRaisesRegex(RuntimeError,'frozen native'):
                    harness.build()
        with patch.object(harness.t1, 'FLAGS', harness.t1.FLAGS+['--use_fast_math']):
            with self.assertRaisesRegex(RuntimeError,'ordered flags'):
                harness.build()
        with patch.object(harness.extraction.pair, 'build', side_effect=AssertionError('pregamma build forbidden')):
            self.assertEqual(harness.build(), (self.source,self.manifest))

    def test_factory_and_guard_negative_real_entry(self):
        namespace,runtime=environment(self.source)
        namespace['H_BUILD_CALLS']=[]
        with self.assertRaisesRegex(RuntimeError,'compiler'):
            prepare_small(namespace)
        self.assertEqual(runtime.builds,[])
        for fault in ('t1guard','corrupt','wrong','stale'):
            namespace,runtime=environment(self.source,fault)
            with self.assertRaisesRegex(RuntimeError,'OVERRUN|paired|nonfinite|empty'):
                prepare_small(namespace)
            self.assertTrue(all(graph.reset_done for graph in runtime.graphs))
            self.assertIs(namespace['load_inline'].__self__,runtime)
            self.assertTrue(all(torch.count_nonzero(tensor)==0 for owner in runtime.states
                for tensor in owner['ma_scratch']+owner['t1_scratch']))

    def test_extra_scratch_ownership_and_capture_restoration_fail_closed(self):
        for fault in ('alias','inventory','budget','routes'):
            namespace,runtime=environment(self.source)
            ownership=namespace['_h_ownership']
            def inject(states):
                if fault=='alias':
                    states[3]['t1_scratch']=states[2]['t1_scratch']
                    states[3]['t1_guards']=states[2]['t1_guards']
                elif fault=='inventory':
                    states[2]['t1_guards']=[]
                result=ownership(states)
                if fault=='budget':
                    namespace['time']=type('Clock',(),{'monotonic':staticmethod(lambda:float('inf'))})
                return result
            namespace['_h_ownership']=inject
            if fault=='routes':
                generate=namespace['generate_input']
                def bad(*args,**kwargs):
                    data=generate(*args,**kwargs);data['topk_ids'][:,1]=data['topk_ids'][:,0];return data
                namespace['generate_input']=bad
            with self.assertRaisesRegex(RuntimeError,'storage|inventory|PREPARE_295S|distinct'):
                prepare_small(namespace)
            self.assertIs(namespace['load_inline'].__self__,runtime)
            self.assertTrue(all(graph.reset_done for graph in runtime.graphs))

    def test_t1_only_regression_set_whole_six_crosses_null_and_drift(self):
        rows=[[seed,tokens,dist,[100.,100.,90.,90.],[tokens,1,tokens],
               [100/90]*4,[0.,0.]] for seed in BANKS for tokens in LOADS for dist in DISTRIBUTIONS]
        self.assertTrue(decision(rows)['component_pass'])
        bad=copy.deepcopy(rows);next(row for row in bad if row[1]==1)[5][0]=1/1.01
        self.assertFalse(decision(bad)['component_pass'])
        changed=copy.deepcopy(rows);next(row for row in changed if row[1]==8)[5][0]=1/1.01
        result=decision(changed)
        self.assertEqual(result['affected_regressions'],[])
        self.assertTrue(any(row[1]==8 for row in result['unchanged_drift']))
        for fault in ('whole','cross','null','missing','nan'):
            bad=copy.deepcopy(rows)
            if fault=='whole':
                for row in bad:row[5]=[1.009]*4
            elif fault=='cross':
                for row in bad:row[5][3]=1.
            elif fault=='null':
                for row in bad:row[6][0]=.1
            elif fault=='missing':bad.pop()
            else:bad[0][3][0]=float('nan')
            if fault in ('missing','nan'):
                with self.assertRaises(RuntimeError):decision(bad)
            else:self.assertFalse(decision(bad)['component_pass'])

    def test_strict_reader_synthetic_schema_source_mode_and_receipt_negatives(self):
        receipt=json.loads((harness.ROOT/'docs/steering/evidence/2026-11-02-worker-moe-component/status-wait.raw.json').read_bytes())
        log=self.full[3];payload=self.source.encode();job=receipt['id']
        upload=dict(job=job,team='yelgabo',mode='check',board=receipt['board'],
            content_hash=receipt['content_hash'],payload_ref=receipt['payload_ref'],
            source_sha256=harness.sha(payload),source_bytes=len(payload))
        attest=dict(job=job,owning_log=receipt['output']['log'],sha256=harness.sha(log),source_sha256=harness.sha(payload))
        self.assertFalse(reader.read(receipt,job,log,self.manifest,payload,upload,attest)['target_qualified'])
        for field,value in (('mode','run'),('board','muse_decode_layer'),('team','other'),('status','running')):
            wrong=copy.deepcopy(receipt);wrong[field]=value
            with self.assertRaises(RuntimeError):reader.read(wrong,job,log,self.manifest,payload,upload,attest)
        for bad in (log+log,log.replace('candidate2','control'),log.replace('378','377'),
                    log.replace('"affected_tokens":[1]','"affected_tokens":[1,8]')):
            with self.assertRaises(RuntimeError):reader.capsule(bad,self.manifest)
        for mutate in ('profile','time','index','success'):
            wrong=copy.deepcopy(receipt)
            if mutate=='profile':next(row for row in wrong['result']['records'] if row['event']=='profile')['case']['seed']+=1
            elif mutate=='time':next(row for row in wrong['result']['records'] if row['event']=='benchmark')['median_ns']=0
            elif mutate=='index':next(row for row in wrong['result']['records'] if row['event']=='test')['index']=1
            else:wrong['result']['passed']=False
            with self.assertRaises(RuntimeError):reader.read(wrong,job,log,self.manifest,payload,upload,attest)
        with self.assertRaises(RuntimeError):reader.read(receipt,job,log,dict(self.manifest,hold=0),payload,upload,attest)
        target,target_manifest=harness.build(target_proposal=True)
        with self.assertRaises(RuntimeError):reader.read(receipt,job,log,target_manifest,payload,upload,attest)
        with self.assertRaises(RuntimeError):reader.read(receipt,job,log,self.manifest,payload+b'\n',upload,attest)
        with self.assertRaises(RuntimeError):reader.read(receipt,job,log,self.manifest,payload,upload,dict(attest,sha256='0'*64))
        for bad in ('{"a":1,"a":2}','{"x":NaN}'):
            with self.assertRaises(RuntimeError):reader.strict_json(bad)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path)
    args,remaining=parser.parse_known_args()
    program=unittest.main(argv=['T1-host-tests',*remaining],verbosity=2,exit=False)
    if args.evidence:
        args.evidence.mkdir(parents=True,exist_ok=True)
        if hasattr(T1Tests,'full'):
            args.evidence.joinpath('host-facade-capsule.log').write_bytes(T1Tests.full[3].encode())
        args.evidence.joinpath('host-receipt.json').write_text(json.dumps(dict(
            tests=program.result.testsRun,passed=program.result.wasSuccessful(),torch=torch.__version__,
            native_math='NOT RUN: facaded; no CUDA numerical oracle',timing='synthetic all-one event times',
            payload_sha256=T1Tests.manifest['source_sha256'],selection=getattr(T1Tests,'selection',[])),indent=2)+'\n')
    raise SystemExit(not program.result.wasSuccessful())
