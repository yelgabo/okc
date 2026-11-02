"""Real-Torch generated-entry host tests; not CUDA or performance evidence."""
import base64
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import types
import unittest
import zlib

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import expert_top100_harness as builder
from expert_top100_harness_contract import validate_rows
from expert_top100_harness_read import capsule


class DeclarationStub(types.ModuleType):
    def __getattr__(self, name):
        if name == 'jit':
            def decorate(function):
                def forbidden(*args, **kwargs):
                    raise RuntimeError('forbidden device execution: ' + function.__name__)
                return forbidden
            return decorate
        return DeclarationStub(name)

    def __call__(self, *args, **kwargs):
        return self


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in tensor_leaves(child)]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in tensor_leaves(child)]
    return []


class CaptureOps(TorchDispatchMode):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = function(*args, **kwargs)
        self.graph.operations.append((function, args, kwargs, result))
        return result


class CpuGraph:
    """Replay recorded ATen operations, updating captured buffers, not Python calls.

    Extra CPU allocations/copies are deliberate; this is not CUDA graph lifetime,
    stream ordering, address stability or performance emulation.
    """
    def __init__(self, host):
        self.host = host
        self.operations = []
        host.graphs.append(self)

    def replay(self):
        self.host.replays += 1
        if self.host.failure == 'replay' and self.host.replays == 10:
            raise RuntimeError('injected replay failure')
        with torch.no_grad():
            for function, args, kwargs, captured in self.operations:
                result = function(*args, **kwargs)
                before, after = tensor_leaves(captured), tensor_leaves(result)
                assert len(before) == len(after)
                for destination, source in zip(before, after):
                    if destination is not source:
                        destination.copy_(source)
        if self.host.failure == 'replay_output' and self.host.layer_outputs:
            self.host.layer_outputs[-1][0].fill_(float('nan'))
        if self.host.failure == 'late_replay_output' and self.host.replays == 8592:
            self.host.layer_outputs[-1][0].fill_(float('nan'))


class CpuStream:
    def wait_stream(self, other):
        pass


class CpuEvent:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.1


class TorchFacade:
    def __init__(self, host):
        self.host = host
        self.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=True)))
        self.cuda = types.SimpleNamespace(
            CUDAGraph=lambda: CpuGraph(host), graph=lambda graph: CaptureOps(graph),
            Stream=CpuStream, current_stream=CpuStream,
            stream=lambda stream: contextlib.nullcontext(), synchronize=lambda: None,
            Event=CpuEvent, get_device_capability=lambda: (9, 0),
            mem_get_info=lambda: (64 << 30, 80 << 30))

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        if kwargs.get('device') == 'cuda':
            assert args == (256 << 20,) and kwargs['dtype'] == torch.uint8
            self.host.flush_requests += 1
            return torch.empty(256, dtype=torch.uint8)
        return torch.empty(*args, **kwargs)


class _GraftLt:
    def __init__(self):
        self.lib = self.handle = self.desc = self.alpha = self.beta = None
        self.workspace = torch.zeros(48, dtype=torch.uint8)


class Host:
    """Executes the entire freshly generated module, with five named boundaries.

    _EH_PREPARE returns a small representative state, not the production state.
    _WL_KERNEL, _ab_inputs, _ab_ref_front and _EH_BACK use deterministic math and
    reduced dimensions. All generated private orchestration remains untouched.
    """
    def __init__(self, source, manifest, failure=None):
        self.failure = failure
        self.source, self.manifest = source, manifest
        self.graphs, self.layer_outputs, self.prepared = [], [], []
        self.reference_modes = []
        self.replays = self.native_calls = self.front_calls = self.back_calls = 0
        self.flush_requests = 0
        self.loads, self.inputs, self.owners = [], [], {}
        self.torch = TorchFacade(self)
        self.namespace = {'__name__': 'expert_generated_cpu_validation'}
        self.stub = DeclarationStub('triton')
        native_import = builtins.__import__

        def import_facade(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'torch':
                return self.torch
            if name == 'torch.nn.functional':
                return native_import(name, globals, locals, fromlist, level)
            if name == 'torch.utils.cpp_extension':
                return types.SimpleNamespace(load_inline=self.load_inline)
            if name.startswith('triton'):
                return self.stub
            return native_import(name, globals, locals, fromlist, level)

        self.namespace['__builtins__'] = dict(vars(builtins), __import__=import_facade)
        exec(compile(source, '<newly-generated-held-expert-harness>', 'exec'), self.namespace)
        self.original = {name: self.namespace[name] for name in (
            'prepare', '_eh_private', '_wl_clone', '_wl_capture', '_eh_routes',
            '_eh_transitions', '_eh_component', '_eh_back', '_qual_oracle',
            '_eh_native', '_EH_NATIVE', '_EH_BUILD', '_ab_ref_back', '_expert_pair_bind')}
        self.namespace.update(_EH_PREPARE=self.prepare_device_state,
                              _WL_KERNEL=self.layer, _ab_inputs=self.make_inputs,
                              _ab_ref_front=self.front, _EH_BACK=self.back)
        self.initial_guard = torch.tensor([7], dtype=torch.int32)
        self.namespace['_GUARDS'].append((self.initial_guard, 1))
        self.namespace['_GUARDED_IDS'].add(id(self.initial_guard))

    def load_inline(self, **kwargs):
        self.loads.append({key: kwargs[key] for key in ('name', 'extra_cuda_cflags')})
        module = types.ModuleType(kwargs['name'])

        def native(weights, scratch, value, ids, scores, output, start, count, sms):
            self.native_calls += 1
            if self.failure == 'native' and self.native_calls == 5:
                raise RuntimeError('injected native failure')
            occupied = ((ids >= 0).float() * scores).sum(-1, keepdim=True)
            output.copy_((value.float() * occupied).to(torch.bfloat16))
            if self.failure == 'empty_leak':
                output.add_(1)
            scratch[0].add_(1)
            if self.failure == 'guard':
                self.namespace['_GUARDS'][-1][0][-1] = 0

        module.okc_run = native
        module.okc_set_pdl = lambda value: None
        return module

    def prepare_device_state(self, weights):
        ns = self.namespace
        raw = torch.full((40,), ns['SCRATCH_SENTINEL'], dtype=torch.int32)
        raw[:32].zero_()
        ns['_GUARDS'].append((raw, 32))
        ns['_GUARDED_IDS'].add(id(raw))
        state = dict(qkv=weights['qkv'], ma_w=[weights['qkv']], ma_scratch=[raw[:32]],
                     scratch_alias=raw[4:20:2], ma_ext={}, ma_sfx=('-DOKC_PRMTSC=1',),
                     ma_ok=True, ma_sc6=True, _fork_native_eligible=True, nsm=132,
                     attention_scratch={'rows': {}, 'cluster': {}, 'retained': []},
                     compose_active={('chain', 32)}, bfp={'disabled': {8}},
                     graft_lt={'plan': {'lt': _GraftLt(), 'weight': weights['qkv'],
                                        'layouts': ('immutable',), 'algo': 4}})
        self.prepared.append(state)
        if self.failure == 'prepare':
            raise RuntimeError('injected prepare failure after guard allocation')
        for tokens in ns['LOADS']:
            ns['_fork_native_moe'](tokens, state)
        if self.failure == 'unknown_owner':
            state['opaque'] = object()
        return state

    def make_inputs(self, batch, length, context, seed):
        tokens = batch * length
        generator = torch.Generator().manual_seed(seed)
        data = dict(stream=torch.randn(tokens, 4, generator=generator),
                    attn_input=torch.randn(tokens, 4, generator=generator),
                    local_start=torch.zeros(1, dtype=torch.int32),
                    config={'tokens': tokens, 'q_len': length, 'page_size': length,
                            'context': context},
                    seq_lens=torch.zeros(batch, dtype=torch.int32),
                    page_table=torch.arange(batch, dtype=torch.int32)[:, None],
                    k_cache=torch.zeros(tokens, 2, 64), v_cache=torch.zeros(tokens, 2, 64),
                    k_scale=torch.ones(()), v_scale=torch.ones(()))
        self.inputs.append({'tokens': tokens, 'context': context, 'seed': seed})
        return data

    def front_values(self, data):
        tokens = data['config']['tokens']
        source = data['stream'] + data['attn_input']
        positions = torch.arange(tokens)[:, None]
        experts = torch.arange(256)[None, :]
        rotation = (source[:, :1].abs() * 29).long() % 256
        biased = -((experts - rotation - positions * 3) % 256).float()
        value = source[:, :1, None].expand(tokens, 2, 64).clone()
        return source, biased, value

    def front(self, data, weights, *args):
        self.front_calls += 1
        source, biased, value = self.front_values(data)
        self.reference_modes.append(torch.is_inference(source))
        return {'source': source, 'biased': biased,
                'local_start': int(data['local_start']),
                'k_new': value.reshape(-1, data['config']['q_len'], 2, 64),
                'v_new': (value * 2).reshape(-1, data['config']['q_len'], 2, 64)}

    def back(self, front, weights, ids):
        self.back_calls += 1
        local = (ids - front['local_start']) % 256
        factor = (local < 32).float().mean(-1, keepdim=True)
        hidden = (front['source'].to(torch.bfloat16).float() * factor).to(torch.bfloat16)
        return hidden, front['source'] + hidden.float()

    def layer(self, data, state):
        ns = self.namespace
        tokens = data['config']['tokens']
        source, biased, value = self.front_values(data)
        ids = biased.topk(8, dim=-1).indices.to(torch.int32)
        local = (ids - data['local_start']) % 256
        local = torch.where(local < 32, local, -1)
        if self.failure == 'local_ids':
            local.fill_(31)
        scores = torch.full((tokens, 8), 0.125)
        hidden = torch.empty_like(source, dtype=torch.bfloat16)
        ext = ns['_fork_native_moe'](tokens, state)
        ext.okc_run(state['ma_w'], state['ma_scratch'], source.to(torch.bfloat16),
                    local, scores, hidden, 0, 32, state['nsm'])
        if self.failure == 'local_ids_after_native':
            local.fill_(31)
        stream = source + hidden.float()
        data['stream'].copy_(stream)
        data['attn_input'].add_(3)
        data['k_cache'].copy_(value)
        data['v_cache'].copy_(value * 2)
        if self.failure == 'output_alias':
            if not hasattr(self, 'shared_hidden'):
                self.shared_hidden = torch.empty_like(hidden)
            self.shared_hidden.copy_(hidden)
            hidden = self.shared_hidden
        output = hidden, stream, ids
        self.layer_outputs.append(output)
        self.owners[id(state)] = state
        return output

    def weights(self):
        return {'qkv': torch.arange(16, dtype=torch.float32).reshape(4, 4),
                'attn_residual_gate': torch.zeros(4), 'ffn_residual_gate': torch.zeros(4)}

    def run(self, weights=None):
        ns = self.namespace
        ns['_EH_OFFLINE_HOLD'] = False
        self.log = io.StringIO()
        with contextlib.redirect_stdout(self.log):
            self.state = ns['prepare'](self.weights() if weights is None else weights)
        encoded = self.log.getvalue().split('EXPERT_TOP100_REPORT ')[1].strip()
        self.report = json.loads(zlib.decompress(base64.b64decode(encoded)))
        return self.report


class GeneratedTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source, cls.manifest = builder.build()
        cls.evidence = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'source_sha256': cls.manifest['source_sha256'],
                        'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        'pins': cls.manifest['pins'], 'runs': []}

    def host(self, failure=None):
        return Host(self.source, self.manifest, failure)

    def assert_cleanup(self, host, success=False):
        ns = host.namespace
        self.assertIsNone(ns['_EH_CACHE'])
        self.assertFalse(ns['_AB_DEQ'])
        self.assertFalse(ns['_EH_BUILDS'])
        self.assertIs(ns['_ma_ext'], host.original['_EH_BUILD'])
        self.assertIs(ns['_fork_native_moe'], host.original['_EH_NATIVE'])
        self.assertIs(ns['_ab_ref_back'], host.original['_ab_ref_back'])
        self.assertTrue(host.torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(len(ns['_GUARDS']), 2 if success else 1)
        self.assertIs(ns['_GUARDS'][0][0], host.initial_guard)
        self.assertEqual(ns['_GUARDED_IDS'], {id(full) for full, words in ns['_GUARDS']})

    def test_generated_entry_all_rows_and_inference_contexts(self):
        for inference, inference_weights in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(inference=inference, inference_weights=inference_weights):
                host = self.host()
                with torch.inference_mode(inference_weights):
                    weights = host.weights()
                self.assertEqual(torch.is_inference(weights['qkv']), inference_weights)
                started = time.monotonic()
                with torch.inference_mode(inference):
                    report = host.run(weights)
                self.assertEqual(len(report['rows']), 36)
                self.assertEqual((report['math_checks'], report['fresh_checks'], report['restores'],
                                  report['transitions']), (144, 1152, 144, 96))
                self.assertEqual(validate_rows(report['rows']), report['decision'])
                self.assertFalse(report['decision']['gate'])
                with self.assertRaisesRegex(AssertionError, 'readonly storage'):
                    capsule(host.log.getvalue(), self.manifest)
                self.assertEqual(report['pins'], self.manifest['pins'])
                self.assertEqual(host.flush_requests, 1)
                self.assertEqual(len(host.inputs), 72)
                self.assertEqual(len(host.owners), 4)
                self.assertEqual(host.replays, 96 + 36 * (4 + 32 + 4 + 4 + 192))
                self.assertEqual(host.native_calls, 624)
                self.assertFalse(any(host.reference_modes))
                self.assertEqual(torch.is_inference(host.prepared[0]['ma_scratch'][0]), inference)
                self.assertTrue(report['cleanup_restored'])
                self.assertIs(host.namespace['_eh_private'], host.original['_eh_private'])
                self.assertIs(host.namespace['_qual_oracle'], host.original['_qual_oracle'])
                for state in host.owners.values():
                    self.assertFalse(torch.is_inference(state['ma_scratch'][0]))
                    self.assertGreaterEqual(state['ma_scratch'][0]._version, 1)
                    self.assertIs(state['qkv'], weights['qkv'])
                    self.assertEqual(state['scratch_alias'].stride(), (2,))
                    self.assertEqual(state['scratch_alias'].storage_offset(), 4)
                    self.assertEqual(state['scratch_alias'].untyped_storage().data_ptr(),
                                     state['ma_scratch'][0].untyped_storage().data_ptr())
                self.assert_cleanup(host, success=True)
                self.evidence['runs'].append({'inference': inference,
                    'inference_created_weights': inference_weights, 'rows': len(report['rows']),
                    'checks': report['math_checks'], 'fresh_checks': report['fresh_checks'],
                    'restores': report['restores'], 'transitions': report['transitions'],
                    'replays': host.replays, 'native_python_calls': host.native_calls,
                    'back_calls': host.back_calls, 'elapsed_host_seconds': time.monotonic() - started,
                    'decision_gate': report['decision']['gate'], 'owners_bytes': report['owners_bytes'],
                    'synthetic_event_us': 100, 'cpu_facade_report': report})

    def test_archived_payload_reproduces_guard_leak(self):
        directory = builder.ROOT / 'docs/steering/evidence/2026-11-02-top100-expert-harness'
        source = (directory / 'harness-draft.py').read_text()
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(builder.sha(source), 'c6f4c25d0b059e0893225515868c9280b3bcb6434692e95b8ed931b9f5c01402')
        host = Host(source, manifest, 'prepare')
        with self.assertRaisesRegex(RuntimeError, 'injected prepare'):
            host.run()
        self.assertEqual(len(host.namespace['_GUARDS']), 2)
        self.assertEqual(len(host.namespace['_GUARDED_IDS']), 2)

    def test_original_hold_prevents_all_device_facades(self):
        host = self.host()
        with self.assertRaisesRegex(RuntimeError, 'STOP_CPU_HOST_GATE'):
            host.namespace['prepare'](host.weights())
        self.assertFalse(host.loads)
        self.assertFalse(host.prepared)
        self.assertFalse(host.graphs)

    def test_midrun_negative_cases_cleanup(self):
        for failure, message in (('native', 'injected native'), ('replay', 'injected replay'),
                                 ('guard', 'GUARD|past a'), ('local_ids', 'EXPERT_ORACLE|LOCAL_IDS'),
                                 ('local_ids_after_native', 'ACTUAL_LOCAL_IDS'),
                                 ('replay_output', 'EXPERT_ORACLE|NONFINITE'),
                                 ('late_replay_output', 'EXPERT_ORACLE|NONFINITE'),
                                 ('output_alias', 'EXPERT_OUTPUT_ALIAS'),
                                 ('empty_leak', 'EXPERT_FIRST_EMPTY'),
                                 ('unknown_owner', 'unreviewed state owner')):
            with self.subTest(failure=failure):
                host = self.host(failure)
                with self.assertRaisesRegex((AssertionError, RuntimeError), message):
                    host.run()
                self.assertNotIn('EXPERT_TOP100_REPORT ', host.log.getvalue())
                self.assert_cleanup(host)

    def test_prepare_failure_cleans_new_guards(self):
        host = self.host('prepare')
        with self.assertRaisesRegex(RuntimeError, 'injected prepare'):
            host.run()
        self.assert_cleanup(host)

    def test_budget_failure_after_preparation_cleans_guards(self):
        host = self.host()
        ticks = iter((0., 296.))
        host.namespace['time'] = types.SimpleNamespace(monotonic=lambda: next(ticks))
        with self.assertRaisesRegex(AssertionError, 'EXPERT_PREPARE_295S'):
            host.run()
        self.assert_cleanup(host)

    def test_storage_clone_aliases_private_policies_and_owner_negative(self):
        host = self.host()
        ns = host.namespace
        state = host.prepare_device_state(host.weights())
        readonly = ns['_wl_readonly'](state, host.weights())
        arms = [ns['_wl_clone'](state, readonly, index >= 2) for index in range(4)]
        ns['_wl_check_owners']([state, *arms], readonly)
        arms[0]['scratch_alias'][0] = 123
        self.assertEqual(int(arms[0]['ma_scratch'][0][4]), 123)
        self.assertEqual(int(arms[1]['ma_scratch'][0][4]), 0)
        arms[0]['compose_active'].add(('new', 1))
        self.assertNotIn(('new', 1), arms[1]['compose_active'])
        arms[0]['bfp']['disabled'].add(32)
        self.assertNotIn(32, arms[1]['bfp']['disabled'])
        self.assertIsNot(arms[0]['ma_ext'], arms[1]['ma_ext'])
        arms[0]['graft_lt']['plan']['lt'].workspace.fill_(8)
        self.assertEqual(int(arms[1]['graft_lt']['plan']['lt'].workspace.sum()), 0)
        arms[1]['ma_scratch'] = arms[0]['ma_scratch']
        with self.assertRaisesRegex(AssertionError, 'mutable storage overlap'):
            ns['_wl_check_owners']([state, *arms], readonly)

    def test_selected_source_negative_runs_generated_entry(self):
        host = self.host()
        host.namespace['_EH_SELECTED']['0:tiny']['sha256'] = '0' * 64
        with self.assertRaisesRegex(AssertionError, 'EXPERT_SELECTED_SOURCE'):
            host.run()
        self.assert_cleanup(host)

    def test_missing_pre_replay_restore_is_detected_by_full_oracle(self):
        restore = "    data['stream'].copy_(saved[0]);data['attn_input'].copy_(saved[1])\n    graph.replay();torch.cuda.synchronize()"
        self.assertEqual(self.source.count(restore), 1)
        mutant = self.source.replace(restore, '    graph.replay();torch.cuda.synchronize()')
        host = Host(mutant, self.manifest)
        with self.assertRaisesRegex(AssertionError, 'EXPERT_ORACLE'):
            host.run()
        self.assert_cleanup(host)

    def test_actual_facade_replay_not_python_callback(self):
        host = self.host()
        source = torch.tensor([1., 2.])
        graph = CpuGraph(host)
        with CaptureOps(graph):
            output = source * 3
            source.add_(1)
        self.assertEqual(source.tolist(), [2., 3.])
        source.copy_(torch.tensor([5., 8.]))
        graph.replay()
        self.assertEqual(output.tolist(), [15., 24.])
        self.assertEqual(source.tolist(), [6., 9.])
        self.assertEqual(host.native_calls, 0)

    def test_final_budget_and_report_failures_rollback_official_guards(self):
        for failure in ('midrun_budget', 'final_budget', 'report_overflow'):
            with self.subTest(failure=failure):
                host = self.host()
                ns = host.namespace
                original_clock = time.monotonic
                if failure == 'midrun_budget':
                    ns['time'] = types.SimpleNamespace(monotonic=lambda:
                        original_clock() + (300 if host.replays >= 10 else 0))
                    message = 'EXPERT_PREPARE_295S'
                elif failure == 'final_budget':
                    ns['time'] = types.SimpleNamespace(monotonic=lambda:
                        original_clock() + (300 if host.replays == 8592 and not ns['_EH_BUILDS'] else 0))
                    message = 'EXPERT_PREPARE_295S'
                else:
                    ns['_EH_PINS'] = {'large': ''.join(format(index * 982451653, 'x') for index in range(10000))}
                    message = 'EXPERT_REPORT_LOG_OVERFLOW'
                with self.assertRaisesRegex(AssertionError, message):
                    host.run()
                self.assertNotIn('EXPERT_TOP100_REPORT ', host.log.getvalue())
                self.assert_cleanup(host)


    def test_reference_cache_real_versions_and_inference_front(self):
        host = self.host()
        ns = host.namespace
        ns['_EH_START'] = time.monotonic()
        ns['_EH_CACHE'] = {}
        data = host.make_inputs(1, 8, 'full', 1102031)
        weights = host.weights()
        front = host.front(data, weights)
        ids = front['biased'].topk(8).indices
        back = ns['_eh_back']
        first = back(front, weights, ids)
        self.assertIs(back(front, weights, ids.clone()), first)
        front['source'].view(-1)[0].add_(1)
        second = back(front, weights, ids)
        self.assertIsNot(second, first)
        changed_ids = (ids + 1) % 256
        self.assertIsNot(back(front, weights, changed_ids), second)
        with self.assertRaisesRegex(AssertionError, 'REFERENCE_OWNER'):
            back(front, dict(weights), ids)
        entry = ns['_EH_CACHE'][id(front)]
        entry['ids'].add_(1)
        with self.assertRaisesRegex(AssertionError, 'IDS_STALE'):
            back(front, weights, ids)
        ns['_EH_CACHE'].clear()
        back(front, weights, ids)[0].add_(1)
        with self.assertRaisesRegex(AssertionError, 'RESULT_STALE'):
            back(front, weights, ids)
        with torch.inference_mode():
            inference_front = host.front(data, weights)
        self.assertTrue(torch.is_inference(inference_front['source']))
        with self.assertRaisesRegex(RuntimeError, 'Inference tensors do not track version counter'):
            back(inference_front, weights, ids)
        ns['_EH_CACHE'].clear()
        with torch.inference_mode():
            inference_ids = ids.clone()
        result = back(front, weights, inference_ids)
        self.assertFalse(torch.is_inference(ns['_EH_CACHE'][id(front)]['ids']))
        self.assertTrue(all(not torch.is_inference(value) for value in result))
        ns['_EH_CACHE'].clear()
        with torch.inference_mode():
            inference_results = tuple(value.clone() for value in result)
        ns['_EH_BACK'] = lambda *args: inference_results
        with self.assertRaisesRegex(RuntimeError, 'Inference tensors do not track version counter'):
            back(front, weights, ids)

    def test_natural_routing_and_local_id_negatives(self):
        host = self.host()
        ns = host.namespace
        data = host.make_inputs(16, 8, 'full', 1102031)
        other = host.make_inputs(16, 8, 'full', 1102037)
        front = host.front(data, host.weights())
        ids = front['biased'].topk(8).indices
        self.assertFalse(torch.equal(ids, host.front(other, host.weights())['biased'].topk(8).indices))
        value = torch.ones(128, 4, dtype=torch.bfloat16)
        local = (ids - 96) % 256
        local = torch.where(local < 32, local, -1).to(torch.int32)
        state = {'_eh_args': (None, None, value, local, torch.ones(128, 8), value)}
        output = value, value.float(), ids
        ns['_eh_routes'](state, output, 96)
        local[0, 0] = 31 if int(local[0, 0]) != 31 else 30
        with self.assertRaisesRegex(AssertionError, 'ACTUAL_LOCAL_IDS'):
            ns['_eh_routes'](state, output, 96)
        ids[:, 1] = ids[:, 0]
        fixed = (ids - 96) % 256
        local.copy_(torch.where(fixed < 32, fixed, -1))
        with self.assertRaisesRegex(AssertionError, 'DUPLICATE_IDS'):
            ns['_eh_routes'](state, output, 96)


if __name__ == '__main__':
    output = None
    if '--evidence' in sys.argv:
        position = sys.argv.index('--evidence')
        output = Path(sys.argv[position + 1])
        del sys.argv[position:position + 2]
    result = unittest.main(exit=False, verbosity=2).result
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        evidence = dict(GeneratedTorch.evidence, tests_run=result.testsRun,
                        failures=len(result.failures), errors=len(result.errors),
                        successful=result.wasSuccessful(), dispatch_ready=False,
                        scope='CPU real storage / deterministic device and mathematical facades')
        output.write_text(json.dumps(evidence, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())


