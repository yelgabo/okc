"""CPU execution of actual adapter functions; native CUDA calls are recorded, not run."""
import ast
import contextlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

import expert_top100_moe_adapter as adapter
from expert_top100_pregamma import capture


class HostTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class TorchFacade:
    def __init__(self):
        self.cuda = types.SimpleNamespace(
            get_device_capability=lambda: (9, 0),
            get_device_properties=lambda device: types.SimpleNamespace(
                major=9, minor=0, multi_processor_count=132),
            device=lambda device: contextlib.nullcontext(),
            is_current_stream_capturing=lambda: False,
            current_device=lambda: None,
            synchronize=lambda *args: None)

    def __getattr__(self, name):
        return getattr(torch, name)


def environment(source):
    namespace = {}
    exec(compile(source, '<actual-generated-payload>', 'exec'), namespace)
    builds, launches = [], []

    class Extension:
        def okc_set_pdl(self, value):
            if value != 0:
                raise AssertionError('unexpected PDL')

        def okc_run(self, *arguments):
            launches.append((self, arguments))
            arguments[5].copy_(arguments[2])

    def load_inline(**kwargs):
        builds.append(kwargs)
        return Extension()

    namespace.update(torch=TorchFacade(), load_inline=load_inline)
    return namespace, builds, launches


def inputs(tokens, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return dict(
        hidden_states=torch.randn((tokens, 4096), generator=generator, dtype=torch.bfloat16).as_subclass(HostTensor),
        topk_ids=torch.randint(0, 256, (tokens, 8), generator=generator, dtype=torch.int32).as_subclass(HostTensor),
        topk_weights=torch.rand((tokens, 8), generator=generator).as_subclass(HostTensor),
        config=dict(bs=tokens, d_hidden=4096, d_expert=4096, n_experts=256,
                    local_start=0, local_count=32, top_k=8, sharding='ep8'))


def local_weights():
    return dict(w13=torch.arange(32, dtype=torch.uint8).reshape(32, 1),
                w2=torch.arange(32, dtype=torch.uint8).reshape(32, 1),
                w13_sf=torch.full((32, 1), 0x40, dtype=torch.uint8).view(torch.float8_e4m3fn),
                w2_sf=torch.full((32, 1), 0x40, dtype=torch.uint8).view(torch.float8_e4m3fn),
                w13_gs=torch.arange(32, dtype=torch.float32) + 1,
                w2_gs=torch.arange(32, dtype=torch.float32) + 2,
                post_expert_norm=torch.linspace(-1, 1, 4096, dtype=torch.bfloat16))


def prepare_small(namespace, entry='prepare'):
    packed_rows = []

    def codes(tensor, rows):
        packed_rows.append((int(tensor[0]), rows.clone()))
        return tensor.to(torch.int32).clone()

    overrides = dict(
        _expert_moe_weights=lambda weights: {key: value.clone() for key, value in local_weights().items()},
        _pack_codes=codes,
        _pack_scales=lambda tensor, rows: tensor.view(torch.uint8).clone(),
        _fat_order=lambda codes, scales: codes.view(torch.uint8).clone(),
        _unit_order=lambda codes, scales: codes.view(torch.uint8).clone(),
        _wgs_copy=lambda codes, scales, rows, sc6: codes.clone(),
        _ma_make_scratch=lambda device: [torch.zeros(80, dtype=dtype) for dtype in
                                       (torch.int32, torch.float32, torch.bfloat16)])
    with patch.dict(namespace, overrides):
        state = namespace[entry]({})
    return state, packed_rows


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source, cls.manifest = adapter.build()

    def setUp(self):
        self.env, self.builds, self.launches = environment(self.source)

    def test_deterministic_pins_closure_and_delivered_payload(self):
        source, manifest = adapter.build()
        self.assertEqual(source, self.source)
        self.assertEqual(manifest, self.manifest)
        delivered = adapter.ROOT / 'docs/steering/evidence/2026-11-02-top100-moe-adapter/payload.py'
        self.assertEqual(delivered.read_bytes(), source.encode())
        self.assertLess(len(source.encode()), 1_000_000)
        self.assertFalse(manifest['dispatch_ready'])
        self.assertEqual(adapter.sha(self.env['MOE_CUDA_SRC']), adapter.CONTROL_NATIVE)
        self.assertEqual(adapter.sha(self.env['_MA_ALT_SRCS']['expert_top100_pregamma']), adapter.CANDIDATE_NATIVE)
        imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('reference' in ast.unparse(node) or 'triton' in ast.unparse(node) for node in imports))

    def test_actual_prepare_and_all_entrypoint_routes(self):
        state, rows = prepare_small(self.env)
        self.assertEqual(len(rows), 64)
        self.assertEqual([expert for expert, _ in rows[::2]], list(range(32)))
        self.assertEqual(rows[0][1][:16].tolist(), list(range(8)) + list(range(4096, 4104)))
        self.assertEqual(rows[1][1].tolist(), list(range(4096)))
        self.assertTrue(torch.equal(state['ma_w'][2], local_weights()['w13_gs']))
        self.assertTrue(torch.equal(state['ma_w'][5], local_weights()['w2_gs']))
        self.assertTrue(torch.equal(state['ma_w'][7], local_weights()['post_expert_norm']))
        self.assertEqual(len(state['ma_w']), 14)
        expected = {1: 'tiny', 4: 'tiny', 8: 'tiny2', 16: 'mid', 32: 'b4',
                    33: 'main', 128: 'main', 256: 'main', 257: 'main'}
        for tokens, route in expected.items():
            for seed in (1102031, 1102037):
                data = inputs(tokens, seed)
                before = set(state)
                output = self.env['kernel_fn'](data, state)
                extension, arguments = self.launches[-1]
                self.assertIs(extension, state['ma_ext'][route]._module)
                for index, key in ((2, 'hidden_states'), (3, 'topk_ids'), (4, 'topk_weights')):
                    self.assertIs(arguments[index], data[key])
                self.assertEqual(arguments[6:], (0, 32, 132))
                self.assertEqual(set(state), before)
                self.assertNotEqual(output.data_ptr(), data['hidden_states'].data_ptr())
                self.assertTrue(torch.equal(output, data['hidden_states']))
        self.assertEqual(len(self.builds), 5)

    def test_actual_compiler_calls_match_pinned_control_and_candidate(self):
        state, _ = prepare_small(self.env)
        original, _ = capture(adapter.pair.BASE.read_text(encoding='utf8'))
        self.assertEqual({call['name']: call for call in self.builds},
                         {call['name']: call for call in original})
        candidate, _ = prepare_small(self.env, 'expert_moe_prepare_candidate')
        self.assertEqual(len(self.builds), 7)
        for call in self.builds[-2:]:
            self.assertEqual(adapter.sha(call['cuda_sources']), adapter.CANDIDATE_NATIVE)
            control = next(item for item in original if item['extra_cuda_cflags'] == call['extra_cuda_cflags'])
            for field in ('cpp_sources', 'functions', 'verbose'):
                self.assertEqual(call[field], control[field])
        with self.assertRaisesRegex(RuntimeError, 'IDENTITY_OR_ARM'):
            self.env['kernel_fn'](inputs(32), candidate)
        with self.assertRaisesRegex(RuntimeError, 'IDENTITY_OR_ARM'):
            self.env['expert_moe_candidate_fn'](inputs(32), state)
        self.env['expert_moe_candidate_fn'](inputs(32), candidate)
        self.assertIs(self.launches[-1][0], candidate['ma_ext']['b4']._module)

    def test_four_independent_preparations_and_guard_corruption(self):
        states, _ = prepare_small(self.env, 'expert_moe_prepare_four')
        self.assertEqual([state['_expert_pair_arm'] for state in states], [False, False, True, True])
        pointers = []
        for state in states:
            pointers.extend(tensor.data_ptr() for tensor in state['ma_w'])
            pointers.extend(full.data_ptr() for full, _ in state['guards'])
            self.env['_expert_moe_check_guards'](state)
        self.assertEqual(len(pointers), len(set(pointers)))
        self.assertEqual(len({id(state['ma_ext']) for state in states}), 4)
        full, words = states[0]['guards'][0]
        full[words] = 0
        with self.assertRaisesRegex(RuntimeError, 'SCRATCH_OVERRUN'):
            self.env['_expert_moe_check_guards'](states[0])
        for state in states[1:]:
            self.env['_expert_moe_check_guards'](state)

    def test_invalid_config_metadata_arm_and_current_device(self):
        state, _ = prepare_small(self.env)
        for key, value in (('local_start', 32), ('local_start', -1), ('local_count', 256),
                           ('local_count', 0), ('d_hidden', 8192), ('top_k', 4),
                           ('n_experts', 32), ('bs', 385), ('bs', True), ('bs', 0)):
            data = inputs(32)
            data['config'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                self.env['kernel_fn'](data, state)
        for field, tensor in (
                ('topk_ids', torch.zeros((32, 8), dtype=torch.int64).as_subclass(HostTensor)),
                ('topk_ids', torch.zeros((32, 16), dtype=torch.int32)[:, ::2].as_subclass(HostTensor)),
                ('topk_ids', torch.zeros(257, dtype=torch.int32)[1:].view(32, 8).as_subclass(HostTensor)),
                ('topk_weights', torch.zeros((33, 8)).as_subclass(HostTensor)),
                ('hidden_states', torch.zeros((32, 4096), dtype=torch.bfloat16))):
            data = inputs(32)
            data[field] = tensor
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'TENSOR_CONTRACT'):
                self.env['kernel_fn'](data, state)
        self.env['torch'].cuda.current_device = lambda: 1
        with self.assertRaisesRegex(RuntimeError, 'CURRENT_DEVICE'):
            self.env['kernel_fn'](inputs(32), state)
        self.assertFalse(self.launches)
        self.env['torch'].cuda.current_device = lambda: None
        state['_expert_pair_arm'] = True
        with self.assertRaisesRegex(RuntimeError, 'IDENTITY_OR_ARM'):
            self.env['kernel_fn'](inputs(32), state)
        with self.assertRaisesRegex(RuntimeError, 'INVALID_REBIND'):
            self.env['_expert_pair_bind'](state, False)

    def test_prepare_rejects_deadline_architecture_scales_and_capture(self):
        with self.assertRaisesRegex(RuntimeError, 'PREPARE_295S'):
            self.env['_expert_moe_prepare']({}, False, deadline=0)
        with self.assertRaisesRegex(RuntimeError, 'ARM_TYPE'):
            self.env['_expert_moe_prepare']({}, 1)
        self.env['torch'].cuda.is_current_stream_capturing = lambda: True
        with self.assertRaisesRegex(RuntimeError, 'DURING_CAPTURE'):
            prepare_small(self.env)
        self.env['torch'].cuda.is_current_stream_capturing = lambda: False
        self.env['torch'].cuda.get_device_properties = lambda device: types.SimpleNamespace(
            major=8, minor=0, multi_processor_count=108)
        with self.assertRaisesRegex(RuntimeError, 'DEVICE_CONTRACT'):
            prepare_small(self.env)
        self.env['torch'].cuda.get_device_properties = lambda device: types.SimpleNamespace(
            major=9, minor=0, multi_processor_count=132)
        with patch.dict(self.env, _ma_scales_prmt_exact=lambda *args: False):
            with self.assertRaisesRegex(RuntimeError, 'SCALE_CONTRACT_NO_FALLBACK'):
                prepare_small(self.env)
        self.assertFalse(self.builds)

    def test_weight_bank_adapter_metadata_and_global_scale_slicing(self):
        class MetadataTensor:
            serial = 0

            def __init__(self, shape, dtype, origin=None):
                self.shape, self.dtype, self.origin = shape, dtype, origin
                self.device, self.is_cuda = torch.device('cuda:0'), True
                MetadataTensor.serial += 1
                self.pointer = 256 * MetadataTensor.serial

            def is_contiguous(self):
                return True

            def data_ptr(self):
                return self.pointer

            def __getitem__(self, index):
                if index != slice(None, 32):
                    raise AssertionError(index)
                return MetadataTensor((32, *self.shape[1:]), self.dtype, (self, index))

            def clone(self):
                return MetadataTensor(self.shape, self.dtype, self.origin or self)

        weights = {
            'w13': MetadataTensor((256, 8192, 2048), torch.uint8),
            'w13_sf': MetadataTensor((256, 8192, 256), torch.float8_e4m3fn),
            'w13_gs': MetadataTensor((256,), torch.float32),
            'w2': MetadataTensor((256, 4096, 2048), torch.uint8),
            'w2_sf': MetadataTensor((256, 4096, 256), torch.float8_e4m3fn),
            'w2_gs': MetadataTensor((256,), torch.float32),
            'post_norm_w': MetadataTensor((4096,), torch.bfloat16)}
        self.env['torch'].Tensor = MetadataTensor
        local = self.env['_expert_moe_weights'](weights)
        for key in weights.keys() - {'post_norm_w'}:
            self.assertEqual(local[key].origin, (weights[key], slice(None, 32)))
            self.assertNotEqual(local[key].pointer, weights[key].pointer)
        self.assertIs(local['post_expert_norm'].origin, weights['post_norm_w'])
        for key in weights:
            original = weights[key].shape
            weights[key].shape = (32,)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'TENSOR_CONTRACT'):
                self.env['_expert_moe_weights'](weights)
            weights[key].shape = original
        weights['w13_gs'].device = torch.device('cuda:1')
        with self.assertRaisesRegex(RuntimeError, 'TENSOR_CONTRACT'):
            self.env['_expert_moe_weights'](weights)

    def test_real_cpu_packing_and_lossless_sc6_wgs_layout(self):
        generator = torch.Generator().manual_seed(1102049)
        codes = torch.randint(0, 256, (256, 2048), dtype=torch.uint8, generator=generator)
        scale_bytes = torch.randint(0x40, 0x7F, (256, 256), dtype=torch.uint8, generator=generator)
        scales, rows = scale_bytes.view(torch.float8_e4m3fn), torch.arange(256)
        plain = self.env['_wgs_copy'](codes, scales, rows, False).view(-1, 9216)
        compact = self.env['_wgs_copy'](codes, scales, rows, True).view(-1, 8960)
        self.assertTrue(torch.equal(plain[:, :8192], compact[:, :8192]))
        packed = compact[:, 8192:]
        low = packed[:, :512].reshape(-1, 128, 4).int()
        high = packed[:, 512:].reshape(-1, 128, 2).int()
        six = torch.cat((low, high), dim=-1)
        last_two = torch.stack(((six[..., 0] & 3) | ((six[..., 1] & 3) << 2) | ((six[..., 2] & 3) << 4),
                                (six[..., 3] & 3) | ((six[..., 4] & 3) << 2) | ((six[..., 5] & 3) << 4)), -1)
        decoded = (torch.cat((six >> 2, last_two), -1) + 0x40).byte().reshape(-1, 1024)
        self.assertTrue(torch.equal(decoded, plain[:, 8192:]))
        words = self.env['_pack_codes'](codes, rows)
        packed_scales = self.env['_pack_scales'](scales, rows)
        self.assertEqual(words.numel() * 4, codes.numel())
        self.assertEqual(packed_scales.numel(), scale_bytes.numel())
        self.assertEqual(self.env['_fat_order'](words, packed_scales).numel(), codes.numel() + scale_bytes.numel())
        self.assertEqual(self.env['_unit_order'](words, packed_scales).numel(), codes.numel() + scale_bytes.numel())
        self.assertTrue(self.env['_ma_scales_prmt_exact'](scales))
        self.assertTrue(self.env['_sc6_ok'](scales))
        scale_bytes[0, 0] = 0x3F
        self.assertFalse(self.env['_sc6_ok'](scales))
        scale_bytes[0, 0] = 0x7F
        self.assertFalse(self.env['_ma_scales_prmt_exact'](scales))

    def test_source_drift_rejects_delivered_files_and_edit_anchors(self):
        read = Path.read_text
        for target in (adapter.pair.BASE, adapter.ROOT / 'tools/expert_top100_candidate.py'):
            def drift(path, *args, **kwargs):
                return read(path, *args, **kwargs) + ('\n' if path == target else '')
            with patch.object(Path, 'read_text', drift):
                with self.assertRaisesRegex(RuntimeError, 'SOURCE_DRIFT'):
                    adapter.build()
        self.env['MOE_CUDA_SRC'] = self.env['MOE_CUDA_SRC'].replace('fork_bf16(d4[h] * gsc) * w0', 'BROKEN')
        with self.assertRaisesRegex(RuntimeError, 'SOURCE_DRIFT'):
            self.env['_expert_pair_source']()
        self.assertFalse(self.builds)

    def test_actual_full_scratch_factory_sizes_initialization_and_guard_views(self):
        scratch = self.env['_ma_make_scratch'](torch.device('cpu'))
        max_routes = 384 * 8
        fixups = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
        xrows = max_routes + 8 * (32 + max_routes // 32 + 8)
        self.assertEqual([tensor.numel() for tensor in scratch],
                         [8 + 1024 + max_routes + 4096,
                          max_routes + max_routes * 4096 + fixups,
                          (max_routes + 2 * xrows) * 4096])
        self.assertEqual(int(torch.count_nonzero(scratch[0])), 0)
        self.assertEqual(int(torch.count_nonzero(scratch[1])), 0)
        inter_bits = scratch[2].view(torch.int16)
        self.assertTrue(bool((inter_bits[128 * 4096:256 * 4096] == -1).all()))
        self.assertEqual(int(torch.count_nonzero(inter_bits[:128 * 4096])), 0)
        self.assertEqual(int(torch.count_nonzero(inter_bits[256 * 4096:])), 0)
        state = dict(device=torch.device('cpu'), guards=[])
        for tensor in scratch:
            guarded = self.env['_expert_moe_guarded'](tensor, state)
            self.assertTrue(torch.equal(guarded.view(torch.uint8), tensor.view(torch.uint8)))
            self.assertEqual(guarded.data_ptr() % 16, 0)
            self.assertNotEqual(guarded.data_ptr(), tensor.data_ptr())
        self.env['_expert_moe_check_guards'](state)


if __name__ == '__main__':
    unittest.main(verbosity=2)
