import ast
from collections import Counter
import io
import struct
import tokenize
import types
import unittest

import g_timed_silu_table as treatment
from g_native_patch_ladder import flags
from prod_size_compile_g import bindings

ORIGINAL = treatment.BASE.read_bytes().decode()
SOURCE, MANIFEST = treatment.build(ORIGINAL)


def capture(source):
    env = bindings(source)
    env.update(flags(source))
    calls, requests = [], []
    class Extension:
        def okc_set_pdl(self, value):
            assert value == 0
    def load_inline(**kwargs):
        calls.append(kwargs)
        return Extension()
    env.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
               load_inline=load_inline, _MA_EXTS={}, _MA_MAIN_NS=3, _MA_PDL=False,
               _RouterExt=lambda value: value)
    for name in ('_main_defines', '_ma_ext', '_fork_native_moe'):
        exec(treatment.function(source, name), env)
    state = dict(ma_sfx=('-DOKC_PRMTSC=1',), ma_sc6=True, ma_ext={})
    for tokens in (1, 8, 16, 32, 33, 128, 256, 384):
        ext = env['_fork_native_moe'](tokens, state)
        requests.append(dict(tokens=tokens, selected_key=next(key for key, value in state['ma_ext'].items() if value is ext)))
    for original_call in list(calls[:2]):
        defines = tuple(flag for flag in original_call['extra_cuda_cflags'] if flag.startswith('-D')
                        and not flag.startswith(('-DOKC_NS=', '-DOKC_FNST=', '-DOKC_NSF=')))
        env['_ma_ext'](('@moe_helper0',) + defines + ('-DOKC_SHORT_FINAL_RELAXED=1',))
    return calls, requests


def coverage(native):
    lines = native.splitlines()
    sites = []
    for index, line in enumerate(lines):
        if 'SILU_DIV(gt,' in line or treatment.NEW_GATE.strip() in line:
            declaration = next(text.strip() for text in reversed(lines[:index]) if text.startswith('__device__'))
            sites.append(dict(line=index + 1, declaration=declaration, text=line.strip()))
    return sites


class Contracts(unittest.TestCase):
    def test_reverse_and_licenses(self):
        self.assertTrue(treatment.validate(SOURCE, ORIGINAL)['exact_reverse'])
        comments = lambda source: [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT]
        self.assertEqual(comments(ORIGINAL), comments(SOURCE))

    def test_actual_seven_build_requests_and_routes(self):
        old, old_routes = capture(ORIGINAL)
        new, routes = capture(SOURCE)
        self.assertEqual(len(old), 7)
        self.assertEqual(old_routes, routes)
        self.assertEqual([row['selected_key'] for row in routes], ['tiny', 'tiny2', 'mid', 'b4', 'main', 'main', 'main', 'main'])
        for index, (before, after) in enumerate(zip(old, new)):
            for key in ('cpp_sources', 'extra_cuda_cflags', 'functions'):
                self.assertEqual(before[key], after[key])
            if index in (3, 4):
                self.assertNotEqual(before['cuda_sources'], after['cuda_sources'])
                self.assertIn(treatment.PRODUCER, after['cuda_sources'])
            else:
                self.assertEqual(before, after)

    def test_actual_prepare_allocates_only_uninitialized_private_table(self):
        allocated = []
        def empty(count, **kwargs):
            result = object()
            allocated.append((count, kwargs, result))
            return result
        env = dict(torch=types.SimpleNamespace(empty=empty, zeros=lambda *args, **kwargs: object(),
                       float32='f32', int32='i32', bfloat16='bf16'), _MA_EMAX=3072, WGS_FIXF=0,
                       WGS_XROWS=1, _df_inter=lambda value: value)
        exec(treatment.function(SOURCE, '_ma_make_scratch'), env)
        states = [dict(ma_scratch=env['_ma_make_scratch']('cuda-stub')) for unused in range(2)]
        self.assertEqual(len(allocated), 2)
        self.assertEqual(allocated[0][:2], (65536, dict(dtype='f32', device='cuda-stub')))
        self.assertIsNot(states[0]['ma_scratch'][3], states[1]['ma_scratch'][3])
        prepare = ast.parse(treatment.function(SOURCE, '_fork_native_ma_prepare')).body[0]
        env.update(st={}, dev='cuda-stub', _guarded=lambda value: ('guard', value))
        exec(compile(ast.Module(body=prepare.body[-2:], type_ignores=[]), '<actual-production-allocation>', 'exec'), env)
        self.assertEqual(len(env['st']['ma_scratch']), 4)
        self.assertIs(env['st']['ma_scratch'][3], allocated[-1][2])
        startup = ast.parse(treatment.function(SOURCE, '_release_expert_self_test')).body[0]
        seen = []
        env.update(weights={'w13': types.SimpleNamespace(device='cuda-stub')}, production={},
                   scratch_factory=env['_ma_make_scratch'], sentinel=123,
                   _release_private_scratch=lambda torch, buffers, sentinel: (seen.append(list(buffers)) or buffers, []))
        env['torch'].Generator = lambda **kwargs: types.SimpleNamespace(manual_seed=lambda seed: object())
        stop = next(index for index, node in enumerate(startup.body) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'production_storage' for target in node.targets))
        exec(compile(ast.Module(body=startup.body[:stop], type_ignores=[]), '<actual-private-startup-allocation>', 'exec'), env)
        self.assertEqual(len(seen[0]), 3)
        self.assertEqual(len(env['private']['ma_scratch']), 4)
        self.assertIs(env['private']['ma_scratch'][3], allocated[-1][2])
        self.assertIn('scratch.size() > 4 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;', bindings(SOURCE)['MOE_CUDA_SRC'])

    def test_domain_embedding_and_index(self):
        counts = Counter()
        for raw in range(65536):
            exponent, fraction = (raw >> 7) & 255, raw & 127
            counts['nan' if exponent == 255 and fraction else 'inf' if exponent == 255 else 'finite'] += 1
            bits = raw << 16
            self.assertEqual(bits >> 16, raw)
            if exponent != 255:
                value = struct.unpack('<f', struct.pack('<I', bits))[0]
                self.assertEqual(struct.unpack('<I', struct.pack('<f', value))[0] >> 16, raw)
        self.assertEqual(counts, dict(finite=65280, inf=2, nan=254))
        self.assertEqual(struct.pack('<f', -0.0), struct.pack('<I', 0x80000000))

    def test_reject_native_and_actual_entry_mutations(self):
        mutations = [
            (treatment.LAUNCH, ''),
            (treatment.LAUNCH, '  if (p.nloc) {\n' + treatment.LAUNCH + '  }\n'),
            (treatment.BINDING, '  p.silu_table = global_table;\n'),
            ('const float* silu_table;', 'static const float* silu_table;'),
            ('fork_bf16(d4[h] * gsc)', '(d4[h] * gsc)'),
            (treatment.NEW_GATE, treatment.OLD_GATE),
            ('__uint_as_float(raw << 16)', '__half2float(raw)'),
            ('<<<256, 256, 0, st>>>', '<<<1, 256, 0, st>>>'),
            (treatment.ALLOCATION, treatment.ALLOCATION.replace('torch.empty', 'torch.zeros')),
            ('buffers[:3], sentinel)', 'buffers, sentinel)'),
            ('scratch.back().data_ptr()', 'scratch[0].data_ptr()'),
            ('_MA_ALT_SRCS[d[1:]]', 'MOE_CUDA_SRC'),
            (treatment.NEW_GATE, treatment.NEW_GATE.replace('>> 16', '>> 17')),
        ]
        for old, new in mutations:
            with self.subTest(old=old):
                self.assertIn(old, SOURCE)
                with self.assertRaises(AssertionError):
                    treatment.validate(SOURCE.replace(old, new), ORIGINAL)

    def test_per_invocation_empty_replay_has_no_conditional_producer(self):
        native = bindings(SOURCE)['MOE_CUDA_SRC']
        host = native[native.index('void okc_run('):]
        self.assertIn(treatment.LAUNCH + '  void* args[]={&p};', host)
        self.assertEqual(host.count('timed_silu_table_build<<<'), 1)
        self.assertNotIn('cudaStreamIsCapturing', host)
        self.assertNotIn('cudaGraph', host)
        self.assertNotIn('static float', native)
        self.assertEqual(treatment.PRODUCER.count('table[raw] ='), 1)
        self.assertNotIn('if (', treatment.PRODUCER)
        self.assertNotIn('return', treatment.PRODUCER)

    def test_independent_main_helper_site_inventory(self):
        before, after = bindings(ORIGINAL), bindings(SOURCE)
        for key, native in [('MAIN', before['MOE_CUDA_SRC']), ('helper', before['_MA_ALT_SRCS']['moe_helper0'])]:
            sites = coverage(native)
            self.assertEqual(len(sites), 5, key)
            self.assertEqual(sum('wgs_epi_tile' in row['declaration'] for row in sites), 1)
        sites = coverage(after['MOE_CUDA_SRC'])
        self.assertEqual(sum('__ldg(p.silu_table' in row['text'] for row in sites), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
