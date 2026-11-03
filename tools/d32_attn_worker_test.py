"""CPU source-entry checks, not CUDA numerical or performance qualification."""
import ast
import os
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from attention_top100_screen import function, literal
from d32_attn_worker_candidate import BASE, generate, native_change, once
from test_attention_layer_continuity import config, environment, invoke


class LookaheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = BASE.read_text()
        cls.candidate, cls.control, cls.treatment, cls.patch = generate(cls.base)
        cls.tree = ast.parse(cls.candidate)

    def entry(self, captures, owners):
        env, control, _ = environment(self.candidate, captures, owners)
        treatment = NS(run=lambda *args: captures.append(('lookahead', args)))
        for name in ('_d32_attention_scope', '_d32_attention_module'):
            if any(isinstance(node, ast.FunctionDef) and node.name == name for node in self.tree.body):
                function(self.tree, name, env)
        env['_prepare_attention_modules7'] = lambda state: state.update(
            d32_attention_module=treatment, d32_attention_eligible=True)
        return env, control, treatment

    def test_actual_final_bindings_preserve_other_cells(self):
        cases = [(b, length, window, 1) for length in (1, 8)
                 for b in (1, 16, 32) for window in (2048, 0)]
        cases += [(b, 1, 0, 1) for b in (17, 24, 31, 33)]
        cases += [(32, 1, 0, 64), (32, 8, 0, 1)]
        captures, references, owners = [], [], []
        env, _, _ = self.entry(captures, owners)
        reference, _, _ = environment(self.base, references, [])
        for b, length, window, page in cases:
            with self.subTest(batch=b, length=length, window=window, page=page):
                cfg = config(b, length, window, page)
                state, data, _ = invoke(env, cfg)
                invoke(reference, cfg)
                target = 17 <= b <= 32 and length == 1 and window == 0 and page == 1
                self.assertEqual(captures[-1][0], 'lookahead' if target else 'control')
                self.assertEqual(captures[-1][1], references[-1][1])
                self.assertIs(owners[-1], state['attention_scratch'])

    def test_wrong_control_and_native_boundary(self):
        with self.assertRaises(ValueError):
            generate(self.base + '\n')
        old, new = native_change(self.control)
        self.assertEqual(once(self.treatment, new, old), self.control)
        with self.assertRaises(ValueError):
            once(self.control.replace(old, ''), old, new)
        self.assertLess(len(self.candidate.encode()), 1_000_000)
        self.assertNotIn('\r', self.candidate)
        self.assertEqual(generate(self.base)[0], self.candidate)

    def test_actual_derived_cuda_and_unchanged_python(self):
        nodes = [node for node in self.tree.body if isinstance(node, (ast.Assign, ast.Assert))
                 and 'ATT7_D32_CUDA_SRC' in ast.unparse(node)]
        env = {'ATT7_CUDA_SRC': self.control}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'derived-cuda', 'exec'), env)
        self.assertEqual(env['ATT7_D32_CUDA_SRC'], self.treatment)
        self.assertEqual(literal(self.tree, 'ATT7_CUDA_SRC'), self.control)
        added = {'_d32_attention_scope', '_d32_attention_module', '_prepare_d32_attention'}
        original = [node for node in ast.parse(self.base).body if isinstance(node, ast.FunctionDef)]
        changed = [node for node in self.tree.body
                   if isinstance(node, ast.FunctionDef) and node.name not in added]
        self.assertEqual(len(original), len(changed))
        self.assertEqual([a.name for a, b in zip(original, changed) if ast.dump(a) != ast.dump(b)],
                         ['prepare', '_launch7', '_decode_attention7'])

    def test_scope_fallbacks_missing_module_and_direct_entry(self):
        captures, owners = [], []
        env, _, module = self.entry(captures, owners)
        cfg = config(32, 1, 0)
        state, data, _ = invoke(env, cfg)
        for key, value in [('batch', 16), ('batch', 33), ('q_len', 8), ('window', 2048),
                           ('page_size', 2), ('head_dim', 128), ('num_q_heads', 32),
                           ('num_kv_heads', 4), ('kv_scale', .015625), ('rope_theta', 10000.),
                           ('sm_scale', 0.), ('qk_scale_factor', float('inf'))]:
            with self.subTest(key=key, value=value):
                other = dict(cfg, **{key: value})
                self.assertIsNone(env['_d32_attention_module'](other, state))
                with self.assertRaises(ValueError):
                    env['_launch7'](dict(data, config=other), _mod=module,
                                    attention_state=state, pdl=False)
        self.assertIsNone(env['_d32_attention_module'](cfg, dict(d32_attention_eligible=False)))
        with self.assertRaises(ValueError):
            env['_d32_attention_module'](cfg, dict(d32_attention_eligible=True))
        for override in [dict(n_splits=132), dict(flat=0), dict(w0=1000), dict(nc=2), dict(cs=2)]:
            with self.assertRaises(ValueError):
                env['_launch7'](data, _mod=module, attention_state=state, pdl=False, **override)
        with self.assertRaises(ValueError):
            env['_launch7'](data, _mod=module, attention_state=state, pdl=True)

    def test_actual_preparation_and_state_isolation(self):
        calls = []
        env = dict(torch=NS(cuda=NS(get_device_capability=lambda: (9, 0)), device=lambda x: x),
                   _att7=lambda: calls.append('control'), _num_sms7=lambda _: 132,
                   ATT7_CPP_SRC=literal(self.tree, 'ATT7_CPP_SRC'), ATT7_D32_CUDA_SRC=self.treatment,
                   ATT7_D32_CUDA_SHA256=literal(self.tree, 'ATT7_D32_CUDA_SHA256'))
        prepare = function(self.tree, '_prepare_d32_attention', env)
        loader = NS(load_inline=lambda **kwargs: calls.append(kwargs) or object())
        a, b = {}, {}
        with patch.dict(sys.modules, {'torch.utils.cpp_extension': loader}), patch.dict(os.environ):
            prepare(a)
            prepare(a)
            prepare(b)
        builds = [call for call in calls if isinstance(call, dict)]
        self.assertEqual(len(builds), 2)
        self.assertIsNot(a['d32_attention_module'], b['d32_attention_module'])
        self.assertTrue(a['d32_attention_eligible'])
        self.assertEqual(builds[0]['cuda_sources'], self.treatment)
        self.assertEqual(builds[0]['extra_cuda_cflags'], ['-DOKC_QDIRECT=0', '-DOKC_PK2=0',
                         '-DOKC_W0ENABLE=1', '-O3', '-DOKC_SM90=1', '--expt-relaxed-constexpr'])
        env['_num_sms7'] = lambda _: 80
        other = {}
        prepare(other)
        self.assertFalse(other['d32_attention_eligible'])
        self.assertNotIn('d32_attention_module', other)
        env['_num_sms7'] = lambda _: 132
        env['torch'].cuda.get_device_capability = lambda: (8, 0)
        other = {}
        prepare(other)
        self.assertFalse(other['d32_attention_eligible'])
        self.assertNotIn('d32_attention_module', other)

    def test_sampled_lookahead_lifetime_and_progress(self):
        def exercise(count, stages, seed, early=False):
            rng = random.Random(seed)
            produced, retired = 0, 0
            slots, scored = {}, set()
            stage_readers = {}
            while retired < count:
                can_produce = produced < count and produced % stages not in slots
                needed = min(retired + 1, count - 1)
                can_consume = produced > needed
                if can_produce and (not can_consume or rng.randrange(2)):
                    slots[produced % stages] = produced
                    stage_readers[produced] = {'QK', 'PV'}
                    produced += 1
                elif can_consume:
                    for tile in range(retired, needed + 1):
                        if tile not in scored:
                            self.assertEqual(slots[tile % stages], tile)
                            stage_readers[tile].remove('QK')
                            scored.add(tile)
                    if not early:
                        stage_readers[retired].remove('PV')
                    if stage_readers[retired]:
                        raise ValueError('raw stage released before final reader')
                    del slots[retired % stages]
                    retired += 1
                else:
                    raise ValueError('lookahead waits for a stage whose release needs current PV')
            self.assertEqual(scored, set(range(count)))
            self.assertEqual(produced, retired)
        for count in (0, 1, 2, 3, 4, 31, 64, 257):
            for seed in range(16):
                exercise(count, 3, seed)
        with self.assertRaises(ValueError):
            exercise(3, 1, 0)
        with self.assertRaises(ValueError):
            exercise(3, 3, 0, early=True)


if __name__ == '__main__':
    unittest.main()
