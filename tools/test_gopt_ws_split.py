"""Check reset lifetime, route export, source selection and complete-call gates."""
import ast
import random
import base64
import contextlib
import io
import json
import time
import zlib
from pathlib import Path
import unittest

from gopt_ws_split import build, contract_source, transform, HELPER_PIN
from gopt_ws_split_runtime import _ws_positions
from cuda_issue_groups import sha
from prod_size_compile_g import bindings
from d32_b4_prefix_audit import body
import expert_top100_moe_adapter as adapter


class SourceAndModel(unittest.TestCase):
    def test_fixed_source_selection_and_internal_publication(self):
        source, manifest, controls, candidates = build()
        ast.parse(source)
        self.assertEqual(len(manifest['compiler_calls']), 8)
        helper = bindings(adapter.build()[0])['_MA_ALT_SRCS']['moe_helper0']
        self.assertEqual(sha(helper), HELPER_PIN)
        edited = transform(helper)
        for signature in ('__device__ void ws_helper(', '__device__ void skinny_ring_ws(', '__device__ void route('):
            self.assertEqual(body(helper, signature), body(edited, signature))
        self.assertNotIn('p.ctr[3]', body(edited, 'void __launch_bounds__(256) ws_finalize(').split('const int i =')[0])
        for control, candidate in zip(controls, candidates):
            self.assertIn('-DOKC_NS=3', candidate['extra_cuda_cflags'])
            self.assertEqual(control['cuda_sources'], candidate['cuda_sources'])
            self.assertIn('-DOKC_WS_SPLIT=1', candidate['extra_cuda_cflags'])
        self.assertIn('ws_finalize<<<p.M * 4, 256, 0, st>>>', edited)
        self.assertNotIn('cudaMemset', edited)

    def test_exported_positions_for_empty_and_mixed_routes(self):
        rng = random.Random(20261103)
        for tokens in (1, 4, 8, 16):
            for _ in range(80):
                ids = [rng.randrange(256) for i in range(tokens*8)]
                pos = _ws_positions(ids)
                occupied = [x for x in pos if x >= 0]
                self.assertEqual(sorted(occupied), list(range(len(occupied))))
                ordered = sorted((expert, i) for i, expert in enumerate(ids) if expert < 32)
                for offset, (expert, index) in enumerate(ordered):
                    self.assertEqual(pos[index], offset)
            self.assertEqual(_ws_positions([32]*tokens*8), [-1]*tokens*8)

    def test_finalizer_interleaving_uses_immutable_generation(self):
        rng = random.Random(8193)
        done = [1]*88
        for tokens in (1, 8, 16, 32, 128, 256, 16, 1)*8:
            gen = getattr(self, 'gen', 0)
            latched = gen
            blocks = list(range(4*tokens))
            rng.shuffle(blocks)
            reads = []
            live = gen
            for block in blocks:
                if block == 0:
                    live = latched+1
                    done[:] = [0]*88
                reads.append(latched&1)
            self.assertEqual(set(reads), {gen&1})
            self.assertEqual(live, gen+1)
            self.assertEqual(done, [0]*88)
            self.gen = live

    def test_retained_target_rows_and_entropy_fit_success_and_failure_bounds(self):
        log = Path('muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/owning.log').read_text()
        line = next(x.split(' ',1)[1] for x in log.splitlines() if x.startswith('EXPERT_MOE_CAPSULE '))
        report = json.loads(line)
        ns = dict(json=json,time=time,H_RUNTIME_ID='test',require=lambda condition,message: condition or (_ for _ in ()).throw(RuntimeError(message)))
        exec(Path('tools/gopt_ws_split_runtime.py').read_text(),ns)
        rng = random.Random(18379)
        for entropy in (False, True):
            data = json.loads(json.dumps(report))
            if entropy:
                for row in data['rows']:
                    row[3] = [rng.uniform(20,300) for _ in range(4)]
                    row[5] = [rng.uniform(.9,1.1) for _ in range(4)]
                    row[6] = [rng.uniform(0,.1) for _ in range(2)]
            blob = ns['_ws_encode'](data)
            self.assertLess(len(blob),12000)
            self.assertEqual(json.loads(zlib.decompress(base64.b64decode(blob))),data)
            ns['_WS_LIVE'].update(rows=data['rows'],counters=data['counters'],began=time.monotonic(),stage='final-budget')
            output = io.StringIO()
            with contextlib.redirect_stdout(output): ns['_ws_partial'](RuntimeError('injected'))
            partial = output.getvalue().split(' ',1)[1].strip()
            self.assertLess(len(partial),12000)
            self.assertEqual(len(json.loads(zlib.decompress(base64.b64decode(partial)))['rows']),36)

    def test_complete_six_load_gate_and_unchanged_drift(self):
        env = {}
        exec(contract_source(), env)
        rows = [[seed, tokens, dist, [100., 100., 97., 97.], [8, 8, 1],
                 [1.04 if tokens<32 else 1. for _ in range(4)], [0.0001, 0.0001]]
                for seed in env['BANKS'] for tokens in env['LOADS'] for dist in env['DISTRIBUTIONS']]
        self.assertTrue(env['decision'](rows)['component_pass'])
        for row in rows:
            if row[1]<32: row[5] = [1.01]*4
        self.assertFalse(env['decision'](rows)['component_pass'])
        for row in rows:
            if row[1]<32: row[5] = [1.04]*4
        row = next(x for x in rows if x[1] == 256)
        row[5] = [1.03]*4
        self.assertFalse(env['decision'](rows)['component_pass'])


if __name__ == '__main__':
    unittest.main()
