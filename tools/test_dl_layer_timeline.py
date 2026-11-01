import ast
import random
import unittest
from pathlib import Path

from dl_make_layer_timeline import ROOT, RUNTIME, build
from dl_timeline_codec import decode_report, encode_report, summarize


class TimelineTests(unittest.TestCase):
    def test_busiest_group_ties_actual_torch_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Run this test in the existing WSL CPU Torch venv')
        parsed = ast.parse(RUNTIME)
        fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_lt_busiest_start')
        ns = {'LOCAL_EXPERTS': 32}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<busiest>', 'exec'), ns)
        for values, expected in [([1, 1, 0, 0, 0, 0, 0, 0], 0),
                                 ([0, 3, 3, 0, 0, 0, 0, 0], 32),
                                 ([0, 0, 0, 0, 2, 0, 0, 0], 128)]:
            self.assertEqual(ns['_lt_busiest_start'](torch.tensor(values)), expected)

    def test_overlapping_streams_and_gap(self):
        trace = summarize([(1000, 5000, 1, 'A'), (3000, 7000, 2, 'B'),
                           (9000, 11000, 1, 'A')])
        self.assertEqual(trace['span_us'], 10)
        self.assertEqual(trace['union_us'], 8)
        self.assertEqual(trace['uncovered_us'], 2)
        self.assertEqual(trace['sum_us'], 10)
        self.assertEqual(trace['events'][1], [2., 6., 2, 'B'])
        with self.assertRaises(ValueError):
            summarize([])

    def test_all12_lossless_transport_and_budget(self):
        rng = random.Random(333)
        rows = []
        for case in ['d1', 'd16', 'd32', 'v1', 'v16', 'v32']:
            for layer in ['s', 'f']:
                for arm in ['owned', 'sglang']:
                    origin = rng.randrange(1_000_000_000)
                    events = [(origin+i*10000+rng.randrange(2000), origin+(i+1)*10000+rng.randrange(4000), i%2,
                               f'long_full_kernel_name<{i%20}, signature parameters>')
                              for i in range(35)]
                    rows.append({'case': case+layer, 'arm': arm, 'trace': summarize(events),
                                 'cold_us': [round(rng.uniform(50,500),3) for _ in range(5)],
                                 'profile_spans_us': [round(rng.uniform(50,500),3) for _ in range(3)],
                                 'routing': {'local_slots_histogram': [rng.randrange(33) for _ in range(9)]}})
        report = {'rows': rows, 'pins': {'candidate': 'x'}}
        self.assertEqual(decode_report(encode_report(report)), report)
        with self.assertRaises(ValueError):
            encode_report(report, limit=10)
        # Input has not been mutated during interning.
        self.assertIsInstance(report['rows'][0]['trace']['events'][0][3], str)

    def test_source_invariants_and_payload(self):
        candidate = (ROOT/'muse_decode_layer/cand/layer_attn31s.py').read_text(encoding='utf-8')
        generated = build(candidate)
        self.assertTrue(generated.startswith(candidate))
        self.assertLess(len(generated.encode()), 1_000_000)
        self.assertTrue(generated.startswith('# okc: problem=muse_decode_layer'))
        original = ast.parse(candidate)
        combined = ast.parse(generated)
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'kernel_fn':
                actual = [n for n in combined.body if isinstance(n, ast.FunctionDef)
                          and n.name == 'kernel_fn']
                self.assertEqual(len(actual), 1)
                self.assertEqual(ast.dump(node), ast.dump(actual[0]))
        # Established helpers are unique; reference source is a literal, never imported.
        for name in ['_ab_inputs', '_ab_ref_front', '_ab_policy']:
            self.assertEqual(sum(isinstance(n, ast.FunctionDef) and n.name == name
                                 for n in combined.body), 1)
        runtime_imports = [n for n in ast.walk(combined) if isinstance(n, ast.Import)]
        self.assertFalse(any(a.name == 'reference' for n in runtime_imports for a in n.names))


if __name__ == '__main__':
    unittest.main()
