import ast
import random
import contextlib
import copy
import types
import unittest
from pathlib import Path

from dl_make_layer_timeline import ROOT, RUNTIME, build
from dl_timeline_codec import decode_report, encode_report, summarize


class TimelineTests(unittest.TestCase):
    def test_capture_outputs_materialized_before_inspection(self):
        node = next(n for n in ast.parse(RUNTIME).body if isinstance(n, ast.FunctionDef)
                    and n.name == '_lt_capture')

        def run(capture_node):
            trace = []
            current = {}

            class Tensor:
                def clone(self):
                    return self
                def copy_(self, other):
                    return self

            class Graph:
                def replay(self):
                    trace.append('replay')
                    current['out']['materialized'] = True

            class Stream:
                def wait_stream(self, other):
                    pass

            @contextlib.contextmanager
            def graph_context(graph):
                current['capturing'] = True
                yield
                current['capturing'] = False

            def fn(data, state):
                current['out'] = {'materialized': not current.get('capturing', False)}
                return current['out']

            cuda = types.SimpleNamespace(Stream=Stream, current_stream=Stream,
                stream=lambda s: contextlib.nullcontext(), CUDAGraph=Graph,
                graph=graph_context, synchronize=lambda: trace.append('sync'))
            ns = {'torch': types.SimpleNamespace(cuda=cuda),
                  '_lt_reset': lambda data, saved: None}
            exec(compile(ast.Module(body=[capture_node], type_ignores=[]), '<capture>', 'exec'), ns)
            graph, out, saved = ns['_lt_capture'](fn, {}, {'stream':Tensor(), 'attn_input':Tensor()})
            return out['materialized'], trace

        valid, trace = run(node)
        self.assertTrue(valid)
        self.assertEqual(trace[-2:], ['replay','sync'])
        # Negative reproduction: capture + synchronize without replay leaves outputs unwritten.
        bad = copy.deepcopy(node)
        bad.body = [n for n in bad.body if not (isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'replay')]
        self.assertFalse(run(bad)[0])

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
                    events = [(origin+i*10000+rng.randrange(2000), origin+(i+1)*10000+rng.randrange(4000), i%3,
                               f'long_full_kernel_name<{i%40}, signature parameters>')
                              for i in range(56)]
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
