"""Execute the actual conformance formatter/gates with real Torch CPU tensors.

Graph facade replays recorded fake-extension calls; no CUDA claim is made.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from dl_fork_moe_empty_probe import empty_conformance


class FakeCuda:
    active = None
    replays = 0

    def synchronize(self):
        pass

    def CUDAGraph(self):
        cuda = self

        class Graph:
            calls = None

            def __init__(self):
                self.calls = []

            def replay(self):
                cuda.replays += 1
                for call in self.calls:
                    call()

        return Graph()

    @contextmanager
    def graph(self, graph):
        self.active = graph
        class RecordFill(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func == torch.ops.aten.fill_.Scalar:
                    graph.calls.append(lambda: func(*args, **kwargs))
                return func(*args, **kwargs)
        try:
            with RecordFill():
                yield
        finally:
            self.active = None


class Facade:
    def __init__(self):
        self.cuda = FakeCuda()

    def __getattr__(self, name):
        return getattr(torch, name)


class Ext:
    def __init__(self, api, corrupt=False, replay_only=False):
        self.api, self.corrupt, self.replay_only = api, corrupt, replay_only

    def okc_run(self, w, scratch, x, ids, tw, out, lo, nloc, nsm):
        assert lo == 0 and nloc == 32

        def call():
            with _disable_current_modes():
                if bool((ids < 0).all()):
                    if self.corrupt and (not self.replay_only or self.api.cuda.replays > 0):
                        return  # deliberately leave the nonzero poison/stale output
                    out.zero_()
                else:
                    out.copy_(x)

        if self.api.cuda.active is not None:
            self.api.cuda.active.calls.append(call)
        call()


class EmptyProbeTests(unittest.TestCase):
    def fixture(self, corrupt=False, replay_only=False):
        api = Facade()
        native = Ext(api, corrupt, replay_only)
        fallback = Ext(api)
        state = dict(_fork_native_eligible=True, qkv=torch.empty(1), ma_w=[], ma_scratch=[], nsm=132)
        select = lambda m, st: native
        control = lambda m, st: fallback
        return api, state, select, select, control

    def test_actual_probe_pass_all_sizes(self):
        args = self.fixture()
        rows = empty_conformance(*args, reps=3)
        self.assertEqual(len(rows), 4)
        self.assertEqual(args[0].cuda.replays, 12)
        self.assertTrue(all(row[2:4] == (0., 0.) for row in rows))

    def test_nonzero_empty_negative_control(self):
        with self.assertRaisesRegex(AssertionError, 'EMPTY_NONZERO'):
            empty_conformance(*self.fixture(corrupt=True), reps=3, sizes=(32,))

    def test_graph_replay_negative_control(self):
        with self.assertRaisesRegex(AssertionError, 'EMPTY_NONZERO'):
            empty_conformance(*self.fixture(corrupt=True, replay_only=True), reps=3, sizes=(32,))

    def test_fallback_cannot_masquerade_as_native(self):
        args = list(self.fixture())
        native = args[3]
        args[2] = lambda m, st: Ext(args[0])
        with self.assertRaisesRegex(AssertionError, 'Native path not selected'):
            empty_conformance(*args, reps=3, sizes=(32,))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
