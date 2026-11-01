from contextlib import redirect_stdout
import io
import time
import unittest

import torch
from torch.utils._python_dispatch import _disable_current_modes

from dl_g1_release_startup import _release_expert_self_test, _release_private_scratch, _release_routes
from test_dl_fork_moe_empty_cpu import Facade


class Expert:
    def __init__(self, api, production_scratch, failure=None):
        self.api = api
        self.production_scratch = production_scratch
        self.failure = failure

    def okc_run(self, w, scratch, x, ids, tw, out, lo, nloc, grid):
        assert (lo, nloc, grid) == (0, 32, 132)
        if scratch:
            assert all(a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
                       for a, b in zip(scratch, self.production_scratch))

        def run():
            with _disable_current_modes():
                if self.failure == 'stale' and self.api.cuda.replays == 2:
                    return
                if self.failure == 'remote' and bool((ids < 0).all()):
                    return
                factors = (torch.clamp_min(ids + 1, 0) * tw).sum(1, keepdim=True)
                out.copy_(x * factors)
                if self.failure == 'canary' and scratch:
                    scratch[0].as_strided((1,), (1,), scratch[0].numel()).fill_(0)

        if self.api.cuda.active is not None:
            self.api.cuda.active.calls.append(run)
        run()


class StartupTests(unittest.TestCase):
    def fixture(self, native=True, failure=None, deadline=None):
        api = Facade()
        production_scratch = [torch.full((16,), 7, dtype=dtype)
                              for dtype in (torch.int32, torch.float32, torch.bfloat16)]
        actual = Expert(api, production_scratch, failure)
        state = {'ma_ext': {'old': 17}, 'ma_scratch': production_scratch, 'ma_w': (),
                 '_fork_native_eligible': native, 'nsm': 132}
        weights = {'w13': torch.empty(1)}

        def select(size, private):
            private['ma_ext'][size] = actual
            return actual if native else private['_fork_marlin_adapter']

        def marlin(_, private):
            private['_fork_marlin_adapter'] = Expert(api, production_scratch)

        with redirect_stdout(io.StringIO()):
            _release_expert_self_test(api, weights, state, select, lambda m, st: actual,
                                      lambda dev: [torch.zeros_like(x) for x in production_scratch],
                                      marlin, 0x5eadbeef, sizes=(4, 1), deadline=deadline)
        self.assertEqual(state['ma_ext'], {'old': 17})
        self.assertTrue(all(bool((v == 7).all()) for v in production_scratch))
        return api.cuda.replays

    def test_real_host_body_keeps_production_private(self):
        self.assertEqual(self.fixture(), 16)
        self.assertEqual(self.fixture(native=False), 16)

    def test_fail_closed(self):
        for failure in ('stale', 'remote', 'canary'):
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.fixture(failure=failure)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            self.fixture(deadline=time.monotonic() - 1)

    def test_local_id_abi_and_initialized_storage(self):
        regular, hot, partial, remote = _release_routes(torch, 33, 'cpu')
        self.assertTrue(bool((regular >= 0).all()) and int(regular.max()) == 31)
        self.assertTrue(bool((hot[0] == hot[-1]).all()))
        self.assertEqual(int((partial[1] >= 0).sum()), 0)
        self.assertEqual(int((partial[0] >= 0).sum()), 1)
        self.assertTrue(bool((remote == -1).all()))
        src = torch.tensor([-1, 9, 3], dtype=torch.int16)
        private, tails = _release_private_scratch(torch, [src], 73)
        self.assertTrue(torch.equal(src, private[0]) and bool((tails[0] == 73).all()))
        private[0].zero_()
        self.assertEqual(src.tolist(), [-1, 9, 3])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
