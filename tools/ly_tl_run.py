"""bx_box_run.py with --kernels also writing each cell's CUDA trace (chrome JSON) for a kernel timeline.

Usage: LY_TRACE_DIR=<dir> python tools/ly_tl_run.py <bx_box_run.py arguments, including --kernels>
Cell traces land in <dir>/cell<N>.json in case order; tools/ly_tl_parse.py turns them into per-kernel
start, duration and the gap to the previous kernel for each of the ten profiled replays.
"""
import os
import runpy
import sys

import torch.profiler as tp

_DIR = os.environ["LY_TRACE_DIR"]
os.makedirs(_DIR, exist_ok=True)
_N = [0]
_Base = tp.profile


class _Traced(_Base):
    def __exit__(self, *exc):
        out = super().__exit__(*exc)
        self.export_chrome_trace(os.path.join(_DIR, f"cell{_N[0]}.json"))
        _N[0] += 1
        return out


tp.profile = _Traced
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "bx_box_run.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
