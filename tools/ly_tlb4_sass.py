#!/usr/bin/env python3
"""tools/ly_micro_sass.py with two extra B4 builds: b4tl (the B4 defines plus -DOKC_TLB=1, the timeline build of
tools/ly_moe_tlb4.py) and b4x (the B4 defines plus the words after --extra), and every nvcc warning printed.

    CUDA_VISIBLE_DEVICES= python tools/ly_tlb4_sass.py FILE OUTDIR [build ...] [--extra -DX=1 ...]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ly_micro_sass as ms  # noqa: E402

extra = []
if "--extra" in sys.argv:
    k = sys.argv.index("--extra")
    extra = sys.argv[k + 1:]
    del sys.argv[k:]
_builds = ms.builds
ms.builds = lambda m: {**_builds(m), "b4tl": _builds(m)["b4"] + ("-DOKC_TLB=1",), "b4x": _builds(m)["b4"] + tuple(extra)}
_run = subprocess.run


def run(cmd, *a, **k):
    r = _run(cmd, *a, **k)
    if cmd and str(cmd[0]).endswith("nvcc") and r.stderr and "arning" in r.stderr:
        print("NVCC", os.path.basename(cmd[-1]), r.stderr.strip()[-3000:], flush=True)
    return r


ms.subprocess.run = run
ms.main()
