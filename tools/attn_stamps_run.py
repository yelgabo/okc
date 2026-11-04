#!/usr/bin/env python3
"""Run a stamps build (tools/attn_stamps_make.py) through bx_box_run.py, then read the attention kernel's per-CTA
stamps left by the last replay of the last cell: entry (slot 0, before the PDL wait), first tile consumed (12),
loop end (3), merge done (4), segment 0 / 1 start and end (16, 17 / 20, 21), SM id (15). Times are globaltimer
ns; printed relative to the earliest entry. Run one load at a time (the full cell is the last one of a load).

    ATTN_STAMPS_OUT=.build/box/attn/stamps_d32 python tools/attn_stamps_run.py <bx_box_run.py arguments>
"""
import os
import runpy
import statistics
import sys

import torch

OUT = os.environ.get("ATTN_STAMPS_OUT", ".build/box/attn/stamps")
STRIDE = 64

sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "bx_box_run.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")

mods = [m for m in list(sys.modules.values()) if isinstance(getattr(m, "_ATTN_DBG", None), torch.Tensor)]
assert mods, "no module with _ATTN_DBG stamps"
d = mods[0]._ATTN_DBG.cpu().view(-1, STRIDE)
torch.save(d, OUT + ".pt")
live = (d[:, 0] != 0).nonzero().flatten()
n = int(live.max()) + 1 if live.numel() else 0
d = d[:n]
print(f"STAMPS {n} CTAs with an entry stamp")
t0 = int(d[:, 0].min())


def col(k):
    return [(int(x) - t0) / 1e3 for x in d[:, k]]


def show(name, xs):
    xs = sorted(xs)
    print(f"  {name:28s} min {xs[0]:8.2f}  p10 {xs[len(xs) // 10]:8.2f}  med {statistics.median(xs):8.2f}  "
          f"p90 {xs[9 * len(xs) // 10]:8.2f}  max {xs[-1]:8.2f}")


entry, first, lend, merged = col(0), col(12), col(3), col(4)
s0s, s0e, s1s, s1e = col(16), col(17), col(20), col(21)
print("per-CTA times in us from the earliest entry:")
show("entry", entry)
show("first tile consumed", first)
show("first tile - entry", [b - a for a, b in zip(entry, first)])
show("loop end (last segment)", lend)
show("merge done", merged)
show("merge done - loop end", [b - a for a, b in zip(lend, merged)])
nseg2 = sum(1 for x in d[:, 20] if int(x) != 0)
print(f"  CTAs with a second segment: {nseg2} of {n}")
show("segment 0 start", s0s)
show("segment 0 end", s0e)
sm = [int(x) for x in d[:, 15]]
pairs = {}
for c, s in enumerate(sm):
    pairs.setdefault(s, []).append(c)
gaps = []
for s, cs in pairs.items():
    if len(cs) == 2:
        gaps.append(abs(merged[cs[0]] - merged[cs[1]]))
print(f"  SMs used: {len(pairs)}; SMs with two CTAs: {sum(1 for v in pairs.values() if len(v) == 2)}; "
      f"with more: {sum(1 for v in pairs.values() if len(v) > 2)}")
if gaps:
    show("pair end gap", gaps)
sm_end = [max(merged[c] for c in cs) for cs in pairs.values()]
show("per-SM end (last CTA)", sm_end)
rate = []
for c in range(n):
    span = merged[c] - first[c]
    rate.append(span)
show("first tile -> merge", rate)
