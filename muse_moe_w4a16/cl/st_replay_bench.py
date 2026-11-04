"""Weight-stream replay bench (docs/runs-decode-layer.md "2026-11-04 - Phase A stream efficiency (box)").

Streams per-CTA lists of 36,864-byte units with st_replay.cu (no math) and reports TB/s, so the access pattern of
helper0's phase A can be compared with alternatives on the same bytes. The unit space is helper0's: 32 local experts,
phase A unit (e, rb) at e * 512 + rb, phase B unit at 32 * 512 + e * 256 + rb.

On the box:
  . tools/box_env.sh
  python muse_moe_w4a16/cl/st_replay_bench.py --lock /root/okc/.build/box/gpu.lock --out F.json \
      [--patterns h0,rr,em] [--k 7] [--depth 3] [--delay 0,1050] [--seq recorded.json]

Patterns (G CTAs, k distinct local experts in ascending id order, L = the phase's units expert-major):
  h0     CTA c streams L[floor(N c / G), floor(N (c + 1) / G)): helper0's muldiv ranges
  rr     unit i of L to CTA i mod G
  rrb<b> blocks of b adjacent units round robin
  em     G split into k groups by muldiv, group g owns expert g, unit rb to CTA rb mod (group size)
  emc    the same groups, contiguous chunks inside the group
Suffix @G sets the CTA count (default 132), e.g. em@56 is the H skeleton's 7 x 8 geometry with static assignment.
Phases: A (512 k units), B (256 k), AB (h0-style per CTA: its A range then its B range in one ring).
"""
import argparse
import fcntl
import json
import os
import random
import statistics
import sys
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = 36864
NLOC = 32
RBA, RBB = 512, 256
FLUSH_WRITE = False
READ_RATE = 4.44e12  # box 4 GiB read, docs/runs-moe.md "T = 8 MoE floor and gap"


class GpuLock:
    def __init__(self, path):
        self.path, self.fd = path, None

    def __enter__(self):
        if self.path:
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
            t = time.monotonic()
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            print(f"LOCK acquired after {time.monotonic() - t:.1f} s", flush=True)
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            print("LOCK released", flush=True)


def build():
    bdir = os.path.expanduser("~/.cache/st_replay_ext")
    os.makedirs(bdir, exist_ok=True)
    return load(name="st_replay_ext", sources=[os.path.join(HERE, "st_replay.cu")], build_directory=bdir,
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)


def md(n, c, g):
    return (n * c) // g


def phase_units(experts, phase):
    if phase == "A":
        return [e * RBA + rb for e in experts for rb in range(RBA)]
    return [NLOC * RBA + e * RBB + rb for e in experts for rb in range(RBB)]


def assign(pattern, experts, phase, G):
    """Per-CTA unit lists."""
    if phase == "AB":
        a, b = assign(pattern, experts, "A", G), assign(pattern, experts, "B", G)
        return [x + y for x, y in zip(a, b)]
    L = phase_units(experts, phase)
    N, k = len(L), len(experts)
    per = N // k
    if pattern == "h0":
        return [L[md(N, c, G):md(N, c + 1, G)] for c in range(G)]
    if pattern == "rr":
        return [L[c::G] for c in range(G)]
    if pattern.startswith("rrb"):
        b = int(pattern[3:])
        out = [[] for _ in range(G)]
        for i in range(0, N, b):
            out[(i // b) % G].extend(L[i:i + b])
        return out
    if pattern in ("em", "emc"):
        out = [[] for _ in range(G)]
        for g in range(k):
            c0, c1 = md(G, g, k), md(G, g + 1, k)
            gs = c1 - c0
            seg = L[g * per:(g + 1) * per]
            if gs == 0:
                continue
            if pattern == "em":
                for i, u in enumerate(seg):
                    out[c0 + i % gs].append(u)
            else:
                for c in range(gs):
                    out[c0 + c].extend(seg[md(per, c, gs):md(per, c + 1, gs)])
        return out
    raise SystemExit(f"unknown pattern {pattern}")


def flat(lists):
    starts, units = [0], []
    for x in lists:
        units.extend(x)
        starts.append(len(units))
    return units, starts


def run(ext, base, flush, lists_fn, depth, delay, replays, evict_first=1, record=True, burn=0):
    lists = lists_fn(0)
    units, starts = flat(lists)
    G, N = len(lists), len(units)
    du = torch.tensor(units, dtype=torch.int32, device="cuda")
    ds = torch.tensor(starts, dtype=torch.int32, device="cuda")
    stamps = torch.zeros(G, 4, dtype=torch.int64, device="cuda")
    ready = torch.zeros(N if record else 0, dtype=torch.int64, device="cuda")
    issued = torch.zeros(N if record else 0, dtype=torch.int64, device="cuda")
    sink = torch.zeros(2 * G, dtype=torch.int32, device="cuda")
    smem = max(depth * UNIT, 118784)  # one CTA per SM, as helper0 (its smem rules out a second)

    def launch():
        ext.launch(base, du, ds, stamps, ready, issued, sink, depth, delay, evict_first, smem, burn)

    launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            launch()
    torch.cuda.synchronize()
    rows = []
    for r in range(replays + 2):
        lst = lists_fn(r)
        u2, s2 = flat(lst)
        assert s2 == starts
        du.copy_(torch.tensor(u2, dtype=torch.int32))
        if FLUSH_WRITE:
            flush.fill_(1.0)
        else:
            flush.sum()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        if r < 2:
            continue
        st = stamps.cpu().tolist()
        t0 = min(x[0] for x in st)
        done = [x[2] for x in st]
        span = (max(done) - t0) * 1e-9
        row = dict(span_us=span * 1e6, event_us=a.elapsed_time(b) * 1e3, tbs=N * UNIT / span / 1e12,
                   end_spread_us=(max(done) - min(done)) * 1e-3,
                   ramp_us=statistics.mean((x[1] - t0) * 1e-3 for x in st))
        if record:
            rd = sorted(ready.cpu().tolist())
            i10, i90 = N // 10, (9 * N) // 10
            row["steady_tbs"] = (i90 - i10) * UNIT / ((rd[i90] - rd[i10]) * 1e-9) / 1e12
            ru = ready.cpu().tolist()
            iv = []
            for c in range(G):
                n = starts[c + 1] - starts[c]
                if n > 1:
                    iv.append((ru[starts[c + 1] - 1] - ru[starts[c]]) / (n - 1) * 1e-3)
            row["cta_unit_us"] = statistics.median(iv) if iv else 0.0
            row["cta_unit_mean_us"] = statistics.mean(iv) if iv else 0.0
            isd = issued.cpu().tolist()
            lat = sorted((r_ - i_) * 1e-3 for r_, i_ in zip(ru, isd))
            row["lat_med_us"], row["lat_p90_us"] = lat[len(lat) // 2], lat[(9 * len(lat)) // 10]
        rows.append(row)
    keys = rows[0].keys()
    med = {k: statistics.median(r[k] for r in rows) for k in keys}
    med["min_span_us"] = min(r["span_us"] for r in rows)
    med["max_span_us"] = max(r["span_us"] for r in rows)
    med["units"], med["grid"] = N, G
    return med, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--patterns", default="h0,rr,em,emc,rrb3,em@56,h0@56")
    ap.add_argument("--phases", default="A,B")
    ap.add_argument("--k", default="7")
    ap.add_argument("--depth", default="3,4")
    ap.add_argument("--delay", default="0,1050")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seq", default="", help="JSON with recorded per-CTA unit lists: {name: [[unit, ...], ...]}")
    ap.add_argument("--evict-first", type=int, default=1)
    ap.add_argument("--burn", default="0", help="extra warps running an integer + mma.sync loop (power load)")
    ap.add_argument("--flush", default="read", choices=("read", "write"),
                    help="1 GiB L2 flush before each replay: read (sum, as the judge) or write (fill_, leaves L2 dirty)")
    a = ap.parse_args()
    global FLUSH_WRITE
    FLUSH_WRITE = a.flush == "write"
    print(f"FLUSH {a.flush}", flush=True)
    ext = build()
    with GpuLock(a.lock):
        nunits = NLOC * (RBA + RBB)
        base = torch.randint(0, 255, (nunits * UNIT,), dtype=torch.uint8, device="cuda")
        flush = torch.ones(1 << 28, dtype=torch.float32, device="cuda")
        buf = torch.ones(1 << 30, dtype=torch.float32, device="cuda")
        ts = []
        for _ in range(7):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); buf.sum(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e-3)
        del buf
        rate = (4 << 30) / min(ts[2:])
        print(f"READ 4 GiB sum {rate / 1e12:.3f} TB/s", flush=True)
        results = []
        rng = random.Random(a.seed)
        if a.seq:
            recs = json.load(open(a.seq))
            for name, lists in recs.items():
                for depth in [int(x) for x in a.depth.split(",")]:
                    for delay in [int(x) for x in a.delay.split(",")]:
                        med, rows = run(ext, base, flush, lambda r, L=lists: L, depth, delay, a.replays,
                                        a.evict_first)
                        med.update(pattern="seq:" + name, phase="rec", k=-1, depth=depth, delay=delay)
                        results.append(dict(summary=med, rows=rows))
                        pr(med, rate)
        for k in [int(x) for x in a.k.split(",")]:
            sets = [sorted(rng.sample(range(NLOC), k)) for _ in range(a.replays + 2)]
            for phase in a.phases.split(","):
                for pat in a.patterns.split(","):
                    name, G = (pat.split("@") + ["132"])[:2]
                    G = int(G)
                    for depth in [int(x) for x in a.depth.split(",")]:
                        for delay in [int(x) for x in a.delay.split(",")]:
                            for burn in [int(x) for x in a.burn.split(",")]:
                                fn = lambda r, n=name, ph=phase, g=G: assign(n, sets[r], ph, g)  # noqa: E731
                                med, rows = run(ext, base, flush, fn, depth, delay, a.replays, a.evict_first,
                                                burn=burn)
                                med.update(pattern=pat, phase=phase, k=k, depth=depth, delay=delay, burn=burn)
                                results.append(dict(summary=med, rows=rows))
                                pr(med, rate)
        if a.out:
            json.dump(dict(read_rate=rate, results=results), open(a.out, "w"))


def pr(m, rate):
    print(f"ST {m['pattern']:>8} {m['phase']:>2} k {m['k']:2d} D {m['depth']} delay {m['delay']:4d} grid {m['grid']:3d}"
          f" burn {m.get('burn', 0):2d} units {m['units']:5d} span {m['span_us']:7.2f} [{m['min_span_us']:.2f}-{m['max_span_us']:.2f}]"
          f" TB/s {m['tbs']:.3f} ({100 * m['tbs'] * 1e12 / rate:5.1f}%) steady {m.get('steady_tbs', 0):.3f}"
          f" ({100 * m.get('steady_tbs', 0) * 1e12 / rate:5.1f}%) ramp {m['ramp_us']:.2f}"
          f" endspread {m['end_spread_us']:.2f} cta_unit {m.get('cta_unit_us', 0):.3f} mean {m.get('cta_unit_mean_us', 0):.3f}"
          f" lat {m.get('lat_med_us', 0):.2f}/{m.get('lat_p90_us', 0):.2f}", flush=True)


if __name__ == "__main__":
    main()
