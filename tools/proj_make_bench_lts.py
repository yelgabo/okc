"""Write the cuBLASLt large-tile split-K bench (round 6): does a bigger tile with an FP32-reduced
K split beat the production pick on qkv_gate (and o) at m = 128 / 256?

    python3 tools/proj_make_bench_lts.py OUT.py

Why: in bench_bg (check cab546f978974515) four strided-batched qkv_gate m = 256 instances ran
at 12.46 us each on cuBLASLt's own 144x128 tile against 17.47 us for one instance on the
production 72x64 tile, so the single-instance loss is the small tile's per-k-step efficiency,
not ramp or tail. One instance has only 32 tiles of 144x128; the production search (lt_sweep at
split-K 1, lt_expand around the top heuristic answers) never pairs that tile with a K split.

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). Per (layer, m) it enumerates explicit configs of algorithm
66 over every layout, tiles TILES, split-K SPLITS (FP32 compute-type reduction, as the
production filter requires), cluster ids, custom options and stage ids; screens every accepted
one by cold-L2 direct launches; checks the best SCREEN_KEEP against an FP64 reference (sampled
rows, probe rows included) and times them with the production preference and an independently
created duplicate of it by cold-L2 graph replay, 60 rounds in a seeded random order. Lines:

  LTS <layer> m=<m> screen   accepted / tried, the best screened configs
  LTS <layer> m=<m> med      median us per finalist after the replay overhead
  LTS <layer> m=<m> d pref   paired median difference finalist - pref (us) / rounds faster
  LTSCHECK                   dropped finalists and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUND = "lts-a"
# cublasLtMatmulTile_t ids (CUDA 12.8 header): 18 128x64, 20 128x128, 24 256x128, 124 72x64,
# 134 80x64, 183 144x64, 184 144x128.
TILES = (18, 20, 24, 124, 134, 183, 184)
SPLITS = (1, 2, 3, 4, 6)
CLUSTERS = (0, 2, 5)
CUSTOM = (0, 2)
STAGES = (0, 35)
TARGETS = (("qkv_gate", 256), ("o", 256), ("qkv_gate", 128))

BENCH = r'''

_LTS_ROUND = @@ROUND@@
_LTS_GRID = @@GRID@@
_LTS_TARGETS = @@TARGETS@@
_LTS_SCREEN_KEEP = 8
import os as _ltsos
import random as _ltsrandom
import math as _ltsmath

_LTS_OUT = []


def _ltsprint(s):
    print(s, flush=True)
    _LTS_OUT.append(s)


def _lts_graph(fn, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn(x)
    torch.cuda.synchronize()
    return g, out


def _lts_time(graphs, rounds, flush, seed):
    rng = _ltsrandom.Random(seed)
    names = list(graphs)
    samples = {n: [] for n in names}
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        evs = []
        for n in order:
            flush.sum()
            a = torch.cuda.Event(enable_timing=True)
            z = torch.cuda.Event(enable_timing=True)
            a.record()
            graphs[n].replay()
            z.record()
            evs.append((n, a, z))
        torch.cuda.synchronize()
        for n, a, z in evs:
            samples[n].append(1000.0 * a.elapsed_time(z))
    return samples


def _lts_case(lt, layer, m, flush, ovh, dev, gen, diag):
    n, k = {"qkv_gate": (2304, 8192), "o": (8192, 1024)}[layer]
    w = (torch.randn((n, k), generator=gen, device=dev) / _ltsmath.sqrt(k)).to(torch.bfloat16).contiguous()
    wt = w.t().contiguous()
    ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)
    fallback = lambda x: x @ w.t()
    cands = {}
    tried = 0
    tiles, splits, clusters, custom, stages = _LTS_GRID
    for layout in LT_LAYOUTS:
        p = lt.lt_plan_new(m, n, k, layout, ws.numel())
        if p < 0:
            continue
        for t in tiles:
            for sk in splits:
                for cl in clusters:
                    for cu in custom:
                        for st in stages:
                            tried += 1
                            cfg = [66, t, sk, 2 if sk > 1 else 0, 0, cu, st, 0, cl]
                            idx = lt.lt_add_config(p, cfg)
                            if idx < 0:
                                continue
                            info = list(lt.lt_info(p, idx))
                            key = f"L{layout}:" + _lt_cfg_str(info)
                            if key not in cands:
                                cands[key] = _lt_runner(lt, p, idx, wt if layout % 2 else w, m, n, ws, fallback)
    x, probes = _gen_x(m, k, gen, dev)
    names = list(cands)
    t_scr = _lt_screen([cands[nm] for nm in names], x, flush, 4)
    ranked = sorted(range(len(names)), key=lambda i: t_scr[i])
    _ltsprint(f"LTS {layer} m={m} screen {len(names)} accepted / {tried} tried; best "
              + " ".join(f"{names[i]} {t_scr[i]:.1f}" for i in ranked[:10]))
    fins = {}
    pref = _lt_pref(lt, {layer: w, "lt_ws": ws}, layer, m, gen, diag, fallback)
    if pref is not None:
        fins["pref"] = pref[1]
        dup = _lt_pref(lt, {layer: w, "lt_ws": ws}, layer, m, gen, diag, fallback)
        if dup is not None:
            fins["prefdup"] = dup[1]
        _ltsprint(f"LTS {layer} m={m} pref L{pref[0][0]}:{_lt_cfg_str(pref[0][1:])}")
    else:
        # qkv_gate m = 128 ships the wd kernel (b1n128c3s8h), not a cuBLASLt config.
        ext = _build_ext()
        cid = WD_INDEX["b1n128c3s8h"]

        def wd_run(x, ext=ext, cid=cid):
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            ext.wd_fwd(cid, w, x, out, n, k, diag)
            return out
        fins["pref"] = wd_run
        _ltsprint(f"LTS {layer} m={m} pref: wd b1n128c3s8h (production pick)")
    fins["torch"] = fallback
    for i in ranked[:_LTS_SCREEN_KEEP]:
        fins[names[i]] = cands[names[i]]
    bad = []
    ref = x.double() @ w.double().t()
    for nm in list(fins):
        try:
            out = fins[nm](x)
            torch.cuda.synchronize()
            ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
            if not ok:
                raise RuntimeError(f"err {err:.1e}/{perr:.1e}")
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{nm}={str(exc)[-80:]}")
            fins.pop(nm)
    graphs = {}
    for nm in list(fins):
        try:
            g, out = _lts_graph(fins[nm], x)
            x2, probes2 = _gen_x(m, k, gen, dev)
            x.copy_(x2)
            g.replay()
            torch.cuda.synchronize()
            ok, err, perr = _check(out, x2.double() @ w.double().t(), probes2, SELFTEST_TOL["bf16"])
            if not ok:
                raise RuntimeError(f"replay err {err:.1e}/{perr:.1e}")
            graphs[nm] = g
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{nm}@replay={str(exc)[-80:]}")
    if bad:
        _ltsprint(f"LTSCHECK {layer} m={m} dropped: " + " ".join(bad))
    if "pref" not in graphs:
        _ltsprint(f"LTS {layer} m={m}: reference arm dropped")
        return
    smp = _lts_time(graphs, 60, flush, 7000 + m + len(layer))
    med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
    dl = {nm: statistics.median([a - b for a, b in zip(v, smp["pref"])]) for nm, v in smp.items()}
    wins = {nm: sum(a < b for a, b in zip(v, smp["pref"])) for nm, v in smp.items()}
    order = sorted(med, key=med.get)
    _ltsprint(f"LTS {layer} m={m} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
    _ltsprint(f"LTS {layer} m={m} d pref " + " ".join(f"{nm}={dl[nm]:+.2f}/{wins[nm]}" for nm in order
                                                      if nm != "pref"))
    graphs = None
    torch.cuda.empty_cache()


def _lts_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    lt = _build_lt()
    _ltsprint(f"LTSBUILD {time.time() - t0:.1f} s, cublasLt {lt.lt_version()}, round {_LTS_ROUND}")
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _lts_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _lts_graph(two_fn, None)
    ov = _lts_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _ltsprint(f"LTSOVH {ovh:.2f} us")
    gen = torch.Generator(device=dev).manual_seed(4646)
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    for layer, m in _LTS_TARGETS:
        try:
            _lts_case(lt, layer, m, flush, ovh, dev, gen, diag)
        except Exception as exc:
            import traceback as _tb
            _ltsprint(f"LTS {layer} m={m} failed: {type(exc).__name__}: {str(exc)[:200]} {_tb.format_exc()[-600:]}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _ltsprint(f"LTSDONE {time.time() - t0:.1f} s")


if not _ltsos.environ.get("PROJ_LTS_DONE"):
    _ltsos.environ["PROJ_LTS_DONE"] = "1"
    try:
        _lts_run()
    except Exception as _exc:
        import traceback as _tb
        _ltsprint(f"LTSBENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _lts_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _lts_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("ltune qkv", "ltune o ", "prepare done")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _LTS_OUT:
            print(line, flush=True)
        return state
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    grid = (TILES, SPLITS, CLUSTERS, CUSTOM, STAGES)
    out.write_text(sub + BENCH.replace("@@ROUND@@", repr(ROUND)).replace("@@GRID@@", repr(grid))
                   .replace("@@TARGETS@@", repr(TARGETS)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
