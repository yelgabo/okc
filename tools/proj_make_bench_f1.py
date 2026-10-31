"""Write the F-PROJ-1 bench: our re-implementations of two donor mechanisms (alygg77
okc-muse-spark-kernels @ 6cae49a, kernels/muse_bf16_proj.py, Apache-2.0) against our release,
same process, paired, with independent duplicate nulls. No donor code is in the bench.

    python3 tools/proj_make_bench_f1.py OUT.py

1. sk at m <= 32: escape corrections read from x in global memory, the escape warps off the
   ring (SK_ESCG, the donor's `_escape_source` scheme), and the donor's sk2 geometries where they
   differ from ours (two units per warp, NACC 2, the stage count its ring cap allows).
2. pooler m = 128: the donor's pinned cuBLASLt config (layout 1 = K-major copy of W, tile id 15,
   custom option 4, cluster id 2, stages id 35) and its neighbours, against our pinned
   preference.

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). Arms per case: "prod" (the production extension's config),
"<name>c" (the same config from the bench extension: the null) and the variants. Every arm is
checked against FP64 with probe rows at adjacent sizes, then replayed twice from a CUDA graph on
fresh inputs. Timing: cold-L2 graph replays, 100 rounds (pooler 128: 60) in a seeded random
order. Lines:

  F1 <layer> m=<m> med      median us per arm after the replay overhead
  F1 <layer> m=<m> d prod   paired median difference arm - prod (us) / rounds faster
  F1CHECK                   dropped arms and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Round f1-a (check a63fa0c391fd40fc): global-x escapes, the donor's sk2 geometries and the
# pooler m = 128 cuBLASLt configs. Round f2-a: the sk codegen differences an independent read of
# both kernels found (compile-time M, escape-loop loads, consumer wait watchdog), an escape-off
# attribution arm, two nulls at qkv8 (its f1-a null was 0.45 us), and the pooler config again.
# Round f2-a (check 6d5554de901143d0) dropped every FULL arm: the FP64 check ran them at adjacent
# sizes, which FULL configs refuse. Round f3-a checks FULL arms at m = TOKR only, adds the
# watchdog-free consumer wait alone and with FULL, and keeps the escape-off attribution arm.
# Round f3-a (check a8de7b162c9f41d5): the watchdog-free consumer wait (n) was the largest sk
# lever (qkv32 -0.51, with FULL -0.77). Round f4-a adds LITEWD (l: the clock read once per 256
# polls, still failing closed), alone and with FULL, against n / fn.
# Round f4-a (check 8712eb94a71e4075): LITEWD kept the vote and recovered little; FULL + no
# watchdog (fn) won up to 0.96 us. Round f5-a: NOVOTE (v: per-thread wait, no vote, no break,
# clock every 256 polls, still fails closed through diag), alone and with FULL, against fn.
# Round f5-a (check eebd764690d5470f) chose NOVOTE + FULL (candidate C1). Round f6-a: the
# register-direct 12-bit kernel rd (smallm2 / v18s / decodediet mechanism) against C1's sk pick
# at pooler 8 / 16 and o 8, rd from the production and the bench extension (the null).
# Round f6-a (check bcabe48caa884e67): rd won at pooler 8 (-0.54 / -0.62) and o 8 (-0.38),
# lost at pooler 16 (+1.1, one unit in flight). Round f7-a: deeper register prefetch (PF), rd on
# qkv_gate 8 / 16 and the unpooler slice, and the two production rd configs again (null: the
# same config from both extensions). The judged path is candidate C2 (C1 + rdp8 / rdo8).
# Round f7-a (check 1759290da58047ab) closed the rd shape search. Round f8-a: the codec
# factorial on C2's shipped sk picks (m = TOKR, so the FULL twins): bytes only (tile 736 / 704
# bytes, outputs garbage, timing only), decode only (+16 / +32 ALU per tile, +16 ALU and 8 LDS),
# and both (p7 = 736 + 16 ALU + 8 LDS, the pair7 model; q12 = 704 + 16 ALU + 8 LDS), with a
# duplicate null of the production config (xc).
ROUND = "f8-a"
NV, FU = 1024, 32
FACT = {"xc": 0, "y736": 2048, "y704": 4096, "a16": 8192, "a32": 16384, "l8": 8192 | 32768,
        "p7": 2048 | 8192 | 32768, "q12": 4096 | 8192 | 32768}
TIMING_ONLY = ("y736", "y704", "p7", "q12")
VARIANTS = {}
CASES = []
for _layer, _m, _base in (("qkv_gate", 8, "qkv8"), ("unpooler", 8, "slice8"), ("qkv_gate", 16, "qkv16"),
                          ("o", 16, "o16"), ("pooler", 16, "pool16"), ("unpooler", 16, "unp16"),
                          ("qkv_gate", 32, "qkv32"), ("o", 32, "o32"), ("pooler", 32, "pool32"),
                          ("unpooler", 32, "unp32")):
    _prod = 1 | NV | (0 if _base == "qkv32" else FU)
    _arms = []
    for _tag, _bits in FACT.items():
        VARIANTS[_base + _tag] = (_base, _prod | _bits)
        _arms.append(_base + _tag)
    CASES.append((_layer, _m, _base, _arms))
FULL_TAGS = tuple(FACT)
LT_ARMS = {}
RD_EXTRA = {}
RD_ARMS = {}

BENCH = r'''

_F1_ROUND = @@ROUND@@
_F1_VARIANTS = @@VARIANTS@@
_F1_CASES = @@CASES@@
_F1_LT = @@LT@@
_F1_FULL = @@FULL@@
_F1_RD = @@RD@@
_F1_RDX = @@RDX@@
_F1_TONLY = @@TONLY@@
import os as _f1os
import random as _f1random
import math as _f1math

_F1_OUT = []


def _f1print(s):
    print(s, flush=True)
    _F1_OUT.append(s)


def _f1_configs():
    """Bench SK_CONFIGS: a copy of each production pick (the null) and every variant."""
    prod = {c[0]: c for c in SK_CONFIGS}
    cfgs = []
    for layer, m, base, names in _F1_CASES:
        c = prod[base]
        cfgs.append((base + "c",) + tuple(c[1:]))
        for nm in names:
            geo, nacc = _F1_VARIANTS[nm]
            if isinstance(geo, str):
                g = prod[geo]
                mt, tokr, nt, nwr, wsk, upw, nstage = g[1:8]
            else:
                mt, tokr, nt, nwr, wsk, upw, nstage = geo
                nstage = _fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, True, False, 1)
            cfgs.append((nm, mt, tokr, nt, nwr, wsk, upw, nstage, True, False, nacc, 1, "bf16", False))
    return cfgs


def _f1_geom(c):
    mt, tokr, nt, nwr, wsk, upw = c[1:7]
    return {"R": nwr * nt * 8, "NT": nt, "NWR": nwr, "WSK": wsk, "UPW": upw, "NW": nwr * wsk,
            "UPS": wsk * upw, "comp": True, "hl": False, "cl": 1, "tokr": tokr, "fuse": False}


def _f1_build_bench(cfgs):
    g = globals()
    saved = {nm: g[nm] for nm in ("SK_CONFIGS", "SK_INDEX", "RD_GEOM", "RD_NAMES")}
    try:
        g["SK_CONFIGS"] = cfgs
        g["SK_INDEX"] = {c[0]: i for i, c in enumerate(cfgs)}
        g["RD_GEOM"] = {**saved["RD_GEOM"], **_F1_RDX}
        g["RD_NAMES"] = list(g["RD_GEOM"])
        src = sk_source() + rd_source()
    finally:
        g.update(saved)
    ext = load_inline(name="okc_f1_bench", cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + _RD_CPP,
                      cuda_sources=src, functions=["sk_fwd", "rd_fwd"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False,
                      no_implicit_headers=True)
    return ext, {c[0]: i for i, c in enumerate(cfgs)}


def _f1_sk_run(ext, cid, stream, tab, pmax, base4, n, k, diag):
    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.sk_fwd(cid, stream, x, out, tab, pmax, base4, n, k, diag)
        return out
    return run


def _f1_graph(fn, xbuf):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(xbuf)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn(xbuf)
    torch.cuda.synchronize()
    return g, out


def _f1_time(graphs, rounds, flush, seed):
    rng = _f1random.Random(seed)
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


def _f1_measure(label, arms, w, n, ms, m, rounds, seed, flush, ovh, gen, dev):
    """FP64 checks at ms, graph replay on fresh inputs, then paired timing against "prod"."""
    k = w.shape[1]
    wd = w[:n].double()
    bad = []
    for mm in ms:
        x, probes = _gen_x(mm, k, gen, dev)
        ref = x.double() @ wd.T
        for nm in list(arms):
            if (mm != m and nm.endswith(_F1_FULL)) or nm.endswith(_F1_TONLY):
                continue
            try:
                out = arms[nm](x)
                torch.cuda.synchronize()
                ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
                if not ok:
                    raise RuntimeError(f"m={mm} err {err:.1e}/{perr:.1e}")
            except Exception as exc:
                torch.cuda.synchronize()
                bad.append(f"{nm}={str(exc)[-80:]}")
                arms.pop(nm)
    xbuf, _ = _gen_x(m, k, gen, dev)
    graphs = {}
    for nm in list(arms):
        try:
            g, out = _f1_graph(arms[nm], xbuf)
            for rep in range(2):
                x2, probes = _gen_x(m, k, gen, dev)
                xbuf.copy_(x2)
                g.replay()
                torch.cuda.synchronize()
                ok, err, perr = _check(out, x2.double() @ wd.T, probes, SELFTEST_TOL["bf16"])
                if not ok and not nm.endswith(_F1_TONLY):
                    raise RuntimeError(f"replay {rep} err {err:.1e}/{perr:.1e}")
            graphs[nm] = g
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{nm}@replay={str(exc)[-80:]}")
    if bad:
        _f1print(f"F1CHECK {label} dropped: " + " ".join(bad))
    if "prod" not in graphs:
        _f1print(f"F1 {label}: prod dropped")
        return
    smp = _f1_time(graphs, rounds, flush, seed)
    med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
    dl = {nm: statistics.median([a - b for a, b in zip(v, smp["prod"])]) for nm, v in smp.items()}
    wins = {nm: sum(a < b for a, b in zip(v, smp["prod"])) for nm, v in smp.items()}
    order = sorted(med, key=med.get)
    _f1print(f"F1 {label} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
    _f1print(f"F1 {label} d prod " + " ".join(f"{nm}={dl[nm]:+.2f}/{wins[nm]}" for nm in order if nm != "prod"))


def _f1_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    prod = _build_ext()
    t1 = time.time()
    cfgs = _f1_configs()
    bench, bidx = _f1_build_bench(cfgs)
    bcfg = {c[0]: c for c in cfgs}
    _f1print(f"F1BUILD prod {t1 - t0:.1f} s, bench {time.time() - t1:.1f} s, {len(cfgs)} sk configs, "
             f"round {_F1_ROUND}: " + " ".join(f"{c[0]}:s{c[7]}a{c[10]}" for c in cfgs))
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _f1_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _f1_graph(two_fn, None)
    ov = _f1_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _f1print(f"F1OVH {ovh:.2f} us")
    gen = torch.Generator(device=dev).manual_seed(4747)
    shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
    weights = {nm: (torch.randn(s, generator=gen, device=dev) / _f1math.sqrt(s[1])).to(torch.bfloat16).contiguous()
               for nm, s in shapes.items()}
    adj = {8: (1, 5, 8), 16: (9, 16), 32: (17, 32)}
    for ci, (layer, m, base, names) in enumerate(_F1_CASES):
        try:
            w = weights[layer]
            n = UNPOOL_SLICE if layer == "unpooler" and m <= 8 else w.shape[0]
            k = w.shape[1]
            arms_cfg = [base, base + "c"] + list(names)
            geoms = {nm: _f1_geom(bcfg[nm]) for nm in arms_cfg[1:]}
            geoms[base] = _f1_geom(bcfg[base + "c"])
            slices = {nm: n for nm in geoms}
            packed = pack_layer(w, geoms, slices, balance=layer in BALANCED_LAYERS)
            if packed is None:
                _f1print(f"F1 {layer} m={m}: matrix rejected by pack_layer")
                continue
            arms = {}
            for nm in arms_cfg:
                g = geoms[nm]
                tab, pmax = packed["tabs"][(g["R"], n)]
                stream = packed["streams"][nm]
                if nm == base:
                    # The production runner, so C1's M == TOKR twin is what runs.
                    arms["prod"] = _sk_runner(prod, base, stream, tab, pmax, packed["base4"], n, k, torch.bfloat16,
                                              diag)
                elif nm == base + "c" and _F1_RD:
                    continue
                else:
                    arms[nm] = _f1_sk_run(bench, bidx[nm], stream, tab, pmax, packed["base4"], n, k, diag)
            bench_rd = list({**RD_GEOM, **_F1_RDX})
            for rnm in _F1_RD.get((layer, m), []):
                g_ = globals()
                saved_geom = g_["RD_GEOM"]
                g_["RD_GEOM"] = {**saved_geom, **_F1_RDX}
                try:
                    pk = rd_pack(w[:n], rnm)
                finally:
                    g_["RD_GEOM"] = saved_geom
                if pk is None:
                    _f1print(f"F1 {layer} m={m}: {rnm} not packable")
                    continue
                _f1print(f"F1 {layer} m={m} {rnm}: {pk['escapes']} escapes, max {pk['max_esc']} per warp")
                for tag, ex in (("", prod), ("c", bench)):
                    if ex is prod and rnm not in RD_NAMES:
                        continue
                    cid = RD_NAMES.index(rnm) if ex is prod else bench_rd.index(rnm)

                    def rd_run(x, pk=pk, cid=cid, ex=ex):
                        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
                        ex.rd_fwd(cid, pk["stream"], pk["esc"], x, out, n, k, pk["base4"])
                        return out
                    arms[rnm + tag] = rd_run
            _f1_measure(f"{layer} m={m}", arms, w, n, adj[m], m, 100, 8000 + ci, flush, ovh, gen, dev)
        except Exception as exc:
            import traceback as _tb
            _f1print(f"F1 {layer} m={m} failed: {type(exc).__name__}: {str(exc)[:200]} {_tb.format_exc()[-500:]}")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if _F1_LT:
        try:
            lt = _build_lt()
            w = weights["pooler"]
            n, k = w.shape
            ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)
            wt = w.t().contiguous()
            fallback = lambda x: x @ w.t()
            arms = {}
            pref = _lt_pref(lt, {"pooler": w, "lt_ws": ws}, "pooler", 128, gen, diag, fallback)
            if pref is not None:
                arms["prod"] = pref[1]
                dup = _lt_pref(lt, {"pooler": w, "lt_ws": ws}, "pooler", 128, gen, diag, fallback)
                if dup is not None:
                    arms["prefdup"] = dup[1]
            for nm, (layout, cfg) in _F1_LT.items():
                p = lt.lt_plan_new(128, n, k, layout, ws.numel())
                idx = lt.lt_add_config(p, list(cfg)) if p >= 0 else -1
                if idx < 0:
                    _f1print(f"F1 pooler m=128: {nm} rejected")
                    continue
                _f1print(f"F1 pooler m=128 cfg {nm} L{layout}:{_lt_cfg_str(list(lt.lt_info(p, idx)))}")
                arms[nm] = _lt_runner(lt, p, idx, wt if layout % 2 else w, 128, n, ws, fallback)
            _f1_measure("pooler m=128", arms, w, n, (128,), 128, 60, 9000, flush, ovh, gen, dev)
        except Exception as exc:
            import traceback as _tb
            _f1print(f"F1 pooler m=128 failed: {type(exc).__name__}: {str(exc)[:200]} {_tb.format_exc()[-500:]}")
    _f1print(f"F1DONE {time.time() - t0:.1f} s")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _f1os.environ.get("PROJ_F1_DONE"):
    _f1os.environ["PROJ_F1_DONE"] = "1"
    try:
        _f1_run()
    except Exception as _exc:
        import traceback as _tb
        _f1print(f"F1BENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _f1_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _f1_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("prepare done", "sk build", "ltune pooler")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _F1_OUT:
            print(line, flush=True)
        return state
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    bench = (BENCH.replace("@@ROUND@@", repr(ROUND)).replace("@@VARIANTS@@", repr(VARIANTS))
             .replace("@@CASES@@", repr(CASES)).replace("@@LT@@", repr(LT_ARMS))
             .replace("@@FULL@@", repr(FULL_TAGS))
             .replace("@@RD@@", repr(RD_ARMS)).replace("@@RDX@@", repr(RD_EXTRA))
             .replace("@@TONLY@@", repr(TIMING_ONLY)))
    out.write_text(sub + bench)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
