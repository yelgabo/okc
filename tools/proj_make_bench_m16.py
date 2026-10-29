"""Write a kernel-level A/B bench for muse_bf16_proj at m = 16 and 32: the control submission,
copies of it with extra skinny geometries (SK_EXTRA candidates), and the public entry, timed in
one process on the same weights and inputs.

    python3 tools/proj_make_bench_m16.py OUT.py --pub PUBLIC.py [--variants geo] [--m 16,32]

Arms per (layer, m), the four BF16 layers:
  sgl          SGLang's call.
  ours, ours2  the control's kernel_fn, captured twice: ours2 - ours is the null difference.
  ours.<pick>  the control's candidate for the bucket, called directly.
  <variant>    the patched copy's kernel_fn (its own tuner's pick).
  <variant>.<nm>  each extra geometry of the patched copy, called directly.
  pub          the public entry's kernel_fn.

Rounds visit the arms in a fresh seeded random order (see proj_make_bench_m8.py for why).
Output lines: ABCHECK (relative error against FP64 per arm), ABCOLD (cold-L2 medians, us),
ABPAIR (paired per-round differences against ours and pub, median [25th, 75th]).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from proj_make_bench_ab import pub_patch  # noqa: E402
from proj_make_bench_gemv import keep_cands  # noqa: E402
from proj_make_bench_m8 import patch  # noqa: E402

# Extra geometries, name -> (MT, TOKR, NT, NWR, WSK, UPW, NSTAGE). Stage counts over K and
# ring depth after _fit_stages are in the comments (compressed build).
EXTRA_GEOM = {
    # qkv_gate: 96 CTAs (R = 24) on 132 SMs, 16 stages of 8 units.
    "qkv16u4": (1, 16, 3, 1, 4, 4, 5),     # 8 stages, ring 3
    "qkv16w8": (1, 16, 3, 1, 8, 2, 4),     # 8 consumer warps, 8 stages, ring 3
    "qkv16w8u1": (1, 16, 3, 1, 8, 1, 5),   # 8 consumer warps, 16 stages, ring 5
    "qkv16r3": (1, 16, 1, 3, 4, 2, 5),     # 12 consumer warps, one 8-row tile each, ring 5
    # R = 16: 144 CTAs at 2 per SM (89 KB), every SM busy if the limit is per CTA, not per SM.
    "qkv16n2": (1, 16, 2, 1, 4, 2, 3),
    "qkv16n2b": (1, 16, 1, 2, 4, 2, 3),    # the same rows as 8 warps of one tile
    "qkv32w8u1": (2, 32, 3, 1, 8, 1, 4),   # 8 consumer warps, 16 stages, ring 4
    "qkv32r3": (2, 32, 1, 3, 4, 2, 4),     # 12 consumer warps, one tile each, ring 4
    "qkv32n2b": (2, 32, 2, 1, 4, 1, 4),    # R = 16, 32 stages, 98 KB: 2 CTAs per SM
    # o: K = 1024, 4 stages, 256 CTAs at 2-3 per SM.
    "o16u2": (1, 16, 2, 2, 1, 2, 4),       # 8 stages, ring 4, 4 CTAs per SM
    "o16w2": (1, 16, 2, 2, 2, 2, 3),       # 4 consumer warps
    "o32w1": (2, 32, 2, 2, 1, 4, 3),       # 2 consumer warps
    # unpooler: unp16 is 256 CTAs at 3 per SM; unp32 128 CTAs at 1.
    "unp16n4": (1, 16, 4, 2, 2, 2, 3),     # R = 64: 128 CTAs, 2 per SM fit
    "unp16n4u4": (1, 16, 4, 2, 2, 4, 3),   # R = 64, 8 stages, ring 3
    "unp32w8": (2, 32, 4, 2, 4, 1, 3),     # 8 consumer warps
    "unp32n2": (2, 32, 2, 2, 2, 2, 3),     # R = 32: 256 CTAs at 2 per SM (98 KB)
    # pooler: 128 CTAs at 1 per SM, 16 stages.
    "pool16w8": (1, 16, 2, 2, 4, 2, 4),    # 8 consumer warps, same stage size
    "pool16n1": (1, 16, 1, 2, 2, 4, 3),    # R = 16: 256 CTAs at 2 per SM (89 KB)
    "pool16n1b": (1, 16, 2, 1, 2, 4, 3),   # the same with 2 consumer warps of two tiles
    "pool32w8": (2, 32, 2, 2, 4, 2, 4),    # 8 consumer warps, same stage size
    "pool32n1b": (2, 32, 1, 2, 2, 2, 4),   # R = 16, 32 stages, 98 KB: 2 CTAs per SM
}
EXTRA_USE = {
    ("qkv_gate", 16): ["qkv16u4", "qkv16w8", "qkv16w8u1", "qkv16r3", "qkv16n2", "qkv16n2b"],
    ("qkv_gate", 32): ["qkv32w8u1", "qkv32r3", "qkv32n2b"],
    ("o", 16): ["o16u2", "o16w2"],
    ("o", 32): ["o32w1"],
    ("unpooler", 16): ["unp16n4", "unp16n4u4"],
    ("unpooler", 32): ["unp32w8", "unp32n2"],
    ("pooler", 16): ["pool16w8", "pool16n1", "pool16n1b"],
    ("pooler", 32): ["pool32w8", "pool32n1b"],
}
# Round 2 (after 01703396ea1a409e: o16w2 -0.22 / -0.38, unp32n2 -0.54 / -0.16, the rest lost):
# more consumer warps on the short-K o projection and the 2-CTA unpooler at 32 tokens.
EXTRA_GEOM2 = {
    "o16w2": (1, 16, 2, 2, 2, 2, 3),       # repeat: 4 consumer warps
    "o16w4": (1, 16, 2, 2, 4, 1, 3),       # 8 consumer warps, one unit each
    "o16r4": (1, 16, 1, 4, 2, 2, 3),       # 8 consumer warps, one tile each
    "o16r4w4": (1, 16, 1, 4, 4, 1, 3),     # 16 consumer warps
    "o32w4": (2, 32, 2, 2, 4, 1, 3),       # 8 consumer warps, one unit each
    "o32r4": (2, 32, 1, 4, 2, 2, 3),       # 8 consumer warps, one tile each
    "unp16w4": (1, 16, 2, 2, 4, 1, 3),     # 8 consumer warps at 3 CTAs per SM
    "unp32n2": (2, 32, 2, 2, 2, 2, 3),     # repeat: R = 32, 2 CTAs per SM
    "unp32n2w4": (2, 32, 2, 2, 4, 1, 3),   # the same with 8 consumer warps
}
EXTRA_USE2 = {
    ("o", 16): ["o16w2", "o16w4", "o16r4", "o16r4w4"],
    ("o", 32): ["o32w4", "o32r4"],
    ("unpooler", 16): ["unp16w4"],
    ("unpooler", 32): ["unp32n2", "unp32n2w4"],
}


def _geo_patch(geoms, use):
    return (
        ("    \"unp32\": (2, 32, 4, 2, 2, 2, 3),\n}\n",
         "    \"unp32\": (2, 32, 4, 2, 2, 2, 3),\n"
         + "".join(f"    {nm!r}: {g!r},\n".replace("'", "\"") for nm, g in geoms.items()) + "}\n"),
        ("SK_EXTRA = {}\n", f"SK_EXTRA = {use!r}\n"),
    )


GEO = _geo_patch(EXTRA_GEOM, EXTRA_USE)
VARIANTS = {"geo": (GEO, EXTRA_USE), "geo2": (_geo_patch(EXTRA_GEOM2, EXTRA_USE2), EXTRA_USE2)}

AB = r'''

_GB_SRCS = {srcs!r}
_GB_EXTRA = {extra!r}
_GB_MS = {ms!r}
import contextlib as _gbctx
import io as _gbio
import math as _gbmath
import os as _gbos
import random as _gbrand
import types as _gbtypes

_GB_OUT = []
_GB_LAYERS = {{"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
              "unpooler": (8192, 4096)}}
_GB_ROUNDS = 200


def _gb_print(s):
    print(s, flush=True)
    _GB_OUT.append(s)


def _gb_quiet(fn, tag):
    buf = _gbio.StringIO()
    t0 = time.time()
    with _gbctx.redirect_stdout(buf), _gbctx.redirect_stderr(buf):
        try:
            res = fn()
        except Exception as exc:
            res = exc
    for line in buf.getvalue().splitlines():
        if "tune" in line and ("m=16" in line or "m=32" in line):
            _gb_print(f"{{tag}}| {{line}}")
        elif "fail" in line.lower() or "disabled" in line:
            _gb_print(f"{{tag}}| {{line}}")
    _gb_print(f"ABPREP {{tag}} {{time.time() - t0:.1f}} s")
    return res


def _gb_graph(fn, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn(x)
    torch.cuda.synchronize()
    return g


def _gb_time(graphs, rounds, flush, sink, rng):
    names = list(graphs)
    samples = {{n: [] for n in names}}
    for r in range(rounds):
        order = names[:]
        rng.shuffle(order)
        evs = []
        for n in order:
            torch.sum(flush, dim=0, out=sink)
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


def _gb_q(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _gb_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(5151)
    weights = {{name: (torch.randn((n, k), generator=gen, device=dev) / _gbmath.sqrt(k)).to(torch.bfloat16)
               for name, (n, k) in _GB_LAYERS.items()}}
    weights["router"] = (torch.randn((256, 8192), generator=gen, device=dev) / 90.5).float()
    st = _gb_quiet(lambda: prepare(weights), "ours")
    if isinstance(st, Exception):
        _gb_print(f"ABPREP ours FAIL {{type(st).__name__}}: {{str(st)[:300]}}")
        return
    mods = {{}}
    for name, src in _GB_SRCS.items():
        src = src.replace('name="okc_sk_v2"', f'name="okc_sk_v2_m16_{{name}}"')
        mod = _gbtypes.ModuleType(f"okc_m16_{{name}}")
        try:
            exec(compile(src, f"m16_{{name}}.py", "exec"), mod.__dict__)
        except Exception as exc:
            _gb_print(f"ABPREP {{name}} FAIL load {{type(exc).__name__}}: {{str(exc)[:300]}}")
            continue
        ps = _gb_quiet(lambda: mod.prepare(weights), name)
        if isinstance(ps, Exception):
            _gb_print(f"ABPREP {{name}} FAIL {{type(ps).__name__}}: {{str(ps)[:300]}}")
            continue
        mods[name] = (mod, ps)
    flush = torch.ones(65536, 1024, dtype=torch.float32, device=dev)
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    xgen = torch.Generator(device=dev).manual_seed(777)
    rng = _gbrand.Random(2026)
    for m in _GB_MS:
        for layer, (n, k) in _GB_LAYERS.items():
            cfg = {{"m": m, "n": n, "k": k, "layer": layer}}
            arms = {{}}
            arms["sgl"] = lambda x, _l=layer: _cublas(x, st[_l])
            arms["ours"] = lambda x, _c=cfg: kernel_fn({{"x": x, "config": _c}}, st)
            arms["ours2"] = lambda x, _c=cfg: kernel_fn({{"x": x, "config": _c}}, st)
            pick = SK_USE[(layer, m)]
            for nm, fn, _ in st.get("cands", {{}}).get((layer, m), []):
                if nm == pick:
                    arms[f"ours.{{nm}}"] = fn
            for name, (mod, ps) in mods.items():
                arms[name] = lambda x, _m=mod, _s=ps, _c=cfg: _m.kernel_fn({{"x": x, "config": _c}}, _s)
                if name == "pub":
                    continue
                for nm, fn, _ in ps.get("cands", {{}}).get((layer, m), []):
                    if nm in _GB_EXTRA.get(f"{{layer}}/{{m}}", ()):
                        arms[f"{{name}}.{{nm}}"] = fn
            for ps_i in range(2):
                x = torch.randn((m, k), generator=xgen, device=dev).to(torch.bfloat16)
                ref = x.double() @ weights[layer].double().T
                errs, live = [], {{}}
                for nm, fn in arms.items():
                    try:
                        out = fn(x)
                        torch.cuda.synchronize()
                        e = float((out.double().reshape(ref.shape) - ref).norm() / ref.norm())
                        errs.append(f"{{nm}}={{e:.1e}}")
                        if e < 1e-2:
                            live[nm] = fn
                    except Exception as exc:
                        errs.append(f"{{nm}}=EXC:{{type(exc).__name__}}:{{str(exc)[:80]}}")
                _gb_print(f"ABCHECK {{'ok' if len(live) == len(arms) else 'FAIL'}} {{layer}} m={{m}} p{{ps_i}} "
                          + " ".join(errs))
                try:
                    graphs = {{nm: _gb_graph(fn, x) for nm, fn in live.items()}}
                    smp = _gb_time(graphs, _GB_ROUNDS, flush, sink, rng)
                except Exception as exc:
                    _gb_print(f"ABCOLD {{layer}} m={{m}} p{{ps_i}} FAIL {{type(exc).__name__}}: {{str(exc)[:200]}}")
                    continue
                graphs = None
                _gb_print(f"ABCOLD {{layer}} m={{m}} p{{ps_i}} "
                          + " ".join(f"{{nm}}={{statistics.median(smp[nm]):.2f}}" for nm in smp))
                pairs = []
                for a in smp:
                    if a in ("ours", "sgl"):
                        continue
                    d = [u - v for u, v in zip(smp[a], smp["ours"])]
                    pairs.append(f"{{a}}-ours={{statistics.median(d):+.2f}}"
                                 f"[{{_gb_q(d, 0.25):+.2f}},{{_gb_q(d, 0.75):+.2f}}]")
                if "pub" in smp:
                    for a in smp:
                        if a in ("pub", "sgl", "ours2") or a.startswith("ours."):
                            continue
                        d = [u - v for u, v in zip(smp[a], smp["pub"])]
                        pairs.append(f"{{a}}-pub={{statistics.median(d):+.2f}}"
                                     f"[{{_gb_q(d, 0.25):+.2f}},{{_gb_q(d, 0.75):+.2f}}]")
                _gb_print(f"ABPAIR {{layer}} m={{m}} p{{ps_i}} " + " ".join(pairs))
            torch.cuda.empty_cache()
    for _, (mod, ps) in mods.items():
        ps.clear()
    st.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _gbos.environ.get("PROJ_M16_DONE"):
    _gbos.environ["PROJ_M16_DONE"] = "1"
    try:
        _gb_run()
    except Exception as _exc:
        _gb_print(f"ABBENCH failed: {{type(_exc).__name__}}: {{str(_exc)[:400]}}")
    _gb_prepare = prepare

    def prepare(weights):
        state = _gb_prepare(weights)
        for line in _GB_OUT:
            print(line, flush=True)
        return state
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    names = list(VARIANTS)
    pub = None
    ms = (16, 32)
    while args:
        a = args.pop(0)
        if a == "--pub":
            pub = Path(args.pop(0)).read_text()
        elif a == "--variants":
            names = args.pop(0).split(",")
        elif a == "--m":
            ms = tuple(int(v) for v in args.pop(0).split(","))
        else:
            raise SystemExit(f"unknown argument {a}")
    base = (ROOT / "muse_bf16_proj/submission.py").read_text()
    srcs = {nm: keep_cands(patch(base, VARIANTS[nm][0], nm)) for nm in names}
    if pub is not None:
        srcs["pub"] = pub_patch(pub)
    extra = {}
    for nm in names:
        for (layer, b), v in VARIANTS[nm][1].items():
            extra[f"{layer}/{b}"] = extra.get(f"{layer}/{b}", ()) + tuple(v)
    out.write_text(keep_cands(base) + AB.format(srcs=srcs, extra=extra, ms=ms))
    print(f"wrote {out} with modules {list(srcs)}")


if __name__ == "__main__":
    main()
