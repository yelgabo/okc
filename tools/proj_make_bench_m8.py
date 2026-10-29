"""Write a kernel-level A/B bench for muse_bf16_proj at m = 8: the control submission, variants
of it made by exact text patches of the consumer path, and the public entry, timed in one
process on the same weights and inputs.

    python3 tools/proj_make_bench_m8.py OUT.py --pub PUBLIC.py [--variants acc,nofence,late,all]

Arms per shape (all five layers at m = 8; the unpooler at its 1024-row slice):
  sgl          SGLang's call.
  ours, ours2  the control's kernel_fn, captured twice: ours2 - ours is the null difference.
  ours.<pick>  the control's m = 8 candidate called directly.
  <variant>    a patched copy of the control (kernel_fn); see VARIANTS.
  pub          the public entry's kernel_fn.

Rounds visit the arms in a fresh seeded random order. The m = 1 GEMV bench rotated a fixed list,
which gives every arm the same predecessor in every round; its kernel_fn arms sat 0.16-0.28 us
above their own candidates at m = 8 (b135dd8ee10f4b24), and this bench measures that offset.
Output lines are ABCHECK / ABCOLD / ABPAIR as in proj_make_bench_gemv.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from proj_make_bench_ab import pub_patch  # noqa: E402
from proj_make_bench_gemv import keep_cands  # noqa: E402

NACC = (
    ("SK_EXTRA = {}\n",
     "SK_EXTRA = {}\nSK_NACC = {\"qkv8\": 2, \"o8\": 2, \"slice8\": 2, \"router8\": 4}\n"),
    ("_fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, False, 1), comp, False, 1, 1, \"bf16\",",
     "_fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, comp, False, 1), comp, False, "
     "SK_NACC.get(name, 1), 1, \"bf16\","),
    ("_fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, False, True, 3), False, True, 2, 3, \"f32\",",
     "_fit_stages(mt, tokr, nt, nwr, wsk, upw, nstage, False, True, 3), False, True, "
     "SK_NACC.get(name, 2), 3, \"f32\","),
)
NOFENCE = (
    ("                // Everything this warp needs from the slot is in registers.\n"
     "                fence_proxy_async();\n",
     "                // Everything this warp needs from the slot is in registers.\n"),
    ("                fence_proxy_async();\n                __syncwarp();\n                if (lane == 0) {\n"
     "                    mbar_arrive(empty0 + 8 * slot);\n                }\n            }\n"
     "            __syncwarp();\n            if (lane == 0) {\n                mbar_arrive(cbar);",
     "                __syncwarp();\n                if (lane == 0) {\n"
     "                    mbar_arrive(empty0 + 8 * slot);\n                }\n            }\n"
     "            __syncwarp();\n            if (lane == 0) {\n                mbar_arrive(cbar);"),
)
LATE = (
    ("            if (uu == UPW - 1) {\n                // Everything this warp needs from the slot is in registers.",
     "            if (UPW == 1) {\n                // Everything this warp needs from the slot is in registers."),
    ("                                     bl[nt][2 * j + 1]);\n                        }\n                    }\n"
     "                }\n            }\n        }\n    }\n",
     "                                     bl[nt][2 * j + 1]);\n                        }\n                    }\n"
     "                }\n            }\n        }\n"
     "        if (UPW > 1) {\n            fence_proxy_async();\n            __syncwarp();\n"
     "            if (lane == 0) {\n                mbar_arrive(empty0 + 8 * slot);\n            }\n        }\n    }\n"),
)
NOVOTE = (
    ("        if (!__all_sync(0xffffffffu, mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8))) {\n"
     "            dead = true;\n            break;\n        }\n",
     "        mbar_wait<false>(full0 + 8 * slot, (st / NSTAGE) & 1, dead, diag, 8);\n"),
)
LATE_NOFENCE = (
    ("        if (UPW > 1) {\n            fence_proxy_async();\n            __syncwarp();\n",
     "        if (UPW > 1) {\n            __syncwarp();\n"),
)
INIT = (
    ("            // Stage 0 goes out before anything else: its latency is the ramp of the kernel.\n"
     "            mbar_init(full0, 1);\n"
     "            fence_mbar_init();\n"
     "            issue(0);\n"
     "            if constexpr (COMP) {\n"
     "                mbar_init(pbar, 1);\n"
     "                fence_mbar_init();\n"
     "                mbar_expect_tx(pbar, tab_ints * 4);\n"
     "                bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);\n"
     "            }\n"
     "            for (int i = 1; i < NSTAGE; ++i) {\n"
     "                mbar_init(full0 + 8 * i, 1);\n"
     "            }\n"
     "            for (int i = 0; i < NSTAGE; ++i) {\n"
     "                mbar_init(empty0 + 8 * i, NW + NESC);\n"
     "            }\n"
     "            mbar_init(cbar, NESC > 0 ? NESC : 1);\n"
     "            fence_mbar_init();\n",
     "            for (int i = 0; i < NSTAGE; ++i) {\n"
     "                mbar_init(full0 + 8 * i, 1);\n"
     "                mbar_init(empty0 + 8 * i, NW + NESC);\n"
     "            }\n"
     "            mbar_init(pbar, 1);\n"
     "            mbar_init(cbar, NESC > 0 ? NESC : 1);\n"
     "            fence_mbar_init();\n"
     "            issue(0);\n"
     "            if constexpr (COMP) {\n"
     "                mbar_expect_tx(pbar, tab_ints * 4);\n"
     "                bulk_g2s(smem_u32(tab), ptab + (size_t)rb * tab_ints, tab_ints * 4, pbar);\n"
     "            }\n"),
)
GEO = (
    ("    \"qkv8\": (1, 8, 3, 1, 4, 2, 5),\n",
     "    \"qkv8\": (1, 8, 3, 1, 4, 2, 5),\n"
     "    \"qkv8u4\": (1, 8, 3, 1, 4, 4, 4),\n"
     "    \"qkv8w8\": (1, 8, 3, 1, 8, 2, 4),\n"
     "    \"qkv8w8u1\": (1, 8, 3, 1, 8, 1, 5),\n"),
    ("    \"router8\": (1, 8, 1, 1, 8, 1, 6),\n",
     "    \"router8\": (1, 8, 1, 1, 8, 1, 6),\n    \"router8u2\": (1, 8, 1, 1, 8, 2, 4),\n"),
    ("SK_EXTRA = {}\n",
     "SK_EXTRA = {(\"qkv_gate\", 8): [\"qkv8u4\", \"qkv8w8\", \"qkv8w8u1\"], (\"router\", 8): [\"router8u2\"]}\n"),
)
VARIANTS = {
    "geo": GEO,
    "geov": GEO + NOVOTE,
    "acc": NACC,
    "nofence": NOFENCE,
    "late": LATE,
    "all": NACC + NOFENCE + LATE + LATE_NOFENCE + NOVOTE,
    "init": INIT,
    "novote": NOVOTE,
    "cand": INIT + NOVOTE,
    "candnf": INIT + NOVOTE + NOFENCE,
}

AB = r'''

_GB_SRCS = {srcs!r}
import contextlib as _gbctx
import io as _gbio
import math as _gbmath
import os as _gbos
import random as _gbrand
import types as _gbtypes

_GB_OUT = []
_GB_LAYERS = {{"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
              "unpooler": (8192, 4096), "router": (256, 8192)}}
_GB_PICK = {{"qkv_gate": "qkv8", "o": "o8", "pooler": "pool8", "unpooler": "slice8", "router": "router8"}}
_GB_SK_EXTRA = {{"qkv_gate": ("qkv8", "qkv8u4", "qkv8w8", "qkv8w8u1"), "router": ("router8", "router8u2")}}
_GB_ROUNDS = 240
_GB_M = 8


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
        if "tune" in line and ("m= 8" in line or "m=8:" in line):
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
    weights = {{name: (torch.randn((n, k), generator=gen, device=dev) / _gbmath.sqrt(k)).to(
                   torch.float32 if name == "router" else torch.bfloat16)
               for name, (n, k) in _GB_LAYERS.items()}}
    st = _gb_quiet(lambda: prepare(weights), "ours")
    if isinstance(st, Exception):
        _gb_print(f"ABPREP ours FAIL {{type(st).__name__}}: {{str(st)[:300]}}")
        return
    mods = {{}}
    for name, src in _GB_SRCS.items():
        src = src.replace('name="okc_sk_v2"', f'name="okc_sk_v2_m8_{{name}}"')
        mod = _gbtypes.ModuleType(f"okc_m8_{{name}}")
        try:
            exec(compile(src, f"m8_{{name}}.py", "exec"), mod.__dict__)
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
    m = _GB_M
    for layer, (nfull, k) in _GB_LAYERS.items():
        n = 1024 if layer == "unpooler" else nfull
        cfg = {{"m": m, "n": n, "k": k, "layer": layer}}
        arms = {{}}
        arms["sgl"] = (lambda x, _l=layer, _n=n: _router_baseline(x, st) if _l == "router"
                       else _cublas(x, st[_l][:_n]))
        arms["ours"] = lambda x: kernel_fn({{"x": x, "config": cfg}}, st)
        arms["ours2"] = lambda x: kernel_fn({{"x": x, "config": cfg}}, st)
        for nm, fn, _ in st.get("cands", {{}}).get((layer, 8), []):
            if nm == _GB_PICK[layer]:
                arms[f"ours.{{nm}}"] = fn
        for name, (mod, ps) in mods.items():
            arms[name] = lambda x, _m=mod, _s=ps: _m.kernel_fn({{"x": x, "config": cfg}}, _s)
            for nm, fn, _ in ps.get("cands", {{}}).get((layer, 8), []):
                if name != "pub" and nm in _GB_SK_EXTRA.get(layer, ()):
                    arms[f"{{name}}.{{nm}}"] = fn
        for ps_i in range(2):
            x = torch.randn((m, k), generator=xgen, device=dev).to(torch.bfloat16)
            ref = x.double() @ weights[layer][:n].double().T
            tol = 1e-4 if layer == "router" else 1e-2
            errs, live = [], {{}}
            for nm, fn in arms.items():
                try:
                    out = fn(x)
                    torch.cuda.synchronize()
                    e = float((out.double().reshape(ref.shape) - ref).norm() / ref.norm())
                    errs.append(f"{{nm}}={{e:.1e}}")
                    if e < tol:
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
                    if a in ("pub", "sgl", "ours"):
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


if not _gbos.environ.get("PROJ_M8_DONE"):
    _gbos.environ["PROJ_M8_DONE"] = "1"
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


def patch(text, edits, tag):
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit(f"variant {tag}: patch does not apply once: {old[:80]!r}")
        text = text.replace(old, new)
    return text


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    names = list(VARIANTS)
    pub = None
    while args:
        a = args.pop(0)
        if a == "--pub":
            pub = Path(args.pop(0)).read_text()
        elif a == "--variants":
            names = args.pop(0).split(",")
        else:
            raise SystemExit(f"unknown argument {a}")
    base = (ROOT / "muse_bf16_proj/submission.py").read_text()
    srcs = {nm: keep_cands(patch(base, VARIANTS[nm], nm)) for nm in names}
    if pub is not None:
        srcs["pub"] = pub_patch(pub)
    out.write_text(keep_cands(base) + AB.format(srcs=srcs))
    print(f"wrote {out} with modules {list(srcs)}")


if __name__ == "__main__":
    main()
