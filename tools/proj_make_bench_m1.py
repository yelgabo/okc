"""Write a kernel-level A/B bench for muse_bf16_proj at m = 1: every m = 1 candidate of the
control submission, its kernel_fn twice, the public entry's kernel_fn and SGLang's call, timed in
one process on the same weights and inputs.

    python3 tools/proj_make_bench_m1.py OUT.py [--pub PUBLIC.py] [NAME=SUBMISSION.py ...]

Arms per shape (qkv_gate, o, pooler, and the unpooler at its 1024-row slice):
  sgl          SGLang's call.
  ours, ours2  the control's kernel_fn, captured twice: ours2 - ours is the null difference.
  ours.<cand>  every self-tested m = 1 candidate except raw, cuBLAS and baseline.
  pub          the public entry's kernel_fn (its m = 1 pick is its cgemv).
  <NAME>       another submission of ours (kernel_fn), e.g. the file before a change.

Rounds visit the arms in a fresh seeded random order (the fixed rotation of
proj_make_bench_gemv.py gave each arm the same predecessor; c991867992d34fb2 measured that
offset at 0.16-0.28 us). Output lines as in proj_make_bench_m8.py: ABCHECK (relative error
against FP64, and whether each cgemv-family arm equals ours.cgemv bit for bit), ABCOLD (median
us per arm), ABPAIR (median [25th, 75th] of per-round differences against ours, ours.cgemv and
pub).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from proj_make_bench_ab import pub_patch  # noqa: E402
from proj_make_bench_gemv import keep_cands  # noqa: E402

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
              "unpooler": (8192, 4096)}}
_GB_ROUNDS = 300
_GB_M = 1


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
        if "tune" in line and ("m= 1" in line or "m=1:" in line):
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


def _gb_pairs(smp, ref):
    out = []
    for a in smp:
        if a in (ref, "sgl"):
            continue
        d = [u - v for u, v in zip(smp[a], smp[ref])]
        out.append(f"{{a}}-{{ref}}={{statistics.median(d):+.2f}}"
                   f"[{{_gb_q(d, 0.25):+.2f}},{{_gb_q(d, 0.75):+.2f}}]")
    return out


def _gb_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(5151)
    weights = {{name: (torch.randn((n, k), generator=gen, device=dev) / _gbmath.sqrt(k)).to(
                   torch.float32 if name == "router" else torch.bfloat16)
               for name, (n, k) in list(_GB_LAYERS.items()) + [("router", (256, 8192))]}}
    st = _gb_quiet(lambda: prepare(weights), "ours")
    if isinstance(st, Exception):
        _gb_print(f"ABPREP ours FAIL {{type(st).__name__}}: {{str(st)[:300]}}")
        return
    mods = {{}}
    for name, src in _GB_SRCS.items():
        if name != "pub":
            src = src.replace('name="okc_sk_v2"', f'name="okc_sk_v2_m1_{{name}}"')
        mod = _gbtypes.ModuleType(f"okc_m1_{{name}}")
        try:
            exec(compile(src, f"m1_{{name}}.py", "exec"), mod.__dict__)
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
        arms["sgl"] = lambda x, _l=layer, _n=n: _cublas(x, st[_l][:_n])
        arms["ours"] = lambda x: kernel_fn({{"x": x, "config": cfg}}, st)
        arms["ours2"] = lambda x: kernel_fn({{"x": x, "config": cfg}}, st)
        for nm, fn, m1only in st.get("cands", {{}}).get((layer, 8), []):
            if nm.endswith("raw") or nm in ("cublas", "baseline"):
                continue
            arms[f"ours.{{nm}}"] = fn
        for name, (mod, ps) in mods.items():
            arms[name] = lambda x, _m=mod, _s=ps: _m.kernel_fn({{"x": x, "config": cfg}}, _s)
        for ps_i in range(2):
            x = torch.randn((m, k), generator=xgen, device=dev).to(torch.bfloat16)
            ref = x.double() @ weights[layer][:n].double().T
            errs, live, outs = [], {{}}, {{}}
            for nm, fn in arms.items():
                try:
                    out = fn(x)
                    torch.cuda.synchronize()
                    e = float((out.double().reshape(ref.shape) - ref).norm() / ref.norm())
                    errs.append(f"{{nm}}={{e:.1e}}")
                    outs[nm] = out.reshape(ref.shape).clone()
                    if e < 1e-2:
                        live[nm] = fn
                except Exception as exc:
                    errs.append(f"{{nm}}=EXC:{{type(exc).__name__}}:{{str(exc)[:80]}}")
            same = []
            if "ours.cgemv" in outs:
                for nm, o in outs.items():
                    if nm.startswith("ours.cgemv") and nm != "ours.cgemv":
                        same.append(f"{{nm}}:{{'bitwise' if torch.equal(o, outs['ours.cgemv']) else 'DIFF'}}")
            _gb_print(f"ABCHECK {{'ok' if len(live) == len(arms) else 'FAIL'}} {{layer}} m={{m}} p{{ps_i}} "
                      + " ".join(errs) + " | " + " ".join(same))
            try:
                graphs = {{nm: _gb_graph(fn, x) for nm, fn in live.items()}}
                smp = _gb_time(graphs, _GB_ROUNDS, flush, sink, rng)
            except Exception as exc:
                _gb_print(f"ABCOLD {{layer}} m={{m}} p{{ps_i}} FAIL {{type(exc).__name__}}: {{str(exc)[:200]}}")
                continue
            graphs = None
            _gb_print(f"ABCOLD {{layer}} m={{m}} p{{ps_i}} "
                      + " ".join(f"{{nm}}={{statistics.median(smp[nm]):.2f}}" for nm in smp))
            pairs = _gb_pairs(smp, "ours")
            if "ours.cgemv" in smp:
                pairs += _gb_pairs({{a: v for a, v in smp.items() if a.startswith("ours.cgemv") or a == "pub"}},
                                   "ours.cgemv")
            if "pub" in smp:
                pairs += _gb_pairs({{a: v for a, v in smp.items() if a.startswith("ours.")}} | {{"pub": smp["pub"]}},
                                   "pub")
            _gb_print(f"ABPAIR {{layer}} m={{m}} p{{ps_i}} " + " ".join(pairs))
        torch.cuda.empty_cache()
    for _, (mod, ps) in mods.items():
        ps.clear()
    st.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _gbos.environ.get("PROJ_M1_DONE"):
    _gbos.environ["PROJ_M1_DONE"] = "1"
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
    srcs = {}
    while args:
        a = args.pop(0)
        if a == "--pub":
            srcs["pub"] = pub_patch(Path(args.pop(0)).read_text())
        else:
            name, _, path = a.partition("=")
            srcs[name] = Path(path).read_text()
    base = (ROOT / "muse_bf16_proj/submission.py").read_text()
    out.write_text(keep_cands(base) + AB.format(srcs=srcs))
    print(f"wrote {out} with modules {list(srcs)}")


if __name__ == "__main__":
    main()
