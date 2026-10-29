"""Write a kernel-level A/B bench for muse_bf16_proj at the decode layer's small token counts
(m = 1 and 8): every candidate kernel of submission.py (the control) and of other whole
submissions of ours, the public entry's picks and its m = 1 cgemv, and SGLang's call, timed in
one process on the same weights and inputs.

    python3 tools/proj_make_bench_gemv.py OUT.py [--pub PUBLIC.py] [NAME=SUBMISSION.py ...]

Arms per shape (layer at m = 1 and 8; the unpooler at its 1024-row slice):
  sgl            SGLang's call (baseline.py).
  <mod>          the module's kernel_fn (its tuned pick); <mod> is "ours" for the control.
  <mod>.<cand>   each self-tested candidate of our modules except raw, cuBLAS and baseline
                 (cgemv, sgemv, rgemv at m = 1, the compressed sk configs), from state["cands"],
                 which the generator makes prepare() keep.
  pub.cgemv      the public cgemv at m = 1, through its _run.

At import, before the judge's tests, the bench prints:
  ABCHECK  per shape: relative L2 error of every arm against FP64 (FAIL lines always kept).
  ABCOLD   per shape and pass: median us of ROUNDS graph replays, each after a 256 MB read,
           arms interleaved in a rotating order. Two passes with different x.
  ABPAIR   per shape and pass: the median of per-round differences (a - b, us) for the pairs
           in PAIRS that exist, with the 25th and 75th percentiles.
The real prepare() prints the AB lines again at its end so they survive the log tail. The
public file is embedded as a string; the output belongs in the session scratchpad.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from proj_make_bench_ab import pub_patch  # noqa: E402

KEEP_CANDS = ('    print(f"prepare done {time.time() - t0:.1f} s")\n',
              '    state["cands"] = cands\n    print(f"prepare done {time.time() - t0:.1f} s")\n')

AB = r'''

_GB_SRCS = {srcs!r}
_GB_PAIRS = {pairs!r}
import contextlib as _gbctx
import io as _gbio
import math as _gbmath
import os as _gbos
import types as _gbtypes

_GB_OUT = []
_GB_LAYERS = {{"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
              "unpooler": (8192, 4096), "router": (256, 8192)}}
_GB_ROUNDS = 200
_GB_SKIP = ("cublas", "baseline")


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
        if ("tune" in line and ("m= 1" in line or "m= 8" in line or "m=1:" in line
                                or "m=8:" in line)):
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


def _gb_time(graphs, rounds, flush, sink):
    names = list(graphs)
    samples = {{n: [] for n in names}}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
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


def _gb_arms(layer, m, n, k, ours, mods):
    arms = {{"sgl": None}}

    def data_fn(kfn):
        return lambda x: kfn({{"x": x, "config": {{"m": m, "n": n, "k": k, "layer": layer}}}})

    for tag, (kfn, st) in ours.items():
        arms[tag] = data_fn(lambda d, _k=kfn, _s=st: _k(d, _s))
        for nm, fn, m1 in st.get("cands", {{}}).get((layer, 8), []):
            if nm in _GB_SKIP or nm.endswith("raw") or (m1 and m != 1):
                continue
            arms[f"{{tag}}.{{nm}}"] = fn
    if "pub" in mods:
        pmod, pst = mods["pub"]
        arms["pub"] = data_fn(lambda d, _m=pmod, _s=pst: _m.kernel_fn(d, _s))
        if m == 1 and layer in pst.get("cg", {{}}):
            arms["pub.cgemv"] = lambda x, _m=pmod, _s=pst: _m._run(("cgemv",), x, layer, m, n, k, _s)
    return arms


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
    ours = {{"ours": (kernel_fn, st)}}
    mods = {{}}
    for name, src in _GB_SRCS.items():
        src = src.replace('name="okc_sk_v2"', f'name="okc_sk_v2_gb_{{name}}"')
        mod = _gbtypes.ModuleType(f"okc_gb_{{name}}")
        try:
            exec(compile(src, f"gb_{{name}}.py", "exec"), mod.__dict__)
        except Exception as exc:
            _gb_print(f"ABPREP {{name}} FAIL load {{type(exc).__name__}}: {{str(exc)[:300]}}")
            continue
        ps = _gb_quiet(lambda: mod.prepare(weights), name)
        if isinstance(ps, Exception):
            _gb_print(f"ABPREP {{name}} FAIL {{type(ps).__name__}}: {{str(ps)[:300]}}")
            continue
        mods[name] = (mod, ps)
        if name != "pub":
            ours[name] = (mod.kernel_fn, ps)
    flush = torch.ones(65536, 1024, dtype=torch.float32, device=dev)
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    xgen = torch.Generator(device=dev).manual_seed(777)
    for m in (1, 8):
        for layer, (nfull, k) in _GB_LAYERS.items():
            n = 1024 if layer == "unpooler" else nfull
            arms = _gb_arms(layer, m, n, k, ours, mods)
            sgl = lambda x, _l=layer, _n=n: (_router_baseline(x, st) if _l == "router"
                                             else _cublas(x, st[_l][:_n]))
            arms["sgl"] = sgl
            for ps in range(2):
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
                line = (f"ABCHECK {{'ok' if len(live) == len(arms) else 'FAIL'}} {{layer}} m={{m}} p{{ps}} "
                        + " ".join(errs))
                if len(live) != len(arms):
                    _gb_print(line)
                else:
                    print(line, flush=True)
                try:
                    graphs = {{nm: _gb_graph(fn, x) for nm, fn in live.items()}}
                    smp = _gb_time(graphs, _GB_ROUNDS, flush, sink)
                except Exception as exc:
                    _gb_print(f"ABCOLD {{layer}} m={{m}} p{{ps}} FAIL {{type(exc).__name__}}: {{str(exc)[:200]}}")
                    continue
                graphs = None
                _gb_print(f"ABCOLD {{layer}} m={{m}} p{{ps}} "
                          + " ".join(f"{{nm}}={{statistics.median(smp[nm]):.2f}}" for nm in smp))
                pairs = []
                for a, b in _GB_PAIRS:
                    a = a.replace("@L", layer)
                    b = b.replace("@L", layer)
                    if a in smp and b in smp:
                        d = [u - v for u, v in zip(smp[a], smp[b])]
                        pairs.append(f"{{a}}-{{b}}={{statistics.median(d):+.2f}}"
                                     f"[{{_gb_q(d, 0.25):+.2f}}," f"{{_gb_q(d, 0.75):+.2f}}]")
                if pairs:
                    _gb_print(f"ABPAIR {{layer}} m={{m}} p{{ps}} " + " ".join(pairs))
        torch.cuda.empty_cache()
    for _, (mod, ps) in mods.items():
        ps.clear()
    st.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _gbos.environ.get("PROJ_GB_DONE"):
    _gbos.environ["PROJ_GB_DONE"] = "1"
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

PAIRS = [
    ("ours.cgemv", "base.cgemv"), ("ours.cgemv", "pub.cgemv"), ("base.cgemv", "pub.cgemv"),
    ("plain.cgemv", "ours.cgemv"), ("lb2.cgemv", "ours.cgemv"), ("nomin.cgemv", "ours.cgemv"),
    ("ours", "pub"), ("base", "pub"), ("ours", "base"),
]


def keep_cands(text):
    old, new = KEEP_CANDS
    if text.count(old) != 1:
        raise SystemExit("prepare() end marker not found once")
    return text.replace(old, new)


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
            srcs[name] = keep_cands(Path(path).read_text())
    src = keep_cands((ROOT / "muse_bf16_proj/submission.py").read_text())
    out.write_text(src + AB.format(srcs=srcs, pairs=PAIRS))
    print(f"wrote {out} with modules {list(srcs)}")


if __name__ == "__main__":
    main()
