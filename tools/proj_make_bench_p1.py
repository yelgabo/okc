"""Write the P1 bench: our candidate projections against two independent preparations of
themselves, SGLang, and (as checks only) the independent reference's router kernels and
pinned cuBLASLt configurations.

    python3 tools/proj_make_bench_p1.py OUT.py [--donor EXPANDED.py]

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import. It calls prepare() twice (A and B, each its own self-tests, cuBLASLt search and picks),
then for every target shape times, as cold-L2 CUDA-graph replays in a seeded random order per
round:

  baseA / baseB  each preparation's pick before the cuBLASLt stage (the eb374e4 policy plus
                 the new router geometries, which are tuned in the main loop)
  newA / newB    each preparation's final pick
  sglang         SGLang's dispatch (router: its hi/lo split above 8 tokens)
  wd geometries  router m = 128 / 256 only, every router wd config by name
  d*             with --donor: NathenFernandes/muse-bf16-proj v18 (MIT), commit 86aff75,
                 its routerwide (m = 128 / 256) and routertail (m = 8 / 16 / 32) kernels and
                 its pinned cuBLASLt configurations restored through our plan code. Check
                 arms only: a file built with --donor must never be submitted.

Every arm is checked against FP64 first (5e-3 BF16, 5e-5 router) and dropped if it fails.
Lines (the judge keeps the last 16 KB of the log, so they are compact):

  P1PREP  per preparation: build lines and the ltune / router tune lines
  P1 <layer> m=<m> med  median us per arm after the replay overhead
  P1 <layer> m=<m> d    paired median difference arm - baseA in us (per round), so baseB and
                        newB - newA are the nulls
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCH = r'''

_P1_DONOR = @@DONOR@@
import os as _p1os
import random as _p1random
import contextlib as _p1ctx
import io as _p1io
import math as _p1math

_P1_OUT = []


def _p1print(s):
    print(s, flush=True)
    _P1_OUT.append(s)


def _p1_graph(fn, x):
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


def _p1_time(graphs, rounds, flush, seed):
    rng = _p1random.Random(seed)
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


def _p1_prepare(tag, weights):
    buf = _p1io.StringIO()
    t0 = time.time()
    with _p1ctx.redirect_stdout(buf):
        st = prepare(weights)
    keep = []
    for ln in buf.getvalue().splitlines():
        if ln.startswith(("lt build", "ltune", "cut build", "sk build", "selftest")):
            keep.append(ln)
        elif ln.startswith("tune router") and ("m=128" in ln or "m=256" in ln or "m= 8" in ln or "m=16" in ln):
            keep.append(ln)
        elif ln.startswith("lt ") and tag == "A":
            keep.append(ln)
    _p1print(f"P1PREP {tag} {time.time() - t0:.1f} s")
    for ln in keep:
        _p1print(f"P1PREP {tag}| {ln}")
    return st


def _p1_donor_router(dev, router):
    """The reference's router kernels, built from its source as published; check arms only."""
    if _P1_DONOR is None:
        return {}
    from torch.utils.cpp_extension import load_inline as _li
    import hashlib as _hl
    arms = {}
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    try:
        packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)],
                             dim=2).contiguous().view(-1, 64)
        for arm in (128, 256):
            label = "rwdl_" + str(arm)
            src = _P1_DONOR["rw"][arm]
            cpp = ("#include <stdint.h>\n#include <pybind11/pybind11.h>\nint " + label
                   + "_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);")
            ext = _li("p1_rw_" + _hl.sha256(src.encode()).hexdigest()[:12], cpp, cuda_sources=src,
                      functions=[label + "_launch"],
                      extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
                      no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
            fn = getattr(ext, label + "_launch")

            def run(x, fn=fn, packed=packed):
                out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
                rc = fn(0, packed.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                        torch.cuda.current_stream().cuda_stream, 0)
                if rc:
                    raise RuntimeError(f"rw launch {rc}")
                return out
            arms[("rw", arm)] = run
    except Exception as exc:
        _p1print(f"P1DONOR rw failed {type(exc).__name__}: {str(exc)[-300:]}")
    try:
        rpacked = rt__pack(router)
        src = _P1_DONOR["rt_cuda"]
        ext = _li("p1_rt_" + _hl.sha256(src.encode()).hexdigest()[:12], _P1_DONOR["rt_cpp"], cuda_sources=src,
                  functions=["routertail_small"],
                  extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
                  no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

        def rt(x, ext=ext, rpacked=rpacked):
            out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
            rc = ext.routertail_small(rpacked.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], 16, 8,
                                      torch.cuda.current_stream().cuda_stream, 0)
            if rc:
                raise RuntimeError(f"rt launch {rc}")
            return out
        arms[("rt", 0)] = rt
    except Exception as exc:
        _p1print(f"P1DONOR rt failed {type(exc).__name__}: {str(exc)[-300:]}")
    return arms


def _p1_donor_lt(st, layer, m):
    """The reference's pinned cuBLASLt configuration for (m, layer), restored by our plan code."""
    if _P1_DONOR is None or "lt_ext" not in st:
        return None
    ent = _P1_DONOR["lt"].get((m, layer))
    if ent is None:
        return None
    their_layout, _, cfg = ent
    # Theirs: bit 0 row order, >= 2 K-major copy. Ours: >= 2 row order, bit 0 K-major copy.
    layout = (2 if their_layout % 2 else 0) + (1 if their_layout >= 2 else 0)
    lt = st["lt_ext"]
    w = st[layer]
    n, k = w.shape
    wts = st.setdefault("lt_wt", {})
    if layer not in wts:
        wts[layer] = w.t().contiguous()
    p = lt.lt_plan_new(m, n, k, layout, st["lt_ws"].numel())
    idx = lt.lt_add_config(p, list(cfg)) if p >= 0 else -1
    if idx < 0:
        _p1print(f"P1DONOR lt {layer} m={m} L{layout} {cfg}: not accepted by this library")
        return None
    return _lt_runner(lt, p, idx, wts[layer] if layout % 2 else w, m, n, st["lt_ws"], lambda x: None)


def _p1_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(5151)
    lay = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
           "unpooler": (8192, 4096), "router": (256, 8192)}
    weights = {nm: (torch.randn((n, k), generator=gen, device=dev) / _p1math.sqrt(k)).to(
                   torch.float32 if nm == "router" else torch.bfloat16) for nm, (n, k) in lay.items()}
    stA = _p1_prepare("A", weights)
    stB = _p1_prepare("B", weights)
    darms = _p1_donor_router(dev, weights["router"])
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one = _p1_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two = _p1_graph(two_fn, None)
    ov = _p1_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _p1print(f"P1OVH {ovh:.2f} us")
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    xgen = torch.Generator(device=dev).manual_seed(777)
    targets = [(layer, m) for m in (128, 256) for layer in BF16_LAYERS]
    targets += [("router", m) for m in (8, 16, 32, 128, 256)]
    rhilo = stA.get("router_hilo")
    for ti, (layer, m) in enumerate(targets):
        w = weights[layer]
        n, k = w.shape
        is_router = layer == "router"
        arms = {"baseA": stA["pick_base"][(layer, m)], "baseB": stB["pick_base"][(layer, m)],
                "newA": stA["pick"][(layer, m)], "newB": stB["pick"][(layer, m)]}
        if is_router:
            arms["sglang"] = lambda x: _router_baseline(x, stA)
            if m >= 128 and "wd_ext" in stA:
                for nm in WD_GEOM:
                    g = WD_GEOM[nm]
                    if g[4] and wd_fits(nm, n, k) and g[1] <= m:
                        arms[nm] = _wd_runner(stA["wd_ext"], nm, rhilo, n, k, torch.float32, diag)
            if m in (128, 256) and ("rw", m) in darms:
                arms["d_rw"] = darms[("rw", m)]
            if m <= 32 and ("rt", 0) in darms:
                arms["d_rt"] = darms[("rt", 0)]
        else:
            arms["sglang"] = lambda x, w=w: _cublas(x, w)
            d = _p1_donor_lt(stA, layer, m)
            if d is not None:
                arms["d_lt"] = d
        x, probes = _gen_x(m, k, xgen, dev)
        ref = x.double() @ w.double().T
        if not is_router:
            ref = ref.to(torch.bfloat16).double()
        tol = SELFTEST_TOL["router" if is_router else "bf16"]
        bad = []
        for nm in list(arms):
            try:
                out = arms[nm](x)
                torch.cuda.synchronize()
                ok, err, perr = _check(out, ref, probes, tol)
                if not ok or tuple(out.shape) != (m, n):
                    bad.append(f"{nm}={err:.1e}/{perr:.1e}")
                    arms.pop(nm)
            except Exception as exc:
                torch.cuda.synchronize()
                bad.append(f"{nm}=EXC:{str(exc)[-80:]}")
                arms.pop(nm)
        if bad:
            _p1print(f"P1CHECK {layer} m={m} dropped: " + " ".join(bad))
        graphs = {}
        for nm in list(arms):
            try:
                graphs[nm] = _p1_graph(arms[nm], x)
            except Exception as exc:
                torch.cuda.synchronize()
                _p1print(f"P1CHECK {layer} m={m} {nm} capture failed: {str(exc)[-80:]}")
        if "baseA" not in graphs:
            continue
        smp = _p1_time(graphs, 100, flush, 1000 + ti)
        graphs = None
        med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
        dl = {nm: statistics.median([a - b for a, b in zip(v, smp["baseA"])]) for nm, v in smp.items()}
        order = sorted(med, key=med.get)
        _p1print(f"P1 {layer} m={m} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
        _p1print(f"P1 {layer} m={m} d " + " ".join(f"{nm}={dl[nm]:+.2f}" for nm in order if nm != "baseA"))
        torch.cuda.empty_cache()
    _p1print(f"P1DONE {time.time() - t0:.1f} s")
    for st in (stA, stB):
        st.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _p1os.environ.get("PROJ_P1_DONE"):
    _p1os.environ["PROJ_P1_DONE"] = "1"
    try:
        _p1_run()
    except Exception as _exc:
        import traceback as _tb
        _p1print(f"P1BENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-600:]}")
    _p1_prepare_judged = prepare

    def prepare(weights):
        buf = _p1io.StringIO()
        with _p1ctx.redirect_stdout(buf):
            state = _p1_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("lt build", "ltune", "prepare done", "cut build", "sk build")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _P1_OUT:
            print(line, flush=True)
        return state
'''


def donor_parts(path):
    """Literal sources and values from the expanded reference file, read without executing it."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    vals = {}
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                vals[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
        elif isinstance(node, ast.FunctionDef) and node.name == "rt__pack":
            funcs[node.name] = ast.get_source_segment(src, node)
    return {"rw": {128: vals["rw_SOURCES"][128], 256: vals["rw_SOURCES"][256]},
            "rt_cuda": vals["rt__CUDA"], "rt_cpp": vals["rt__CPP"],
            "rt_pack": "import torch\n" + funcs["rt__pack"],
            "lt": vals["tt_lt__LT_H200_FIXED"]}


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    donor = None
    while args:
        a = args.pop(0)
        if a == "--donor":
            donor = donor_parts(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    bench = BENCH.replace("@@DONOR@@", repr(donor))
    if donor is not None:
        bench = bench.replace("\n\ndef _p1_run():", "\n\n" + donor["rt_pack"] + "\n\n\ndef _p1_run():", 1)
    out.write_text(sub + bench)
    print(f"wrote {out}, donor arms {'on' if donor else 'off'}")


if __name__ == "__main__":
    main()
