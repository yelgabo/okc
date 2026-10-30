"""Write the round-2 router bench: which design difference makes the independent reference's
router faster, toggled one at a time in our kernels, with independent duplicate nulls.

    python3 tools/proj_make_bench_r2.py OUT.py [--donor EXPANDED.py] [--cu OUT.cu]

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). It builds two extensions: the production one (_build_ext,
cached for the judged prepare) and a bench one holding only the router configs below. Arms:

  m = 128 / 256, wd router (wgmma hi/lo), every variant of each base geometry by its VAR bits
  (WV_* in _WD_SRC): N no producer warp, K packed hi/lo box, L 64 B L2 promotion, U push
  reduction, S no wait watchdog, E early prologue, D duplicate (no effect), T exit barrier with
  push. "prod" is the base config from the production extension, "v0" the same config from the
  bench extension and "vD" its duplicate: prod and vD are the two nulls of the base; every
  variant set that is timed also has a D twin.
  m = 8 / 16 / 32, sk router picks (prod and bench-extension copies) and rt geometries (RT_GEOM
  plus the bench ones below; prod rt16w8 against the bench copy is its null).
  d_rw / d_rt with --donor: NathenFernandes/muse-bf16-proj v18 (MIT), its router kernels as
  published. Check arms only: a file built with --donor must never be submitted.

Every arm passes FP64 checks first (random x with probe rows, at adjacent sizes, on the given
router and on one with outlier and tiny weights), then a CUDA-graph replay check with fresh
inputs written into the captured buffer twice (replay-safe scratch), else it is dropped. Timing:
cold-L2 graph replays, 100 rounds in a seeded random order. Lines:

  R2 <m> med   median us per arm after the replay overhead
  R2 <m> d     paired median difference arm - prod (us)
  R2CHECK      dropped arms and why

--cu writes the device-only source with the bench tables for a local nvcc compile.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

N, K, L, U, S, E, D, T = 1, 2, 4, 8, 16, 32, 64, 128
# ALL is every difference that keeps our barriers and watchdog; S is tested alone and on ALL.
ALL = N | K | L | U | E
SINGLES = (N, K, L, U, S, E)
LOO = (N, K, L, U, E)
WD_BASES = {128: ("r64c8s8",), 256: ("r128c8s6", "r64c4s8")}
RT_BENCH = {"rt16w8": (16, 8, 4, False), "rt16w8t": (16, 8, 4, True), "rt8w8": (8, 8, 4, False),
            "rt16w4": (16, 4, 4, False), "rt16w8p8": (16, 8, 8, False), "rt8w16": (8, 16, 4, False)}


def var_sets():
    sets = [0, D] + list(SINGLES) + [ALL, ALL | D, ALL | S] + [ALL & ~b for b in LOO] + [ALL | T]
    return list(dict.fromkeys(sets))


def var_tag(v):
    if v == 0:
        return "v0"
    return "v" + "".join(c for b, c in zip((N, K, L, U, S, E, D, T), "NKLUSEDT") if v & b)


def bench_wd():
    return {f"{base}_{var_tag(v)}": (base, v) for bases in WD_BASES.values() for base in bases for v in var_sets()}


BENCH = r'''

_R2_DONOR = @@DONOR@@
_R2_WD = @@WD@@
_R2_RT = @@RT@@
_R2_BASES = @@BASES@@
import os as _r2os
import random as _r2random
import math as _r2math

_R2_OUT = []


def _r2print(s):
    print(s, flush=True)
    _R2_OUT.append(s)


def _r2_build_bench():
    """The bench extension: the router sk configs, the wd variants and the rt geometries only."""
    g = globals()
    saved = {nm: g[nm] for nm in ("SK_CONFIGS", "SK_INDEX", "WD_GEOM", "WD_VAR", "WD_NAMES", "WD_INDEX",
                                  "WD_ABL", "RT_GEOM", "RT_NAMES")}
    try:
        g["SK_CONFIGS"] = [c for c in saved["SK_CONFIGS"] if c[0] in ROUTER_GEOM]
        g["SK_INDEX"] = {c[0]: i for i, c in enumerate(g["SK_CONFIGS"])}
        g["WD_GEOM"] = {nm: saved["WD_GEOM"][b] for nm, (b, v) in _R2_WD.items()}
        g["WD_VAR"] = {nm: v for nm, (b, v) in _R2_WD.items()}
        g["WD_ABL"] = {}
        g["WD_NAMES"] = list(_R2_WD)
        g["WD_INDEX"] = {nm: i for i, nm in enumerate(_R2_WD)}
        g["RT_GEOM"] = dict(_R2_RT)
        g["RT_NAMES"] = list(_R2_RT)
        src = sk_source() + wd_source() + rt_source()
        idx = (dict(g["SK_INDEX"]), dict(g["WD_INDEX"]), list(g["RT_NAMES"]))
    finally:
        g.update(saved)
    ext = load_inline(name="okc_r2_bench", cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + _WD_CPP + _RT_CPP,
                      cuda_sources=src, functions=["sk_fwd", "wd_fwd", "rt_fwd"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False,
                      no_implicit_headers=True)
    return ext, idx


def _r2_wd_run(ext, cid, w, n, k, diag):
    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


def _r2_sk_run(ext, cid, stream, n, k, diag):
    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
        ext.sk_fwd(cid, stream, x, out, diag, 0, 0, n, k, diag)
        return out
    return run


def _r2_rt_run(ext, cid, wp, diag):
    def run(x):
        out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
        ext.rt_fwd(cid, wp, x, out, diag)
        return out
    return run


def _r2_donor(router):
    """The reference's router kernels, built from its source as published; check arms only."""
    if _R2_DONOR is None:
        return {}
    import hashlib as _hl
    arms = {}
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    try:
        packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)],
                             dim=2).contiguous().view(-1, 64)
        for arm in (128, 256):
            label = "rwdl_" + str(arm)
            src = _R2_DONOR["rw"][arm]
            cpp = ("#include <stdint.h>\n#include <pybind11/pybind11.h>\nint " + label
                   + "_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);")
            ext = load_inline("r2_rw_" + _hl.sha256(src.encode()).hexdigest()[:12], cpp, cuda_sources=src,
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
            arms[arm] = run
    except Exception as exc:
        _r2print(f"R2DONOR rw failed {type(exc).__name__}: {str(exc)[-300:]}")
    try:
        rpacked = rt__pack(router)
        src = _R2_DONOR["rt_cuda"]
        ext = load_inline("r2_rt_" + _hl.sha256(src.encode()).hexdigest()[:12], _R2_DONOR["rt_cpp"], cuda_sources=src,
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
        arms["rt"] = rt
    except Exception as exc:
        _r2print(f"R2DONOR rt failed {type(exc).__name__}: {str(exc)[-300:]}")
    return arms


def _r2_graph(fn, xbuf):
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


def _r2_time(graphs, rounds, flush, seed):
    rng = _r2random.Random(seed)
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


def _r2_weights(dev, gen, outliers):
    k = 8192
    w = torch.randn((256, k), generator=gen, device=dev) / _r2math.sqrt(k)
    if outliers:
        # Large entries (hi carries them, lo must stay exact) and tiny ones (lo underflow region).
        big = torch.rand((256, k), generator=gen, device=dev) < 1 / 512
        tiny = torch.rand((256, k), generator=gen, device=dev) < 1 / 64
        w = torch.where(big, w * 4096.0, w)
        w = torch.where(tiny, w * 1e-6, w)
    return w.float().contiguous()


def _r2_arms(m, st, dev, diag):
    """Arm name -> fn for router shape m, from the production and bench extensions."""
    prod, bench, (sk_idx, wd_idx, rt_names), wts = st["prod"], st["bench"], st["idx"], st["wts"]
    arms = {}
    if m >= 128:
        for base in _R2_BASES[m]:
            arms[f"{base}:prod"] = _r2_wd_run(prod, WD_INDEX[base], wts["hilo"], 256, 8192, diag)
            for nm, (b, v) in _R2_WD.items():
                if b == base:
                    w = wts["pack"] if v & WV_PACK else wts["hilo"]
                    arms[f"{base}:{nm.split('_', 1)[1]}"] = _r2_wd_run(bench, wd_idx[nm], w, 256, 8192, diag)
        if m in st["donor"]:
            arms["d_rw"] = st["donor"][m]
    else:
        sk = SK_USE[("router", 8 if m <= 8 else 16 if m <= 16 else 32)]
        arms[f"{sk}:prod"] = _r2_sk_run(prod, SK_INDEX[sk], wts["sk"][sk], 256, 8192, diag)
        arms[f"{sk}:v0"] = _r2_sk_run(bench, sk_idx[sk], wts["sk"][sk], 256, 8192, diag)
        arms["rt16w8:prod"] = _r2_rt_run(prod, RT_NAMES.index("rt16w8"), wts["rt"], diag)
        for nm in rt_names:
            arms[f"{nm}:v0"] = _r2_rt_run(bench, rt_names.index(nm), wts["rt"], diag)
        if "rt" in st["donor"]:
            arms["d_rt"] = st["donor"]["rt"]
    arms["sglang"] = lambda x: _router_baseline(x, st["sgl"])
    return arms


def _r2_state(router, prod, bench, idx, dev):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    rgeoms = {nm: _sk_geom(nm) for nm in ROUTER_GEOM}
    _, _, rstreams = pack_router(router, rgeoms)
    return {"prod": prod, "bench": bench, "idx": idx,
            "wts": {"hilo": torch.cat([hi, lo], 0).contiguous(), "pack": router_pack(hi, lo),
                    "rt": rt_pack(hi, lo), "sk": rstreams},
            "sgl": {"router": router, "router_hi_t": hi.t().contiguous(), "router_lo_t": lo.t().contiguous()},
            "donor": {}}


def _r2_spin_twin(nm):
    """The watchdog twin of a no-watchdog arm: a broken pipeline hangs a spin wait forever, so a
    spin arm runs only after its twin passed."""
    base, _, tag = nm.partition(":")
    if not tag.startswith("v") or "S" not in tag:
        return None
    t = tag.replace("S", "")
    return f"{base}:{t if t != 'v' else 'v0'}"


def _r2_check_all(arms, router, ms, gen, dev, label):
    bad = []
    order = sorted(arms, key=lambda nm: _r2_spin_twin(nm) is not None)
    for m in ms:
        x, probes = _gen_x(m, 8192, gen, dev)
        ref = x.double() @ router.double().T
        for nm in order:
            if nm not in arms:
                continue
            twin = _r2_spin_twin(nm)
            if twin is not None and twin not in arms:
                bad.append(f"{nm}@{label}{m}=twin {twin} failed")
                arms.pop(nm)
                continue
            try:
                out = arms[nm](x)
                torch.cuda.synchronize()
                ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["router"])
                if not ok or tuple(out.shape) != (m, 256):
                    bad.append(f"{nm}@{label}{m}={err:.1e}/{perr:.1e}")
                    arms.pop(nm)
            except Exception as exc:
                torch.cuda.synchronize()
                bad.append(f"{nm}@{label}{m}=EXC:{str(exc)[-80:]}")
                arms.pop(nm)
    return bad


def _r2_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(4242)
    prod = _build_ext()
    t1 = time.time()
    bench, idx = _r2_build_bench()
    _r2print(f"R2BUILD prod {t1 - t0:.1f} s, bench {time.time() - t1:.1f} s, {len(_R2_WD)} wd arms")
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    w_main = _r2_weights(dev, gen, False)
    w_out = _r2_weights(dev, gen, True)
    st_main = _r2_state(w_main, prod, bench, idx, dev)
    st_out = _r2_state(w_out, prod, bench, idx, dev)
    st_main["donor"] = _r2_donor(w_main)
    st_out["donor"] = _r2_donor(w_out)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _r2_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _r2_graph(two_fn, None)
    ov = _r2_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _r2print(f"R2OVH {ovh:.2f} us")
    adj = {8: (1, 5, 7, 8), 16: (9, 13, 16), 32: (17, 31, 32), 128: (65, 100, 128), 256: (129, 200, 256)}
    for ti, m in enumerate((8, 16, 32, 128, 256)):
        arms = _r2_arms(m, st_main, dev, diag)
        oarms = _r2_arms(m, st_out, dev, diag)
        bad = _r2_check_all(arms, w_main, adj[m], gen, dev, "")
        bad += _r2_check_all(oarms, w_out, adj[m], gen, dev, "o")
        for nm in list(arms):
            if nm not in oarms and nm != "sglang":
                arms.pop(nm)
        xbuf, _ = _gen_x(m, 8192, gen, dev)
        graphs = {}
        for nm in list(arms):
            try:
                g, out = _r2_graph(arms[nm], xbuf)
                for rep in range(2):
                    x2, probes = _gen_x(m, 8192, gen, dev)
                    xbuf.copy_(x2)
                    g.replay()
                    torch.cuda.synchronize()
                    ok, err, perr = _check(out, x2.double() @ w_main.double().T, probes, SELFTEST_TOL["router"])
                    if not ok:
                        raise RuntimeError(f"replay {rep} err {err:.1e}/{perr:.1e}")
                graphs[nm] = g
            except Exception as exc:
                torch.cuda.synchronize()
                bad.append(f"{nm}@replay={str(exc)[-80:]}")
        if bad:
            _r2print(f"R2CHECK m={m} dropped: " + " ".join(bad))
        base = next((nm for nm in graphs if nm.endswith(":prod") and not nm.startswith("rt")), None)
        if base is None:
            continue
        smp = _r2_time(graphs, 100, flush, 2000 + ti)
        graphs = None
        med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
        dl = {nm: statistics.median([a - b for a, b in zip(v, smp[base])]) for nm, v in smp.items()}
        order = sorted(med, key=med.get)
        _r2print(f"R2 m={m} base {base} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
        _r2print(f"R2 m={m} d " + " ".join(f"{nm}={dl[nm]:+.2f}" for nm in order if nm != base))
        torch.cuda.empty_cache()
    _r2print(f"R2DONE {time.time() - t0:.1f} s")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _r2os.environ.get("PROJ_R2_DONE"):
    _r2os.environ["PROJ_R2_DONE"] = "1"
    try:
        _r2_run()
    except Exception as _exc:
        import traceback as _tb
        _r2print(f"R2BENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _r2_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _r2_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("tune router", "prepare done", "sk build", "lt build")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _R2_OUT:
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
            "rt_pack": "import torch\n" + funcs["rt__pack"]}


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    donor = None
    cu = None
    while args:
        a = args.pop(0)
        if a == "--donor":
            donor = donor_parts(args.pop(0))
        elif a == "--cu":
            cu = Path(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    wd = bench_wd()
    bench = (BENCH.replace("@@DONOR@@", repr(donor)).replace("@@WD@@", repr(wd)).replace("@@RT@@", repr(RT_BENCH))
             .replace("@@BASES@@", repr(WD_BASES)))
    if donor is not None:
        bench = bench.replace("\n\ndef _r2_run():", "\n\n" + donor["rt_pack"] + "\n\n\ndef _r2_run():", 1)
    out.write_text(sub + bench)
    print(f"wrote {out}: {len(wd)} wd arms, {len(RT_BENCH)} rt arms, donor arms {'on' if donor else 'off'}")
    if cu is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sub", ROOT / "muse_bf16_proj/submission.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.WD_GEOM = {nm: mod.WD_GEOM[b] for nm, (b, v) in wd.items()}
        mod.WD_VAR = {nm: v for nm, (b, v) in wd.items()}
        mod.WD_ABL = {}
        mod.WD_NAMES = list(wd)
        mod.RT_GEOM = dict(RT_BENCH)
        mod.RT_NAMES = list(RT_BENCH)
        mod.SK_CONFIGS = [c for c in mod.SK_CONFIGS if c[0] in mod.ROUTER_GEOM]
        cu.write_text(mod.sk_source(device_only=True))
        print(f"wrote {cu}")


if __name__ == "__main__":
    main()
