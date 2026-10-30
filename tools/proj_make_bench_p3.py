"""Write the P3 bench: cw (12-bit pooler weights reconstructed into wgmma A tiles in shared
memory) against its raw-BF16 twin of the same compute layout, the production pick and cuBLAS,
pooler m = 128, with independent duplicates.

    python3 tools/proj_make_bench_p3.py OUT.py

OUT.py is muse_bf16_proj/submission.py (the judged path unchanged, cw offered at pooler m=128)
plus a block that runs at import, before the judged prepare(). Arms (synthetic pooler weights,
randn * 0.01 in BF16, which gives about the judge's escape rate):

  lt          the production cuBLASLt preference LT_PREF[("pooler", 128)]
  cublas      SGLang's dispatch
  wd8         wd b1n128c2s8h, the pick before cuBLASLt
  twin:<e>    wd b2n128c4s6 (raw BF16, cw's compute layout) from extension e
  <cw>:<e>    cw configs from extension e
  prod is the judged extension (_build_ext); A and B are bench extensions, each compiled
  separately and each packing the weights itself (pack_cw), so A against B is an independent
  null of build and preparation for both the twin and cw.
  Diagnostics, extension A only, wrong output, timed but not checked: cw6d3n (decoders skip the
  decode and stores), cw6d3m (no wgmma), cw6d3nm (both), twin_a1 (twin without wgmma).

Checks before timing (else the arm is dropped and listed in P3CHECK): every cw arm equals its
extension's twin bit for bit at m = 65 / 100 / 128 on the synthetic weights and on a matrix with
many escapes (up to the slot's capacity); every arm passes the FP64 check; two CUDA-graph
replays with fresh inputs written into the captured buffer. Timing: cold-L2 graph replays,
100 rounds in a seeded random order. Lines:

  P3 med        median us per arm after the replay overhead
  P3 d          paired median difference arm - lt (us)
  P3 pair       named paired contrasts (cw - twin per extension, A - B nulls, prod - A)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CW_BENCH = {"cw6d3": (6, 3, 0), "cw6d2": (6, 2, 0), "cw5d3": (5, 3, 0),
            "cw6d3n": (6, 3, 1), "cw6d3m": (6, 3, 2), "cw6d3nm": (6, 3, 3)}
CW_DUP = ("cw6d3", "cw6d2", "cw5d3")
CW_DIAG = ("cw6d3n", "cw6d3m", "cw6d3nm")
PAIRS = [("cw6d3:A", "twin:A"), ("cw6d3:B", "twin:B"), ("cw6d2:A", "twin:A"), ("cw5d3:A", "twin:A"),
         ("twin:A", "twin:B"), ("cw6d3:A", "cw6d3:B"), ("twin:prod", "twin:A"), ("cw6d3:prod", "cw6d3:A"),
         ("cw6d3n:A", "cw6d3:A"), ("cw6d3m:A", "cw6d3:A"), ("cw6d3nm:A", "twin_a1:A"), ("twin:A", "lt"),
         ("cw6d3:A", "lt"), ("twin:A", "wd8")]

BENCH = r'''

_P3_CW = @@CW@@
_P3_DUP = @@DUP@@
_P3_DIAG = @@DIAG@@
_P3_PAIRS = @@PAIRS@@
import os as _p3os
import random as _p3random

_P3_OUT = []


def _p3print(s):
    print(s, flush=True)
    _P3_OUT.append(s)


def _p3_build(tag, cw):
    """A bench extension: the router sk configs (as in round 2), the twin and its no-wgmma
    ablation, and the given cw configs."""
    g = globals()
    keep = ("SK_CONFIGS", "SK_INDEX", "WD_GEOM", "WD_VAR", "WD_NAMES", "WD_INDEX", "WD_ABL", "CW_GEOM", "CW_NAMES")
    saved = {nm: g[nm] for nm in keep}
    try:
        g["SK_CONFIGS"] = [c for c in saved["SK_CONFIGS"] if c[0] in set(ROUTER_GEOM)]
        g["SK_INDEX"] = {c[0]: i for i, c in enumerate(g["SK_CONFIGS"])}
        g["WD_GEOM"] = {CW_TWIN: saved["WD_GEOM"][CW_TWIN]}
        g["WD_VAR"] = {}
        g["WD_ABL"] = {CW_TWIN + "_a1": (CW_TWIN, 1)}
        g["WD_NAMES"] = [CW_TWIN, CW_TWIN + "_a1"]
        g["WD_INDEX"] = {nm: i for i, nm in enumerate(g["WD_NAMES"])}
        g["CW_GEOM"] = dict(cw)
        g["CW_NAMES"] = list(cw)
        src = sk_source() + wd_source() + rt_source() + cw_source()
        idx = (dict(g["WD_INDEX"]), list(g["CW_NAMES"]))
    finally:
        g.update(saved)
    ext = load_inline(name="okc_p3_" + tag, cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + _WD_CPP + _RT_CPP
                      + _CW_CPP, cuda_sources=src, functions=["sk_fwd", "wd_fwd", "rt_fwd", "cw_fwd"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-DOKC_P3_TAG_{tag}=1"],
                      verbose=False, no_implicit_headers=True)
    return ext, idx


def _p3_wd(ext, cid, w, diag):
    n, k = w.shape

    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


def _p3_cw(ext, cid, packed, n, k, diag):
    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        ext.cw_fwd(cid, packed["stream"], packed["eby"], packed["base4"], x, out, n, k, diag)
        return out
    return run


def _p3_graph(fn, xbuf):
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


def _p3_time(graphs, rounds, flush, seed):
    rng = _p3random.Random(seed)
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


def _p3_arms(w, exts, lt, lt_st, gen, diag, bad):
    """name -> fn on weights w; the twin of each extension under 'twin:<tag>'."""
    n, k = w.shape
    arms = {"cublas": lambda x, w=w: _cublas(x, w),
            "wd8": _p3_wd(exts["prod"][0], WD_INDEX["b1n128c2s8h"], w, diag)}
    for tag, (ext, (wdi, cwn)) in exts.items():
        arms[f"twin:{tag}"] = _p3_wd(ext, wdi[CW_TWIN], w, diag)
        packed, why = pack_cw(w)
        if packed is None:
            bad.append(f"pack:{tag}={why}")
            continue
        for nm in cwn:
            if tag != "A" and nm not in _P3_DUP:
                continue
            arms[f"{nm}:{tag}"] = _p3_cw(ext, cwn.index(nm), packed, n, k, diag)
        if tag == "A":
            arms["twin_a1:A"] = _p3_wd(ext, wdi[CW_TWIN + "_a1"], w, diag)
    if lt is not None:
        lt_st["pooler"] = w
        lt_st.pop("lt_wt", None)
        pref = _lt_pref(lt, lt_st, "pooler", 128, gen, diag, arms["cublas"])
        if pref is None:
            bad.append("lt=pref unavailable")
        else:
            arms["lt"] = pref[1]
    return arms


def _p3_diag_arm(nm):
    base = nm.split(":")[0]
    return base in _P3_DIAG or base == "twin_a1"


def _p3_check(arms, w, gen, dev, label, bad):
    """Bitwise cw = twin per extension, then FP64, at m = 65 / 100 / 128; failures drop arms."""
    n, k = w.shape
    for m in (65, 100, 128):
        x, probes = _gen_x(m, k, gen, dev)
        ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
        outs = {}
        for nm in list(arms):
            if _p3_diag_arm(nm):
                continue
            try:
                torch.cuda.synchronize()
                outs[nm] = arms[nm](x)
                torch.cuda.synchronize()
                ok, err, perr = _check(outs[nm], ref, probes, SELFTEST_TOL["bf16"])
                if not ok:
                    bad.append(f"{nm}@{label}{m}={err:.1e}/{perr:.1e}")
                    arms.pop(nm)
            except Exception as exc:
                torch.cuda.synchronize()
                bad.append(f"{nm}@{label}{m}=EXC:{str(exc)[-100:]}")
                arms.pop(nm)
        for nm in list(arms):
            base, _, tag = nm.partition(":")
            if not base.startswith("cw") or _p3_diag_arm(nm) or nm not in outs:
                continue
            twin = outs.get(f"twin:{tag}")
            if twin is None or not torch.equal(outs[nm], twin):
                diffs = -1 if twin is None else int((outs[nm] != twin).sum())
                bad.append(f"{nm}@{label}{m}=bitwise {diffs} differ")
                arms.pop(nm)


def _p3_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(3303)
    prod = _build_ext()
    t1 = time.time()
    exts = {"prod": (prod, (dict(WD_INDEX), list(CW_NAMES)))}
    for tag in ("A", "B"):
        exts[tag] = _p3_build(tag, _P3_CW)
    t2 = time.time()
    lt = None
    lt_st = {}
    try:
        lt = _build_lt()
        lt_st["lt_ws"] = torch.zeros(LT_WS_BYTES, dtype=torch.uint8, device=dev)
    except Exception as exc:
        _p3print(f"P3LT build failed {str(exc)[-200:]}")
    _p3print(f"P3BUILD prod {t1 - t0:.1f} s, bench A+B {t2 - t1:.1f} s, lt {time.time() - t2:.1f} s")
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    n, k = 4096, 8192
    w = (torch.randn((n, k), generator=gen, device=dev) * 0.01).to(torch.bfloat16)
    # Escape-heavy matrix for the correctness checks only: tiny and huge weights at rates that
    # fill blocks up to about half the slot's capacity.
    we = torch.randn((n, k), generator=gen, device=dev) * 0.01
    we = torch.where(torch.rand((n, k), generator=gen, device=dev) < 2e-3, we * 1e-9, we)
    we = torch.where(torch.rand((n, k), generator=gen, device=dev) < 5e-4, we * 1e6, we).to(torch.bfloat16)
    bad = []
    for label, wt in (("", w), ("e", we)):
        p, why = pack_cw(wt)
        _p3print(f"P3PACK{label} " + (why if p is None else f"base {p['base']} escapes {p['escapes']} "
                                      f"cmax {p['cmax']} eby {p['eby']} bytes {p['stream'].numel()}"))
    earms = _p3_arms(we, exts, None, lt_st, gen, diag, bad)
    _p3_check(earms, we, gen, dev, "e", bad)
    arms = _p3_arms(w, exts, lt, lt_st, gen, diag, bad)
    _p3_check(arms, w, gen, dev, "", bad)
    for nm in list(arms):
        base, _, tag = nm.partition(":")
        if base.startswith("cw") and not _p3_diag_arm(nm) and nm not in earms:
            bad.append(f"{nm}=failed on the escape matrix")
            arms.pop(nm)
    earms = None
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _p3_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _p3_graph(two_fn, None)
    ov = _p3_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _p3print(f"P3OVH {ovh:.2f} us")
    m = 128
    xbuf, _ = _gen_x(m, k, gen, dev)
    graphs = {}
    for nm in list(arms):
        try:
            g, out = _p3_graph(arms[nm], xbuf)
            if not _p3_diag_arm(nm):
                for rep in range(2):
                    x2, probes = _gen_x(m, k, gen, dev)
                    xbuf.copy_(x2)
                    g.replay()
                    torch.cuda.synchronize()
                    ref = (x2.double() @ w.double().T).to(torch.bfloat16).double()
                    ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
                    if not ok:
                        raise RuntimeError(f"replay {rep} err {err:.1e}/{perr:.1e}")
            graphs[nm] = g
        except Exception as exc:
            torch.cuda.synchronize()
            bad.append(f"{nm}@replay={str(exc)[-100:]}")
    _p3print(f"P3CHECK dropped: " + (" ".join(bad) if bad else "none") + f"; diag {int(diag[0])}")
    if "lt" not in graphs:
        _p3print("P3 no lt arm; differences against cublas")
    base = "lt" if "lt" in graphs else "cublas"
    smp = _p3_time(graphs, 100, flush, 4404)
    med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
    dl = {nm: statistics.median([a - b for a, b in zip(v, smp[base])]) for nm, v in smp.items()}
    order = sorted(med, key=med.get)
    _p3print(f"P3 m={m} base {base} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
    _p3print(f"P3 m={m} d " + " ".join(f"{nm}={dl[nm]:+.2f}" for nm in order if nm != base))
    parts = []
    for a, b in _P3_PAIRS:
        if a in smp and b in smp:
            d = [p - q for p, q in zip(smp[a], smp[b])]
            wins = sum(v < 0 for v in d)
            parts.append(f"{a}-{b}={statistics.median(d):+.2f}({wins}/{len(d)})")
    _p3print("P3 pair " + " ".join(parts))
    graphs = None
    _p3print(f"P3DONE {time.time() - t0:.1f} s")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _p3os.environ.get("PROJ_P3_DONE"):
    _p3os.environ["PROJ_P3_DONE"] = "1"
    try:
        _p3_run()
    except Exception as _exc:
        import traceback as _tb
        _p3print(f"P3BENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _p3_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _p3_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if (ln.startswith(("cw ", "tune pooler", "ltune pooler", "prepare done", "sk build", "lt build",
                               "selftest pooler")) or "disabled" in ln):
                print("judged| " + ln[:600], flush=True)
        for line in _P3_OUT:
            print(line, flush=True)
        return state
'''


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    sub = (ROOT / "muse_bf16_proj/experiments/cw_p3_submission.py").read_text()
    bench = (BENCH.replace("@@CW@@", repr(CW_BENCH)).replace("@@DUP@@", repr(CW_DUP))
             .replace("@@DIAG@@", repr(CW_DIAG)).replace("@@PAIRS@@", repr(PAIRS)))
    out.write_text(sub + bench)
    print(f"wrote {out}: {len(CW_BENCH)} cw configs per bench extension")


if __name__ == "__main__":
    main()
