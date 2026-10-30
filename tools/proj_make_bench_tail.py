"""Write the exit-barrier bench: the conforming exit cluster barrier of the push reductions (wd
router at m = 128 / 256, rt router at m = 8 / 16 / 32) in its shipped form against the split
form, with independent duplicate nulls.

    python3 tools/proj_make_bench_tail.py OUT.py [--cu OUT.cu]

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). It builds the production extension (_build_ext, cached for
the judged prepare) and a bench extension with the arms below. Tags are the wd VAR bits:

  T  WV_TAILSYNC, the exit barrier shipped in ce25f67 (release arrive, acquire wait, after the
     output stores)
  Y  WV_TAILSPLIT, arrive right after the receive wait, wait just before exit
  Z  WV_TAILRLX, that arrive relaxed
  D  duplicate (no effect), the independent null
  KLUE without T or Y: no exit barrier (the non-conforming upper bound, bench-only)
  rt <geom>_<mode>: 1 the ce25f67 tail, 2 split with a release arrive, 3 split with a relaxed
     arrive, 0 none (bench-only).
  "prod" arms come from the production extension (whatever submission.py ships).

Every arm passes FP64 checks first (random x with probe rows, at adjacent sizes, on the given
router and on one with outlier and tiny weights), then a CUDA-graph replay check with fresh
inputs written into the captured buffer twice, else it is dropped. Timing: cold-L2 graph
replays, 100 rounds in a seeded random order. Lines:

  TB m=<m> med          median us per arm after the replay overhead
  TB m=<m> d <ref>      paired median difference arm - ref (us) / rounds the arm was faster
  TBCHECK               dropped arms and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

N, K, L, U, S, E, D, T, W, B, X, Y, Z = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
BITS = (N, K, L, U, S, E, D, T, W, B, X, Y, Z)
TAGS = "NKLUSEDTWBXYZ"
KLUE = K | L | U | E
ROUND = "tail-a"
_SET = [KLUE | T, KLUE | T | D, KLUE | Y, KLUE | Y | Z, KLUE | Y | Z | D, KLUE]
WD_SETS = {"r64c8s8": _SET, "r128c8s6": _SET}
WD_BASES = {128: ("r64c8s8",), 256: ("r128c8s6",)}
# Production names of the shipped router picks per base geometry.
WD_PROD = {"r64c8s8": "r64c8s8k", "r128c8s6": "r128c8s6k"}
RT_BENCH = {f"{g}_{mode}": (s_, wk, p_, mode) for g, (s_, wk, p_) in
            (("rt8w8", (8, 8, 4)), ("rt16w8", (16, 8, 4)), ("rt16w8p8", (16, 8, 8))) for mode in (1, 2, 3, 0)}
RT_BUCKET = {8: "rt8w8", 16: "rt16w8", 32: "rt16w8p8"}


def var_tag(v):
    if v == 0:
        return "v0"
    return "v" + "".join(c for b, c in zip(BITS, TAGS) if v & b)


def bench_wd():
    return {f"{base}_{var_tag(v)}": (base, v) for base, sets in WD_SETS.items() for v in dict.fromkeys(sets)}


BENCH = r'''

_R2_WD = @@WD@@
_R2_RT = @@RT@@
_R2_BASES = @@BASES@@
_R2_PROD = @@PROD@@
_R2_RTB = @@RTB@@
_R2_ROUND = @@ROUND@@
import os as _r2os
import random as _r2random
import math as _r2math

_R2_OUT = []


def _r2print(s):
    print(s, flush=True)
    _R2_OUT.append(s)


def _r2_build_bench(name, flags, wd, rt):
    """A bench extension: the sk router configs, the given wd variants and rt geometries only."""
    g = globals()
    saved = {nm: g[nm] for nm in ("SK_CONFIGS", "SK_INDEX", "WD_GEOM", "WD_VAR", "WD_NAMES", "WD_INDEX",
                                  "WD_ABL", "RT_GEOM", "RT_NAMES")}
    sk_keep = set(ROUTER_GEOM)
    try:
        g["SK_CONFIGS"] = [c for c in saved["SK_CONFIGS"] if c[0] in sk_keep]
        g["SK_INDEX"] = {c[0]: i for i, c in enumerate(g["SK_CONFIGS"])}
        g["WD_GEOM"] = {nm: saved["WD_GEOM"][b] for nm, (b, v) in wd.items()}
        g["WD_VAR"] = {nm: v for nm, (b, v) in wd.items()}
        g["WD_ABL"] = {}
        g["WD_NAMES"] = list(wd)
        g["WD_INDEX"] = {nm: i for i, nm in enumerate(wd)}
        g["RT_GEOM"] = dict(rt)
        g["RT_NAMES"] = list(rt)
        src = sk_source() + wd_source() + rt_source()
        idx = (dict(g["SK_INDEX"]), dict(g["WD_INDEX"]), list(g["RT_NAMES"]))
    finally:
        g.update(saved)
    ext = load_inline(name=name, cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + _WD_CPP + _RT_CPP,
                      cuda_sources=src, functions=["sk_fwd", "wd_fwd", "rt_fwd"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + flags, verbose=False,
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
    """Arm name -> fn for router shape m, from the production and bench extensions, and the
    reference arm (the ce25f67 exit barrier, bench copy) the d line is paired against."""
    prod, bench, (sk_idx, wd_idx, rt_names), wts = st["prod"], st["bench"], st["idx"], st["wts"]
    arms = {}
    if m >= 128:
        base = _R2_BASES[m][0]
        pn = _R2_PROD[base]
        arms[f"{pn}:prod"] = _r2_wd_run(prod, WD_INDEX[pn], wts["pack"], 256, 8192, diag)
        for nm, (b, v) in _R2_WD.items():
            if b == base:
                w = wts["pack"] if v & WV_PACK else wts["hilo"]
                arms[f"{base}:{nm.split('_', 1)[1]}"] = _r2_wd_run(bench, wd_idx[nm], w, 256, 8192, diag)
        ref = f"{base}:vKLUET"
    else:
        g = _R2_RTB[m]
        arms[f"{g}:prod"] = _r2_rt_run(prod, RT_NAMES.index(g), wts["rt"], diag)
        for nm in rt_names:
            if nm.rsplit("_", 1)[0] == g:
                arms[f"{nm}:v0"] = _r2_rt_run(bench, rt_names.index(nm), wts["rt"], diag)
        ref = f"{g}_1:v0"
    arms["sglang"] = lambda x: _router_baseline(x, st["sgl"])
    return arms, ref


def _r2_state(router, prod, bench, idx, dev):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    rgeoms = {nm: _sk_geom(nm) for nm in ROUTER_GEOM}
    _, _, rstreams = pack_router(router, rgeoms)
    return {"prod": prod, "bench": bench, "idx": idx,
            "wts": {"hilo": torch.cat([hi, lo], 0).contiguous(), "pack": router_pack(hi, lo),
                    "rt": rt_pack(hi, lo), "sk": rstreams},
            "sgl": {"router": router, "router_hi_t": hi.t().contiguous(), "router_lo_t": lo.t().contiguous()}}


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
    bench, idx = _r2_build_bench("okc_tb_bench", [], _R2_WD, _R2_RT)
    _r2print(f"TBBUILD prod {t1 - t0:.1f} s, bench {time.time() - t1:.1f} s, {len(_R2_WD)} wd arms, "
             f"{len(_R2_RT)} rt arms, round {_R2_ROUND}")
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    w_main = _r2_weights(dev, gen, False)
    w_out = _r2_weights(dev, gen, True)
    st_main = _r2_state(w_main, prod, bench, idx, dev)
    st_out = _r2_state(w_out, prod, bench, idx, dev)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _r2_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _r2_graph(two_fn, None)
    ov = _r2_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _r2print(f"TBOVH {ovh:.2f} us")
    adj = {8: (1, 5, 7, 8), 16: (9, 13, 16), 32: (17, 31, 32), 128: (65, 100, 128), 256: (129, 200, 256)}
    for ti, m in enumerate((8, 16, 32, 128, 256)):
        arms, ref = _r2_arms(m, st_main, dev, diag)
        oarms, _ = _r2_arms(m, st_out, dev, diag)
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
            _r2print(f"TBCHECK m={m} dropped: " + " ".join(bad))
        base = ref if ref in graphs else None
        if base is None:
            _r2print(f"TB m={m} reference {ref} dropped")
            continue
        smp = _r2_time(graphs, 100, flush, 2000 + ti)
        graphs = None
        med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
        dl = {nm: statistics.median([a - b for a, b in zip(v, smp[base])]) for nm, v in smp.items()}
        wins = {nm: sum(a < b for a, b in zip(v, smp[base])) for nm, v in smp.items()}
        order = sorted(med, key=med.get)
        _r2print(f"TB m={m} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
        _r2print(f"TB m={m} d {base} " + " ".join(f"{nm}={dl[nm]:+.2f}/{wins[nm]}" for nm in order if nm != base))
        torch.cuda.empty_cache()
    _r2print(f"TBDONE {time.time() - t0:.1f} s")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _r2os.environ.get("PROJ_TB_DONE"):
    _r2os.environ["PROJ_TB_DONE"] = "1"
    try:
        _r2_run()
    except Exception as _exc:
        import traceback as _tb
        _r2print(f"TBBENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _r2_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _r2_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("tune router", "ltune", "prepare done", "sk build", "lt build")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _R2_OUT:
            print(line, flush=True)
        return state
'''


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    cu = None
    while args:
        a = args.pop(0)
        if a == "--cu":
            cu = Path(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    wd = bench_wd()
    bench = (BENCH.replace("@@WD@@", repr(wd)).replace("@@RT@@", repr(RT_BENCH))
             .replace("@@BASES@@", repr(WD_BASES)).replace("@@PROD@@", repr(WD_PROD))
             .replace("@@RTB@@", repr(RT_BUCKET)).replace("@@ROUND@@", repr(ROUND)))
    out.write_text(sub + bench)
    print(f"wrote {out}: {len(wd)} wd arms, {len(RT_BENCH)} rt arms")
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
        keep = set(mod.ROUTER_GEOM)
        mod.SK_CONFIGS = [c for c in mod.SK_CONFIGS if c[0] in keep]
        cu.write_text(mod.sk_source(device_only=True))
        print(f"wrote {cu}")


if __name__ == "__main__":
    main()
