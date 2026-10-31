"""Write the router reference-difference bench (round 6): the three differences between our wd
router and the independent reference's that no earlier bench toggled (docs/ruled-out.md, wd XA
row), one at a time and together, at m = 128 / 256, with independent duplicate nulls.

    python3 tools/proj_make_bench_rg.py OUT.py [--donor EXPANDED.py] [--cu OUT.cu]

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import, before the judged prepare(). It builds the production extension (_build_ext, cached for
the judged prepare) and a bench extension with the arms below. Tags are the wd VAR bits:

  K L U E  the production KLUE set (packed hi/lo box, 64 B promotion, push reduction, early
           prologue); Y Z the shipped split exit barrier with a relaxed arrive
  G  WV_PGROUP, a 4-warp producer group (the reference's 256-thread CTA at m = 256)
  A  WV_ACQCL, consumer full and producer empty waits as try_wait.parity.acquire.cluster
  S  WV_SPIN, no watchdog; with A it is the reference's single-instruction branch loop
  Q  WV_SCALAR, scalar output stores in the XA epilogue (the reference's emit)
  X  WV_XA, x as the wgmma A operand (the reference's m = 256 orientation), W no suspend hint
  D  duplicate (no effect), the independent null
  "prod" arms come from the production extension (whatever submission.py ships).
  d_rw with --donor: NathenFernandes/muse-bf16-proj v18 (MIT), its router kernels as published,
  the positive control. A file built with --donor is a check-only bench: never submit it.

Every arm passes FP64 checks first (random x with probe rows, at adjacent sizes, on the given
router and on one with outlier and tiny weights), then a CUDA-graph replay check with fresh
inputs written into the captured buffer twice, else it is dropped. A spin (S) arm runs only
after its watchdog twin passed. Timing: cold-L2 graph replays, 100 rounds in a seeded random
order. Lines:

  RG m=<m> med          median us per arm after the replay overhead
  RG m=<m> d <ref>      paired median difference arm - ref (us) / rounds the arm was faster
  RGCHECK               dropped arms and why
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

N, K, L, U, S, E, D, T, W, B, X, Y, Z = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
G, A, Q, R, M, J, O, H, F, V = 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304
BITS = (N, K, L, U, S, E, D, T, W, B, X, Y, Z, G, A, Q, R, M, J, O, H, F, V)
TAGS = "NKLUSEDTWBXYZGAQRMJOHFV"
KLUE = K | L | U | E
P = KLUE | Y | Z
# Round a (check 97869fa8302041b5) lacked the watchdog twins of the G|A|S arms, so the spin rule
# dropped them; round b adds them.
# Round c (F-PROJ-1): R = WV_GRED, the K split without a cluster (red.add into a global
# accumulator, last-arriver copy), the donor rk router's mechanism, at its CTA counts: 64 rows x
# 64 tokens, split 12 at m = 128 (96 CTAs) and 8 at m = 256 (128 CTAs), with neighbours.
# Round d: R + M (WV_GZERO), out zeroed by a memset node and the partials added into it as the
# CTA's last stores (round c's counter join cost about 4 us: rk's reds are fire-and-forget).
# Round rk-a (stamp the donor): the donor rk router itself (--donor-rk, check only) beside our
# wd at its geometry (64 x 64 tiles, split 12 at m = 128 and 8 at m = 256, memset-zeroed red.add
# join R + M), then its differences swapped into ours one at a time: J (WV_E1, one empty arrive
# per stage), O (WV_RKORD, rk block order), S + W (spin consumer waits, no producer hint), no
# packed box (K off), 6 stages, and all of them together. t_zero times the zeroing node alone.
# Round rk-b (after 3fd0c7a7cea74ab4: the rk block order O is worth about 1.1 us and the
# memset node about 1.1 us): H = WV_GTICKET, the tile cleared inside the kernel by the CTA with
# the first ticket (no zeroing node, no co-residency), with O, against the GZ + O form.
# Round rk-c (round 9, after a SASS diff of rk against wd): F = WV_PFTM (prefetch both tensor
# maps, which rk does and wd does not), V = WV_UDESC (uniform wgmma descriptors, no R2UR per
# HGMMA), on the production routers; and the order-only form without a new cross-CTA protocol
# (O with the memset-zeroed red.add, R + M) at m = 256, alone and with F + V.
ROUND = "rk-c"
KLE = K | L | E
GEOM_EXTRA = {"r64c12s8": (1, 64, 12, 1, True, 8), "r64c16s8": (1, 64, 16, 1, True, 8),
              "r64c6s8": (1, 64, 6, 1, True, 8), "r64c8s6": (1, 64, 8, 1, True, 6),
              "r64c12s6": (1, 64, 12, 1, True, 6)}
# Production picks r64c8s8 (m = 128) and r128c8s6 (m = 256) with every untested difference alone
# and together; at m = 256 also the reference's own geometry and orientation (r64c4s8, XA), where
# the gap at identical tiling was 2.4 us (6cc72e840403402b).
_SET = [P, P | D, P | A, P | A | S, P | G, P | G | A, P | G | A | D, P | G | A | S, P | G | A | S | D]
WD_SETS_B = {
    "r64c8s8": _SET,
    "r128c8s6": _SET,
    "r64c4s8": [P | W | X, P | W | X | D, P | W | X | Q, P | W | X | A, P | W | X | A | S, P | W | X | G,
                P | W | X | G | A | Q, P | W | X | G | A | Q | D, P | W | X | G | A | S | Q,
                P | W | X | G | A | S | Q | D],
}
LE = L | E
GZ = R | M
TK = R | H
WD_SETS = {
    "r64c8s8": [P, P | D, P | F, P | V, P | F | V, P | F | V | D, KLE | GZ | O, KLE | GZ | O | F | V],
    "r128c8s6": [P, P | D, P | F, P | V, P | F | V, P | F | V | D],
}
WD_BASES = {128: ("r64c8s8",), 256: ("r128c8s6", "r64c8s8")}
WD_PROD = {"r64c8s8": "r64c8s8k", "r128c8s6": "r128c8s6k"}
WD_REF = {128: "r64c8s8:vKLUEYZ", 256: "r128c8s6:vKLUEYZ"}


def var_tag(v):
    if v == 0:
        return "v0"
    return "v" + "".join(c for b, c in zip(BITS, TAGS) if v & b)


def bench_wd():
    return {f"{base}_{var_tag(v)}": (base, v) for base, sets in WD_SETS.items() for v in dict.fromkeys(sets)}


BENCH = r'''

_RG_WD = @@WD@@
_RG_BASES = @@BASES@@
_RG_PROD = @@PROD@@
_RG_GEOM_EXTRA = @@GEOMX@@
_RG_REF = @@REF@@
_RG_ROUND = @@ROUND@@
_RG_DONOR = @@DONOR@@
import os as _rgos
import random as _rgrandom
import math as _rgmath

_RG_OUT = []


def _rgprint(s):
    print(s, flush=True)
    _RG_OUT.append(s)


def _rg_build_bench(name, wd):
    # Swaps the module tables for the bench variants just long enough to render the source.
    g = globals()
    saved = {nm: g[nm] for nm in ("SK_CONFIGS", "SK_INDEX", "WD_GEOM", "WD_VAR", "WD_NAMES", "WD_INDEX", "WD_ABL")}
    try:
        g["SK_CONFIGS"] = [c for c in saved["SK_CONFIGS"] if c[0] in ROUTER_GEOM]
        g["SK_INDEX"] = {c[0]: i for i, c in enumerate(g["SK_CONFIGS"])}
        g["WD_GEOM"] = {nm: {**saved["WD_GEOM"], **_RG_GEOM_EXTRA}[b] for nm, (b, v) in wd.items()}
        g["WD_VAR"] = {nm: v for nm, (b, v) in wd.items()}
        g["WD_ABL"] = {}
        g["WD_NAMES"] = list(wd)
        g["WD_INDEX"] = {nm: i for i, nm in enumerate(wd)}
        src = sk_source() + wd_source()
        idx = dict(g["WD_INDEX"])
    finally:
        g.update(saved)
    ext = load_inline(name=name, cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + _WD_CPP,
                      cuda_sources=src, functions=["sk_fwd", "wd_fwd"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False,
                      no_implicit_headers=True)
    return ext, idx


def _rg_wd_run(ext, cid, w, n, k, diag):
    def run(x):
        out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
        ext.wd_fwd(cid, w, x, out, n, k, diag)
        return out
    return run


def _rg_donor_rk(router):
    # Check-only: the donor's rk kernel compiled from its source, as published.
    import hashlib as _hl
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    wc = torch.cat([hi, lo], dim=1).contiguous()
    zc = torch.zeros(16, dtype=torch.int32, device=router.device)
    src = ("#include <cuda.h>\n#include <cudaTypedefs.h>\n#include <ATen/ATen.h>\n"
           "#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>\n" + _RG_DONOR["rk"])
    ext = load_inline("rk_stamp_" + _hl.sha256(src.encode()).hexdigest()[:12],
                      "#include <torch/extension.h>\nvoid rk(at::Tensor Wc, at::Tensor x, at::Tensor out, int64_t S, "
                      "at::Tensor zc);", cuda_sources=src, functions=["rk"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    arms = {}
    for m, sp in ((128, 12), (256, 8)):
        def run(x, sp=sp):
            out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
            ext.rk(wc, x, out, sp, zc)
            return out
        arms[m] = run
    return arms


def _rg_donor(router):
    # Check-only positive control: the reference's router kernels compiled from its source.
    if _RG_DONOR is None:
        return {}
    if "rk" in _RG_DONOR:
        try:
            return _rg_donor_rk(router)
        except Exception as exc:
            _rgprint(f"RGDONOR rk failed {type(exc).__name__}: {str(exc)[-400:]}")
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
            src = _RG_DONOR[arm]
            cpp = ("#include <stdint.h>\\n#include <pybind11/pybind11.h>\\nint " + label
                   + "_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);")
            ext = load_inline("rg_rw_" + _hl.sha256(src.encode()).hexdigest()[:12], cpp, cuda_sources=src,
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
        _rgprint(f"RGDONOR failed {type(exc).__name__}: {str(exc)[-300:]}")
    return arms


def _rg_graph(fn, xbuf):
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


def _rg_time(graphs, rounds, flush, seed):
    rng = _rgrandom.Random(seed)
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


def _rg_weights(dev, gen, outliers):
    k = 8192
    w = torch.randn((256, k), generator=gen, device=dev) / _rgmath.sqrt(k)
    if outliers:
        big = torch.rand((256, k), generator=gen, device=dev) < 1 / 512
        tiny = torch.rand((256, k), generator=gen, device=dev) < 1 / 64
        w = torch.where(big, w * 4096.0, w)
        w = torch.where(tiny, w * 1e-6, w)
    return w.float().contiguous()


def _rg_state(router, prod, bench, idx):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {"prod": prod, "bench": bench, "idx": idx,
            "wts": {"hilo": torch.cat([hi, lo], 0).contiguous(), "pack": router_pack(hi, lo)},
            "donor": _rg_donor(router)}


def _rg_arms(m, st, diag):
    prod, bench, wd_idx, wts = st["prod"], st["bench"], st["idx"], st["wts"]
    arms = {}
    for base in _RG_BASES[m]:
        if base in _RG_PROD:
            pn = _RG_PROD[base]
            arms[f"{pn}:prod"] = _rg_wd_run(prod, WD_INDEX[pn], wts["pack"], 256, 8192, diag)
        for nm, (b, v) in _RG_WD.items():
            if b == base:
                w = wts["pack"] if v & WV_PACK else wts["hilo"]
                arms[f"{base}:{nm.split('_', 1)[1]}"] = _rg_wd_run(bench, wd_idx[nm], w, 256, 8192, diag)
    if m in st["donor"]:
        arms["d_rw"] = st["donor"][m]

    def t_zero(x):
        return torch.zeros((x.shape[0], 256), device=x.device, dtype=torch.float32)
    arms["t_zero"] = t_zero
    return arms


def _rg_spin_twin(nm):
    base, _, tag = nm.partition(":")
    if not tag.startswith("v") or "S" not in tag:
        return None
    return f"{base}:{tag.replace('S', '')}"


def _rg_check_all(arms, router, ms, gen, dev, label):
    bad = []
    order = sorted(arms, key=lambda nm: _rg_spin_twin(nm) is not None)
    for m in ms:
        x, probes = _gen_x(m, 8192, gen, dev)
        ref = x.double() @ router.double().T
        for nm in order:
            if nm not in arms:
                continue
            if nm.startswith("t_"):
                continue
            twin = _rg_spin_twin(nm)
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


def _rg_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(4343)
    prod = _build_ext()
    t1 = time.time()
    bench, idx = _rg_build_bench("okc_rg_bench", _RG_WD)
    _rgprint(f"RGBUILD prod {t1 - t0:.1f} s, bench {time.time() - t1:.1f} s, {len(_RG_WD)} wd arms, "
             f"donor {'on' if _RG_DONOR else 'off'}, round {_RG_ROUND}")
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    w_main = _rg_weights(dev, gen, False)
    w_out = _rg_weights(dev, gen, True)
    st_main = _rg_state(w_main, prod, bench, idx)
    st_out = _rg_state(w_out, prod, bench, idx)
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one, _ = _rg_graph(lambda x: tiny.add_(1), None)

    def two_fn(x):
        tiny.add_(1)
        tiny.add_(1)
    two, _ = _rg_graph(two_fn, None)
    ov = _rg_time({"one": one, "two": two}, 200, flush, 1)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _rgprint(f"RGOVH {ovh:.2f} us")
    adj = {128: (65, 100, 128), 256: (129, 200, 256)}
    for ti, m in enumerate((128, 256)):
        arms = _rg_arms(m, st_main, diag)
        oarms = _rg_arms(m, st_out, diag)
        bad = _rg_check_all(arms, w_main, adj[m], gen, dev, "")
        bad += _rg_check_all(oarms, w_out, adj[m], gen, dev, "o")
        for nm in list(arms):
            if nm not in oarms:
                arms.pop(nm)
        xbuf, _ = _gen_x(m, 8192, gen, dev)
        graphs = {}
        for nm in list(arms):
            try:
                g, out = _rg_graph(arms[nm], xbuf)
                for rep in range(2):
                    x2, probes = _gen_x(m, 8192, gen, dev)
                    xbuf.copy_(x2)
                    g.replay()
                    torch.cuda.synchronize()
                    ok, err, perr = _check(out, x2.double() @ w_main.double().T, probes, SELFTEST_TOL["router"])
                    if not ok and not nm.startswith("t_"):
                        raise RuntimeError(f"replay {rep} err {err:.1e}/{perr:.1e}")
                graphs[nm] = g
            except Exception as exc:
                torch.cuda.synchronize()
                bad.append(f"{nm}@replay={str(exc)[-80:]}")
        if bad:
            _rgprint(f"RGCHECK m={m} dropped: " + " ".join(bad))
        base = _RG_REF[m]
        if base not in graphs:
            _rgprint(f"RG m={m} reference {base} dropped")
            continue
        smp = _rg_time(graphs, 100, flush, 3000 + ti)
        graphs = None
        med = {nm: statistics.median(v) - ovh for nm, v in smp.items()}
        dl = {nm: statistics.median([a - b for a, b in zip(v, smp[base])]) for nm, v in smp.items()}
        wins = {nm: sum(a < b for a, b in zip(v, smp[base])) for nm, v in smp.items()}
        order = sorted(med, key=med.get)
        _rgprint(f"RG m={m} med " + " ".join(f"{nm}={med[nm]:.2f}" for nm in order))
        _rgprint(f"RG m={m} d {base} " + " ".join(f"{nm}={dl[nm]:+.2f}/{wins[nm]}" for nm in order if nm != base))
        torch.cuda.empty_cache()
    _rgprint(f"RGDONE {time.time() - t0:.1f} s")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _rgos.environ.get("PROJ_RG_DONE"):
    _rgos.environ["PROJ_RG_DONE"] = "1"
    try:
        _rg_run()
    except Exception as _exc:
        import traceback as _tb
        _rgprint(f"RGBENCH failed: {type(_exc).__name__}: {str(_exc)[:300]} {_tb.format_exc()[-900:]}")
    _rg_prepare_judged = prepare

    def prepare(weights):
        import contextlib as _ctx
        import io as _io
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            state = _rg_prepare_judged(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith(("tune router", "prepare done", "sk build")) or "disabled" in ln:
                print("judged| " + ln, flush=True)
        for line in _RG_OUT:
            print(line, flush=True)
        return state
'''


def donor_rk(path):
    # The donor's RK_SRC string, read as a literal without executing its file.
    import ast
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "RK_SRC":
            return {"rk": ast.literal_eval(node.value)}
    raise SystemExit(f"{path}: no RK_SRC")


def donor_rw(path):
    # The reference's two router sources, read as literals without executing its file.
    import ast
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") == "rw_SOURCES":
            v = ast.literal_eval(node.value)
            return {128: v[128], 256: v[256]}
    raise SystemExit(f"{path}: no rw_SOURCES")


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    cu = None
    donor = None
    while args:
        a = args.pop(0)
        if a == "--cu":
            cu = Path(args.pop(0))
        elif a == "--donor":
            donor = donor_rw(args.pop(0))
        elif a == "--donor-rk":
            donor = donor_rk(args.pop(0))
    sub = (ROOT / "muse_bf16_proj/submission.py").read_text()
    wd = bench_wd()
    bench = (BENCH.replace("@@WD@@", repr(wd)).replace("@@BASES@@", repr(WD_BASES))
             .replace("@@PROD@@", repr(WD_PROD)).replace("@@GEOMX@@", repr(GEOM_EXTRA)).replace("@@REF@@", repr(WD_REF))
             .replace("@@ROUND@@", repr(ROUND)).replace("@@DONOR@@", repr(donor)))
    out.write_text(sub + bench)
    print(f"wrote {out}: {len(wd)} wd arms, donor arms {'on (check only, never submit)' if donor else 'off'}")
    if cu is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sub", ROOT / "muse_bf16_proj/submission.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.WD_GEOM = {nm: {**mod.WD_GEOM, **GEOM_EXTRA}[b] for nm, (b, v) in wd.items()}
        mod.WD_VAR = {nm: v for nm, (b, v) in wd.items()}
        mod.WD_ABL = {}
        mod.WD_NAMES = list(wd)
        mod.SK_CONFIGS = [c for c in mod.SK_CONFIGS if c[0] in mod.ROUTER_GEOM]
        cu.write_text(mod.sk_source(device_only=True))
        print(f"wrote {cu}")


if __name__ == "__main__":
    main()
