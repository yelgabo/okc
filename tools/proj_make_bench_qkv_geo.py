"""qkv_gate m = 128 wd geometry bench on the exact public C2 bytes; never submit the output.

Hypothesis (QKV128-GEO-01): the 2.4-2.7 us kernel-body gap to comparator 3c6a2baf at qkv_gate
m = 128 (wd b1n128c3s8h 13.9-14.3 us profiled vs rq 11.6 us) is set by the wd memory pipeline's
geometry. b1n128c3s8h streams 64 weight rows against 128 x tokens per stage, so every SM pulls
twice as many x bytes from L2 as weight bytes from HBM, on 108 of 132 SMs. Same kernel, new
template arguments only: 128-row CTAs (two consumer warpgroups) at K split 6 / 7, x multicast
across two 64-row CTAs, and one more ring stage. Ablations of the control and of b2c6 attribute
the time to the memory pipeline, the wgmma and the cluster reduction.

Arms (all on one fresh judge-like m = 128 input per held-out round set, cold L2, CUDA graphs,
shuffled order, 100 paired rounds, 3 held-out inputs):
  ctrl     production pick from C2's own prepare (b1n128c3s8h in every recorded log)
  ctrl2    the same template arguments compiled in this bench's module (independent-build null)
  cand     b2c6, b2c7, b2c6s7, mc3, g0s9; ablations g0_a1, g0_a2, b2c6_a1, b2c6_a2 (wrong output)

Pre-registered keep rule: a candidate is kept if its median paired delta against ctrl and against
ctrl2 is <= -0.5 us on all 3 inputs, it is faster in >= 90 of 100 rounds on each, and it passes
every FP64 / probe / captured-replay gate. Only the qkv_gate m = 128 bucket is touched.

    python3 tools/proj_make_bench_qkv_geo.py muse_bf16_proj/experiments/qkv_geo.py
"""
import ast
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import proj_sk_xor_control  # noqa: E402

# name -> (WG, BN, CLK, CLR, HL, NSTAGE); every arm carries the L2 policy like the production pick.
GEO = {
    "g0": (1, 128, 3, 1, False, 8),
    "b2c6": (2, 128, 6, 1, False, 6),
    "b2c7": (2, 128, 7, 1, False, 6),
    "b2c6s7": (2, 128, 6, 1, False, 7),
    "mc3": (1, 128, 3, 2, False, 8),
    "g0s9": (1, 128, 3, 1, False, 9),
}
ABL = {"g0_a1": ("g0", 1), "g0_a2": ("g0", 2), "b2c6_a1": ("b2c6", 1), "b2c6_a2": ("b2c6", 2)}

INFO = r'''
#include <vector>
template <int WG, int BN, int CLK, int CLR, bool HL, int NST, typename OutT, int ABL, bool POL, int VAR = 0>
std::vector<int64_t> qg_info_one(int N, int M) {
    using C = WdCfg<WG, BN, CLK, CLR, HL, NST, VAR>;
    auto k = wd_kernel<WG, BN, CLK, CLR, HL, NST, OutT, ABL, POL, VAR>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
    if (C::LCL > 8) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }
    cudaFuncAttributes a{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, k));
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3((unsigned)(N / (C::BM * CLR) * C::CL), (unsigned)((M + BN - 1) / BN));
    lc.blockDim = dim3(C::THREADS);
    lc.dynamicSmemBytes = C::SMEM;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = C::LCL;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    lc.attrs = &attr;
    lc.numAttrs = C::LCL > 1 ? 1 : 0;
    int clusters = 0, blocks = 0;
    if (C::LCL > 1) {
        C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, k, &lc));
    }
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, C::THREADS, C::SMEM));
    return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, C::SMEM, blocks, clusters,
            (int64_t)lc.gridDim.x * lc.gridDim.y, C::LCL};
}
std::vector<int64_t> qg_info(int64_t cfg, int64_t n, int64_t m) {
    switch (cfg) {
@@CASES@@
    default:
        TORCH_CHECK(false, "qg_info: unknown config ", cfg);
    }
    return {};
}
'''

RUN = r'''
import random as _qgrandom
import statistics as _qgstats
_QG_GEO = @@GEO@@
_QG_ABL = @@ABL@@
_QG_INFO = @@INFO@@
_QGLOG = []


def _qgprint(s):
    _QGLOG.append(s)


def _qgbuild():
    keys = ("SK_CONFIGS", "SK_INDEX", "WD_GEOM", "WD_VAR", "WD_NAMES", "WD_INDEX", "WD_ABL", "WD_POL")
    scope = globals()
    snapshot = {key: scope[key] for key in keys}
    try:
        scope["SK_CONFIGS"] = []
        scope["SK_INDEX"] = {}
        scope["WD_GEOM"] = dict(_QG_GEO)
        scope["WD_VAR"] = {}
        scope["WD_ABL"] = dict(_QG_ABL)
        scope["WD_POL"] = set(_QG_GEO)
        scope["WD_NAMES"] = list(_QG_GEO) + list(_QG_ABL)
        scope["WD_INDEX"] = {nm: i for i, nm in enumerate(scope["WD_NAMES"])}
        names = list(scope["WD_NAMES"])
        cases = "\n".join(f"    case {i}: return qg_info_one<{_wd_targs(nm)}>((int)n, (int)m);"
                          for i, nm in enumerate(names))
        cuda = sk_source() + wd_source() + _QG_INFO.replace("@@CASES@@", cases)
    finally:
        scope.update(snapshot)
    mod = load_inline(name="okc_qkv_geo", cpp_sources="#include <torch/extension.h>\n" + _SK_CPP + _WD_CPP
                      + "std::vector<int64_t> qg_info(int64_t cfg, int64_t n, int64_t m);\n",
                      cuda_sources=cuda, functions=["wd_fwd", "qg_info"], no_implicit_headers=True,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return mod, {nm: i for i, nm in enumerate(names)}


def _qgbench(weights, state, started):
    t1 = time.time()
    mod, index = _qgbuild()
    _qgprint(f"QG build {time.time() - t1:.1f} s")
    w = state["qkv_gate"]
    n, k = w.shape
    dev = w.device
    for nm, i in index.items():
        r = mod.qg_info(i, n, 128)
        waves = "one-wave" if r[7] == 1 or r[5] * r[7] >= r[6] else "MULTI-WAVE"
        _qgprint(f"QGRES {nm} regs={r[0]} local={r[1]} static={r[2]} dyn={r[3]} blocks/SM={r[4]} "
                 f"maxclusters={r[5]} grid={r[6]} cluster={r[7]} {waves}")
    gen = torch.Generator(device=dev).manual_seed(20261103)
    diag = torch.zeros(4, dtype=torch.int32, device=dev)
    arms = {"ctrl": state["pick"][("qkv_gate", 128)]}

    def make(i):
        def run(x):
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            mod.wd_fwd(i, w, x, out, n, k, diag)
            return out
        return run
    for nm, i in index.items():
        arms["ctrl2" if nm == "g0" else nm] = make(i)
    ablated = {nm for nm in _QG_ABL}
    failed = set()
    for m in (65, 100, 127, 128):
        for tag, fn in arms.items():
            if tag in ablated or tag in failed:
                continue
            x, probes = _gen_x(m, k, gen, dev)
            diag.zero_()
            try:
                out = fn(x)
                torch.cuda.synchronize()
                ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
                ok, err, perr = _check(out, ref, probes, SELFTEST_TOL["bf16"])
                if not ok or int(diag.abs().max()):
                    failed.add(tag)
                    _qgprint(f"QGFAIL {tag} m={m} err={err:.3e} probe={perr:.3e} diag={int(diag.abs().max())}")
            except Exception as exc:
                torch.cuda.synchronize()
                failed.add(tag)
                _qgprint(f"QGFAIL {tag} m={m} {type(exc).__name__}: {str(exc)[-300:]}")
    if "ctrl" in failed or "ctrl2" in failed:
        _qgprint("QGSTOP a control failed numerics; no timing")
        return
    for tag in failed:
        arms.pop(tag)
    _qgprint(f"QGCHECK FP64+probe m=65/100/127/128 PASS {sorted(t for t in arms if t not in ablated)}")
    for tag, fn in list(arms.items()):
        if tag in ablated:
            continue
        x, _ = _gen_x(128, k, gen, dev)
        fn(x)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn(x)
        for _ in range(3):
            x.copy_(_gen_x(128, k, gen, dev)[0])
            g.replay()
            torch.cuda.synchronize()
            ref = (x.double() @ w.double().T).to(torch.bfloat16).double()
            err = float((out.double() - ref).norm() / ref.norm())
            if not err < SELFTEST_TOL["bf16"]:
                arms.pop(tag)
                _qgprint(f"QGFAIL {tag} captured replay err={err:.3e}")
                break
    _qgprint(f"QGREPLAY 3 fresh captured inputs PASS {sorted(t for t in arms if t not in ablated)}")
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    rounds = 100
    for inp in range(3):
        if time.time() - started > 240:
            _qgprint(f"QGSTOP time guard before input {inp}")
            break
        x, _ = _gen_x(128, k, gen, dev)
        graphs = {}
        for tag, fn in arms.items():
            fn(x)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(x)
            graphs[tag] = g
        times = {tag: [] for tag in arms}
        rng = _qgrandom.Random(20261103 + inp)
        for rd in range(rounds):
            order = list(arms)
            rng.shuffle(order)
            for tag in order:
                flush.zero_()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                graphs[tag].replay()
                b.record()
                b.synchronize()
                times[tag].append(1000 * a.elapsed_time(b))
        null = [p - q for p, q in zip(times["ctrl2"], times["ctrl"])]
        nmed = _qgstats.median(null)
        nmad = _qgstats.median(abs(z - nmed) for z in null)
        _qgprint(f"QGNULL in{inp} ctrl2-ctrl={nmed:+.3f} MAD={nmad:.3f} ctrl={_qgstats.median(times['ctrl']):.3f}")
        for tag, ts in times.items():
            if tag == "ctrl":
                continue
            d1 = [p - q for p, q in zip(ts, times["ctrl"])]
            d2 = [p - q for p, q in zip(ts, times["ctrl2"])]
            m1, m2 = _qgstats.median(d1), _qgstats.median(d2)
            f1 = sum(z < 0 for z in d1)
            keep = tag not in ablated and m1 <= -0.5 and m2 <= -0.5 and f1 >= 90 and sum(z < 0 for z in d2) >= 90
            _qgprint(f"QGTIME in{inp} {tag} raw={_qgstats.median(ts):.3f} d_ctrl={m1:+.3f} d_ctrl2={m2:+.3f} "
                     f"faster={f1}/{rounds}{' KEEP' if keep else ''}")
    if int(diag.abs().max()):
        _qgprint(f"QGFAIL watchdog diag={diag.tolist()}")
    _qgprint(f"QGDONE elapsed={time.time() - started:.1f} s")


_qg_original_prepare = prepare


def prepare(weights):
    started = time.time()
    state = _qg_original_prepare(weights)
    try:
        _qgbench(weights, state, started)
    except Exception as exc:
        torch.cuda.synchronize()
        _qgprint(f"QGERROR {type(exc).__name__}: {str(exc)[-1500:]}")
    for line in _QGLOG:
        print(line, flush=True)
    return state
'''


def make(source):
    out = (source + RUN.replace("@@GEO@@", repr(GEO)).replace("@@ABL@@", repr(ABL))
           .replace("@@INFO@@", repr(INFO)))
    ast.parse(out)
    return out


def _stub_torch():
    class Any:
        def __getattr__(self, name):
            return Any()

        def __call__(self, *a, **k):
            return Any()

    def mod(name):
        m = types.ModuleType(name)
        m.__getattr__ = lambda attr: Any()
        return m
    for name in ("torch", "torch.nn", "torch.nn.functional", "torch.utils", "torch.utils.cpp_extension"):
        sys.modules.setdefault(name, mod(name))


def host_gate(text):
    """Run the bench's real builder on CPU with a fake compiler: the emitted CUDA must hold every
    arm's instantiation, and every C2 global it swaps must come back unchanged."""
    _stub_torch()
    ns = {"__name__": "qkv_geo_gate"}
    exec(compile(text, "qkv_geo.py", "exec"), ns)
    keys = ("SK_CONFIGS", "SK_INDEX", "WD_GEOM", "WD_VAR", "WD_NAMES", "WD_INDEX", "WD_ABL", "WD_POL")
    before = {k: repr(ns[k]) for k in keys}
    seen = {}
    ns["load_inline"] = lambda **kw: seen.update(kw) or "mod"
    mod, index = ns["_qgbuild"]()
    assert mod == "mod" and all(repr(ns[k]) == before[k] for k in keys), "C2 globals not restored"
    cuda = seen["cuda_sources"]
    assert list(index) == list(GEO) + list(ABL)
    for nm, (wg, bn, clk, clr, hl, nst) in GEO.items():
        targs = f"{wg}, {bn}, {clk}, {clr}, false, {nst}, __nv_bfloat16, 0, true, 0"
        assert f"launch_wd<{targs}>" in cuda and f"qg_info_one<{targs}>" in cuda, nm
    for nm, (base, abl) in ABL.items():
        wg, bn, clk, clr, hl, nst = GEO[base]
        assert f"launch_wd<{wg}, {bn}, {clk}, {clr}, false, {nst}, __nv_bfloat16, {abl}, true, 0>" in cuda, nm
    for nm, (wg, bn, clk, clr, hl, nst) in GEO.items():
        smem = ns["wd_smem"](wg, bn, clk, clr, hl, nst)
        assert smem <= 227 * 1024, (nm, smem)
        assert 2304 % (64 * wg * clr) == 0 and clk * clr <= 8, nm
    assert "case 0:" in cuda and cuda.count("qg_info_one<") == len(GEO) + len(ABL)
    return cuda


if __name__ == "__main__":
    src = proj_sk_xor_control.restore(ROOT).decode("utf8").replace("\r\n", "\n")
    text = make(src)
    cuda = host_gate(text)
    Path(sys.argv[1]).write_text(text, encoding="utf-8")
    print(f"host builder gate PASS: {len(GEO)} geometries + {len(ABL)} ablations, CUDA {len(cuda)} chars")
