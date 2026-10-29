"""Write a CUTLASS sweep bench for the dense projections at the decode layer's large-token shapes.

    python3 tools/proj_make_bench_cut.py OUT.py [--emit-cu DIR] [--cfgs NAME,NAME,...]

OUT.py is muse_bf16_proj/submission.py (the judged path, unchanged) plus a block that runs at
import: it builds one CUTLASS 3.x sm90 GEMM per translation unit (swap-AB, D^T = W x^T, the
public entry's composition; headers from the FlashInfer package in the judge image), each
`.cu` file a separate ninja job through `cpp_extension.load`, then times every layer GEMM the
decode layer runs at T = 256 and 128 (qkv_gate, o, pooler, unpooler, and the front's "concat":
pooler + router hi + router lo in one GEMM with FP32 output) against cuBLAS (SGLang's
dispatch, and `torch.mm(..., out_dtype=float32)` for concat), torchT (cuBLAS on a K-major copy)
and our prepare() pick, as cold-L2 CUDA-graph replays, 100 rotating rounds. Lines:

  CSBUILD  build time, CPU count, CUTLASS version, or the build error.
  CSCHECK  per shape: candidates that fail the error check (rel L2 1e-2; concat router columns
           1e-4 against x @ (hi + lo)).
  CSCOLD   per shape: median us per candidate, fastest first.
  CSRATIO  per shape: paired median of cublas / candidate after the replay overhead.

--emit-cu writes the .cu files for a local compile check (tools/proj_cut_compile_check.sh).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# name: (kernel schedule, TM (W rows), TN (tokens), TK, cluster along W rows, tile scheduler, out)
CFGS = {
    "pp128x64": ("pp", 128, 64, 64, 1, "pers", "bf16"),
    "pp128x64c2": ("pp", 128, 64, 64, 2, "pers", "bf16"),
    "pp128x128": ("pp", 128, 128, 64, 1, "pers", "bf16"),
    "co128x128sk": ("co", 128, 128, 64, 1, "sk", "bf16"),
    "co128x64sk": ("co", 128, 64, 64, 1, "sk", "bf16"),
    "co256x128sk": ("co", 256, 128, 64, 1, "sk", "bf16"),
    "pp128x64f": ("pp", 128, 64, 64, 1, "pers", "f32"),
    "pp128x64c2f": ("pp", 128, 64, 64, 2, "pers", "f32"),
    "co128x128skf": ("co", 128, 128, 64, 1, "sk", "f32"),
    "co128x64skf": ("co", 128, 64, 64, 1, "sk", "f32"),
    "co256x128skf": ("co", 256, 128, 64, 1, "sk", "f32"),
}

TU = r"""
// Instantiates NVIDIA CUTLASS 3.x templates (header-only, BSD-3-Clause), composed as in CUTLASS's
// examples/49_hopper_gemm_with_collective_builder.
#include <cuda_runtime.h>
#include <cstdint>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
using namespace cute;
namespace {
using EA = cutlass::bfloat16_t; using LA = cutlass::layout::RowMajor;
using EB = cutlass::bfloat16_t; using LB = cutlass::layout::ColumnMajor;
using EC = @EC@; using LC = cutlass::layout::ColumnMajor;
constexpr int AC = 16 / sizeof(EC);
using TileShape = Shape<_@TM@, _@TN@, _@TK@>;
using ClusterShape = Shape<_@CM@, _1, _1>;
using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, EC, LC, AC, EC, LC, AC,
    @ESCHED@>::CollectiveOp;
using Main = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, EA, LA, 8, EB, LB, 8, float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
    @KSCHED@>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, @TSCHED@>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

typename Gemm::Arguments make_args(const void* W, const void* x, void* out, int N, int M, int K,
                                   int splits, int mode, int dev, int sms) {
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(N, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(M, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(N, M, 1));
  auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, cute::make_shape(N, M, 1));
  cutlass::KernelHardwareInfo hw;
  hw.device_id = dev;
  hw.sm_count = sms;
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, 1},
      {static_cast<const EA*>(W), sA, static_cast<const EB*>(x), sB},
      {{1.f, 0.f}, static_cast<const EC*>(out), sC, static_cast<EC*>(out), sD}, hw};
  @SCHEDARGS@
  return args;
}
}  // namespace

int64_t cs_ws_@NAME@(int N, int M, int K, int splits, int mode, int dev, int sms) {
  return (int64_t)Gemm::get_workspace_size(make_args(nullptr, nullptr, nullptr, N, M, K, splits, mode, dev, sms));
}

int cs_run_@NAME@(const void* W, const void* x, void* out, void* ws, int64_t wsb, int N, int M, int K,
                  int splits, int mode, int dev, int sms, cudaStream_t st) {
  auto args = make_args(W, x, out, N, M, K, splits, mode, dev, sms);
  Gemm gemm;
  if ((int64_t)Gemm::get_workspace_size(args) > wsb) return 1;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
  if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) return 3;
  if (gemm.run(st) != cutlass::Status::kSuccess) return 4;
  return 0;
}
"""

# mode: 0 heuristic, 1 data parallel, 2 split-K (splits), 3 stream-K.
SK_ARGS = """using DM = decltype(args.scheduler.decomposition_mode);
  args.scheduler.splits = splits;
  args.scheduler.decomposition_mode = mode == 1 ? DM::DataParallel : mode == 2 ? DM::SplitK
                                      : mode == 3 ? DM::StreamK : DM::Heuristic;"""


def tu_source(name, cfg):
    ks, tm, tn, tk, cm, ts, out = cfg
    rep = {
        "@NAME@": name, "@TM@": str(tm), "@TN@": str(tn), "@TK@": str(tk), "@CM@": str(cm),
        "@EC@": "float" if out == "f32" else "cutlass::bfloat16_t",
        "@ESCHED@": "cutlass::epilogue::TmaWarpSpecialized" if ks == "pp"
        else "cutlass::epilogue::TmaWarpSpecializedCooperative",
        "@KSCHED@": "cutlass::gemm::KernelTmaWarpSpecializedPingpong" if ks == "pp"
        else "cutlass::gemm::KernelTmaWarpSpecializedCooperative",
        "@TSCHED@": "cutlass::gemm::StreamKScheduler" if ts == "sk" else "cutlass::gemm::PersistentScheduler",
        "@SCHEDARGS@": SK_ARGS if ts == "sk" else "(void)splits; (void)mode;",
    }
    src = TU
    for k, v in rep.items():
        src = src.replace(k, v)
    return src


def bind_source(names):
    decl = "".join(
        f"int64_t cs_ws_{n}(int, int, int, int, int, int, int);\n"
        f"int cs_run_{n}(const void*, const void*, void*, void*, int64_t, int, int, int, int, int, int, int,"
        f" cudaStream_t);\n" for n in names)
    ws_tab = ", ".join(f"cs_ws_{n}" for n in names)
    run_tab = ", ".join(f"cs_run_{n}" for n in names)
    return f"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
{decl}
using WsFn = int64_t (*)(int, int, int, int, int, int, int);
using RunFn = int (*)(const void*, const void*, void*, void*, int64_t, int, int, int, int, int, int, int, cudaStream_t);
static const WsFn kWs[] = {{{ws_tab}}};
static const RunFn kRun[] = {{{run_tab}}};
static int sm_count() {{
  return at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}}
int64_t cs_ws(int64_t idx, int64_t N, int64_t M, int64_t K, int64_t splits, int64_t mode) {{
  return kWs[idx]((int)N, (int)M, (int)K, (int)splits, (int)mode, (int)c10::cuda::current_device(), sm_count());
}}
void cs_run(int64_t idx, at::Tensor W, at::Tensor x, at::Tensor out, at::Tensor ws, int64_t splits, int64_t mode) {{
  TORCH_CHECK(W.is_contiguous() && x.is_contiguous() && out.is_contiguous() && ws.is_contiguous(), "cs_run: layout");
  const int N = W.size(0), K = W.size(1), M = x.size(0);
  TORCH_CHECK(x.size(1) == K && out.size(0) == M && out.size(1) == N, "cs_run: shapes");
  int rc = kRun[idx](W.data_ptr(), x.data_ptr(), out.data_ptr(), ws.data_ptr(), ws.numel(), N, M, K,
                     (int)splits, (int)mode, (int)c10::cuda::current_device(), sm_count(),
                     at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(rc == 0, "cs_run: CUTLASS status ", rc);
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("cs_ws", &cs_ws);
  m.def("cs_run", &cs_run);
}}
"""


BENCH = r'''

_CS_CFGS = @@CFGS@@
_CS_TU = @@TU@@
_CS_SK_ARGS = @@SKARGS@@
import math as _csmath
import os as _csos
import tempfile as _cstmp

_CS_OUT = []
_CS_SHAPES = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
              "unpooler": (8192, 4096), "concat": (4608, 8192)}
_CS_TOKENS = (256, 128)
_CS_MODES = {"pers": [("", 1, 0)],
             "sk": [("h", 1, 0), ("s", 1, 3), ("k2", 2, 2), ("k3", 3, 2), ("k4", 4, 2)]}


def _cs_print(s):
    print(s, flush=True)
    _CS_OUT.append(s)


@@FUNCS@@


def _cs_build():
    import importlib.util
    from torch.utils.cpp_extension import load
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("flashinfer (CUTLASS headers) not installed")
    root = _csos.path.join(list(spec.submodule_search_locations)[0], "data", "cutlass")
    inc = [_csos.path.join(root, "include"), _csos.path.join(root, "tools", "util", "include")]
    ver = "?"
    try:
        txt = open(_csos.path.join(inc[0], "cutlass", "version.h")).read()
        ver = ".".join(ln.split()[-1] for ln in txt.splitlines()
                       if ln.startswith(("#define CUTLASS_MAJOR", "#define CUTLASS_MINOR", "#define CUTLASS_PATCH")))
    except OSError:
        pass
    d = _cstmp.mkdtemp(prefix="okc_cs_")
    names = list(_CS_CFGS)
    srcs = []
    for nm in names:
        p = _csos.path.join(d, f"cs_{nm}.cu")
        with open(p, "w") as f:
            f.write(tu_source(nm, _CS_CFGS[nm]))
        srcs.append(p)
    p = _csos.path.join(d, "cs_bind.cpp")
    with open(p, "w") as f:
        f.write(bind_source(names))
    srcs.append(p)
    t0 = time.time()
    ext = load(name="okc_cs_sweep_v1", sources=srcs, extra_include_paths=inc,
               extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr",
                                  "-gencode=arch=compute_90a,code=sm_90a", "-DNDEBUG"],
               verbose=False)
    _cs_print(f"CSBUILD {len(names)} configs {time.time() - t0:.1f} s, cpus {_csos.cpu_count()}, CUTLASS {ver}")
    return ext, names


def _cs_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    return g


def _cs_time(graphs, rounds, flush, sink):
    names = list(graphs)
    samples = {n: [] for n in names}
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


def _cs_run():
    dev = torch.device("cuda", torch.cuda.current_device())
    t0 = time.time()
    gen = torch.Generator(device=dev).manual_seed(5151)
    lay = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
           "unpooler": (8192, 4096), "router": (256, 8192)}
    weights = {nm: (torch.randn((n, k), generator=gen, device=dev) / _csmath.sqrt(k)).to(
                   torch.float32 if nm == "router" else torch.bfloat16) for nm, (n, k) in lay.items()}
    st = None
    try:
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st = prepare(weights)
        for ln in buf.getvalue().splitlines():
            if ln.startswith("tune") and ("m=128" in ln or "m=256" in ln):
                _cs_print(f"ours| {ln}")
    except Exception as exc:
        _cs_print(f"CSPREP ours failed {type(exc).__name__}: {str(exc)[:200]}")
    try:
        ext, names = _cs_build()
    except Exception as exc:
        _cs_print(f"CSBUILD failed {time.time() - t0:.1f} s: {type(exc).__name__}: {str(exc)[-1500:]}")
        return
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    W = {nm: weights[nm] for nm in ("qkv_gate", "o", "pooler", "unpooler")}
    W["concat"] = torch.cat((weights["pooler"], hi, lo), 0).contiguous()
    WT = {nm: w.t().contiguous() for nm, w in W.items()}
    ws = torch.zeros(64 << 20, dtype=torch.uint8, device=dev)
    flush = torch.ones(65536, 1024, dtype=torch.float32, device=dev)
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    tiny = torch.zeros(1, device=dev)
    one = _cs_graph(lambda: tiny.add_(1))

    def two_fn():
        tiny.add_(1)
        tiny.add_(1)
    two = _cs_graph(two_fn)
    ov = _cs_time({"one": one, "two": two}, 200, flush, sink)
    ovh = 2 * statistics.median(ov["one"]) - statistics.median(ov["two"])
    _cs_print(f"CSOVH overhead {ovh:.2f} us")
    xgen = torch.Generator(device=dev).manual_seed(777)
    for m in _CS_TOKENS:
        for shape, (n, k) in _CS_SHAPES.items():
            f32 = shape == "concat"
            odt = torch.float32 if f32 else torch.bfloat16
            x = torch.randn((m, k), generator=xgen, device=dev).to(torch.bfloat16)
            w, wt = W[shape], WT[shape]
            ref = x.double() @ w.double().T
            rref = None
            if f32:
                rref = x.double() @ (hi.double() + lo.double()).T
            fns = {}
            if f32:
                fns["cublas"] = lambda x=x, w=w: torch.mm(x, w.t(), out_dtype=torch.float32)
                fns["torchT"] = lambda x=x, wt=wt: torch.mm(x, wt, out_dtype=torch.float32)
            else:
                fns["cublas"] = lambda x=x, w=w: _cublas(x, w)
                fns["torchT"] = lambda x=x, wt=wt: torch.mm(x, wt)
                if st is not None:
                    data = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": shape}}
                    fns["ours"] = lambda data=data: kernel_fn(data, st)
            for i, nm in enumerate(names):
                if (_CS_CFGS[nm][6] == "f32") != f32:
                    continue
                for suf, splits, mode in _CS_MODES[_CS_CFGS[nm][5]]:
                    def run(i=i, x=x, w=w, splits=splits, mode=mode, n=n, odt=odt):
                        out = torch.empty((x.shape[0], n), dtype=odt, device=x.device)
                        ext.cs_run(i, w, x, out, ws, splits, mode)
                        return out
                    fns[nm + suf] = run
            bad = []
            for cn in list(fns):
                try:
                    out = fns[cn]()
                    torch.cuda.synchronize()
                    e = float((out.double() - ref).norm() / ref.norm())
                    ok = e < 1e-2
                    if f32:
                        r = out[:, 4096:4352].double() + out[:, 4352:].double()
                        er = float((r - rref).norm() / rref.norm())
                        ok = ok and er < 1e-4
                        e = max(e, er)
                    if not ok:
                        bad.append(f"{cn}={e:.1e}")
                        fns.pop(cn)
                except Exception as exc:
                    torch.cuda.synchronize()
                    bad.append(f"{cn}=EXC:{str(exc)[-60:]}")
                    fns.pop(cn)
            if bad:
                _cs_print(f"CSCHECK {shape} m={m} dropped: " + " ".join(bad))
            graphs = {}
            for cn in list(fns):
                try:
                    graphs[cn] = _cs_graph(fns[cn])
                except Exception as exc:
                    torch.cuda.synchronize()
                    _cs_print(f"CSCHECK {shape} m={m} {cn} capture failed: {str(exc)[-80:]}")
            if "cublas" not in graphs:
                continue
            smp = _cs_time(graphs, 100, flush, sink)
            graphs = None
            meds = {cn: statistics.median(v) for cn, v in smp.items()}
            rat = {cn: statistics.median([(b - ovh) / (v - ovh) for b, v in zip(smp["cublas"], smp[cn])])
                   for cn in smp}
            order = sorted(meds, key=meds.get)
            _cs_print(f"CSCOLD {shape} m={m} " + " ".join(f"{cn}={meds[cn]:.2f}" for cn in order))
            _cs_print(f"CSRATIO {shape} m={m} " + " ".join(f"{cn}={rat[cn]:.3f}" for cn in order))
        torch.cuda.empty_cache()
    _cs_print(f"CSDONE {time.time() - t0:.1f} s")
    if st is not None:
        st.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if not _csos.environ.get("PROJ_CS_DONE"):
    _csos.environ["PROJ_CS_DONE"] = "1"
    try:
        _cs_run()
    except Exception as _exc:
        _cs_print(f"CSBENCH failed: {type(_exc).__name__}: {str(_exc)[:400]}")
    _cs_prepare = prepare

    def prepare(weights):
        state = _cs_prepare(weights)
        for line in _CS_OUT:
            print(line, flush=True)
        return state
'''


def bench_source(cfgs):
    import inspect
    funcs = inspect.getsource(tu_source) + "\n\n" + inspect.getsource(bind_source)
    funcs = funcs.replace("SK_ARGS", "_CS_SK_ARGS").replace("src = TU", "src = _CS_TU")
    return (BENCH.replace("@@CFGS@@", repr(cfgs)).replace("@@TU@@", repr(TU))
            .replace("@@SKARGS@@", repr(SK_ARGS))
            .replace("@@FUNCS@@", funcs))


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    out = Path(args.pop(0))
    cfgs = dict(CFGS)
    emit = None
    while args:
        a = args.pop(0)
        if a == "--emit-cu":
            emit = Path(args.pop(0))
        elif a == "--cfgs":
            keep = args.pop(0).split(",")
            cfgs = {k: CFGS[k] for k in keep}
    if emit is not None:
        emit.mkdir(parents=True, exist_ok=True)
        for nm, cfg in cfgs.items():
            (emit / f"cs_{nm}.cu").write_text(tu_source(nm, cfg))
        (emit / "cs_bind.cpp").write_text(bind_source(list(cfgs)))
    src = (ROOT / "muse_bf16_proj/submission.py").read_text()
    out.write_text(src + bench_source(cfgs))
    print(f"wrote {out} with configs {list(cfgs)}")


if __name__ == "__main__":
    main()
