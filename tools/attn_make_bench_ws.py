"""Write muse_attention_decode/bench.py for the q_len 8 work: the submission, the shared timing
helpers from tools/attn_make_bench.py, and a sweep run at import with three parts:

  GATHER   cold-L2 scattered-gather microbenchmark on the scored cache layout (page size 1,
           pages shuffled across the pool): bandwidth of reading every K and V row of each
           request through the page table, by row size (128 bytes = both KV heads, 64 = one
           head, as FA3 reads), batch, CTAs per request, rows in flight per thread, and whether
           page entries load a step ahead or right before their row (dependent). Then the
           latency of one dependent page-entry + row load chain.
  BENCH    CUDA-graph replays after a 256 MB flush (median of 40) of each variant per shape,
           with relative error against the first variant.
  STAMPS   one run of the warp-specialized kernel with clock64 phase totals (mode 2), per tile
           and per warp, plus the timeout flags.

    python3 tools/attn_make_bench_ws.py [gather] VARIANT [VARIANT ...]

The GATHER part runs only when "gather" is among the arguments (its 150 lines crowd the
16,000-character log tail).

VARIANT is name@s=SPLITS,n=STAGES,m=MODE,p=LOADER_WARPS,v=VARIANT,c=CLUSTER (all optional).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_make_bench  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

GATHER_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace {

// Each CTA reads its slice of one request's rows, as an attention split would. With ROWB 64,
// grid.z picks the KV head and the CTA reads that head's half of each row.
template <int ROWB, int U, bool PREF>
__global__ void __launch_bounds__(256) gather_kernel(const int* __restrict__ pt, int pt_stride,
                                                     const int* __restrict__ seq_lens,
                                                     const uint4* __restrict__ kc,
                                                     const uint4* __restrict__ vc, int splits,
                                                     unsigned* out) {
    constexpr int CH = ROWB / 16;
    constexpr int RPI = 256 / CH;
    const int split = blockIdx.x;
    const int b = blockIdx.y;
    const int head = blockIdx.z;
    const int seq = seq_lens[b];
    const int per = (seq + splits - 1) / splits;
    const int lo = min(seq, split * per);
    const int hi = min(seq, lo + per);
    const int sub = threadIdx.x / CH;
    const int ck = threadIdx.x % CH + (ROWB == 64 ? head * 4 : 0);
    const int* ptb = pt + static_cast<size_t>(b) * pt_stride;
    uint32_t x = 0;
    int nxt[U];
    if (PREF) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int pos = lo + u * RPI + sub;
            nxt[u] = pos < hi ? __ldg(ptb + pos) : -1;
        }
    }
    for (int base = lo; base < hi; base += RPI * U) {
        int e[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (PREF) {
                e[u] = nxt[u];
                const int pos = base + RPI * U + u * RPI + sub;
                nxt[u] = pos < hi ? __ldg(ptb + pos) : -1;
            } else {
                const int pos = base + u * RPI + sub;
                e[u] = pos < hi ? __ldg(ptb + pos) : -1;
            }
        }
        uint4 kv[U][2];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (e[u] >= 0) {
                kv[u][0] = __ldcg(kc + static_cast<size_t>(e[u]) * 8 + ck);
                kv[u][1] = __ldcg(vc + static_cast<size_t>(e[u]) * 8 + ck);
            } else {
                kv[u][0] = kv[u][1] = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            x ^= kv[u][0].x ^ kv[u][0].w ^ kv[u][1].y ^ kv[u][1].z;
        }
    }
    if (x == 0x9e3779b9u) {
        out[0] = x;
    }
}

// One thread walks n steps. ROW_ONLY 0: page entry at position 32 i (a fresh 128-byte line each
// step), then its K row, each address depending on the previous row's data. ROW_ONLY 1: only
// the row load, at a hashed row.
template <int ROW_ONLY>
__global__ void chase_kernel(const int* __restrict__ pt, const uint32_t* kc, int rows, int n,
                             int64_t* out) {
    uint32_t x = 0;
    int pos = 0;
    const long long t0 = clock64();
    for (int i = 0; i < n; ++i) {
        int e;
        if (ROW_ONLY) {
            e = static_cast<int>((static_cast<uint32_t>(i) * 2654435761u + (x & 1)) % rows);
        } else {
            e = *(volatile const int*)(pt + pos);
        }
        const uint32_t v = *(volatile const uint32_t*)(kc + static_cast<size_t>(e) * 32);
        x += v;
        pos += 32 + (v == 0x80000000u ? 1 : 0);
    }
    out[0] = clock64() - t0;
    out[1] = x;
}

}  // namespace

template <int R, int UU, bool P>
void run_gather(torch::Tensor& pt, torch::Tensor& seq_lens, torch::Tensor& kc, torch::Tensor& vc,
                int splits, torch::Tensor& out) {
    const int batch = seq_lens.size(0);
    dim3 grid(splits, batch, R == 64 ? 2 : 1);
    gather_kernel<R, UU, P><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        pt.data_ptr<int>(), static_cast<int>(pt.size(1)), seq_lens.data_ptr<int>(),
        reinterpret_cast<const uint4*>(kc.data_ptr()), reinterpret_cast<const uint4*>(vc.data_ptr()),
        splits, reinterpret_cast<unsigned*>(out.data_ptr<int>()));
}

template <int R, bool P>
void gather_u(torch::Tensor& pt, torch::Tensor& seq_lens, torch::Tensor& kc, torch::Tensor& vc,
              int64_t u, int splits, torch::Tensor& out) {
    if (u == 1) {
        run_gather<R, 1, P>(pt, seq_lens, kc, vc, splits, out);
    } else if (u == 2) {
        run_gather<R, 2, P>(pt, seq_lens, kc, vc, splits, out);
    } else if (u == 4) {
        run_gather<R, 4, P>(pt, seq_lens, kc, vc, splits, out);
    } else {
        run_gather<R, 8, P>(pt, seq_lens, kc, vc, splits, out);
    }
}

void gather(torch::Tensor pt, torch::Tensor seq_lens, torch::Tensor kc, torch::Tensor vc,
            int64_t rowb, int64_t u, bool pref, int64_t splits, torch::Tensor out) {
    const int sp = static_cast<int>(splits);
    if (rowb == 128) {
        if (pref) {
            gather_u<128, true>(pt, seq_lens, kc, vc, u, sp, out);
        } else {
            gather_u<128, false>(pt, seq_lens, kc, vc, u, sp, out);
        }
    } else {
        if (pref) {
            gather_u<64, true>(pt, seq_lens, kc, vc, u, sp, out);
        } else {
            gather_u<64, false>(pt, seq_lens, kc, vc, u, sp, out);
        }
    }
}

void chase(torch::Tensor pt, torch::Tensor kc, int64_t rows, int64_t n, bool row_only,
           torch::Tensor out) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* k = reinterpret_cast<const uint32_t*>(kc.data_ptr());
    if (row_only) {
        chase_kernel<1><<<1, 1, 0, stream>>>(pt.data_ptr<int>(), k, rows, n, out.data_ptr<int64_t>());
    } else {
        chase_kernel<0><<<1, 1, 0, stream>>>(pt.data_ptr<int>(), k, rows, n, out.data_ptr<int64_t>());
    }
}
'''

GATHER_CPP = """
void gather(torch::Tensor pt, torch::Tensor seq_lens, torch::Tensor kc, torch::Tensor vc,
            int64_t rowb, int64_t u, bool pref, int64_t splits, torch::Tensor out);
void chase(torch::Tensor pt, torch::Tensor kc, int64_t rows, int64_t n, bool row_only,
           torch::Tensor out);
"""

RUNNER = r'''

# Generated by tools/attn_make_bench_ws.py.
_WSB_VARIANTS = {variants!r}
_WSB_GATHER = {gather!r}
_WSB_SHAPES = [
    (32, "full", 8), (16, "full", 8), (1, "full", 8), (32, "sliding", 8), (16, "sliding", 8),
    (1, "sliding", 8), (32, "full", 1),
]
_GATHER_SRC = {gather_src!r}
_GATHER_CPP = {gather_cpp!r}


def _ref32(data):
    """FP32 attention with unrounded queries over the cache after this step's own write (done
    here the way the prep kernel does it), for the error column: graph-capturable torch."""
    cfg = data["config"]
    batch, lq = cfg["batch"], cfg["q_len"]
    hq, hkv, d = 16, 2, 64
    tokens = batch * lq
    qkv = data["qkv"].view(tokens, -1).float()
    q = qkv[:, :hq * d].view(tokens, hq, d)
    k = qkv[:, hq * d:(hq + hkv) * d].view(tokens, hkv, d)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d].view(tokens, hkv, d)
    gate = qkv[:, (hq + 2 * hkv) * d:].view(tokens, hq, d)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    pos = data["positions"]
    if cfg["rope_theta"]:
        ang = pos.float()[:, None] * _inv_freq(cfg["rope_theta"], d, qkv.device)[None, :]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]

        def rope(x):
            a, b2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((a * cos - b2 * sin, a * sin + b2 * cos), -1).flatten(-2)
        q, k = rope(q), rope(k)
    kc = data["k_cache"].view(-1, hkv, d)
    vc = data["v_cache"].view(-1, hkv, d)
    loc = data["cache_loc"]
    kc[loc] = (k / data["k_scale"]).to(FP8)
    vc[loc] = (v / data["v_scale"]).to(FP8)
    pt = data["page_table"].long().clamp(min=0)
    kk = kc[pt].float() * data["k_scale"]
    vv = vc[pt].float() * data["v_scale"]
    qq = q.view(batch, lq, hkv, hq // hkv, d)
    sc = torch.einsum("bjhgd,bphd->bjhgp", qq, kk) * (cfg["sm_scale"] * cfg["qk_scale_factor"])
    p_idx = torch.arange(pt.shape[1], device=qkv.device)
    qpos = pos.view(batch, lq)
    ok = p_idx[None, None, :] <= qpos[:, :, None]
    if cfg["window"]:
        ok = ok & (p_idx[None, None, :] > qpos[:, :, None] - cfg["window"])
    sc = sc.masked_fill(~ok[:, :, None, None, :], float("-inf"))
    a = torch.einsum("bjhgp,bphd->bjhgd", sc.softmax(-1), vv).reshape(tokens, hq, d)
    a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + RMS_EPS) * torch.sigmoid(gate)
    return a.to(torch.bfloat16).view(batch, lq, hq * d)


def _event_us(fn, flush, iters=7):
    times = []
    for _ in range(iters):
        flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        z.record()
        torch.cuda.synchronize()
        times.append(1000 * a.elapsed_time(z))
    return sorted(times)[len(times) // 2]


def _gather_bench(flush):
    ext = load_inline(name="okc_attn_gather_bench_v1", cpp_sources=_GATHER_CPP,
                      cuda_sources=_GATHER_SRC, functions=["gather", "chase"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    empty = _event_us(lambda: out.add_(1), flush)
    print(f"GATHER empty-launch {{empty:.1f}} us (subtract for small reads)", flush=True)
    for batch, split_list in ((32, (4, 8, 16, 32)), (16, (8, 16, 32)), (1, (64, 132, 264))):
        data = _bench_inputs(batch, "full", 1)
        seq = data["seq_lens"]
        pt = data["page_table"]
        kc = data["k_cache"].view(torch.uint8)
        vc = data["v_cache"].view(torch.uint8)
        nbytes = int(seq.sum()) * 256
        row = []
        for rowb in (128, 64):
            for pref in (False, True):
                for splits in split_list:
                    for u in (1, 2, 4, 8):
                        us = _event_us(lambda: ext.gather(pt, seq, kc, vc, rowb, u, pref, splits,
                                                          out), flush)
                        row.append((rowb, pref, splits, u, us))
        best = min(row, key=lambda r: r[4])
        print(f"GATHER b={{batch}} bytes={{nbytes / 1e6:.0f}}MB floor@4.8TB/s="
              f"{{nbytes / 4.8e6:.1f}}us best={{best[4]:.1f}}us "
              f"({{nbytes / best[4] / 1e6:.2f}} TB/s rowb={{best[0]}} pref={{int(best[1])}} "
              f"splits={{best[2]}} u={{best[3]}})", flush=True)
        for rowb, pref, splits, u, us in row:
            print(f"GATHER b={{batch}} rowb={{rowb}} pref={{int(pref)}} splits={{splits}} u={{u}} "
                  f"{{us:.1f}}us {{nbytes / us / 1e6:.2f}}TB/s", flush=True)
        if batch == 32:
            res = torch.zeros(2, dtype=torch.int64, device="cuda")
            rows = kc.numel() // 128
            for row_only in (False, True):
                n = 2000
                flush.zero_()
                ext.chase(pt[0], kc, rows, n, row_only, res)
                torch.cuda.synchronize()
                cyc = res[0].item() / n
                print(f"GATHER latency {{'row' if row_only else 'entry+row'}}: {{cyc:.0f}} "
                      f"cycles per dependent step (cold, one thread)", flush=True)
        del data
        torch.cuda.empty_cache()


_WS_PHASES = ["wait-data", "qk", "softmax", "pv", "-", "-", "tail", "prologue"]
_WS_LOADER = ["wait-free-stage", "cp.async-issue", "page-entries", "wait-own-data",
              "v-convert+release"]


def _ws_stamps(data, cfg_splits, pw=4):
    dbg = torch.zeros(17, dtype=torch.int64, device="cuda")
    _ws(data, splits=cfg_splits, mode=2, pw=pw, dbg=dbg)
    torch.cuda.synchronize()
    t = dbg.tolist()
    cfg = data["config"]
    seq = data["seq_lens"].tolist()
    splits = cfg_splits or _ws_splits(cfg)
    start = [max(0, s - cfg["window"] + 1) if cfg["window"] else 0 for s in seq]
    tiles = sum((s - a + 63) // 64 for s, a in zip(seq, start)) + cfg["batch"] * 0
    ctas = splits * cfg["batch"]
    math = " ".join(f"{{n}}={{t[1 + k] / 8 / max(tiles, 1):.0f}}" for k, n in enumerate(_WS_PHASES))
    nload = max(t[16], 1)
    load = " ".join(f"{{n}}={{t[9 + k] / nload / max(tiles, 1):.0f}}"
                    for k, n in enumerate(_WS_LOADER))
    return (f"pw={{t[16]}} timeout_flags={{t[0]}} tiles={{tiles}} ctas={{ctas}} cycles per tile per "
            f"math warp: {{math}} | per loader warp per tile: {{load}}")


def _ws_bench():
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    try:
        if _WSB_GATHER:
            _gather_bench(flush)
    except Exception as exc:
        msg = str(exc)
        errs = [ln for ln in msg.splitlines() if "error" in ln.lower()]
        print(f"GATHER failed: {{type(exc).__name__}}: {{msg[:300]}} ... "
              f"{{' | '.join(errs)[:3000]}}", flush=True)
    import functools
    keys = {{"s": "splits", "n": "stages", "m": "mode", "v": "variant", "c": "cluster",
            "p": "pw"}}
    fns = {{}}
    for spec in _WSB_VARIANTS:
        name, _, args = spec.partition("@")
        kw = {{keys[k]: int(v) for k, v in (a.split("=") for a in args.split(",") if a)}}
        if "cluster" in kw:
            kw["cluster"] = bool(kw["cluster"])
        fns[spec] = functools.partial(globals()[name], **kw)
    for batch, layer, q_len in _WSB_SHAPES:
        data = _bench_inputs(batch, layer, q_len)
        ref = None
        row = []
        for name, fn in fns.items():
            if q_len == 1 and name.startswith("_ws"):
                continue
            try:
                graph, out = _bench_graph(fn, data)
                us = _bench_time(graph, flush)
                graph.replay()
                torch.cuda.synchronize()
                err = ""
                if ref is None:
                    ref = out.float().clone()
                elif out.shape == ref.shape:
                    e = ((out.float() - ref).norm() / ref.norm().clamp(min=1e-12)).item()
                    err = f" e={{e:.1e}}"
                row.append(f"{{name}}={{us:.1f}}{{err}}")
                del graph
            except Exception as exc:
                row.append(f"{{name}}=FAIL({{type(exc).__name__}}: {{str(exc)[:200]}})")
        print(f"BENCH b={{batch:2d}} {{layer:7s}} L={{q_len}} " + " ".join(row), flush=True)
        if q_len == 8 and _WS is not None:
            for pw in (4, 8):
                try:
                    print(f"STAMPS b={{batch:2d}} {{layer:7s}} L={{q_len}} "
                          + _ws_stamps(data, None, pw), flush=True)
                except Exception as exc:
                    print(f"STAMPS failed: {{type(exc).__name__}}: {{exc}}", flush=True)
        del data
        torch.cuda.empty_cache()


import os as _os

if not _os.environ.get("ATTN_BENCH_DONE"):
    _os.environ["ATTN_BENCH_DONE"] = "1"
    try:
        _ws_bench()
    except Exception as _exc:
        print(f"BENCH failed: {{type(_exc).__name__}}: {{_exc}}")
'''


def main():
    variants = [v for v in sys.argv[1:] if v != "gather"]
    variants = variants or ["_ref32", "_fused_prep", "_probe_fa3", "_ws"]
    helpers = attn_make_bench.HARNESS.format(variants=[], stamps=[])
    helpers = helpers.split("\nimport os as _os\n")[0]
    src = (ROOT / "muse_attention_decode/submission.py").read_text()
    src += helpers + RUNNER.format(variants=variants, gather_src=GATHER_SRC,
                                   gather_cpp=GATHER_CPP, gather="gather" in sys.argv[1:])
    (ROOT / "muse_attention_decode/bench.py").write_text(src)
    print(f"wrote bench.py with {variants}")


if __name__ == "__main__":
    main()
