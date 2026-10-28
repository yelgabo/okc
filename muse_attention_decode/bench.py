# okc: problem=muse_attention_decode
"""SGLang's attention for Muse Spark Open on H200 (FA3 backend, FP8 E4M3 KV cache), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: the kernels between the QKV and output projections
(MuseSpark1xAttention.gated_attention), in order:

  FusedQKNorm (Triton)                 per-head RMSNorm of q and k from the fused projection
                                       output, with interleaved RoPE fused in on sliding layers
  set_kv_buffer                        k and v cast to E4M3 (divided by the cache scales when
                                       set), stored at the step's cache rows (store_cache)
  q.to(float8_e4m3fn)                  as prepare_paged_mha_query does for an FP8 cache
  flash_attn_with_kvcache (FA3)        paged, causal, window (2047, 0) on sliding layers,
                                       5.473/8 softmax scale
  rms_norm_gate (JIT CUDA)             per-head RMSNorm x sigmoid(gate)

Differs from SGLang: the step's positions, cache rows, FA3 lengths and FA3 scheduler metadata
come in with the inputs (SGLang builds them once per step for all layers)."""

from typing import TypedDict

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: q (Hq*64), keys (Hkv*64),
                               # values (Hkv*64), the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding builds it, cached per theta."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _sglang(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)

    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        # The 5.473 query factor is folded into the softmax scale, as SGLang does.
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


def prepare(*args, **kwargs):
    print(f"prepare called with {len(args)} args, keys {list(kwargs)}")
    return None


def kernel_fn(data, state=None):
    return _sglang(data)


# Generated by tools/attn_make_bench.py: a timing sweep run at import.
_BENCH_VARIANTS = ['_sglang', '_probe_prep', '_probe_fa3', '_probe_gate', '_probe_empty']
_BENCH_SHAPES = [
    (1, "sliding", 1), (1, "full", 1), (16, "sliding", 1), (16, "full", 1),
    (32, "sliding", 1), (32, "full", 1), (1, "sliding", 8), (1, "full", 8),
    (16, "sliding", 8), (16, "full", 8), (32, "sliding", 8), (32, "full", 8),
]


def _bench_inputs(batch, layer, q_len, ctx=80000, seed=1):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    hq, hkv, d = 16, 2, 64
    u = torch.rand((batch,), generator=gen, device=dev)
    seq_lens = (ctx * (1 + 0.075 * (2 * u - 1))).long()
    longest = int(ctx * 1.075) + 1
    max_pages = longest + q_len
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    lens = (seq_lens + q_len).tolist()
    at = 0
    for b in range(batch):
        page_table[b, :lens[b]] = perm[at:at + lens[b]]
        at += lens[b]
    k_cache = torch.empty((total, 1, hkv, d), dtype=FP8, device=dev)
    v_cache = torch.empty((total, 1, hkv, d), dtype=FP8, device=dev)
    for c in (k_cache, v_cache):
        flat = c.view(-1, hkv, d)
        for i in range(0, total, 1 << 20):
            n = min(1 << 20, total - i)
            flat[i:i + n] = torch.randn((n, hkv, d), generator=gen, device=dev).to(FP8)
    qkv = torch.randn((batch, q_len, (2 * hq + 2 * hkv) * d), generator=gen, device=dev)
    qkv = (2 * qkv).to(torch.bfloat16)
    positions = (seq_lens[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    cache_loc = torch.gather(page_table.long(), 1, positions.view(batch, q_len)).reshape(-1)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu = torch.arange(0, batch * q_len + 1, q_len, dtype=torch.int32, device=dev)
    meta = None
    if layer == "full" and q_len == 1:
        from sgl_kernel.flash_attn import get_scheduler_metadata
        meta = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=hq,
            num_heads_k=hkv, headdim=d, cache_seqlens=cache_seqlens, qkv_dtype=FP8,
            cu_seqlens_q=cu, page_size=1, causal=True, num_splits=0)
    return {
        "qkv": qkv, "positions": positions, "cache_loc": cache_loc,
        "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu, "scheduler_metadata": meta,
        "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "config": {
            "batch": batch, "q_len": q_len, "num_q_heads": hq, "num_kv_heads": hkv,
            "head_dim": d, "page_size": 1, "window": 2048 if layer == "sliding" else 0,
            "rope_theta": 500000.0 if layer == "sliding" else 0.0,
            "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0,
        },
    }


def _probe_empty(data):
    return data["qkv"].view(-1)[:1] + 1


def _probe_prep(data):
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"].view(tokens, -1)
    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, qkv.device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    store_cache(k.reshape(tokens, hkv * d).to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                data["v_cache"].view(-1, hkv * d).view(torch.uint8), data["cache_loc"],
                reserved_skip_index=-1)
    return q.view(tokens, hq, d).to(FP8)


def _probe_fa3(data):
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    q = data["qkv"].view(tokens, -1)[:, :16 * 64].reshape(tokens, 16, 64).to(FP8)
    window = cfg["window"]
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=cfg["q_len"], softmax_scale=0.684, causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"])
    return out[0] if isinstance(out, tuple) else out


def _probe_gate(data):
    cfg = data["config"]
    tokens = cfg["batch"] * cfg["q_len"]
    qkv = data["qkv"].view(tokens, -1)
    return rms_norm_gate(qkv[:, :1024].contiguous(), qkv[:, 1152:], None, head_dim=64, eps=RMS_EPS,
                         has_norm=True)


def _bench_time(graph, flush, iters=40):
    events = []
    for _ in range(iters):
        flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in events)
    return 1000 * ms[len(ms) // 2]


def _bench_graph(fn, data):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data)
    return graph, out


def _run_bench():
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    fns = {name: globals()[name] for name in _BENCH_VARIANTS}
    for batch, layer, q_len in _BENCH_SHAPES:
        data = _bench_inputs(batch, layer, q_len)
        ref = None
        row = []
        for name, fn in fns.items():
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
                    err = f" e={e:.1e}"
                row.append(f"{name}={us:.1f}{err}")
                del graph
            except Exception as exc:
                row.append(f"{name}=FAIL({type(exc).__name__}: {str(exc)[:200]})")
        print(f"BENCH b={batch:2d} {layer:7s} L={q_len} " + " ".join(row), flush=True)
        del data
        torch.cuda.empty_cache()


import os as _os

if not _os.environ.get("ATTN_BENCH_DONE"):
    _os.environ["ATTN_BENCH_DONE"] = "1"
    try:
        _run_bench()
    except Exception as _exc:
        print(f"BENCH failed: {type(_exc).__name__}: {_exc}")
