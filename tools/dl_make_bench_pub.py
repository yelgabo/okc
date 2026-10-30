"""Write a same-process whole-layer bench: a candidate layer against one other layer on identical
inputs and the judge's weights: the frozen incumbent source (inc) or the public layer file (pub).
Four arms: A and B, and a second copy of each (A2, B2) as that side's null. With --other=inc,
A = inc and B = cand; with --other=pub, A = cand and B = pub.

    python3 tools/dl_make_bench_pub.py OUT.py OTHER.py --other=inc|pub --loads=d1,d16,...
        [--cand=LAYER.py] [--cand-strip=1] [--rounds=10] [--deadline=255] [--seed0=50001]
        [--order-seed=7] [--prep-seed=N] [--dup=own|shared] [--cand-report=0|1]

Why four arms (2026-10-30): the three-arm bench timed cand twice (cand, null) and the other
file once. cand and null shared one state and one compiled module and inc had no copy, so
the null measured cand's noise only; an effect specific to the inc arm (it ran about 5 us
slow at decode 1 on identical code in 50c4c1328ab742f4 and 720079090ee44cc3) showed in
cand-inc and null-inc alike and passed as a gain. Now both files have a copy, and the
decision pools the four cross ratios (A/B, A/B2, A2/B, A2/B2), which cancels anything the
two copies of one file share, and needs that pooled ratio beyond twice the larger null.

The judge takes files under 1,000,000 bytes and the layer alone is about 800 KB, so one check
carries one other arm: OTHER.py with whole-line comments removed (strip_comments: `//` lines
and non-directive `#` lines, refusing anything that could change compiled code) and patched,
then lzma-compressed and base85-encoded, decompressed and exec'd at import. inc: the scored
layer source (for example `git show a923f11:muse_decode_layer/submission.py`), its load_inline
names prefixed `okc_inc_` so both copies load in one process. pub: the public layer, patched
(no threading import; its prepare() calls its builds in turn, already done at import). --cand
names the candidate (default: the repository's muse_decode_layer/submission.py); --cand-strip=1
sends it through the same whole-line comment strip (for a candidate too large to fit beside the
other arm; ABPIN records both hashes). The copies A2 and B2 reuse their file's module and
compiled extensions, so they cost no bytes. OUT.py belongs in the session scratchpad; only the
logs of its runs are committed.

  --dup     own (default for inc): each copy gets its own prepare() (own allocations, own
            tuner picks, so the nulls include run-to-run policy noise); shared (default for
            pub, whose prepare() has never run twice in one process): the copy reuses its
            file's state, as the old null did.
  import    every arm's extensions build on the layer's build threads (_bg_start), so no build
            counts against prepare()'s 300 s.
  ABPIN     SHA256 and git blob id of each source as read and as sent.
  ABPREP    the prepare() calls in a seeded random order (--prep-seed, default --order-seed),
            each timed; then per scored shape of the chosen loads and per input (held-out seeds
            from --seed0, never the judge's case seeds; two routing distributions: the local
            range is the reference routing's busiest 32-expert range, as the judge picks it, or
            its second busiest):
  ABPOL     each arm's prepare-time picks side by side (MoE variant and W4 configs per token
            count, front, attention boundary, sk projection picks, wide picks, M1 policy, stage
            flags): "same", or DIFF with every arm's value.
  ABCHECK   every arm against the restated FP32 reference with the judge's rules (routing
            acceptance, residual update and next input within 2.5x the E4M3-query error
            clamped to [0.03, 0.15], rows within 5x, new cache entries within 8e-2): PASS or
            FAIL, top-8 agreement with the reference, local routes / experts against the
            reference's. A FAIL marks the arm INVALID on that shape's ABTIME line.
  ABTIME    each arm on its own copy of the inputs (kernel_fn writes the cache), graphs
            captured in a seeded random arm order, CUDA graph replays after a 256 MB L2 flush,
            arms in a seeded random order per round; medians, the four cross paired
            differences (B - A, B2 - A, B - A2, B2 - A2) and the two nulls (A2 - A, B2 - B).
  ABLOAD    46 x sliding + 16 x full per arm (per 62 steps): the four cross ratios A/B...,
            their geometric mean (pooled) and the nulls A2/A and B2/B.
  ABSCORE   the same ratios as geometric means over the loads covered.
  ABDECIDE  B faster / B slower / not resolved: all four cross ratios on one side of 1 and
            |log pooled| above twice the larger |log null|.
  ABPRED    per arm, the median deviation of its replays from its per-input median, grouped
            by the arm replayed just before it; ABPOS the same by position in the round;
            ABTWIN each arm after its own copy minus after the other arms. These test
            carry-over between back-to-back replays: state the 256 MB memset does not clear
            (instruction caches, TLB entries) is warm only after a same-module predecessor.
  ABSTOP    shapes skipped by the deadline: missing coverage, not a pass.
  ABTL      one replay per shape for A and B under torch.profiler, kernels as
            code@start+duration.
  ABK       per shape for A and B, every kernel of the step in launch order: its code,
            the median over five profiled replays of its summed duration per step, xN when it
            runs N times; span is the median replay's first start to last end, sum the summed
            durations (above span when kernels overlap, PDL secondaries included from launch,
            so their time counts waits in griddepcontrol.wait). ABKN maps the codes to names.
  ABCHK     per arm, checks passed of checks run; failing checks print as ABCHKFAIL.
  ABCAND    the candidate's prepare-time reports (m1_report, the MoE gate, _FAILED); off with
            --cand-report=0.
  ABZ       for the shapes named by --tlend=v1s,d32f,...: the median replay's kernels in start
            order as code@start+duration/exposed (exposed: how far the kernel moves the
            latest end so far, so the exposed times sum to the span).
  ABEND     seconds into prepare() when the bench finished (prepare() has 300 s).
The tail is sorted so the most expendable lines come first and fall out of the judge's 16 KB
log tail first (ABTL, passing ABCHECK, ABBUILD, ABOTHER, ABCAND, ABKN / ABK / ABZ); the
decision, scores, loads, timings, policies and carry-over tables come last.
"""

import ast
import base64
import hashlib
import lzma
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PUB_PATCHES = [
    # The judge's static check rejected a bench with the public proj file's build thread as
    # "starts processes" (tools/proj_make_bench_ab.py); the builds run inline instead.
    ("import threading\n", ""),
    ("    ths = [threading.Thread(target=_build, args=(f,)) for f in (_att7, _moe, _moe_tiny, "
     "_moe_tiny2, _moe_mid)]  # ~85 s of nvcc, in parallel\n"
     "    for th in ths:\n"
     "        th.start()\n"
     "    for th in ths:\n"
     "        th.join()\n",
     "    for f in (_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid):\n"
     "        _build(f)\n"),
]

BODY = r'''

# Generated by tools/dl_make_bench_pub.py: same-process whole-layer bench of this file (cand)
# against one other layer, the frozen incumbent source (inc) or the public layer (pub), with a
# second copy of each as that side's null.
_AB_OUT = []
_AB_T0 = time.time()
_AB_LOAD_SHAPE = {"d1": (1, 1), "d16": (16, 1), "d32": (32, 1), "v1": (1, 8), "v16": (16, 8),
                  "v32": (32, 8)}
# A is the reference arm and B the arm under test; A2 and B2 are second copies of the same
# files, timed like the others, so each side of the comparison has its own null.
_AB_A = "inc" if _AB_OTHER == "inc" else "cand"
_AB_B = "cand" if _AB_OTHER == "inc" else _AB_OTHER
_AB_A2, _AB_B2 = _AB_A + "2", _AB_B + "2"
_AB_ARMS = (_AB_A, _AB_A2, _AB_B, _AB_B2)


def _ab_print(line):
    _AB_OUT.append(line)
    print(line, flush=True)


class _AbSource:
    """A loader that only serves the arm's source, registered with linecache.lazycache, so
    inspect.getsource (which Triton's @jit calls on the public file's kernels) finds code that
    was exec'd from a string: check 121dff3ab17a4245 failed with "@jit functions should be
    defined in a Python file". No module is registered anywhere."""

    def __init__(self, src):
        self.src = src

    def get_source(self, name):
        return self.src


class _AbArm:
    """The other arm's globals (a plain dict the source was exec'd in), read as attributes."""

    def __init__(self, ns):
        self.ns = ns

    def __getattr__(self, key):
        try:
            return self.ns[key]
        except KeyError:
            raise AttributeError(key) from None


def _ab_module(name, packed):
    # The judge takes files under 1,000,000 bytes; the other arm's source travels compressed.
    import base64
    import linecache
    import lzma
    src = lzma.decompress(base64.b85decode(packed)).decode()
    ns = {"__name__": name, "__loader__": _AbSource(src), "__builtins__": __builtins__}
    linecache.lazycache(f"{name}.py", ns)
    exec(compile(src, f"{name}.py", "exec"), ns)
    return _AbArm(ns)


def _ab_import():
    """The other arm's module and every extension build, on the layer's build threads
    (_bg_start), so prepare() only repacks, self-tests and tunes."""
    other = _ab_module(f"okc_ab_{_AB_OTHER}", _AB_OTHER_XZ)
    builds = {}
    tagged = [(globals(), "cand")]
    if _AB_OTHER == "pub":
        # Its prepare() sets this from the weights before building; the judge's block scales
        # are positive normal E4M3 (our _w4_scales_ok holds on them), which selects it.
        other.ns["_MA_SFX"] = other._MA_SFX_IF_PRMT if other._MA_PRMT_SCALES else ()
        for name in ("_att7", "_moe", "_moe_tiny", "_moe_tiny2", "_moe_mid", "bfp__get_ext"):
            builds["pub" + name] = getattr(other, name)
    else:
        other.ns["_timeline"] = lambda state: None
        tagged.append((other.ns, "inc"))
    for mod, tag in tagged:
        for key, flag in (("_one_load", "PORT_ATTN_ONE"), ("_moe1_load", "PORT_MOE_FUSED"),
                          ("_skp_load", "PORT_PROJ_SK"), ("_w4_ext", "PORT_MOE_W4"),
                          ("_ext", None)):
            if flag is None or mod[flag]:
                builds[f"{tag}{key}"] = mod[key]
    # Our own modules' loaders cache into their module globals (_EXT, _W4_EXT) or their
    # _BG_BUILDS; the latter are keyed by these names in each module.
    for name, fn in builds.items():
        _bg_start(name, fn)
    for name, fn in builds.items():
        try:
            ext = _bg_ext(name, fn)
            _ab_print(f"ABBUILD {name} {_bg_seconds(name):.0f} s")
        except Exception as exc:
            _ab_print(f"ABBUILD {name} failed: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        for mod, tag in tagged:
            for key, bg in (("_one_load", "attn_one"), ("_moe1_load", "moe1"), ("_skp_load", "skp")):
                if name == tag + key:
                    mod["_BG_BUILDS"][bg] = {"started": 0.0, "done": _bg_seconds(name),
                                             "ext": ext, "thread": None}
    _ab_print(f"ABBUILD import done at {time.time() - _AB_T0:.0f} s")
    return other


try:
    _AB_OTHER_MOD = _ab_import()
except Exception as _exc:
    _AB_OTHER_MOD = None
    _ab_print(f"ABBUILD failed: {type(_exc).__name__}: {str(_exc)[:300]}")


def _ab_inputs(batch, q_len, layer, seed):
    """reference.py's generate_input for a scored case (ctx 80000, page 1, dist soft, spread
    agentic above batch 1, kv_scale 1), local_start 0."""
    import math
    dev = "cuda"
    ctx, page = 80000, 1
    gen = torch.Generator(device=dev).manual_seed(seed)
    tokens = batch * q_len
    if batch > 1:
        longest = math.ceil(ctx * 1.075)
        u = torch.rand((batch,), generator=gen, device=dev)
        seq_lens = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
    else:
        longest = ctx
        seq_lens = torch.full((batch,), ctx, dtype=torch.int64, device=dev)
    pages_per = (seq_lens + q_len + page - 1) // page
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=dev)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=dev)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page
                 + pos % page).reshape(-1)
    shape = (total, page, 2, 64)
    soft = torch.Generator(device=dev).manual_seed(seed + 0x50F7)
    uu = torch.randn((2, 64), generator=soft, device=dev)
    uu = uu * torch.rsqrt(uu.pow(2).mean(-1, keepdim=True))
    lo, hi = (0.3, 0.75) if layer == "sliding" else (0.5, 0.62)
    c2 = lo + (hi - lo) * torch.rand((2, 1), generator=soft, device=dev)
    mix = (c2.sqrt() * uu, (1 - c2).sqrt())
    k_cache = torch.empty(shape, dtype=FP8, device=dev)
    v_cache = torch.empty(shape, dtype=FP8, device=dev)
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, 2, 64)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            xr = torch.randn((n, 2, 64), generator=gen, device=dev)
            if keys:
                xr = mix[0] + mix[1] * xr
            flat[at:at + n] = xr.to(FP8)
    g0 = torch.Generator().manual_seed(_REF_STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=g0).to(dev)
    shared = torch.randn((_REF_STREAM_RANK, HIDDEN), generator=g0).to(dev)
    coef = torch.randn((tokens, _REF_STREAM_RANK), generator=gen, device=dev)
    scale = torch.ones((batch, q_len, 1), device=dev)
    if 1 < q_len <= 8:
        scale[:, 1:] = _REF_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (_REF_STREAM_MEAN * mean
                       + math.sqrt(_REF_STREAM_SHARED / _REF_STREAM_RANK) * coef @ shared)
              + (1 - _REF_STREAM_SHARED * scale * scale).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=dev))
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=dev)
    sched = None
    if layer == "full" and q_len == 1:
        try:
            from sgl_kernel.flash_attn import get_scheduler_metadata
            sched = get_scheduler_metadata(
                batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()),
                num_heads=16, num_heads_k=2, headdim=64, cache_seqlens=cache_seqlens,
                qkv_dtype=FP8, cu_seqlens_q=cu_seqlens_q, page_size=page, causal=True,
                num_splits=0)
        except Exception:
            sched = None
    attn_input = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS))
    return {
        "stream": stream, "attn_input": attn_input.to(torch.bfloat16), "positions": positions,
        "cache_loc": cache_loc, "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu_seqlens_q,
        "scheduler_metadata": sched, "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(1.0, dtype=torch.float32, device=dev),
        "v_scale": torch.tensor(1.0, dtype=torch.float32, device=dev),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "local_start": torch.tensor([0], dtype=torch.int32, device=dev),
        "config": {"batch": batch, "q_len": q_len, "tokens": tokens, "hidden": HIDDEN,
                   "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64, "page_size": page,
                   "window": 2048 if layer == "sliding" else 0,
                   "rope_theta": ROPE_THETA if layer == "sliding" else 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": 64 ** -0.5,
                   "kv_scale": 1.0, "n_experts": N_EXPERTS, "top_k": TOP_K,
                   "local_experts": LOCAL_EXPERTS},
    }



# The judge's output checks, restated from reference.py and checks.py (the judge's own copies
# may not be imported): FP32 reference with BF16 or E4M3 queries, routing acceptance, the
# relative L2 bounds and the cache-entry check.
_AB_DEQ = {}


def _ab_rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _ab_rope(x, pos, theta):
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, device=x.device).float() / 64))
    ang = pos.float()[:, None] * inv
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _ab_coeffs(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return alpha, beta


def _ab_attention(data, x, weights, qdtype):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    qkv = x @ weights["qkv_gate"].float().T
    q, k, v, gate = qkv.split([1024, 128, 128, 1024], dim=-1)
    pos = (data["seq_lens"].long()[:, None] + torch.arange(length, device=x.device)).reshape(-1)
    q = _ab_rms(q.view(-1, 16, 64))
    k = _ab_rms(k.view(-1, 2, 64))
    if cfg["rope_theta"]:
        q, k = _ab_rope(q, pos, cfg["rope_theta"]), _ab_rope(k, pos, cfg["rope_theta"])
    if qdtype == torch.bfloat16:
        q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16).float()
    else:
        q = q.to(qdtype).float() * cfg["qk_scale_factor"]
    q = q.view(batch, length, 16, 64)
    ks, vs = data["k_scale"], data["v_scale"]
    k_new = (k.float() / ks).to(FP8).view(batch, length, 2, 64)
    v_new = (v.view(-1, 2, 64).float() / vs).to(FP8).view(batch, length, 2, 64)
    k_flat = data["k_cache"].view(-1, 2, 64)
    v_flat = data["v_cache"].view(-1, 2, 64)
    window = cfg["window"]
    out = torch.empty((batch, length, 16, 64), device=x.device)
    steps = torch.arange(length, device=x.device)
    for b in range(batch):
        seq = int(data["seq_lens"][b])
        first = max(0, seq - window + 1) if window else 0
        past = torch.arange(first, seq, device=x.device)
        size = cfg["page_size"]
        rows = data["page_table"][b].long()[past // size] * size + past % size
        keys = torch.cat([k_flat[rows].float() * ks, k_new[b].float() * ks])
        vals = torch.cat([v_flat[rows].float() * vs, v_new[b].float() * vs])
        key_pos = torch.cat([past, seq + steps])
        query_pos = seq + steps
        visible = key_pos[None, :] <= query_pos[:, None]
        if window:
            visible &= key_pos[None, :] > query_pos[:, None] - window
        att = torch.einsum("lkgd,tkd->lkgt", q[b].reshape(length, 2, 8, 64), keys) * cfg["sm_scale"]
        att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
        out[b] = torch.einsum("lkgt,tkd->lkgd", att, vals).reshape(length, 16, 64)
    out = _ab_rms(out).reshape(-1, 1024) * torch.sigmoid(gate)
    return out, k_new, v_new


def _ab_ref_front(data, weights, qdtype=torch.bfloat16):
    o, k_new, v_new = _ab_attention(data, data["attn_input"].float(), weights, qdtype)
    a = o @ weights["o"].float().T
    alpha1, beta1 = _ab_coeffs(weights["attn_residual_gate"])
    stream1 = alpha1 * data["stream"].float() + beta1 * _ab_rms(a, POST_NORM_EPS)
    h = _ab_rms(stream1) * (1.0 + weights["pre_ffn_norm"].float())
    logits = h.double() @ weights["router"].double().T
    scores = torch.sigmoid(logits * ROUTER_MULTIPLIER).float()
    return {"stream1": stream1, "h": h, "scores": scores,
            "biased": scores + weights["router_bias"].float(), "k_new": k_new, "v_new": v_new,
            "local_start": int(data["local_start"])}


def _ab_deq(weights, e):
    if e not in _AB_DEQ:
        def dq(packed, sf, gs):
            p = packed.view(torch.uint8)
            codes = torch.stack((p & 0x0F, (p >> 4) & 0x0F), dim=-1).flatten(-2)
            grid = torch.tensor(_E2M1, dtype=torch.float32, device=p.device)
            mag = grid[(codes & 0x7).long()]
            vals = torch.where((codes & 0x8) != 0, -mag, mag)
            blocks = vals.view(*vals.shape[:-1], vals.shape[-1] // 16, 16)
            return (blocks * (sf.float() * gs.float()).unsqueeze(-1)).view_as(vals)
        _AB_DEQ[e] = (dq(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]),
                      dq(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]))
    return _AB_DEQ[e]


def _ab_ref_back(front, weights, ids):
    picked = front["scores"].gather(1, ids.long())
    tw = picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS)
    p = front["h"] @ weights["pooler"].float().T
    p = _ab_rms(p) * (1.0 + weights["pre_expert_norm"].float())
    local = (ids.long() - front["local_start"]) % N_EXPERTS
    moe = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
    for e in torch.unique(local).tolist():
        if not 0 <= e < LOCAL_EXPERTS:
            continue
        tok, slot = (local == e).nonzero(as_tuple=True)
        w13, w2 = _ab_deq(weights, e)
        hh = p[tok] @ w13.T
        inter = torch.nn.functional.silu(hh[:, :D_EXPERT]) * hh[:, D_EXPERT:]
        y = (inter @ w2.T) * (1.0 + weights["post_expert_norm"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        moe.index_add_(0, tok, y * tw[tok, slot, None])
    if len(moe) <= SLICED_UNPOOL_MAX_TOKENS:
        f = torch.zeros_like(front["stream1"])
        f[:, :UNPOOL_SLICE] = moe @ weights["unpooler"][:UNPOOL_SLICE].float().T
    else:
        f = moe @ weights["unpooler"].float().T
    f = f * (1.0 + weights["post_ffn_norm"].float())
    alpha2, beta2 = _ab_coeffs(weights["ffn_residual_gate"])
    stream2 = alpha2 * front["stream1"] + beta2 * _ab_rms(f, POST_NORM_EPS)
    hidden = _ab_rms(stream2) * (1.0 + weights["next_pre_attn_norm"].float())
    return hidden.to(torch.bfloat16), stream2


def _ab_local(ids, start):
    local = ids[(ids - start) % N_EXPERTS < LOCAL_EXPERTS]
    return int(local.numel()), int(torch.unique(local).numel())


def _ab_check_routing(ids, biased, start):
    if ids.min() < 0 or ids.max() >= N_EXPERTS:
        return "expert id out of range"
    ranked = ids.long().sort(-1).values
    if (ranked[:, 1:] == ranked[:, :-1]).any():
        return "a token routed to the same expert twice"
    best = biased.topk(TOP_K, dim=-1)
    gap = best.values[:, -1:] - biased.gather(1, ids.long())
    if gap.max() > 5e-2:
        return f"pick {float(gap.max()):.2e} below the 8th best"
    ninth = biased.topk(TOP_K + 1, dim=-1).values[:, -1:]
    keep_tol = 0.03 if len(ids) <= 256 else 0.04
    kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices - start) % N_EXPERTS < LOCAL_EXPERTS
    if (local & ~kept & (best.values - ninth > keep_tol)).any():
        return "dropped a clear local expert"
    routes, distinct = _ab_local(ids.long(), start)
    rr, rd = _ab_local(best.indices, start)
    if routes < rr * 0.92 - 4:
        return f"{routes} local routes against {rr}"
    if distinct < rd * 0.85 - 2:
        return f"{distinct} local experts against {rd}"
    return None


def _ab_compare(out, ref, tol):
    if out.shape != ref.shape or out.dtype != ref.dtype:
        return False, f"shape/dtype {tuple(out.shape)} {out.dtype}"
    o, r = out.float(), ref.float()
    if not torch.isfinite(o).all():
        return False, "NaN or Inf"
    d = o - r
    err = float(d.norm() / r.norm().clamp(min=1e-12))
    rows_d = d.reshape(-1, r.shape[-1]).norm(dim=-1)
    rows_r = r.reshape(-1, r.shape[-1]).norm(dim=-1)
    worst = float((rows_d / torch.maximum(rows_r, (0.1 * rows_r.mean()).clamp(min=1e-12))).max())
    return err <= tol and worst <= 5 * tol, f"{err:.2e}/{worst:.2e}<={tol:.3f}"


def _ab_validate(base, out, arm_data, weights, front, front8):
    """The judge's check_implementation on one arm's outputs: (ok, message)."""
    hidden, stream, ids = out
    err = _ab_check_routing(ids, front["biased"], front["local_start"])
    if err:
        return False, f"routing: {err}"
    ref_h, ref_s = _ab_ref_back(front, weights, ids)
    e8_h, e8_s = _ab_ref_back(front8, weights, ids)
    a1, _ = _ab_coeffs(weights["attn_residual_gate"])
    a2, _ = _ab_coeffs(weights["ffn_residual_gate"])
    carried = a1 * a2 * base["stream"].float()

    def bound(e):
        return min(0.15, max(0.03, 2.5 * e))

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))

    ok1, m1 = _ab_compare(stream - carried, ref_s - carried, bound(rel(e8_s - carried, ref_s - carried)))
    ok2, m2 = _ab_compare(hidden, ref_h, bound(rel(e8_h, ref_h)))
    cfg = base["config"]
    pos = base["seq_lens"].long()[:, None] + torch.arange(cfg["q_len"], device=ids.device)
    slots = (torch.gather(base["page_table"].long(), 1, pos // cfg["page_size"])
             * cfg["page_size"] + pos % cfg["page_size"])
    ok3, m3 = True, ""
    for name, cache, want, sc in (("k", "k_cache", front["k_new"], base["k_scale"]),
                                  ("v", "v_cache", front["v_new"], base["v_scale"])):
        got = arm_data[cache].view(-1, 2, 64)[slots]
        ok, m = _ab_compare(got.float() * sc, want.float() * sc, 8e-2)
        ok3, m3 = ok3 and ok, m3 + f" {name} {m}"
    return ok1 and ok2 and ok3, f"update {m1} hidden {m2} cache{m3}"


_AB_CODES = {}
_AB_USED = set()
_AB_NCHECK = {}
_AB_NFAIL = {}


def _ab_code(name):
    name = name.replace("void ", "").replace("(anonymous namespace)::", "")
    for pre in ("at::native::", "sglang::", "cutlass::"):
        name = name.replace(pre, "")
    cut = name.find("(")
    name = (name[:cut] if cut > 0 else name).replace(" ", "")[:40]
    if name not in _AB_CODES:
        _AB_CODES[name] = f"K{len(_AB_CODES)}"
    return _AB_CODES[name]


def _ab_timeline(graph, flush, reps=5):
    """One replay's kernels (the median span of reps), as (stream rank, start, duration, code)
    relative to the first kernel; the 256 MB flush fill is left out."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            flush.zero_()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
    evs = []
    for ev in prof.profiler.kineto_results.events():
        if ev.device_type() != DeviceType.CUDA:
            continue
        t0, t1 = _ev_ns(ev)
        name = ev.name()
        if "FillFunctor<unsigned char>" in name and t1 - t0 > 40000:
            continue
        evs.append((t0, t1, ev.device_resource_id(), name))
    evs.sort()
    segs, cur = [], []
    for e in evs:
        if cur and e[0] - max(c[1] for c in cur) > 20000:
            segs.append(cur)
            cur = []
        cur.append(e)
    if cur:
        segs.append(cur)
    if not segs:
        return None, [], []
    spans = sorted((max(e[1] for e in s) - s[0][0], i) for i, s in enumerate(segs))
    seg = segs[spans[len(spans) // 2][1]]
    base = seg[0][0]
    rank = {}
    for e in seg:
        rank.setdefault(e[2], len(rank))
    # Per kernel name, the median over the replays of its summed duration in one replay: one
    # replay's single-kernel durations swing by about 0.5 us, the size of the gaps to attribute.
    sums = []
    for s in segs:
        d = {}
        for e in s:
            c = _ab_code(e[3])
            n, t = d.get(c, (0, 0.0))
            d[c] = (n + 1, t + (e[1] - e[0]) / 1000)
        sums.append(d)
    agg = []
    for e in seg:
        c = _ab_code(e[3])
        if any(a[0] == c for a in agg):
            continue
        vals = sorted(d.get(c, (0, 0.0))[1] for d in sums)
        agg.append((c, vals[len(vals) // 2], sums[spans[len(spans) // 2][1]][c][0]))
    return spans[len(spans) // 2][0] / 1000, [
        (rank[e[2]], (e[0] - base) / 1000, (e[1] - e[0]) / 1000, _ab_code(e[3])) for e in seg], agg




def _ab_exposed(ks):
    """Kernels in start order as code@start+duration/exposed: exposed is how far the kernel's
    end moves the latest end so far (its share of the span); a trailing -idle marks a gap with
    nothing running before it started; s1: a second stream."""
    out, last = [], 0.0
    for r, st, d, c in sorted(ks, key=lambda e: e[1]):
        idle = st - last
        end = st + d
        exp = max(0.0, end - max(last, st))
        out.append(f"{'s1:' if r else ''}{c}@{st:.1f}+{d:.1f}/{exp:.1f}"
                   + (f"-{idle:.1f}" if idle > 0.05 else ""))
        last = max(last, end)
    return " ".join(out)


def _ab_clone(data):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}


def _ab_dev_table(seq, key):
    """Per arm, the median deviation (percent of that arm's median on the same input) of its
    replays grouped by key (the arm replayed just before it, or its position in the round)."""
    import statistics
    groups = {}
    for arm, pred, pos, dev in seq:
        groups.setdefault(arm, {}).setdefault(key(pred, pos), []).append(dev)
    out = []
    for arm in _AB_ARMS:
        g = groups.get(arm, {})
        out.append(f"{arm}: " + " ".join(
            f"{k} {statistics.median(v):+.2f}% ({len(v)})" for k, v in sorted(g.items())))
    return " | ".join(out)


def _ab_run(arms, weights, t_prep):
    """arms: name -> (kernel_fn, state). Every shape of _AB_LOADS: per input (held-out seeds,
    the routing variants in _AB_ROUTINGS), the judge's check of every arm first, then graphs
    captured in a seeded random arm order, then cold graph replays in a seeded random arm
    order per round, each arm on its own copy of the inputs."""
    import math
    import statistics
    names = list(arms)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    order_gen = torch.Generator().manual_seed(_AB_ORDER_SEED)
    res, missing, invalid = {}, [], set()
    seq = []
    for li, load in enumerate(_AB_LOADS):
        batch, q_len = _AB_LOAD_SHAPE[load]
        for layer in ("sliding", "full"):
            label = f"{load}{layer[0]}"
            if time.time() - t_prep > _AB_DEADLINE:
                missing.append(label)
                continue
            samples = {n: [] for n in names}
            tls = None
            for k, routing in enumerate(_AB_ROUTINGS):
                seed = _AB_SEED0 + 1000 * li + 100 * (layer == "full") + k
                base = _ab_inputs(batch, q_len, layer, seed)
                front0 = _ab_ref_front(base, weights)
                ids0 = front0["biased"].topk(TOP_K, dim=-1).indices
                pairs = torch.bincount((ids0 // LOCAL_EXPERTS).flatten(),
                                       minlength=N_EXPERTS // LOCAL_EXPERTS)
                rank = pairs.argsort(descending=True)
                base["local_start"].fill_(LOCAL_EXPERTS * int(rank[0 if routing == "busiest" else 1]))
                front = dict(front0, local_start=int(base["local_start"]))
                front8 = _ab_ref_front(base, weights, FP8)
                front8["local_start"] = front["local_start"]
                rr, rd = _ab_local(ids0, front["local_start"])
                datas = {n: _ab_clone(base) for n in names}
                for n, (fn, st) in arms.items():
                    out = [o.clone() for o in fn(datas[n], st)]
                    torch.cuda.synchronize()
                    ok, msg = _ab_validate(base, out, datas[n], weights, front, front8)
                    same = float((out[2].long().sort(-1).values == ids0.sort(-1).values)
                                 .all(-1).float().mean())
                    lr, ld = _ab_local(out[2].long(), front["local_start"])
                    if not ok:
                        invalid.add((label, n))
                    if ok:
                        msg = msg.split(" cache")[0]
                    _AB_NCHECK[n] = _AB_NCHECK.get(n, 0) + 1
                    _AB_NFAIL[n] = _AB_NFAIL.get(n, 0) + (not ok)
                    _ab_print(f"{'ABCHECK' if ok else 'ABCHKFAIL'} {label} {seed} {routing[:4]} "
                              f"{n} {'PASS' if ok else 'FAIL'} {msg} top8=ref {same:.3f} local "
                              f"routes {lr}/{rr} experts {ld}/{rd}")
                datas = {n: _ab_clone(base) for n in names}
                # Capture order decides which arm's graph pool takes the memory freed by the
                # previous shape, so it is randomized like the replay order.
                cap = [names[i] for i in torch.randperm(len(names), generator=order_gen)]
                graphs = {}
                for n in cap:
                    fn, st = arms[n]
                    graphs[n] = _graph_build(lambda fn=fn, st=st, d=datas[n]: fn(d, st))
                torch.cuda.synchronize()
                evs, prev = [], None
                for r in range(_AB_ROUNDS):
                    order = [names[i] for i in torch.randperm(len(names), generator=order_gen)]
                    for pos, n in enumerate(order):
                        flush.zero_()
                        e0 = torch.cuda.Event(enable_timing=True)
                        e1 = torch.cuda.Event(enable_timing=True)
                        e0.record()
                        graphs[n].replay()
                        e1.record()
                        evs.append((n, prev, pos, e0, e1))
                        prev = n
                torch.cuda.synchronize()
                here = {n: [] for n in names}
                for n, pred, pos, e0, e1 in evs:
                    here[n].append((pred, pos, 1000 * e0.elapsed_time(e1)))
                for n, rows in here.items():
                    samples[n].extend(t for _, _, t in rows)
                    mid = statistics.median(t for _, _, t in rows)
                    seq.extend((n, pred, pos, 100 * (t / mid - 1)) for pred, pos, t in rows)
                if k == 0:
                    tls = {n: _ab_timeline(graphs[n], flush) for n in (_AB_A, _AB_B)}
                del graphs, datas, base
                torch.cuda.empty_cache()
            med = {n: statistics.median(v) for n, v in samples.items()}
            res[(load, layer)] = med

            def pdiff(a, b):
                return statistics.median([x - y for x, y in zip(samples[a], samples[b])])

            flags = "".join(f" INVALID:{n}" for n in names if (label, n) in invalid)
            _ab_print(f"ABTIME {label} " + " ".join(f"{n} {med[n]:.1f}" for n in names)
                      + f" | paired {_AB_B}-{_AB_A} {pdiff(_AB_B, _AB_A):+.1f} "
                      f"{_AB_B2}-{_AB_A} {pdiff(_AB_B2, _AB_A):+.1f} "
                      f"{_AB_B}-{_AB_A2} {pdiff(_AB_B, _AB_A2):+.1f} "
                      f"{_AB_B2}-{_AB_A2} {pdiff(_AB_B2, _AB_A2):+.1f} | null "
                      f"{_AB_A2}-{_AB_A} {pdiff(_AB_A2, _AB_A):+.1f} "
                      f"{_AB_B2}-{_AB_B} {pdiff(_AB_B2, _AB_B):+.1f} "
                      f"(n={len(samples[_AB_A])}){flags}")
            for n in (_AB_A, _AB_B):
                span, ks, agg = tls[n]
                if span is None:
                    continue
                _ab_print(f"ABTL {label} {n} {span:.1f}: " + " ".join(
                    f"{'s1:' if r else ''}{c}@{st:.1f}+{d:.1f}" for r, st, d, c in ks))
                _AB_USED.update(c for c, _, _ in agg)
                if label in _AB_TLEND:
                    _AB_USED.update(c for _, _, _, c in ks)
                    _ab_print(f"ABZ {label} {n} span {span:.1f}: " + _ab_exposed(ks))
                _ab_print(f"ABK {label} {n} span {span:.1f} sum {sum(t for _, t, _ in agg):.1f}: "
                          + " ".join(f"{c} {t:.1f}" + (f"x{k}" if k > 1 else "")
                                     for c, t, k in agg))
    if missing:
        _ab_print("ABSTOP missing coverage (not timed, not a pass): " + " ".join(missing))
    if seq:
        # A same-module copy replayed just before an arm is the one carry-over the old
        # three-arm bench gave only cand and null (they shared one state and one compiled
        # module, inc had no twin); these tables show whether it moves the timings.
        _ab_print("ABPRED dev from the arm's per-input median by the arm replayed before it: "
                  + _ab_dev_table(seq, lambda pred, pos: pred or "first"))
        _ab_print("ABPOS dev by position in the round: "
                  + _ab_dev_table(seq, lambda pred, pos: f"p{pos}"))
        twin = []
        for a, b in ((_AB_A, _AB_A2), (_AB_B, _AB_B2)):
            for x, y in ((a, b), (b, a)):
                after = [d for arm, p, _, d in seq if arm == x and p == y]
                other = [d for arm, p, _, d in seq if arm == x and p is not None and p != y]
                if after and other:
                    twin.append(f"{x} after {y} minus after others "
                                f"{statistics.median(after) - statistics.median(other):+.2f}%")
        _ab_print("ABTWIN " + "; ".join(twin))
    cross = [(a, b) for a in (_AB_A, _AB_A2) for b in (_AB_B, _AB_B2)]
    logs = {key: [] for key in cross + [(_AB_A2, _AB_A), (_AB_B2, _AB_B)]}
    for load in _AB_LOADS:
        if (load, "sliding") not in res or (load, "full") not in res:
            continue
        c = {n: 46 * res[(load, "sliding")][n] + 16 * res[(load, "full")][n] for n in names}
        for a, b in logs:
            logs[(a, b)].append(math.log(c[a] / c[b]))
        pooled = math.exp(sum(math.log(c[a] / c[b]) for a, b in cross) / len(cross))
        _ab_print(f"ABLOAD {load} (46 sliding + 16 full) / 62: "
                  + " ".join(f"{n} {c[n] / 62:.1f}" for n in names)
                  + " | cross " + " ".join(f"{a}/{b} {c[a] / c[b]:.4f}" for a, b in cross)
                  + f" pooled {pooled:.4f} | null {_AB_A2}/{_AB_A} {c[_AB_A2] / c[_AB_A]:.4f} "
                  f"{_AB_B2}/{_AB_B} {c[_AB_B2] / c[_AB_B]:.4f}")
    nl = len(logs[cross[0]])
    if nl:
        geo = {key: math.exp(sum(v) / nl) for key, v in logs.items()}
        pooled = math.exp(sum(math.log(geo[key]) for key in cross) / len(cross))
        noise = max(abs(math.log(geo[(_AB_A2, _AB_A)])), abs(math.log(geo[(_AB_B2, _AB_B)])))
        _ab_print(f"ABSCORE geometric mean over {nl} loads of load-time ratios (above 1: the "
                  f"right-hand arm faster): {_AB_A}/{_AB_B} {geo[(_AB_A, _AB_B)]:.4f}; cross "
                  + " ".join(f"{a}/{b} {geo[(a, b)]:.4f}" for a, b in cross)
                  + f"; pooled {pooled:.4f}; null {_AB_A2}/{_AB_A} "
                  f"{geo[(_AB_A2, _AB_A)]:.4f} {_AB_B2}/{_AB_B} {geo[(_AB_B2, _AB_B)]:.4f}")
        lo, hi = min(geo[key] for key in cross), max(geo[key] for key in cross)
        gain = math.log(pooled)
        if lo > 1 and gain > 2 * noise:
            verdict = f"{_AB_B} faster"
        elif hi < 1 and -gain > 2 * noise:
            verdict = f"{_AB_B} slower"
        else:
            verdict = "not resolved"
        _ab_print(f"ABDECIDE {verdict}: pooled {_AB_A}/{_AB_B} {pooled:.4f} (log {gain:+.4f}), "
                  f"cross range {lo:.4f}-{hi:.4f}, larger null |log| {noise:.4f}; rule: all "
                  "four cross ratios on one side of 1 and |log pooled| above twice the larger "
                  "null")
    for name, code in _AB_CODES.items():
        if code in _AB_USED:
            _ab_print(f"ABKN {code} {name}")
    for n in names:
        fails = sum(1 for lab, arm in invalid if arm == n)
        _ab_print(f"ABCHK {n} {_AB_NCHECK.get(n, 0) - _AB_NFAIL.get(n, 0)} of "
                  f"{_AB_NCHECK.get(n, 0)} PASS; shapes with a FAIL {fails}")


def _ab_policy(ns, state):
    """The prepare-time picks of one arm, as short strings keyed by what was picked; ns is the
    globals of the module that prepared state."""
    out = {}

    def put(key, fn):
        try:
            out[key] = fn()
        except Exception as exc:
            out[key] = f"? {type(exc).__name__}"

    put("moe", lambda: " ".join(
        f"{bs}:{ns['_moe1_name'](ns['_moe_opts'](bs, state)['fused'])}"
        for bs in ns["TUNE_TOKENS"]))
    put("w4", lambda: " ".join(
        "{}:{}/{}".format(bs, *(ns["_moe_opts"](bs, state)[g] for g in ("g1", "g2")))
        for bs in ns["TUNE_TOKENS"]))
    put("front", lambda: " ".join(f"{k}:{v}" for k, v in sorted(state["front_policy"].items())))
    put("abnd", lambda: " ".join(f"{k}:{v}" for k, v in sorted(
        state.get("abnd_policy", {}).items())))
    put("skp", lambda: " ".join(
        f"{layer[0]}{m}:{'-' if pick is None else pick[0]}"
        for (layer, m), pick in sorted(state.get("skp_pick", {}).items())))
    put("wide", lambda: " ".join(f"{k}:{v[0]}" for k, v in sorted(
        state.get("skp_wide_report", {}).items())))
    put("m1", lambda: str(state.get("m1_policy", "none")))
    # Layers with the attention workspace pick (tools/dl_one_port_patch.py) record it here; a
    # pick is per prepare, so the four arms differ by design.
    put("wspick", lambda: str((state.get("one") or {}).get("ws_pick", "-")))
    put("stages", lambda: "".join(str(int(bool(state.get(k)))) for k in (
        "moe_ours", "w4_ok", "moe1_ok", "router_ours", "proj_ours", "skp_ok", "attn_one_ours",
        "attn_ws_ours", "attn_fused_ours", "bnd_ours", "abnd_ours", "norm_ours",
        "route_side_ok", "route_ours")))
    return out


_ab_prepare = prepare
_timeline = lambda state: None  # noqa: E731  (the bench prints its own timelines)


def _ab_prepare_arm(arm, weights):
    """One arm's prepare() on the judge weights: (kernel_fn, state, module globals). Only
    cand's prepare prints in full; the others' output is scanned for failures."""
    import contextlib
    import io
    mod = globals() if arm in ("cand", "cand2") else _AB_OTHER_MOD.ns
    fn = _ab_prepare if mod is globals() else mod["prepare"]
    t1 = time.time()
    if arm == "cand":
        st = fn(weights)
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st = fn(weights)
        keys = ("fallback", "failed", "disabled", "Error", "SUMMARY stages", "SUMMARY failed")
        hits = [ln for ln in buf.getvalue().splitlines() if any(k in ln for k in keys)]
        for ln in hits[:8]:
            _ab_print(f"ABOTHER {arm} prepare: {ln[:300]}")
    torch.cuda.synchronize()
    _ab_print(f"ABPREP {arm} {time.time() - t1:.0f} s")
    return mod["kernel_fn"], st, mod


def prepare(weights):
    for name, pin in _AB_PINS.items():
        _ab_print(f"ABPIN {name} {pin}")
    t = time.time()
    # Each arm's copy is prepared on its own (own state, own tuner picks) unless _AB_DUP is
    # "shared", where the copy reuses its twin's state as the old null arm did. The order of
    # the prepare calls is seeded random: the first-prepared state gets the first allocations
    # and runs its tuners on a GPU nothing else has touched yet.
    want = ["cand", _AB_OTHER] + (["cand2", _AB_OTHER + "2"] if _AB_DUP == "own" else [])
    gen = torch.Generator().manual_seed(_AB_PREP_SEED)
    order = [want[i] for i in torch.randperm(len(want), generator=gen)]
    if _AB_OTHER_MOD is None:
        order = ["cand"]
    _ab_print(f"ABPREP order {' '.join(order)} (dup {_AB_DUP}, seed {_AB_PREP_SEED})")
    prepared, state = {}, None
    for arm in order:
        try:
            prepared[arm] = _ab_prepare_arm(arm, weights)
        except Exception as exc:
            if arm == "cand":
                raise
            _ab_print(f"ABFAIL prepare {arm}: {type(exc).__name__}: {str(exc)[:300]}")
    state = prepared["cand"][1]
    _ab_print("ABDIAG cand attention fallback on fresh rows vs FP32: "
              + str(state.get("one_fallback_fp32", "not run"))
              + f"; attn_one_ours={int(bool(state.get('attn_one_ours')))}")
    if _AB_CAND_REPORT:
        for item in state.get("m1_report", []):
            _ab_print(f"ABCAND m1 {item[:400]}")
        for name, item in state.get("moe1_gate", {}).items():
            _ab_print(f"ABCAND moe gate {name} {item}")
        for key in ("route_gate", "gate_negctl"):
            for name, item in state.get(key, {}).items():
                _ab_print(f"ABCAND {key} {name} {item}")
        if "_gate_manifest" in globals():
            _ab_print(f"ABCAND {_gate_manifest(state)}")
        for item in _FAILED:
            _ab_print(f"ABCAND failed {item[:200]}")
    if _AB_OTHER_MOD is not None and all(a in prepared for a in want):
        try:
            if _AB_DUP != "own":
                prepared["cand2"] = prepared["cand"]
                prepared[_AB_OTHER + "2"] = prepared[_AB_OTHER]
            arms = {n: prepared[n][:2] for n in _AB_ARMS}
            pols = {n: _ab_policy(prepared[n][2], prepared[n][1]) for n in _AB_ARMS}
            _ab_print(f"ABPREP all at {time.time() - t:.0f} s")
            for key in pols[_AB_ARMS[0]]:
                vals = [pols[n][key] for n in _AB_ARMS]
                same = all(v == vals[0] for v in vals)
                _ab_print(f"ABPOL {key} {'same' if same else 'DIFF'}: "
                          + (vals[0] if same else " | ".join(
                              f"{n} {v}" for n, v in zip(_AB_ARMS, vals))))
            _ab_run(arms, weights, t)
            _ab_print(f"ABEND prepare at {time.time() - t:.0f} s")
        except Exception as exc:
            import traceback
            _ab_print(f"ABFAIL {type(exc).__name__}: {str(exc)[:300]} "
                      + traceback.format_exc()[-800:].replace("\n", " | "))
    _AB_DEQ.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # The log keeps only its last 16,000 characters. Most expendable first: raw timelines,
    # passing checks, build and candidate reports, the per-kernel tables; last, the arms'
    # policies, the carry-over tables, the timings and the decision.
    order = ("ABTL", "ABCHECK", "ABBUILD", "ABOTHER", "ABCAND", "ABKN", "ABK ", "ABZ ", "ABPIN",
             "ABDIAG", "ABPREP", "ABEND", "ABFAIL", "ABCHKFAIL", "ABCHK", "ABSTOP", "ABPOS",
             "ABPRED", "ABTWIN", "ABPOL", "ABTIME", "ABLOAD", "ABSCORE", "ABDECIDE")
    rank = lambda line: next((i for i, p in enumerate(order) if line.startswith(p)), 0)  # noqa: E731
    for line in sorted(_AB_OUT, key=rank):
        print(line, flush=True)
    return state
'''


KEEP_HASH = ("#define", "#include", "#if", "#ifdef", "#ifndef", "#endif", "#else", "#elif",
             "#pragma", "#undef", "#error")


def strip_comments(text):
    """Whole-line comments out: `//` lines (C++) and `#` lines that are not preprocessor
    directives (Python). Fails on a dropped line that is part of a backslash continuation, or
    on a `#  directive` with spaces, so nothing that compiles can change."""
    lines = text.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines):
        t = line.strip()
        if re.match(r"#\s+(define|include|if|ifdef|ifndef|endif|pragma|else|elif|undef)\b", t):
            raise SystemExit(f"spaced preprocessor directive at line {i + 1}")
        drop = t.startswith("//") or (t.startswith("#") and not t.startswith(KEEP_HASH))
        # A comment line the code names in a string (a substitution marker such as
        # //@MOE1_DISPATCH@, which _moe1_source() replaces) stays: dropping it broke the
        # incumbent's MoE and W4 dispatch in check 135ab2a9f22b49a5.
        if drop and any(q + t in text or t + q in text for q in ('"', "'")):
            drop = False
        if drop and i > 0 and (line.rstrip().endswith("\\") or lines[i - 1].rstrip().endswith("\\")):
            raise SystemExit(f"comment line {i + 1} inside a continuation")
        if not drop:
            out.append(line)
    stripped = "".join(out)
    ast.parse(stripped)
    return stripped


def pub_patch(text):
    for old, new in PUB_PATCHES:
        if text.count(old) != 1:
            raise SystemExit(f"public patch does not apply: {old[:80]!r}")
        text = text.replace(old, new)
    if "threading" in text:
        raise SystemExit("the public file still names threading")
    return text


def inc_patch(text):
    n = text.count('name="okc_decode_layer_')
    if n != 5:
        raise SystemExit(f"expected the incumbent's five load_inline names, found {n}")
    return text.replace('name="okc_decode_layer_', 'name="okc_inc_decode_layer_')


def pin(data):
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return f"sha256 {hashlib.sha256(data).hexdigest()} git-blob {blob}"


def main():
    opts = {"rounds": "10", "deadline": "255", "seed0": "50001", "order-seed": "7", "loads": "",
            "other": "", "cand": str(ROOT / "muse_decode_layer/submission.py"), "tlend": "",
            "cand-strip": "", "dup": "", "prep-seed": "", "cand-report": "1"}
    pos = []
    for a in sys.argv[1:]:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            if k not in opts:
                raise SystemExit(f"unknown option --{k}")
            opts[k] = v
        else:
            pos.append(a)
    if len(pos) != 2 or not opts["loads"] or opts["other"] not in ("inc", "pub"):
        raise SystemExit(__doc__)
    # The public file's prepare() has never been run twice in one process.
    dup = opts["dup"] or ("own" if opts["other"] == "inc" else "shared")
    if dup not in ("own", "shared"):
        raise SystemExit(f"--dup={dup}: own or shared")
    out, other = Path(pos[0]), Path(pos[1])
    loads = tuple(opts["loads"].split(","))
    if not set(loads) <= {"d1", "d16", "d32", "v1", "v16", "v32"}:
        raise SystemExit(f"unknown load in {loads}")
    raw = other.read_bytes()
    patched = (pub_patch if opts["other"] == "pub" else inc_patch)(raw.decode())
    sent = strip_comments(patched)
    packed = base64.b85encode(lzma.compress(sent.encode(), preset=9 | lzma.PRESET_EXTREME))
    cand = Path(opts["cand"])
    pins = {opts["other"]: f"{pin(raw)} as read ({other.name}); comment-stripped and patched "
                           f"as sent: {pin(sent.encode())}",
            "cand": f"{pin(cand.read_bytes())} ({cand.name})"}
    cand_text = cand.read_text()
    if opts["cand-strip"]:
        head = cand_text.splitlines(keepends=True)[0]
        cand_text = strip_comments(cand_text)
        if head.startswith("# okc:"):
            cand_text = head + cand_text
        pins["cand"] += f"; comment-stripped as sent: {pin(cand_text.encode())}"
    params = (f"\n_AB_PINS = {pins!r}\n"
              f"_AB_OTHER = {opts['other']!r}\n"
              f"_AB_OTHER_XZ = {packed.decode()!r}\n"
              f"_AB_LOADS = {loads!r}\n_AB_ROUNDS = {int(opts['rounds'])!r}\n"
              "# prepare() has 300 s; no new shape starts past this many seconds into it.\n"
              f"_AB_DEADLINE = {int(opts['deadline'])!r}\n"
              f"_AB_SEED0 = {int(opts['seed0'])!r}\n"
              f"_AB_ORDER_SEED = {int(opts['order-seed'])!r}\n"
              f"_AB_PREP_SEED = {int(opts['prep-seed'] or opts['order-seed'])!r}\n"
              f"_AB_DUP = {dup!r}\n"
              f"_AB_CAND_REPORT = {opts['cand-report'] == '1'!r}\n"
              "_AB_ROUTINGS = (\"busiest\", \"second\")\n"
              f"_AB_TLEND = {tuple(x for x in opts['tlend'].split(',') if x)!r}\n")
    text = cand_text + params + BODY
    if len(text.encode()) >= 1_000_000:
        raise SystemExit(f"{len(text.encode())} bytes: the judge takes under 1,000,000")
    out.write_text(text)
    print(f"wrote {out} ({len(text.encode())} bytes): loads {loads}; dup {dup}; "
          + "; ".join(f"{k} {v}" for k, v in pins.items()))


if __name__ == "__main__":
    main()
