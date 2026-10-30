"""Write a same-process A/B bench of the fused MoE kernel against the public entry's MoE file.

    python3 tools/moe_make_bench_ab.py OUT.py PUBLIC.py [--base=28] [--seeds=1,2,...]
        [--sizes=32,128,256] [--routings=layer,std,skewed] [--pairs=A-B,...] [--gate=1]
        [--skew=1] [--work=1] [--skst=8:3:2,...] [ARM=MODE ...]

--work=1 prints ABWORK per size and routing: the work each design gives each CTA (our jobs and
owned phase A units, the public kernel's stream-K stage ranges and cut units ported from its
wgs_range, the skinny path's split jobs, weights streamed) and our claimed phase B units per CTA
from a counter twin. --skst adds stamped skinny variants (kMode 2) outside --small and prints
their ABSK at 32 tokens and below, next to the fat stamps.

--gate=1 is for a pre-registered keep-or-drop run: no stamped or counter twins, no stamped
public runs (ABSTAMP, ABDIAG, ABPUB are not printed, so three routings fit the 16 KB log tail),
the arms' modes added to MAIN_CANDIDATES (prepare's FP32 self-test covers them), and ABCHECKALL,
each arm's output against the base on every seed.

OUT.py is muse_moe_w4a16/submission.py with MAIN_MODES = (0, base, the arms, their stamped
twins), the gate and the stamped diagnostics off, and a block that runs at import, before the
judge's prepare(), on random NVFP4 weights (32 experts):

  ABROUTE  busiest local expert per seed for each routing: "std" is reference.py's
           generate_input (the standalone board), "layer" a port of the layer's judge routing
           (reference.py of muse_decode_layer: residual stream, boundary, sigmoid router with
           bias, busiest 32-expert range), with the router seed picked from 1-40 whose busiest
           expert matches the layer's SUMMARY lines (21 / 53 / 107 at 32 / 128 / 256 tokens).
  ABCHECK  per size and routing: the base mode and the public kernel against the FP32
           reference, every arm against the base.
  ABTIME   per size and routing: graph replay after a 256 MB L2 flush, median of 20, arms
           interleaved and reversed on every other seed; the mean per arm.
  ABPAIR   per-seed paired deltas of every arm against the base, the public kernel against
           the base, and the --pairs.
  ABSTAMP  per size and routing, mean over the seeds of one cold launch each: thread 0's
           globaltimer at the phase boundaries of the stamped twins (light stamps) and of the
           public kernel built with its own OKC_DEBUG stamps (DBG(3) moved to the CTA-wide
           point before the grid barrier).
  ABDIAG   for the phase A counter twins (31, 33, 38, 40, 41, 44 of 28, 32, 36, 39, 37, 43):
           per-CTA time (consumers start to phase A end) as mean / p10 / p50 / p90 / max, owned
           units, MB streamed (36,864 B of weights per stage plus 4,096 B of x per n-tile per
           stage) and the rank correlation of time with MB across CTAs; per consumer warp, us in
           weights
           waits and x waits, the rest of the units (compute and ring bookkeeping), the epilogue
           (per unit too) and the deferred publish; the producers' phase A waits for a free slot.
           Since 2026-10-29 every wait is timed from before its first try_wait (it can suspend
           the thread); earlier runs counted only waits whose first try failed, and only the
           time after that try.
  ABDIAGB  for the twins with phase B counters (41, 44): per consumer warp over phase B, us in
           weights waits, x waits (which carry the done_as dependency on other CTAs' phase A),
           waits for a claimed unit, the rest of the units, the epilogue, units, stages and
           n-tiles per unit; and the CTA's phase B span (thread 0's phase A end to the CTA's
           barrier after phase B, light stamps). ABDIAG and ABDIAGB end with the average
           cycles per wait (weights and x wait cycles over twice the stages; since 2026-10-29).
  ABPUB    the public kernel's per-CTA phase A time, its stream-K stage ranges (patched to land
           in DBG slot 7) and the number of cut units per CTA.
  ABSKEW   with --skew=1, per phase (A, B), mean over CTAs and seeds of one cold launch each, for
           our skew twin (49 of 47) and, with --pubskew=1, the public kernel built with the same
           per-warp stamps (OKC_SKEW, its own build). Not usable as of 2026-10-30: at its 96
           registers any consumer stamp makes the public DEBUG build spill 400-1,260 B and three
           of four stamp subsets serialize its wgmmas (C7512; local nvcc 12.8, sm_90a), so its
           skew would not be its own. Columns: stages per CTA, the
           per-warp weights and x waits, cycles per wait, the spread across the 16 warps of a
           CTA of the stage start (also across the 4 warpgroups' means and within a
           warpgroup), of the weights wait end and of the slot release; per wait the share a
           zero-skew release would remove (min(wait, slowest warp's release of the slot minus
           this warp's)), the producer's delay after the slowest release, the copy latency
           (first waiting warp's wait end minus the producer's issue), this warp's lead (its
           stage start minus its own release of the slot) and the x copy latency.

The judge's prepare() that follows runs with its output filtered to its summary lines (POLICY,
PREPARE, BENCH at 128 and 256 tokens, each variant's worst self-test error), so the run log's
15 KB tail keeps the A/B. Twins: 26 -> 27 and 37 -> 42 (light stamps), 28 -> 31, 32 -> 33, 36 -> 38,
39 -> 40, 37 -> 41 and 43 -> 44 (light stamps and phase A counters, 41 and 44 with phase B
counters too; name a twin as an arm to time it so its cost shows). An arm "v301" (the staged pair on 37's components) also
builds modes 45 and 46 and needs 37 built (its fallback above 512 tokens).

The public file is embedded as a string; OUT.py belongs in the session scratchpad, never in
the repository (only the logs of its runs are committed).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIGHT_TWINS = {26: 27, 37: 42, 47: 48, 52: 54}
SKEW_TWINS = {47: 49}
PA_TWINS = {28: 31, 32: 33, 36: 38, 39: 40, 37: 41, 43: 44}
PB_TWINS = (41, 44)
# Variant arms that need fat modes built: v301 is the staged pair on 37's components (45 then 46).
VARIANT_MODES = {"v301": (45, 46)}


# Per-unit stamps of the skinny ring (--unitstamps), the same events and layout as our kMode 16:
# per CTA 8 header words (units in phase A, units in all, clock64 and globaltimer at the start and
# at the end) and 4 clock64 per unit q < 256 (copy issued, thread 0 starts waiting, the wait
# ends, the slot is free after the unit barrier), after the G * 8 DBG words.
US_MAX = 256
US_PER = 8 + 4 * US_MAX
UNIT_STAMP_PAIRS = [
    ("#define DBG(k) do {",
     "#define UST(q, k) do { if (OKC_DEBUG && p.dbg && (q) < %d) p.dbg[gridDim.x * 8 + "
     "blockIdx.x * %d + 8 + (q) * 4 + (k)] = (unsigned long long)clock64(); } while (0)\n"
     "#define USH(k, v) do { if (OKC_DEBUG && p.dbg) p.dbg[gridDim.x * 8 + blockIdx.x * %d + (k)]"
     " = (unsigned long long)(v); } while (0)\n#define DBG(k) do {" % (US_MAX, US_PER, US_PER)),
    ("  const int sl = q % OKC_NS;\n  if (q < nA) ring_issue<false>",
     "  const int sl = q % OKC_NS;\n  UST(q, 0);\n  if (q < nA) ring_issue<false>"),
    ('"r"(smem_u32(S.bars + i)) L2H_ARG : "memory");\n      }\n      S.preissued',
     '"r"(smem_u32(S.bars + i)) L2H_ARG : "memory");\n        UST(i, 0);\n      }\n'
     '      S.preissued'),
    ("    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n"
     "    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc",
     "    if (tid == 0) UST(q, 1);\n    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n"
     "    if (tid == 0) UST(q, 2);\n"
     "    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc"),
    ("    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0]",
     "    CTA_SYNC();\n    if (tid == 0) {\n      UST(q, 3);\n      const int nA = S.wsr[1] - S.wsr[0]"),
    ("  const int tid = threadIdx.x;\n  DBG(0);\n",
     "  const int tid = threadIdx.x;\n  DBG(0);\n"
     "  if (tid == 0) { USH(2, clock64()); USH(3, gtimer()); }\n"),
    ("    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n",
     "    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n"
     "    USH(0, S.wsr[1] - S.wsr[0]);\n"
     "    USH(1, S.wsr[1] - S.wsr[0] + S.wsr[3] - S.wsr[2]);\n  }\n"),
    ("  DBG(5);  // nothing to reset at exit",
     "  if (tid == 0) { USH(4, clock64()); USH(5, gtimer()); }\n  DBG(5);  // nothing to reset at exit"),
]


# Per-warp stage stamps of the public kernel (--pubskew=1; see ABSKEW for why unused), the
# layout of our skew_st modes: per CTA 8 header words, then per stage q < SK_Q 16 x 4 consumer
# words (stage start, weights wait done, x wait done, weights slot released) and 2 producer
# words (weights and x copy issued), after the G * 8 DBG words and the G * US_PER unit-stamp
# words. Built only with -DOKC_SKEW=1.
SK_Q = 256
SK_ROW = 66
SK_PER = 8 + SK_Q * SK_ROW
SKEW_PAIRS = [
    ("#define DBG(k) do {",
     "#ifndef OKC_SKEW\n#define OKC_SKEW 0\n#endif\n"
     "#define SKW(q, k, on) do { if (OKC_SKEW && p.dbg) { const unsigned long long c_ = clock64(); "
     "unsigned long long* a_ = p.dbg + gridDim.x * %d + blockIdx.x * %d + 8 + (q) * %d + (k); "
     "asm volatile(\"{\\n .reg .pred sp;\\n setp.ne.s32 sp, %%2, 0;\\n @sp st.global.b64 "
     "[%%0], %%1;\\n}\" :: \"l\"(a_), \"l\"(c_), \"r\"((int)((on) && (q) < %d))); } } "
     "while (0)\n#define SKH(k, v) do { if (OKC_SKEW && p.dbg) p.dbg[gridDim.x * %d + blockIdx.x"
     " * %d + (k)] = (unsigned long long)(v); } while (0)\n#define DBG(k) do {"
     % (8 + US_PER, SK_PER, SK_ROW, SK_Q, 8 + US_PER, SK_PER)),
    ("    const int ws = seq % WGS_NS, xs = seq % WGS_NX;\n"
     "    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n"
     "    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n"
     "    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n",
     "    const int ws = seq % WGS_NS, xs = seq % WGS_NX;\n    SKW(seq, warp * 4, lane == 0);\n"
     "    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n"
     "    SKW(seq, warp * 4 + 1, lane == 0);\n"
     "    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n"
     "    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n"
     "    SKW(seq, warp * 4 + 2, lane == 0);\n"),
    ("        mbar_arrive1(S.bars + BW_E + ws);  // every code",
     "        mbar_arrive1(S.bars + BW_E + ws);\n"
     "        SKW(seq, warp * 4 + 3, true);  // every code"),
    ("      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n",
     "      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n"
     "      SKW(n, 64, true);\n"),
    ("      uint64_t* bar = S.bars + BX_F + xs;\n",
     "      SKW(n, 65, true);\n      uint64_t* bar = S.bars + BX_F + xs;\n"),
    ("    int seq = 0;\n    wgs_phase<false>",
     "    int seq = 0;\n"
     "    if (threadIdx.x == 0) { SKH(0, WGS_NS); SKH(1, clock64()); SKH(2, gtimer()); }\n"
     "    wgs_phase<false>"),
    ("  DBG(5);  // nothing to reset at exit",
     "  if (threadIdx.x == 0) { SKH(3, clock64()); SKH(4, gtimer()); }\n"
     "  DBG(5);  // nothing to reset at exit"),
]


def patch_public(src, small=False, ranges=False, unitstamps=False, skew=False):
    for name in () if small else ("TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES"):
        src, n = re.subn(rf"^{name} = \(.*?\)\n(?=\S)", f"{name} = ()\n", src, count=1,
                         flags=re.S | re.M)
        assert n == 1, name
    pairs = [
        ("#define OKC_DEBUG 0\n", "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n"),
        ("    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n",
         "    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n"),
        ("  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);",
         "  CTA_SYNC();\n  DBG(3);\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);"),
    ]
    # The stream-K stage ranges of both phases, 16 bits each, over DBG(7)'s stamp. Timing is
    # not usable with it: the DEBUG MAIN build then spills 392 / 896 B (none without it) and
    # ran 30 us over its replay in checks c7db20e053254b52, 0b684d5db0834dbc, 473d87d03de94e96.
    pairs += [] if not ranges else [
        ("  WS_JOIN();\n  DBG(6);\n",
         "  WS_JOIN();\n  DBG(6);\n  if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) {\n"
         "    p.dbg[blockIdx.x * 8 + 7] = (long long)S.wgr[0] | ((long long)S.wgr[1] << 16)\n"
         "        | ((long long)S.wgr[2] << 32) | ((long long)S.wgr[3] << 48);\n  }\n"),
    ]
    pairs += UNIT_STAMP_PAIRS if unitstamps else []
    pairs += SKEW_PAIRS if skew else []
    for a, b in pairs:
        assert src.count(a) == 1, a[:60]
        src = src.replace(a, b)
    return src


AB = r'''

# Generated by tools/moe_make_bench_ab.py: the same-process A/B against the public MoE file.
_AB_PUB_SRC = @PUB_SRC@
_AB_BASE = @BASE@
_AB_ARMS = @ARMS@
_AB_PAIRS = @PAIRS@
_AB_SEEDS = @SEEDS@
_AB_SIZES = @SIZES@
_AB_ROUTINGS = @ROUTINGS@
_AB_LIGHT = @LIGHT@
_AB_PA = @PA@
_AB_PB = @PB@
_AB_SMALL = @SMALL@
_AB_GATE = @GATE@
_AB_US = @US@
_AB_US_MAX = @USMAX@
_AB_US_PER = 8 + 4 * _AB_US_MAX
_AB_STAMP_SIZES = @STAMPSIZES@
_AB_SKEW = @SKEW@
_AB_PUBSKEW = @PUBSKEW@
_AB_SK_PER = @SKPER@
_AB_WORK = @WORK@


def _ab_main():
    import math
    import types
    t_start = time.time()
    dev = torch.device("cuda")
    # The judge rejects files that import reference.py. Timing does not depend on the values;
    # the scales stay positive normal E4M3, as scales_ok and the public PRMT decode require.
    g = torch.Generator(device=dev).manual_seed(5252)

    def sf(n, k):
        e = torch.randint(-3, 4, (32, n, k // 16), generator=g, device=dev).float()
        return torch.exp2(e).to(torch.float8_e4m3fn)

    weights = {
        "w13": torch.randint(0, 256, (32, 8192, 2048), generator=g, device=dev,
                             dtype=torch.uint8),
        "w13_sf": sf(8192, 4096),
        "w13_gs": torch.full((N_EXPERTS,), 1e-3, device=dev),
        "w2": torch.randint(0, 256, (32, 4096, 2048), generator=g, device=dev,
                            dtype=torch.uint8),
        "w2_sf": sf(4096, 4096),
        "w2_gs": torch.full((N_EXPERTS,), 1e-3, device=dev),
        "post_norm_w": (0.1 * torch.randn((HIDDEN,), generator=g, device=dev)).to(
            torch.bfloat16),
    }
    torch.cuda.empty_cache()
    state = {"w": _repack(weights, 0, MAX_LOCAL),
             "nw": (weights["post_norm_w"].float() + 1.0).contiguous(),
             "scratch": _scratch(dev), "main_ok": True, "policy": {}, "main_bad": set()}
    _ext()
    print(f"ABBUILD ours {time.time() - t_start:.0f} s", flush=True)
    pub = types.ModuleType("okc_ab_public")
    exec(compile(_AB_PUB_SRC, "public_moe.py", "exec"), pub.__dict__)
    pstate = pub.prepare(weights)
    dbg_defs = tuple(pub._main_defines(pstate["sfx"])) + pstate["sfx"] + ("-DOKC_DEBUG=1",)
    # The small sizes stamp their own builds (pdbgs); the MAIN debug build is not used there.
    pdbg = None if _AB_SMALL or _AB_GATE else pub._ext(dbg_defs)

    def pub_defs(m):
        # kernel_fn's build choice by token count, with the debug stamps.
        sfx = pstate["sfx"]
        if pub.TINY_DEFINES and m <= pub.TINY_MAX_M:
            d = tuple(pub.TINY_DEFINES)
        elif pub.TINY2_DEFINES and m <= pub.TINY2_MAX_M:
            d = tuple(pub.TINY2_DEFINES)
        elif pub.MID_DEFINES and m <= pub.MID_MAX_M:
            d = tuple(pub.MID_DEFINES)
        else:
            return dbg_defs
        return d + sfx + ("-DOKC_DEBUG=1",)

    pdbgs = {t: pub._ext(pub_defs(t)) for t in _AB_SIZES} if _AB_SMALL else {}
    pskew = pub._ext(dbg_defs + ("-DOKC_SKEW=1",)) if _AB_PUBSKEW else None
    print(f"ABBUILD pub {time.time() - t_start:.0f} s", flush=True)
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    def vmode(m):
        # An int is a fat mode, "vN" a variant index (the skinny variants).
        return int(m[1:]) if isinstance(m, str) else VARIANT_MAIN0 + MAIN_MODES.index(m)

    base = f"o{_AB_BASE}"

    # The layer's judge routing (muse_decode_layer/reference.py), its constants restated.
    H8 = 8192
    gb = torch.Generator().manual_seed(5354)
    mean = torch.randn((H8,), generator=gb).to(dev)
    shared = torch.randn((4, H8), generator=gb).to(dev)
    gw = torch.Generator(device=dev).manual_seed(4242)
    gate = 0.3 * torch.randn((H8,), generator=gw, device=dev)
    pre_ffn = 0.1 * torch.randn((H8,), generator=gw, device=dev)
    wg = gate / 0.3
    beta = torch.sigmoid(wg)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-wg) * (1.0 + beta), min=1e-3))

    def rms(x, eps=1e-5):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def layer_h(tokens, q_len, seed):
        gen = torch.Generator(device=dev).manual_seed(seed)
        coef = torch.randn((tokens, 4), generator=gen, device=dev)
        scale = torch.ones((tokens // q_len, q_len, 1), device=dev)
        if 1 < q_len <= 8:
            scale[:, 1:] = 0.6
        scale = scale.reshape(tokens, 1)
        stream = (scale * (0.7 * mean + math.sqrt(0.9 / 4) * coef @ shared)
                  + (1 - 0.9 * scale * scale).sqrt()
                  * torch.randn((tokens, H8), generator=gen, device=dev))
        attn = torch.randn((tokens, H8), generator=gen, device=dev)
        x = torch.randn((tokens, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
        return rms(alpha * stream + beta * rms(attn, 1e-8)) * (1.0 + pre_ffn), x

    def router(rseed):
        gen = torch.Generator(device=dev).manual_seed(rseed)
        rows = 0.045 * torch.randn((N_EXPERTS, H8), generator=gen, device=dev)
        align = 0.07 * torch.randn((N_EXPERTS, 1), generator=gen, device=dev)
        rows = rows + align * mean[None, :] / math.sqrt(H8)
        bias = 0.055 * torch.randn((N_EXPERTS,), generator=gen, device=dev)
        return rows, bias

    def route(h, rt):
        scores = torch.sigmoid((h.double() @ rt[0].double().T) * 0.1767766952966369).float()
        ids = (scores + rt[1]).topk(TOP_K, dim=-1).indices
        picked = scores.gather(1, ids)
        tw = picked / (picked.sum(-1, keepdim=True) + 1e-15)
        pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
        ids = (ids - 32 * int(pairs.argmax())) % N_EXPERTS
        return ids.to(torch.int32).contiguous(), tw.float().contiguous()

    def busiest(ids):
        loc = ids.long().flatten()
        return int(torch.bincount(loc[loc < 32], minlength=32).max())

    # The router seed is picked on all three sizes and seeds 1-6, as in the earlier checks, so
    # the layer routing stays the same whatever --sizes and --seeds say.
    sizes_all = ((32, 1), (128, 8), (256, 8))
    # The layer's loads: decode 1 and 16 (q_len 1), verify 1 (8 tokens, q_len 8).
    qlen = {**dict(sizes_all), 1: 1, 8: 8, 16: 1}
    target = {32: 21.0, 128: 53.0, 256: 107.0}
    pick_seeds = (1, 2, 3, 4, 5, 6)
    hs = {}

    def hget(t, s):
        if (t, s) not in hs:
            hs[t, s] = layer_h(t, qlen[t], 1000 * t + s)
        return hs[t, s]

    best = None
    for rs in range(1, 41):
        rt = router(rs)
        err = 0.0
        for t, _ in sizes_all:
            b = sum(busiest(route(hget(t, s)[0], rt)[0]) for s in pick_seeds) / len(pick_seeds)
            err += abs(b / target[t] - 1)
        if best is None or err < best[0]:
            best = (err, rs)
    rt = router(best[1])
    print(f"ABROUTE layer router seed {best[1]} (mean relative miss {best[0] / 3:.3f})")

    def inputs(t, routing, s):
        if routing == "std":
            return _make_inputs(t, 977 * s + t)
        if routing == "skewed":
            return _make_inputs(t, 977 * s + t, "skewed")
        h, x = hget(t, s)
        ids, tw = route(h, rt)
        return {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                "config": {"bs": t, "d_hidden": HIDDEN, "d_expert": HIDDEN,
                           "n_experts": N_EXPERTS, "local_start": 0, "local_count": 32,
                           "top_k": TOP_K, "sharding": "ep8"}}

    def pub_stamped(data):
        dbg = torch.zeros(G * 8 + (G * _AB_US_PER if _AB_US else 0), dtype=torch.int64,
                          device=dev)
        x = data["hidden_states"]
        out = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=dev)
        ext = pdbgs.get(x.shape[0], pdbg)
        for k in range(2):
            if k:
                flush.zero_()
                torch.cuda.synchronize()
            ext.okc_run(pstate["w"], pstate["scratch"] + [dbg], x, data["topk_ids"],
                         data["topk_weights"], out, 0, 32, G)
        torch.cuda.synchronize()
        raw = dbg[: G * 8].view(G, 8).cpu()
        us = dbg[G * 8:].view(G, _AB_US_PER).double().cpu() if _AB_US else None
        return raw[:, [0, 1, 6, 2, 3, 4, 5]].double(), raw[:, 7].clone(), us

    def skew_stats(sk, aend):
        # sk: [G, SK_PER] int64 clock64 stamps (0 = not written); aend: per CTA, the clock64 at
        # thread 0's phase A end. Returns, per phase, the columns of skew_cols (cycles unless
        # named us).
        G_ = sk.shape[0]
        hdr = sk[:, :8].double()
        ghz = float(((hdr[:, 3] - hdr[:, 1]) / (hdr[:, 4] - hdr[:, 2])).mean())
        nsw = int(hdr[0, 0])
        st = sk[:, 8:].view(G_, -1, 66)
        Q = st.shape[1]
        C = st[:, :, :64].reshape(G_, Q, 16, 4).double()
        P = st[:, :, 64].double()
        X = st[:, :, 65].double()
        valid = (C > 0).all(-1).all(-1)
        S0, Ew, Ex, R = C[..., 0], C[..., 1], C[..., 2], C[..., 3]
        Ww = Ew - S0
        Wx = Ex - Ew
        isA = S0.min(-1).values < aend[:, None]
        wgm = S0.view(G_, Q, 4, 4).mean(-1)
        s4 = S0.view(G_, Q, 4, 4)
        within = (s4.max(-1).values - s4.min(-1).values).mean(-1)
        sskew = S0.max(-1).values - S0.min(-1).values
        eskew = Ew.max(-1).values - Ew.min(-1).values
        rskew = R.max(-1).values - R.min(-1).values
        Rp = torch.zeros_like(R)
        Rp[:, nsw:] = R[:, :-nsw]
        vp = torch.zeros_like(valid)
        vp[:, nsw:] = valid[:, nsw:] & valid[:, :-nsw]
        rmaxp = Rp.max(-1).values
        skw = (rmaxp[..., None] - Rp).clamp(min=0)
        expl = torch.minimum(Ww.clamp(min=0), skw)
        react = P - rmaxp
        waiting = Ww > 64
        big = torch.where(waiting, Ew, torch.full_like(Ew, float("inf")))
        arr = big.min(-1).values
        haswait = waiting.any(-1)
        lat = arr - P
        lead = S0 - Rp
        wx = Wx > 64
        arrx = torch.where(wx, Ex, torch.full_like(Ex, float("inf"))).min(-1).values
        latx = arrx - X
        out = []
        for ph in (True, False):
            m = valid & (isA == ph)
            mp = vp & (isA == ph)
            nst = float(m.sum()) / G_
            mw = m[..., None].expand_as(Ww)
            mpw = mp[..., None].expand_as(Ww)
            ml = mp & haswait & (P > 0)
            mx = m & wx.any(-1) & (X > 0)
            mean = lambda t, k: float(t[k].mean()) if bool(k.any()) else 0.0
            out.append([nst, float(Ww[mw].sum()) / (16 * G_) / ghz / 1e3,
                        float(Wx[mw].sum()) / (16 * G_) / ghz / 1e3,
                        float((Ww + Wx)[mw].sum()) / max(2.0 * float(mw.sum()), 1.0),
                        mean(sskew, m), mean(wgm.max(-1).values - wgm.min(-1).values, m),
                        mean(within, m), mean(eskew, m), mean(rskew, m),
                        float(expl[mpw].sum()) / max(float(Ww[mpw].clamp(min=0).sum()), 1.0),
                        float(waiting[mw].double().mean()) if bool(mw.any()) else 0.0,
                        mean(react, mp & (P > 0)), mean(lat, ml), mean(lead, mpw),
                        mean(latx, mx), ghz])
        return out

    skew_cols = ("stages", "warp w-wait us", "x-wait us", "cycles per wait", "start skew",
               "wg-mean skew", "within-wg skew", "w-ready skew", "release skew",
               "w-wait share from release skew", "w-waits >64 cyc", "producer delay",
               "copy latency", "lead", "x copy latency", "GHz")

    def ours_skew(data, v):
        reg = state["scratch"]["inter"].reshape(-1).view(torch.int64)
        _run(data, state, v)
        flush.zero_()
        reg[1 << 20: (1 << 20) + G * _AB_SK_PER].zero_()
        torch.cuda.synchronize()
        _run(data, state, v)
        torch.cuda.synchronize()
        sk = reg[1 << 20: (1 << 20) + G * _AB_SK_PER].view(G, _AB_SK_PER).cpu()
        light = reg[: G * 16].view(G, 16).double().cpu()
        hdr = sk[:, :8].double()
        rate = (hdr[:, 3] - hdr[:, 1]) / (hdr[:, 4] - hdr[:, 2])
        aend = hdr[:, 1] + (light[:, 3] - hdr[:, 2]) * rate
        return skew_stats(sk, aend)

    def pub_skew(data):
        off = G * 8 + G * _AB_US_PER
        dbg = torch.zeros(off + G * _AB_SK_PER, dtype=torch.int64, device=dev)
        x = data["hidden_states"]
        out = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=dev)
        for k in range(2):
            if k:
                flush.zero_()
                dbg.zero_()
                torch.cuda.synchronize()
            pskew.okc_run(pstate["w"], pstate["scratch"] + [dbg], x, data["topk_ids"],
                          data["topk_weights"], out, 0, 32, G)
        torch.cuda.synchronize()
        raw = dbg[: G * 8].view(G, 8).double().cpu()
        sk = dbg[off:].view(G, _AB_SK_PER).cpu()
        hdr = sk[:, :8].double()
        rate = (hdr[:, 3] - hdr[:, 1]) / (hdr[:, 4] - hdr[:, 2])
        aend = hdr[:, 1] + (raw[:, 2] - hdr[:, 2]) * rate
        return skew_stats(sk, aend)

    def ours_stamped(data, v, width):
        _run(data, state, v)
        flush.zero_()
        torch.cuda.synchronize()
        _run(data, state, v)
        torch.cuda.synchronize()
        inter = state["scratch"]["inter"]
        d = inter.reshape(-1).view(torch.int64)[: G * width].view(G, width).double().cpu()
        if width == 16:
            light = d
            full = None
        else:
            full = d.view(G, 18, 8)
            light = full[:, 16:18, :].reshape(G, 16)
        return torch.cat([light[:, :7], light[:, 8:14]], 1), full

    def summarize(d):
        t0 = float(d[:, 0].min())
        r = (d - t0) / 1e3
        return [float((r[:, 1] - r[:, 0]).mean()), float(r[:, 2].mean()),
                float((r[:, 3] - r[:, 2]).mean()), float(r[:, 3].max()),
                float((r[:, 4] - r[:, 3]).mean()), float(r[:, 4].max() - r[:, 4].mean()),
                float((r[:, 6] - r[:, 4]).mean()), float(r[:, 6].max())]

    cols = ("route", "start", "phase A", "A done max", "phase B", "tail", "barrier+final",
            "end")

    def spearman(a, b):
        ra = a.argsort().argsort().double()
        rb = b.argsort().argsort().double()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        den = float((ra.pow(2).sum() * rb.pow(2).sum()).sqrt())
        return float((ra * rb).sum()) / den if den > 0 else 0.0

    def dist(x):
        q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], dtype=x.dtype))
        return [float(x.mean()), float(q[0]), float(q[1]), float(q[2]), float(x.max())]

    def pa_summary(light, full):
        pa = (light[:, 3] - light[:, 2]) / 1e3
        ghz = float((full[:, 17, 7] / (light[:, 3] - light[:, 2])).mean())
        us = lambda c: c / ghz / 1e3
        w = full[:, :16, :]
        units = w[:, 0, 5]
        mb = (36864 * w[:, 0, 4] + 65536 * w[:, 0, 6]) / 1e6
        ww = float(us(w[:, :, 0]).mean())
        xw = float(us(w[:, :, 1]).mean())
        seg = float(us(w[:, :, 2]).mean())
        epi = float(us(w[:, :, 3]).mean())
        pubc = float(us(w[:, :, 7]).mean())
        return (dist(pa) + [float(units.min()), float(units.mean()), float(units.max()),
                            float(mb.mean()), float(mb.max()), spearman(pa, mb),
                            ww, xw, seg - ww - xw, epi, epi / max(float(units.mean()), 1.0),
                            pubc, float(us(full[:, 16, 7]).mean()),
                            float(us(full[:, 17, 6]).mean()), ghz,
                            float((w[:, :, 0] + w[:, :, 1]).sum()
                                  / (2 * w[:, :, 4].sum()).clamp(min=1))])

    def pb_summary(light, full, pb):
        ghz = float((full[:, 17, 7] / (light[:, 3] - light[:, 2])).mean())
        us = lambda c: float((c / ghz / 1e3).mean())
        span = float(((light[:, 4] - light[:, 3]) / 1e3).mean())
        ww, xw, qw = us(pb[:, :, 0]), us(pb[:, :, 1]), us(pb[:, :, 7])
        units = float(pb[:, :, 5].mean())
        return [span, ww, xw, qw, us(pb[:, :, 2]) - ww - xw, us(pb[:, :, 3]), units,
                float(pb[:, :, 4].mean()),
                float(pb[:, :, 6].sum() / pb[:, :, 5].sum().clamp(min=1)),
                float((pb[:, :, 0] + pb[:, :, 1]).sum() / (2 * pb[:, :, 4].sum()).clamp(min=1))]

    pb_cols = ("phase B span", "warp: w-wait", "x-wait", "claim wait", "compute", "epilogue",
               "units", "stages", "n-tiles per unit", "cycles per wait")

    pa_cols = ("phase A mean", "p10", "p50", "p90", "max", "units min", "mean", "max",
               "MB mean", "max", "rho(time, MB)", "warp: w-wait", "x-wait", "compute",
               "epilogue", "per unit", "publish", "wproducer wait", "xproducer wait", "GHz",
               "cycles per wait")

    def pub_summary(d, rng):
        pa = (d[:, 3] - d[:, 2]) / 1e3
        a0 = (rng & 0xffff).double()
        a1 = ((rng >> 16) & 0xffff).double()
        st = a1 - a0
        cuts = ((a0.long() % 16) != 0).double() + ((a1.long() % 16) != 0).double()
        return dist(pa) + [float(st.min()), float(st.mean()), float(st.max()),
                           float(cuts.mean()), spearman(pa, st)]

    pub_cols = ("phase A mean", "p10", "p50", "p90", "max", "stages min", "mean", "max",
                "cut units per CTA", "rho(time, stages)")

    def sk_summary(d):
        # Our skinny stamps (kMode 2, 4, 8): see the kernel; cycles to us by the CTA's own clock.
        # Since 2026-10-29 (kMode 8) the weights waits include the first try_wait, which can
        # suspend the thread; 14 and 15 are thread 0's partial-buffer wait and unit barrier.
        r = (d[:, :8] - float(d[:, 0].min())) / 1e3
        ghz = (d[:, 13] - d[:, 12]) / (d[:, 7] - d[:, 0])
        us = lambda c: float((c / ghz / 1e3).mean())
        seg = [r[:, 1] - r[:, 0], r[:, 2] - r[:, 1], r[:, 3] - r[:, 2], r[:, 4] - r[:, 3],
               r[:, 5] - r[:, 4], r[:, 6] - r[:, 5], r[:, 7] - r[:, 6]]
        return ([float(x.mean()) for x in seg] + [float(r[:, 3].max()), float(r[:, 5].max()),
                float(r[:, 7].max())] + [us(d[:, 8]), us(d[:, 9]), us(d[:, 10]), us(d[:, 11]),
                                         us(d[:, 14]), us(d[:, 15])])

    sk_cols = ("route", "positions", "phase A", "A->B wait", "phase B", "barrier", "final",
               "A done max", "B done max", "end", "t0 weights wait A", "B", "helper idle",
               "credits", "t0 buffer wait", "t0 unit barrier")

    def pubsk_summary(d):
        # Public small builds: DBG 0 start, 1 routed, 2 phase A done, 3 before the barrier,
        # 4 after it, 5 end (columns 0, 1, 3, 4, 5, 6 of pub_stamped's order).
        r = (d - float(d[:, 0].min())) / 1e3
        seg = [r[:, 1] - r[:, 0], r[:, 3] - r[:, 1], r[:, 4] - r[:, 3], r[:, 5] - r[:, 4],
               r[:, 6] - r[:, 5]]
        return [float(x.mean()) for x in seg] + [float(r[:, 3].max()), float(r[:, 4].max()),
                                                 float(r[:, 6].max())]

    pubsk_cols = ("route", "phase A", "phase B", "barrier", "final", "A done max",
                  "B done max", "end")

    # --work=1: the work each design gives each CTA, from the routing alone (our fat jobs and owned
    # phase A deal, the public kernel's cost-weighted stream-K ranges ported from its wgs_prefix /
    # wgs_range with its own OKC_WGSA / OKC_WGSB tables, the skinny path's split jobs) plus our
    # claimed phase B units per CTA from the counter twin.
    def _pub_tab(name):
        import re
        m = re.search(r"#define %s ([0-9, ]+)" % name, _AB_PUB_SRC)
        return [int(x) for x in m.group(1).split(",")]

    pub_ca, pub_cb = (_pub_tab("OKC_WGSA"), _pub_tab("OKC_WGSB")) if _AB_WORK else ([], [])

    def work_of(ids, G_):
        loc = ids.long().flatten()
        cnt = [int(x) for x in torch.bincount(loc[loc < 32], minlength=32).cpu()]
        ours = [min(64, c - 64 * k) for c in cnt for k in range((c + 63) // 64)]
        J = len(ours)
        ua = 32 * J
        own = [sum(1 for r in range(ua // G_ + 2) if r * G_ + ((G_ - 1 - c) if r & 1 else c) < ua)
               for c in range(G_)]
        fat = [min(64, c - 64 * k) for c in cnt if c > 8 for k in range((c + 63) // 64)]
        pj = fat + [c for c in cnt if 0 < c <= 8]
        prow = []
        for tab, sh in ((pub_ca, 9), (pub_cb, 8)):
            pre = [0]
            for c in pj:
                pre.append(pre[-1] + (tab[(c + 7) // 8 - 1] << sh))
            tot = pre[-1]

            def at(x):
                lo = max(i for i in range(len(pj)) if pre[i] <= x) if pj else 0
                return (lo << sh) + (x - pre[lo]) // tab[(pj[lo] + 7) // 8 - 1]
            g = [at(tot * c // G_) for c in range(G_)] + [len(pj) << sh]
            st = [g[c + 1] - g[c] for c in range(G_)]
            cuts = [(g[c] % 16 != 0) + (g[c + 1] % 16 != 0) for c in range(G_)]
            prow += [min(st), sum(st) / G_, max(st), sum(x > 0 for x in st), sum(cuts) / G_]
        jsk = sum((c + 7) // 8 for c in cnt)
        mb = lambda units: units * 16 * 36864 / 1e6
        hist = [sum(1 for n in ours if (n + 7) // 8 == k) for k in range(1, 9)]
        return ([J, min(own), sum(own) / G_, max(own), sum(o > 0 for o in own),
                 mb(48 * J), len(pj)] + prow + [jsk, 512 * jsk / G_, 256 * jsk / G_,
                                                 512 * 3 * jsk * 36864 / 1e6 / 2,
                                                 sum(c > 64 for c in cnt)]
                + hist)

    work_cols = ("ours jobs", "A units/CTA min", "mean", "max", "CTAs with A",
                 "weights MB", "pub jobs", "pub A stages/CTA min", "mean", "max",
                 "CTAs with A", "cut units", "pub B stages/CTA min", "mean", "max",
                 "CTAs with B", "cut units", "skinny jobs", "A units/CTA", "B units/CTA",
                 "skinny weights MB", "experts over 64", "ntt1", "ntt2", "ntt3", "ntt4", "ntt5",
                 "ntt6", "ntt7", "ntt8")

    def unit_stats(u, ns):
        # u: [G, US_PER] (header, then per unit: copy issued I, thread 0 starts waiting S, the
        # wait ends E, slot free F), clock64 converted to us by each CTA's own clock. A unit whose
        # wait lasted over 0.05 us is "stalled": the wait's end is its arrival, within one poll.
        # Sums are per CTA; everything is averaged over the CTAs.
        ghz = (u[:, 4] - u[:, 2]) / (u[:, 5] - u[:, 3])
        rows = []
        buck = {b: [0.0, 0] for b in ("A1", "A", "Az", "B1", "B")}
        for c in range(u.shape[0]):
            na, n = int(u[c, 0]), min(int(u[c, 1]), _AB_US_MAX)
            if na <= ns or n <= na:
                continue
            t = u[c, 8: 8 + 4 * n].view(n, 4) / (1e3 * float(ghz[c]))
            I, S_, E, F = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            w = E - S_
            st = w > 0.05
            sa = st[:na]
            lat = float((E - I)[:na][sa].mean()) if bool(sa.any()) else 0.0
            rd = I[ns:n] - F[: n - ns]
            infl = [int(((I <= S_[q]) & (E > S_[q])).sum()) for q in range(na)]
            rows.append([float(w[:na].sum()), float((F - E)[:na].sum()),
                         float(F[na - 1] - S_[0]), float(sa.double().mean()), lat,
                         float((S_ - I)[:na].mean()), float(rd[:na].mean()), sum(infl) / na,
                         float(E[0] - I[0]), float(w[na:].sum()), float((F - E)[na:].sum()),
                         float(E[na] - F[na - 1])])
            for q in range(n):
                if q < na:
                    b = "A1" if q < ns else ("Az" if q >= na - ns else "A")
                else:
                    b = "B1" if q < na + ns else "B"
                buck[b][0] += float(w[q])
                buck[b][1] += 1
        m = max(len(rows), 1)
        out = [sum(r[i] for r in rows) / m for i in range(12)]
        return out + [buck[b][0] / max(buck[b][1], 1) for b in buck] + [float(ghz.mean())]

    us_cols = ("A wait", "A hold", "A span", "A stalled", "lat", "lead", "refill delay",
               "in flight", "first copy", "B wait", "B hold", "A->B gap", "wait/unit A1",
               "A", "Az", "B1", "B", "GHz")

    timed = {base: vmode(_AB_BASE)}
    for name, m in _AB_ARMS.items():
        timed[name] = vmode(m)
    light = {n: vmode(m) for n, m in _AB_LIGHT.items()}
    skewv = {n: vmode(m) for n, m in _AB_SKEW.items()}
    padiag = {n: vmode(m) for n, m in _AB_PA.items()}
    for t in _AB_SIZES:
        for routing in _AB_ROUTINGS:
            tag = f"T={t} {routing}"
            b = [busiest(inputs(t, routing, s)["topk_ids"]) for s in _AB_SEEDS]
            print(f"ABROUTE {tag} busiest local expert routes {b}", flush=True)
            data = inputs(t, routing, _AB_SEEDS[0])
            ref = _ref_forward(data, weights)
            ob = _run(data, state, timed[base]).float()
            op = pub.kernel_fn(data, pstate).float()
            line = (f"ABCHECK {tag} vs fp32 {base} {_rel(ob, ref):.2e} pub {_rel(op, ref):.2e};"
                    f" vs {base}:")
            for name, v in {**timed, **padiag}.items():
                if name != base:
                    line += f" {name} {_rel(_run(data, state, v).float(), ob):.1e}"
            print(line, flush=True)
            arms = {n: (lambda vv: lambda d: _run(d, state, vv))(v) for n, v in timed.items()}
            arms["pub"] = lambda d: pub.kernel_fn(d, pstate)
            if t <= 32 and not _AB_SMALL:
                arms["sk8w4s"] = lambda d: _run(d, state, 0)
            tim = {a: [] for a in arms}
            stamps = {a: [] for a in [*light, *padiag, "pub"]}
            diags = {a: [] for a in padiag}
            pbdiags = {a: [] for a in padiag if a in _AB_PB}
            skd_ = {a: [] for a in [*skewv, *(["pub"] if pskew is not None else [])]}
            pubd = []
            chk = {a: [] for a in timed if a != base}
            stamp_here = not _AB_STAMP_SIZES or t in _AB_STAMP_SIZES
            skst = [i for i, v in enumerate(SKINNY_VARIANTS) if v[2] in (2, 4, 8)]
            usst = [i for i, v in enumerate(SKINNY_VARIANTS) if v[2] & 16]
            if not stamp_here:
                skst, usst = [], []
            skd = {**{i: [] for i in skst}, "pub": []}
            usd = {**{i: [] for i in usst}, "pub": []}
            works = []
            pbu = {a: [] for a in padiag if a in _AB_PB}
            for si, s in enumerate(_AB_SEEDS):
                d = inputs(t, routing, s)
                order = list(arms.items())
                if si % 2:
                    order.reverse()
                for a, fn in order:
                    tim[a].append(_graph_time(lambda: fn(d), flush))
                if _AB_WORK:
                    works.append(work_of(d["topk_ids"], G))
                if _AB_GATE:
                    ob = _run(d, state, timed[base]).float()
                    for a in chk:
                        chk[a].append(_rel(_run(d, state, timed[a]).float(), ob))
                    continue
                for a, v in light.items():
                    stamps[a].append(summarize(ours_stamped(d, v, 16)[0]))
                for a, v in skewv.items():
                    skd_[a].append(ours_skew(d, v))
                if pskew is not None:
                    skd_["pub"].append(pub_skew(d))
                for a, v in padiag.items():
                    lt, full = ours_stamped(d, v, 144)
                    stamps[a].append(summarize(lt))
                    diags[a].append(pa_summary(lt, full))
                    if a in pbdiags:
                        pb = state["scratch"]["inter"].reshape(-1).view(torch.int64)[
                            G * 144: G * 144 + G * 128].view(G, 16, 8).double().cpu()
                        pbdiags[a].append(pb_summary(lt, full, pb))
                        u = pb[:, 0, 5]
                        pbu[a].append([float(u.min()), float(u.mean()), float(u.max()),
                                       float((u > 0).sum())])
                if _AB_SMALL and not stamp_here:
                    continue
                if not _AB_SMALL and t <= 32:
                    for i in skst:
                        _run(d, state, i)
                        flush.zero_()
                        torch.cuda.synchronize()
                        _run(d, state, i)
                        torch.cuda.synchronize()
                        sd = state["scratch"]["inter"].reshape(-1).view(torch.int64)[: G * 16]
                        skd[i].append(sk_summary(sd.view(G, 16).double().cpu()))
                pd, prng, pus = pub_stamped(d)
                if _AB_SMALL:
                    for i in usst:
                        _run(d, state, i)
                        flush.zero_()
                        torch.cuda.synchronize()
                        _run(d, state, i)
                        torch.cuda.synchronize()
                        ud = state["scratch"]["inter"].reshape(-1).view(torch.int64)[
                            1 << 21: (1 << 21) + G * _AB_US_PER]
                        usd[i].append(unit_stats(ud.view(G, _AB_US_PER).double().cpu(),
                                                 SKINNY_VARIANTS[i][1]))
                    if _AB_US:
                        usd["pub"].append(unit_stats(pus, 3))
                    for i in skst:
                        _run(d, state, i)
                        flush.zero_()
                        torch.cuda.synchronize()
                        _run(d, state, i)
                        torch.cuda.synchronize()
                        sd = state["scratch"]["inter"].reshape(-1).view(torch.int64)[: G * 16]
                        skd[i].append(sk_summary(sd.view(G, 16).double().cpu()))
                    skd["pub"].append(pubsk_summary(pd))
                    continue
                stamps["pub"].append(summarize(pd))
                pubd.append(pub_summary(pd, prng))
            n = len(_AB_SEEDS)
            print(f"ABTIME {tag} mean of {n} " + " ".join(f"{a} {sum(v) / n:.1f}"
                                                         for a, v in tim.items()))
            pairs = [(a, base) for a in arms if a != base] + list(_AB_PAIRS)
            for a, bb in pairs:
                if a in tim and bb in tim:
                    dd = [x - y for x, y in zip(tim[a], tim[bb])]
                    print(f"ABPAIR {tag} {a} minus {bb} " + " ".join(f"{x:+.1f}" for x in dd)
                          + f" (mean {sum(dd) / n:+.1f}, {sum(x < 0 for x in dd)} of {n} lower)")
            # chk is filled only by --gate=1 runs; an empty list here crashed ed3b4fdc76b54263.
            if works:
                m = [sum(r[i] for r in works) / n for i in range(len(works[0]))]
                print(f"ABWORK {tag} " + ", ".join(f"{c} {v:.4g}" for c, v in zip(work_cols, m)))
            for a, rows in pbu.items():
                m = [sum(r[i] for r in rows) / n for i in range(4)]
                print(f"ABWORK {tag} {a} claimed B units/CTA min {m[0]:.2f}, mean {m[1]:.2f}, "
                      f"max {m[2]:.2f}, CTAs with B {m[3]:.1f}")
            for a, v in ((a, v) for a, v in chk.items() if v):
                print(f"ABCHECKALL {tag} {a} vs {base} per seed " + " ".join(f"{x:.1e}" for x in v)
                      + f" (max {max(v):.1e})")
            if (_AB_SMALL and not stamp_here) or _AB_GATE:
                continue
            if _AB_SMALL:
                for a, rows, cc in ([(_vname(i), skd[i], sk_cols) for i in skst]
                                    + [("pub", skd["pub"], pubsk_cols)]):
                    m = [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]
                    print(f"ABSK {tag} {a} " + ", ".join(f"{c} {v:.1f}" for c, v in zip(cc, m)))
                for a, rows in ([(_vname(i), usd[i]) for i in usst]
                                + ([("pub", usd["pub"])] if _AB_US and stamp_here else [])):
                    m = [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]
                    print(f"ABUNIT {tag} {a} " + ", ".join(f"{c} {v:.2f}"
                                                           for c, v in zip(us_cols, m)))
                continue
            for a, rows in stamps.items():
                m = [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]
                print(f"ABSTAMP {tag} {a} " + ", ".join(f"{c} {v:.1f}" for c, v in zip(cols, m)))
            for i in (skst if t <= 32 else []):
                m = [sum(r[k] for r in skd[i]) / n for k in range(len(skd[i][0]))]
                print(f"ABSK {tag} {_vname(i)} " + ", ".join(f"{c} {v:.1f}"
                                                            for c, v in zip(sk_cols, m)))
            for a, rows in diags.items():
                m = [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]
                print(f"ABDIAG {tag} {a} " + ", ".join(f"{c} {v:.2f}" for c, v in zip(pa_cols, m)))
            for a, rows in pbdiags.items():
                m = [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]
                print(f"ABDIAGB {tag} {a} " + ", ".join(f"{c} {v:.2f}" for c, v in zip(pb_cols, m)))
            m = [sum(r[i] for r in pubd) / n for i in range(len(pubd[0]))]
            print(f"ABPUB {tag} " + ", ".join(f"{c} {v:.2f}" for c, v in zip(pub_cols, m)))
            for a, rows in skd_.items():
                for k, ph in enumerate("AB"):
                    m = [sum(r[k][i] for r in rows) / n for i in range(len(skew_cols))]
                    print(f"ABSKEW {tag} {a} phase {ph} "
                          + ", ".join(f"{c} {v:.3g}" for c, v in zip(skew_cols, m)))
    flags = int(state["scratch"]["ctr"][8].item())
    state["scratch"]["ctr"][8] = 0
    print(f"ABMOE spin flags {flags}; total {time.time() - t_start:.0f} s", flush=True)
    del state, pstate, weights
    torch.cuda.empty_cache()


try:
    _ab_main()
except Exception as _exc:
    import traceback
    print(f"ABMOE FAIL {type(_exc).__name__}: {_exc}", flush=True)
    traceback.print_exc()

_ab_prepare = prepare


def prepare(weights):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st = _ab_prepare(weights)
    keep = ("POLICY", "PREPARE", "BENCH bs=128", "BENCH bs=256", "BENCH spin")
    for line in buf.getvalue().splitlines():
        if line.startswith(keep) or "worst rel err" in line or "FAIL" in line:
            print(line)
    return st
'''


def main():
    out, pub = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    opts = {"base": "28", "seeds": "1,2,3,4,5,6", "sizes": "32,128,256", "routings": "layer,std",
            "pairs": "", "small": "", "pubranges": "", "skextra": "", "unitstamps": "",
            "stampsizes": "", "gate": "", "skew": "", "pubskew": "", "skst": "",
            "work": ""}
    arms = {}
    for a in rest:
        if a.startswith("--"):
            k, v = a[2:].split("=", 1)
            assert k in opts, k
            opts[k] = v
        else:
            k, v = a.split("=")
            arms[k] = v if v.startswith("v") else int(v)
    small = bool(opts["small"])
    base = opts["base"] if opts["base"].startswith("v") else int(opts["base"])
    ints = lambda s: tuple(int(x) for x in s.split(","))
    pairs = tuple(tuple(x.split("-")) for x in opts["pairs"].split(",") if x)
    gate = bool(opts["gate"])
    named = {m for m in (base, *arms.values()) if isinstance(m, int)}
    twinned = () if gate else named
    light = {f"o{LIGHT_TWINS[m]}": LIGHT_TWINS[m] for m in twinned if m in LIGHT_TWINS}
    pa = {f"o{PA_TWINS[m]}": PA_TWINS[m] for m in twinned if m in PA_TWINS}
    skewt = ({f"o{SKEW_TWINS[m]}": SKEW_TWINS[m] for m in twinned if m in SKEW_TWINS}
             if opts["skew"] else {})
    # A counter twin that is also an arm is timed and stamped; it is not built twice.
    staged = {m for a in arms.values() for m in VARIANT_MODES.get(a, ())}
    modes = sorted({0, *named, *light.values(), *pa.values(), *skewt.values(), *staged})
    src = (ROOT / "muse_moe_w4a16" / "submission.py").read_text()
    extra = "".join(f"({x.replace(':', ', ')}), " for x in opts["skextra"].split(",") if x)
    if opts["skst"]:
        # Stamped skinny variants outside --small (ABSK at 32 tokens and below, next to the fat
        # stamps), e.g. --skst=8:3:2 for sk8w3s with mode 0's stamps.
        assert not small
        add = "".join(f", ({x.replace(':', ', ')})" for x in opts["skst"].split(",") if x)
        src, n = re.subn(r"^SKINNY_VARIANTS = \(\((.*)\)\)$",
                         lambda mm: f"SKINNY_VARIANTS = (({mm.group(1)}){add})", src, flags=re.M)
        assert n == 1
    if small:
        # The stamped skinny variant (kMode 2) goes last; ABSK runs it.
        src, n = re.subn(r"^SKINNY_VARIANTS = \(\((.*)\)\)$",
                         lambda mm: f"SKINNY_VARIANTS = (({mm.group(1)}), {extra}(8, 4, 2))", src,
                         flags=re.M)
        assert n == 1
    src, n = re.subn(r"^MAIN_MODES = \(.*?\)$", f"MAIN_MODES = {tuple(modes)}", src, flags=re.M)
    assert n == 1
    src, n = re.subn(r"^GATE_RUNS = \(.*?\)$", "GATE_RUNS = ()", src, flags=re.M)
    assert n == 1
    if gate:
        add = tuple(m for m in arms.values() if isinstance(m, int))
        src, n = re.subn(r"^MAIN_CANDIDATES = \((.*?)\)$",
                         lambda mm: f"MAIN_CANDIDATES = ({mm.group(1)}" + "".join(
                             f", {m}" for m in add) + ")", src, flags=re.M)
        assert n == 1
    pub_src = patch_public(Path(pub).read_text(), small, bool(opts["pubranges"]),
                           bool(opts["unitstamps"]), bool(opts["pubskew"]))
    block = AB
    for key, val in (("@PUB_SRC@", repr(pub_src)), ("@BASE@", repr(base)),
                     ("@ARMS@", repr(arms)), ("@PAIRS@", repr(pairs)),
                     ("@SEEDS@", repr(ints(opts["seeds"]))), ("@SIZES@", repr(ints(opts["sizes"]))),
                     ("@ROUTINGS@", repr(tuple(opts["routings"].split(",")))),
                     ("@LIGHT@", repr(light)), ("@PA@", repr(pa)),
                     ("@PB@", repr(tuple(a for a, m in pa.items() if m in PB_TWINS))),
                     ("@SMALL@", repr(small)), ("@GATE@", repr(gate)),
                     ("@US@", repr(bool(opts["unitstamps"]))), ("@USMAX@", repr(US_MAX)),
                     ("@STAMPSIZES@", repr(ints(opts["stampsizes"]) if opts["stampsizes"] else ())),
                     ("@SKEW@", repr(skewt)), ("@SKPER@", repr(SK_PER)),
                     ("@PUBSKEW@", repr(bool(opts["pubskew"]))),
                     ("@WORK@", repr(bool(opts["work"])))):
        assert block.count(key) == 1, key
        block = block.replace(key, val)
    Path(out).write_text(src + block)


if __name__ == "__main__":
    main()
