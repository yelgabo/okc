#!/usr/bin/env python3
"""ly_rpf: at decode 1 the route kernel L2-prefetches the first weight unit of every MoE CTA (the MoE's ramp).

Base: muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d..., checked). Target: the decode 1 MoE ramp (TINY build,
docs/runs-decode-layer.md "Decode 1 MoE ramp, handoff, tail"). The MoE is a cooperative launch with PDL off, so it
starts only after the route kernel completes; each CTA then loads the routed ids (an L2 round trip), and its FAST1
warp issues the CTA's first two 36,864-byte units at about entry + 0.8 us. All 132 CTAs issue at once, so the 9.7 MB
first wave streams from an idle DRAM and the first ring slot fills about 3 us after entry (MOE-TL-01: 3.1).

Change: the T = 1 route launch gets NSM (the MoE grid) extra programs. Each runs the same sigmoid / bias / top-8 as
the storing program on the same logits (bitwise the same selection), stores nothing, and issues one
cp.async.bulk.prefetch.L2.global of the unit the MoE's CTA c = program - PARTS - 1 requests first: the TINY build's
FAST1 split (no dynamic pool in TINY): local expert mask m from the local ids, v0 = floor(popc(m) * 512 * c / G),
expert = the (v0 >> 9)-th set bit of m, address wAu + (expert * 512 + (v0 & 511)) * 36864. The prefetch is a hint
with no effect on any value; the MoE's own copies of the same lines then hit L2 or merge with the pending fill.
The ids / lids / wts stores, the xm program, every wait and trigger, the MoE launch and the MoE source are
unchanged; T = 8 passes NPF = 0 and launches the base grid. The non-prq route kernel gets the same parameters and
runs its top-8 branch only in program PARTS, so extra programs there do nothing.

    python3 tools/dl_ly_rpf_patch.py [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfx.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_rpf.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

SIG = ("SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr):")
SIG_NEW = ("SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr, "
           "NPF: tl.constexpr, WAUA, NSM: tl.constexpr):")
BASE_FN = "def _sl_tile_route_cg(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, "
PRQ_FN = "def _prq_sl_tile_route_cg(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, "
TOP_ELSE = "    else:\n        e = tl.arange(0, 256)\n"
STORES = """        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)
"""
STORES_NEW = """        lid = (ids - ls + 256) % 256
        if kind == PARTS:
            tl.store(IDS + t * 8 + k8, ids)
            tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
            tl.store(WTS + t * 8 + k8, wv)
        elif NPF > 0:
            # rpf: the TINY MoE's FAST1 split (OKC_FAST1, no dynamic pool) for CTA kind - PARTS - 1
            k32 = tl.arange(0, 32)
            pres = tl.max((lid[:, None] == k32[None, :]).to(tl.int32), 0)
            ne = tl.sum(pres, 0)
            rank = tl.cumsum(pres, 0) - pres
            v = (ne * 512 * (kind - PARTS - 1)) // NSM
            ex = tl.sum(tl.where((pres > 0) & (rank == (v >> 9)), k32, 0), 0)
            addr = tl.zeros([1], dtype=tl.int64) + (WAUA + (ex * 512 + (v & 511)).to(tl.int64) * 36864)
            ok = tl.zeros([1], dtype=tl.int32) + (ne > 0).to(tl.int32)
            tl.inline_asm_elementwise(
                "{ .reg .pred p, q; .reg .u32 r; mov.u32 r, %tid.x; setp.eq.u32 p, r, 0; setp.ne.s32 q, $2, 0; "
                "and.pred p, p, q; @p cp.async.bulk.prefetch.L2.global [$1], 36864; mov.u32 $0, 0; }",
                "=r,l,r", [addr, ok], dtype=tl.int32, is_pure=False, pack=1)
"""
CALL = "_sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)]("
CALL_NEW = "_sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1+_rpf_n)]("
KW = 'PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))'
KW_NEW = ('PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"), '
          'NPF=_rpf_n, WAUA=_rpf_a, NSM=state["nsm"])')
XM = "    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)\n    if S == -1:\n"
XM_NEW = ("    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)\n"
          "    _rpf_n = state[\"nsm\"] if (T == 1 and use_ma and T <= MOE_TINY_MAX_M) else 0  # rpf: TINY's FAST1 split\n"
          "    _rpf_a = state[\"ma_w\"][10].data_ptr() if _rpf_n else 0\n"
          "    if S == -1:\n")
HEADER = ("# ly_rpf: ly_rfx with the decode 1 route launch L2-prefetching each MoE CTA's first weight unit from NSM extra "
          "top-8 programs (tools/dl_ly_rpf_patch.py).\n")


def sub(text, old, new, n):
    if text.count(old) != n:
        sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old[:120]!r}")
    return text.replace(old, new)


def fn_region(text, head):
    a = text.index(head)
    return a, text.index("\n\n@triton.jit", a)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a, b = fn_region(text, BASE_FN)
    seg = sub(text[a:b], SIG, SIG_NEW, 1)
    seg = sub(seg, TOP_ELSE, "    elif kind == PARTS:\n        e = tl.arange(0, 256)\n", 1)
    text = text[:a] + seg + text[b:]
    a, b = fn_region(text, PRQ_FN)
    seg = sub(text[a:b], SIG, SIG_NEW, 1)
    seg = sub(seg, STORES, STORES_NEW, 1)
    text = text[:a] + seg + text[b:]
    i = text.index("def _pdl8_sl_tile_after_attention(")
    j = text.index("\ndef ", i + 10)
    seg = sub(text[i:j], XM, XM_NEW, 1)
    seg = sub(seg, CALL, CALL_NEW, 2)
    seg = sub(seg, KW, KW_NEW, 2)
    text = text[:i] + seg + text[j:]
    if not text.startswith("# ly_rfx:"):
        sys.exit("base header line is not ly_rfx's")
    text = HEADER + text[text.index("\n") + 1:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
