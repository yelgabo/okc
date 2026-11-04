#!/usr/bin/env python3
"""Routing slack at T = 16 / 32: writes muse_decode_layer/cand/g1b4_rs2.py from g1b4_rs.py (01426b89).

g1b4_rs applies TASK.md's near-tie routing slack at T = 1 only (tools/dl_rs_patch.py). This adds it at decode 16
and decode 32. Which after-attention path runs there is chosen per process in prepare by timing (the
compose_graft_enabled policy): the graft path's route launches in _pdl8_graft_after_attention (pooled / logits
split, and the fused 4608-column form) and the one-program-per-token path's _route_split both become
_rs_tile_routeN, one program that reads every token's FP32 router logits (.cg,
after griddepcontrol.wait), forms the same biased scores and sort key, keeps a top 16 per token and drops whole
local experts: an expert is droppable when every token routed to it beats its 9th best by at most
RS2_KEEP_TOL - RS2_KEEP_MARGIN and still has an unused non-local candidate ranked 9th or lower within
RS2_ROUTE_TOL - RS2_WIN_MARGIN of its 8th best. Up to RS2_KMAX experts go, fewest routes first (then the lower
id), each only while distinct experts and routes stay RS2_COUNT_MARGIN above TASK.md's bounds, with the
reference counts raised by local experts within RS2_RANK_EPS of a token's 8th best. Each token takes its best
non-local candidates for its dropped slots. Weights are the picked unbiased scores over their sum, ids in score
order as before. The normalize programs are the base kernel's. T = 1 and T = 8 run g1b4_rs's code, every PDL
edge and gate and every CUDA source are unchanged.

Box result (docs/runs-decode-layer.md "Routing slack at T = 16 / 32"): a measurement prototype, not for dispatch.
The safe margins leave 0.1-0.6 experts per input to remove, and this single-program route costs 42 / 104 us.

    python3 tools/dl_rs2_patch.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/g1b4_rs.py"
DST = ROOT / "muse_decode_layer/cand/g1b4_rs2.py"
RS_PIN = "01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07"

KERNEL = r'''
RS2_T = (16, 32)
RS2_KEEP_TOL = 0.03   # reference.KEEP_TOL up to 256 tokens
RS2_ROUTE_TOL = 0.05  # reference.ROUTE_TOL
# T = 16 / 32 route from the bfp router projection: over 800 box inputs (tools/box_rs_check.py) the 9th-best lead
# of these scores is off the judge's FP64 one by at most 9.97e-3 and the distance below the 8th best by at most
# 1.43e-2, against 5.4e-3 / 1.16e-2 at T = 1, so the margins are wider than RS_KEEP_MARGIN / RS_WIN_MARGIN
RS2_KEEP_MARGIN = 0.016
RS2_WIN_MARGIN = 0.021
RS2_RANK_EPS = 0.016
RS2_COUNT_MARGIN = 0.25
RS2_KMAX = 4


@triton.jit
def _rs_tile_routeN(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr,
                    TB: tl.constexpr, PARTS: tl.constexpr, KT: tl.constexpr, WIN: tl.constexpr,
                    RE: tl.constexpr, CM: tl.constexpr, KMAX: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        _gdc_launch()
        _gdc_wait()
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c, cache_modifier='.cg').to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c, cache_modifier='.cg').to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        elif SPLIT:
            vals = tl.load(POOLED + t * 4096 + oc, cache_modifier='.cg').to(tl.float32)
        else:
            vals = tl.load(ALL + t * 4608 + oc, cache_modifier='.cg').to(tl.bfloat16).to(tl.float32)
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    elif t == 0:
        e = tl.arange(0, 256)
        tt = tl.arange(0, TB)
        j = tl.arange(0, 32)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        _gdc_launch()
        _gdc_wait()
        if SPLIT:
            logit = tl.load(LOGITS + tt[:, None] * 256 + e[None, :], cache_modifier='.cg')
        else:
            logit = (tl.load(ALL + tt[:, None] * 4608 + 4096 + e[None, :], cache_modifier='.cg')
                     + tl.load(ALL + tt[:, None] * 4608 + 4352 + e[None, :], cache_modifier='.cg'))
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias[None, :]
        # the base kernel's key: biased score (order-preserving bits), then the lower expert id
        kbits = sel.to(tl.uint32, bitcast=True)
        ordered = tl.where((kbits & 0x80000000) != 0, ~kbits, kbits ^ 0x80000000).to(tl.uint32)
        packed = (ordered.to(tl.uint64) << 32) | (255 - e)[None, :].to(tl.uint64)
        top = tl.sort(tl.topk(packed, 16, dim=1), dim=1, descending=True)
        es = (255 - (top & 255)).to(tl.int32)
        ss = tl.gather(sel, es, 1)
        scs = tl.gather(sc, es, 1)
        rk = tl.arange(0, 16)[None, :]
        v8 = tl.sum(tl.where(rk == 7, ss, 0.0), 1)
        v9 = tl.sum(tl.where(rk == 8, ss, 0.0), 1)
        le = (es - ls + 256) % 256
        isl = le < 32
        in8 = rk < 8
        refl = in8 & isl
        opt = refl & (ss - v9[:, None] <= KT)
        nlc = (le >= 32) & (rk >= 8) & (ss >= v8[:, None] - WIN)
        navail = tl.sum(nlc.to(tl.int32), 1)
        hit = (le[:, :, None] == j[None, None, :]) & refl[:, :, None]
        uses = tl.sum(tl.reshape(hit.to(tl.int32), (TB * 16, 32)), 0)
        routes = tl.sum(uses, 0)
        distinct = tl.sum((uses > 0).to(tl.int32), 0)
        # the judge counts its own top 8: a local expert ranked 9th or lower here but within RE of the 8th best
        # may be in it, so the bounds use those larger counts
        near = isl & (rk >= 8) & (ss >= v8[:, None] - RE)
        nearj = tl.max(tl.reshape(((le[:, :, None] == j[None, None, :]) & near[:, :, None]).to(tl.int32),
                                  (TB * 16, 32)), 0)
        rhi = (routes + tl.sum(tl.sum(near.to(tl.int32), 1), 0)).to(tl.float32)
        dhi = (distinct + tl.sum(((nearj > 0) & (uses == 0)).to(tl.int32), 0)).to(tl.float32)
        rfloor = 0.92 * rhi - 4.0 + CM
        dfloor = 0.85 * dhi - 2.0 + CM
        nlused = tl.zeros([TB], dtype=tl.int32)
        drop = tl.zeros([TB, 16], dtype=tl.int32)
        for it in tl.static_range(KMAX):
            okts = opt & (navail > nlused)[:, None]
            bad = tl.sum(tl.reshape((hit & (okts == 0)[:, :, None]).to(tl.int32), (TB * 16, 32)), 0)
            feas = ((uses > 0) & (bad == 0) & ((routes - uses).to(tl.float32) >= rfloor)
                    & ((distinct - 1).to(tl.float32) >= dfloor))
            go = tl.max(feas.to(tl.int32), 0)
            best = tl.argmin(tl.where(feas, uses * 64 + j, 1 << 20), 0)
            dj = ((le == best) & refl & (go > 0)).to(tl.int32)
            drop = drop | dj
            nlused = nlused + tl.sum(dj, 1)
            routes = routes - tl.sum(tl.where(j == best, uses, 0), 0) * go
            distinct = distinct - go
            uses = tl.where((j == best) & (go > 0), 0, uses)
        chosen = (in8 & (drop == 0)) | (nlc & (tl.cumsum(nlc.to(tl.int32), 1) <= nlused[:, None]))
        slot = tl.cumsum(chosen.to(tl.int32), 1) - 1
        wsum = tl.sum(tl.where(chosen, scs, 0.0), 1)
        off = tt[:, None] * 8 + slot
        tl.store(IDS + off, es, mask=chosen)
        tl.store(WTS + off, scs / (wsum[:, None] + 1e-15), mask=chosen)
        tl.store(LIDS + off, tl.where(isl, le, -1), mask=chosen)


def _rs2_route_launch(pooled, logits, allp, split, state, ids, wts, xm, ls, lids, T, nw, parts):
    if _RS_DEBUG is not None:
        _RS_DEBUG.append(logits.clone() if split else allp[:, 4096:4352] + allp[:, 4352:4608])
    if RS_ON and T in RS2_T:
        _rs_tile_routeN[(T, parts + 1)](
            pooled, logits, allp, state["bias"], state["g_preexp"], ids, wts, xm, ls, lids, SPLIT=split, TB=T,
            PARTS=parts, KT=RS2_KEEP_TOL - RS2_KEEP_MARGIN, WIN=RS2_ROUTE_TOL - RS2_WIN_MARGIN, RE=RS2_RANK_EPS,
            CM=RS2_COUNT_MARGIN, KMAX=RS2_KMAX, num_warps=nw, launch_pdl=_PDL8)
        return
    (_sl_tile_route_cg if _PDL8 else _sl_tile_route)[(T, parts + 1)](pooled, logits, allp, state["bias"], state["g_preexp"], ids, wts, xm,
        ls, lids, SPLIT=split, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
        num_warps=nw, launch_pdl=_PDL8)


def _rs2_vj_route(pooled, logits, state, ids, wts, xm, ls, lids, T):
    """The one-program-per-token path's _route_split at T in RS2_T (the expert backend reads xm unpermuted):
    the same normalize, then _rs_tile_routeN's picks."""
    if _RS_DEBUG is not None:
        _RS_DEBUG.append(logits.clone())
    _rs_tile_routeN[(T, 2)](
        pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm, ls, lids, SPLIT=True, TB=T,
        PARTS=1, KT=RS2_KEEP_TOL - RS2_KEEP_MARGIN, WIN=RS2_ROUTE_TOL - RS2_WIN_MARGIN, RE=RS2_RANK_EPS,
        CM=RS2_COUNT_MARGIN, KMAX=RS2_KMAX, num_warps=8, launch_pdl=_pdl(T))

'''

CALL_VJ_OLD = '''        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
'''
CALL_VJ_NEW = '''        if RS_ON and T in RS2_T and (use_marlin or use_ma):
            _rs2_vj_route(pooled, logits, state, ids, wts, xm, data["local_start"], lids, T)
        else:
            _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                               data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
'''

CALL_SPLIT_OLD = '''            (_sl_tile_route_cg if _PDL8 else _sl_tile_route)[(T, parts + 1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=True, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=_PDL8)
'''
CALL_SPLIT_NEW = '''            _rs2_route_launch(pooled, logits, logits, True, state, ids, wts, xm, data["local_start"], lids, T, nw, parts)
'''
CALL_FUSED_OLD = '''            (_sl_tile_route_cg if _PDL8 else _sl_tile_route)[(T, parts + 1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=False, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=_PDL8)
'''
CALL_FUSED_NEW = '''            _rs2_route_launch(allp, allp, allp, False, state, ids, wts, xm, data["local_start"], lids, T, nw, parts)
'''
ANCHOR = "\n\ndef _pdl8_graft_after_attention("


def sub(s, old, new, n=1):
    if s.count(old) != n:
        sys.exit(f"anchor {old[:70]!r}: {s.count(old)} found, {n} expected")
    return s.replace(old, new)


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RS_PIN:
        sys.exit("g1b4_rs.py is not 01426b89")
    s = raw.decode()
    s = sub(s, CALL_VJ_OLD, CALL_VJ_NEW)
    s = sub(s, CALL_SPLIT_OLD, CALL_SPLIT_NEW)
    s = sub(s, CALL_FUSED_OLD, CALL_FUSED_NEW)
    s = sub(s, ANCHOR, "\n" + KERNEL + ANCHOR)
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
