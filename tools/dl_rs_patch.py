#!/usr/bin/env python3
"""Routing slack at T = 1: writes muse_decode_layer/cand/g1b4_rs.py from g1b4_pdl8d.py (7cb74240).

TASK.md lets routing differ from the reference where scores nearly tie (reference.check_routing): a local top-8
pick that beats the 9th best by at most 0.03 may be dropped, every pick must be within 0.05 of the 8th best,
and this GPU keeps at least the reference's routes minus 8% and 4 and its distinct experts minus 15% and 2.
moe_fused costs about 6.3 us per distinct local expert (docs/box.md "Routing"), so at T = 1 the route
program of _sl_tile_route_cg is replaced by one program that routes the whole call: the same FP32 scores
(sigmoid(logit * mult) + router_bias from the FP32 router GEMV, .cg reads after griddepcontrol.wait), the
reference top 8, then a greedy that drops whole local experts, fewest routes first. An expert is dropped only
when each token routed to it holds it as an optional pick and has a replacement inside the window: a non-local
expert (preferred: one fewer local route) or a local expert this call already uses. A drop must keep
routes >= 0.92 * ref - 4 + RS_ROUTE_MARGIN and distinct >= 0.85 * ref - 2 + RS_DISTINCT_MARGIN, and the two score
tolerances are tightened by RS_KEEP_MARGIN / RS_WIN_MARGIN against the gap between these FP32 scores and the
judge's FP64 ones. Replacements are the highest-scoring candidates, as far inside the window as they come.
Weights are the picked unbiased scores over their sum, as before. The pooled-normalize programs, every other
kernel, every PDL edge and gate, and every T other than 1 run pdl8d's code (T = 8: see RS_T).

    python3 tools/dl_rs_patch.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/g1b4_pdl8d.py"
DST = ROOT / "muse_decode_layer/cand/g1b4_rs.py"
PDL8D_PIN = "7cb74240d6ab5edd1c72e869777637157d17c67b615d508565a6084496028904"

KERNEL = r'''
RS_ON = True
# T = 8 runs the layer's FP8-query attention: its scores sit up to 0.019 from the judge's BF16-query FP64 ones and
# the 9th-best lead / 8th-best distance up to 0.021 / 0.026 (tools/box_rs_check.py), too close to the 0.03 / 0.05
# tolerances, so T = 8 keeps the reference top 8 (the kernel handles any TB <= 8)
RS_T = (1,)
RS_KEEP_TOL = 0.03   # reference.KEEP_TOL up to 256 tokens
RS_ROUTE_TOL = 0.05  # reference.ROUTE_TOL
# T = 1, 800 box inputs (tools/box_rs_check.py): the 9th-best lead of these scores is off the judge's by at most
# 5.4e-3 (next 3.7e-3), the distance below the 8th best by at most 1.16e-2 (next 5.4e-3); every timed call is checked
RS_KEEP_MARGIN = 0.010
RS_WIN_MARGIN = 0.015
RS_ROUTE_MARGIN = 1.0     # our reference top 8 can differ from the judge's by a near tie: one route / expert
RS_DISTINCT_MARGIN = 1.0
_RS_DEBUG = None  # tests only: a list that collects each eager call's logits tensor


@triton.jit
def _rs_tile_route(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, TB: tl.constexpr, PARTS: tl.constexpr,
                   KT: tl.constexpr, WIN: tl.constexpr, RM: tl.constexpr, DM: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        _gdc_launch()
        _gdc_wait()
        pooled = tl.load(POOLED + t * 4096 + c, cache_modifier='.cg').to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        else:
            vals = tl.load(POOLED + t * 4096 + oc, cache_modifier='.cg').to(tl.float32)
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    elif t == 0:
        e = tl.arange(0, 256)
        tt = tl.arange(0, TB)
        j = tl.arange(0, 32)
        j64 = j.to(tl.int64)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        _gdc_launch()
        _gdc_wait()
        logit = tl.load(LOGITS + tt[:, None] * 256 + e[None, :], cache_modifier='.cg')
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias[None, :]
        work = sel
        in8i = tl.zeros([TB, 256], dtype=tl.int32)
        for k in tl.static_range(8):
            idx = tl.argmax(work, 1)
            hit = e[None, :] == idx[:, None]
            in8i = in8i | hit.to(tl.int32)
            work = tl.where(hit, float("-inf"), work)
        in8 = in8i != 0
        v9 = tl.max(work, 1)
        v8 = tl.min(tl.where(in8, sel, float("inf")), 1)
        le = (e - ls + 256) % 256
        isl = (le < 32)[None, :]
        nloc = (le >= 32)[None, :]
        lb = tl.where(le < 32, tl.full([256], 1, tl.int64) << (le & 31).to(tl.int64), 0)[None, :]
        refl = in8 & isl
        pick = tl.sum(tl.where(refl, lb, 0), 1)
        opt = tl.sum(tl.where(refl & (sel - v9[:, None] <= KT), lb, 0), 1)
        inwin = sel >= v8[:, None] - WIN
        lsel = tl.gather(sel, tl.broadcast_to(((ls + j) % 256)[None, :], [TB, 32]), 1)
        cand = tl.sum(tl.where(isl & inwin, lb, 0), 1)
        nlc = nloc & (in8i == 0) & inwin
        navail = tl.sum(nlc.to(tl.int32), 1)
        bits = ((pick[:, None] >> j64[None, :]) & 1).to(tl.int32)
        uses = tl.sum(bits, 0)
        routes = tl.sum(uses, 0)
        distinct = tl.sum((uses > 0).to(tl.int32), 0)
        U = tl.sum(tl.where(uses > 0, tl.full([32], 1, tl.int64) << j64, 0), 0)
        rfloor = routes.to(tl.float32) * 0.92 - 4.0 + RM
        dfloor = distinct.to(tl.float32) * 0.85 - 2.0 + DM
        nlused = tl.zeros([TB], dtype=tl.int32)
        go = tl.max(tl.where(tt >= 0, (opt != 0).to(tl.int32), 0), 0)
        while go > 0:
            bits = ((pick[:, None] >> j64[None, :]) & 1).to(tl.int32)
            uses = tl.sum(bits, 0)
            ob = ((opt[:, None] >> j64[None, :]) & 1).to(tl.int32)
            hasnl = (navail - nlused) > 0
            locav = cand & U & ~pick
            hasloc = locav != 0
            either = (hasnl | hasloc).to(tl.int32)
            okte = (bits == 0) | ((ob == 1) & (either[:, None] == 1))
            allok = tl.min(okte.to(tl.int32), 0)
            nlonly = tl.sum(bits * (hasnl & (locav == 0)).to(tl.int32)[:, None], 0)
            feas = ((allok == 1) & (uses > 0) & ((routes - nlonly).to(tl.float32) >= rfloor)
                    & ((distinct - 1).to(tl.float32) >= dfloor))
            best = tl.argmin(tl.where(feas, uses * 64 + j, 1 << 20), 0)
            go = tl.max(feas.to(tl.int32), 0)
            eb = (tl.full([], 1, tl.int64) << best.to(tl.int64)) * go.to(tl.int64)
            aff = (pick & eb) != 0
            nlo = tl.sum(tl.where(j == best, nlonly, 0), 0)
            budget = tl.floor(routes.to(tl.float32) - rfloor) - nlo.to(tl.float32)
            both = aff & hasnl & hasloc
            rank = tl.cumsum(both.to(tl.int32), 0)
            gonl = aff & hasnl & ((locav == 0) | (rank.to(tl.float32) <= budget))
            goloc = aff & (gonl == 0)
            lav = ((locav[:, None] >> j64[None, :]) & 1) != 0
            nb = tl.where(goloc, tl.full([TB], 1, tl.int64) << tl.argmax(tl.where(lav, lsel, float("-inf")), 1).to(tl.int64), 0)
            pick = tl.where(aff, (pick & ~eb) | nb, pick)
            opt = tl.where(aff, (opt & ~eb) | nb, opt)
            nlused = nlused + gonl.to(tl.int32)
            routes = routes - tl.sum(gonl.to(tl.int32), 0)
            distinct = distinct - go
            U = U & ~eb
        mine = ((pick[:, None] >> (le & 31).to(tl.int64)[None, :]) & 1) != 0
        chosen = (in8 & nloc) | (isl & mine)
        work2 = tl.where(nlc, sel, float("-inf"))
        for k in tl.static_range(8):
            idx = tl.argmax(work2, 1)
            hit = (e[None, :] == idx[:, None]) & (nlused[:, None] > k)
            chosen = chosen | hit
            work2 = tl.where(hit, float("-inf"), work2)
        slot = tl.cumsum(chosen.to(tl.int32), 1) - 1
        wsum = tl.sum(tl.where(chosen, sc, 0.0), 1)
        off = tt[:, None] * 8 + slot
        tl.store(IDS + off, tl.broadcast_to(e[None, :], [TB, 256]), mask=chosen)
        tl.store(WTS + off, sc / (wsum[:, None] + 1e-15), mask=chosen)
        tl.store(LIDS + off, tl.broadcast_to(tl.where(isl, le[None, :], -1), [TB, 256]), mask=chosen)


def _rs_route_launch(pooled, logits, state, ids, wts, xm, ls, lids, T):
    if _RS_DEBUG is not None:
        _RS_DEBUG.append(logits.clone())
    if RS_ON and T in RS_T:
        _rs_tile_route[(T, SL_TILE_CONFIG[T][1] + 1)](
            pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm, ls, lids, TB=T,
            PARTS=SL_TILE_CONFIG[T][1], KT=RS_KEEP_TOL - RS_KEEP_MARGIN, WIN=RS_ROUTE_TOL - RS_WIN_MARGIN,
            RM=RS_ROUTE_MARGIN, DM=RS_DISTINCT_MARGIN, num_warps=SL_TILE_CONFIG[T][0],
            launch_pdl=_pdl8_edge("route"))
        return
    _sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                         ls, lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))

'''

CALL_OLD = '''        _sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))
'''
CALL_NEW = '''        _rs_route_launch(pooled, logits, state, ids, wts, xm, data["local_start"], lids, T)
'''
ANCHOR = "\n\n@triton.jit\ndef _sl_cluster_finish_cg("


def sub(s, old, new, n=1):
    if s.count(old) != n:
        sys.exit(f"anchor {old[:70]!r}: {s.count(old)} found, {n} expected")
    return s.replace(old, new)


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PDL8D_PIN:
        sys.exit("g1b4_pdl8d.py is not 7cb74240")
    s = raw.decode()
    s = sub(s, CALL_OLD, CALL_NEW)
    s = sub(s, ANCHOR, "\n" + KERNEL + ANCHOR)
    DST.write_text(s)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
