#!/usr/bin/env python3
"""Blueprint micro-changes on ly_micro2.py (0bc2bd8b), one per --variant; the output SHA256 is printed.

    python3 tools/dl_ly_fab_patch.py --variant b2pre [--variant ...] --out muse_decode_layer/cand/ly_fab_<v>.py

Variants (docs/runs-decode-layer.md "Blueprint builds (fable lane)"):
  b2pre   T = 16 boundary 2 (glue_cluster_boundary POST, the PDL8 graft glue): the entry griddepcontrol.wait no longer
          precedes the weight preloads (alpha, beta, pre, post are prepare-time constants) and each thread prefetches its
          s1 chunk into L2; the wait comes right after the preloads and the s1 (.cg) loads stay behind it. The second
          wait before the branch (unpooler output) loads is unchanged. SLICED keeps its entry wait.
  b1d32   T = 32 boundary 1 (_graft_row_boundary_pre_cg, POST = False): the stream load moves before gdc_wait. For
          boundary 1 the stream is data['stream'], the harness input, which nothing in the call writes; boundary 2
          (POST) keeps its stream (s1) load behind the wait with .cg.
  finpre  T <= 8 boundary 2 (_sl_cluster_finish_cg): the NEXT weight load (the next input's norm weight, a prepare-time
          constant) moves before gdc_wait; carry, tail and the slice stay behind the wait with .cg.
Diagnostics (timing only, wrong routing, run with --no-check; never candidates):
  rtdiag_ctl  T = 1: the route kernel reads the previous call's pooled / logits (state buffers copied after the finish
          kernel), so its routing is stale by one replay; kernel order unchanged.
  rtdiag_var  the same stale route launched before the pooler + router, so the MoE follows the pooler directly: the
          difference to rtdiag_ctl bounds what a route-in-MoE fusion could save at decode 1.
No fence, barrier, mbarrier or griddepcontrol.wait removed; no new PDL edge; no launch attribute change.
"""
import argparse
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"


def once(t, old, new, what):
    n = t.count(old)
    if n != 1:
        sys.exit(f"{what}: anchor found {n} times")
    return t.replace(old, new)


B2PRE_EDITS = [
    ('void glue_cluster_boundary(GlueArgs a) {\n  if constexpr (SLICED || POST) asm volatile("griddepcontrol.wait;" ::: "memory");',
     'void glue_cluster_boundary(GlueArgs a) {\n  if constexpr (SLICED) asm volatile("griddepcontrol.wait;" ::: "memory");', 1),
    ('    glfcg<V>(a.stream+row*8192+col,s+r*V);\n    glf<V>(a.alpha+col,alpha+r*V);\n',
     '    if constexpr(POST) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.stream+row*8192+col));\n'
     '    else glfcg<V>(a.stream+row*8192+col,s+r*V);\n'
     '    glf<V>(a.alpha+col,alpha+r*V);\n', 1),
    ('  // DSMEM destinations must exist before the first remote store.',
     '  if constexpr(POST) {  // b2: the stream is s1 from boundary 1 of this call, so it waits; the weights above are prepare-time constants\n'
     '    asm volatile("griddepcontrol.wait;" ::: "memory");\n'
     '    #pragma unroll\n'
     '    for(int r=0;r<repeats;r++) {\n'
     '      int col=logical*V+r*logical_threads*V;\n'
     '      glfcg<V>(a.stream+row*8192+col,s+r*V);\n'
     '    }\n'
     '  }\n'
     '  // DSMEM destinations must exist before the first remote store.', 1),
]

EDITS_LINE = "_PDL8_GRAFT_GLUE_EDITS = ["

B1D32_OLD_1 = """    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)"""
B1D32_NEW_1 = """    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    else:
        s = _gl.load(Stream + row * H + c)  # boundary 1: the stream is the harness input, nothing in the call writes it
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)"""
B1D32_OLD_2 = """    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
    ns = a * s + beta * b"""
B1D32_NEW_2 = """    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    if POST:
        s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
    ns = a * s + beta * b"""

FIN_OLD = """    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    carry = tl.load(CARRY+t*8192+c, cache_modifier='.cg')
    tail = tl.load(TAIL+t, cache_modifier='.cg')
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o, cache_modifier='.cg')
    gn = tl.load(NEXT+o).to(tl.float32)
"""
FIN_NEW = """    o = tile*BC+tl.arange(0,BC)
    gn = tl.load(NEXT+o).to(tl.float32)  # the next input's norm weight is a prepare-time constant
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    carry = tl.load(CARRY+t*8192+c, cache_modifier='.cg')
    tail = tl.load(TAIL+t, cache_modifier='.cg')
    outcarry = tl.load(CARRY+t*8192+o, cache_modifier='.cg')
"""

RT_ANCHOR = """    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))
"""
RT_CTL = """    if S == -1:
        pooled, logits = P
        if T == 1:  # rtdiag: route from the previous call's pooled / logits (stale by one replay; timing only)
            rb = state.setdefault("rtd", {})
            if "p" not in rb:
                rb["p"], rb["l"] = torch.zeros_like(pooled), torch.zeros_like(logits)
            rtp, rtl = rb["p"], rb["l"]
            rb["src"] = (pooled, logits)
        else:
            rtp, rtl = pooled, logits
        _graft_al(rtp, rtl, xm, ids, wts, lids)
        _sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)](rtp, rtl, rtl, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))
"""
RT_VAR = """    if S == -1:
        pooled, logits = P
        if xm_rt is not None:
            xm = xm_rt
            state["rtd"]["src"] = (pooled, logits)
        else:
            if T == 1:
                rb = state.setdefault("rtd", {})
                rb["p"], rb["l"] = torch.zeros_like(pooled), torch.zeros_like(logits)
                rb["src"] = (pooled, logits)
            _graft_al(pooled, logits, xm, ids, wts, lids)
            _sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))
"""
RT_POOL_OLD = """    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        if xm_rt"""
RT_POOL_NEW = """    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    xm_rt = None
    if T == 1 and "p" in state.get("rtd", {}):  # rtdiag: the stale route goes before the pooler, so the MoE follows the pooler directly
        rtp, rtl = state["rtd"]["p"], state["rtd"]["l"]
        xm_rt = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        _graft_al(rtp, rtl, xm_rt, ids, wts, lids)
        _sl_tile_route_cg[(T,SL_TILE_CONFIG[T][1]+1)](rtp, rtl, rtl, state["bias"], state["g_preexp"], ids, wts, xm_rt,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=_pdl8_edge("route"))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        if xm_rt"""
RT_RET_OLD = """    _sl_cluster_finish_cg[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=_pdl8_edge("fin"))
    return out, s2, ids


def _pdl8_graft_after_attention("""
RT_RET_NEW = """    _sl_cluster_finish_cg[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=_pdl8_edge("fin"))
    if T == 1 and state.get("rtd", {}).get("src") is not None:  # rtdiag: this call's pooled / logits feed the next call's route
        _rp, _rl = state["rtd"].pop("src")
        state["rtd"]["p"].copy_(_rp)
        state["rtd"]["l"].copy_(_rl)
    return out, s2, ids


def _pdl8_graft_after_attention("""


def apply(s, v):
    if v in ("rtdiag_ctl", "rtdiag_var"):
        i = s.index("def _pdl8_sl_tile_after_attention(")
        j = s.index("\ndef _pdl8_graft_after_attention(", i)
        t = s[i:j + 40]
        t = once(t, RT_ANCHOR, RT_CTL if v == "rtdiag_ctl" else RT_VAR, v)
        if v == "rtdiag_var":
            t = once(t, RT_POOL_OLD, RT_POOL_NEW, v)
        t = once(t, RT_RET_OLD, RT_RET_NEW, v)
        return s[:i] + t + s[j + 40:]
    if v == "finpre":
        return once(s, FIN_OLD, FIN_NEW, v)
    if v == "b2pre":
        i = s.index(EDITS_LINE)
        j = s.index("\n", i) + 1
        stmt = "_PDL8_GRAFT_GLUE_EDITS += " + repr(B2PRE_EDITS) + "   # fab b2pre\n"
        return s[:j] + stmt + s[j:]
    if v == "b1d32":
        i = s.index("def _graft_row_boundary_pre_cg(")
        j = s.index("\ndef ", i + 1)
        t = s[i:j]
        t = once(t, B1D32_OLD_1, B1D32_NEW_1, v)
        t = once(t, B1D32_OLD_2, B1D32_NEW_2, v)
        return s[:i] + t + s[j:]
    sys.exit(f"unknown variant {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("base SHA mismatch")
    s = raw.decode()
    for v in a.variant:
        s = apply(s, v)
    out = ROOT / a.out
    out.write_text(s)
    py_compile.compile(str(out), doraise=True)
    print(out.name, hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
