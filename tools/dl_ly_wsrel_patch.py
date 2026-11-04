"""ly_wsrel: ly_mpdl with the doneA credit released after the warp / CTA relay.

Every doneA credit site in both MoE sources (MOE_CUDA_SRC and the embedded moe_helper0 used by TINY / TINY2 / MID)
had: every writer thread fences, the relay (__syncwarp or CTA_SYNC), then the leader does a relaxed atomicAdd. The
leader's release pattern (its fence, then its strong write) starts before the relay, so the other threads' stores are
not causality-ordered before it. The fix adds a leader fence after the relay, so the release pattern is
relay -> fence.sc.gpu -> atom: every store ordered before the relay precedes the first operation of the release
pattern (PTX ISA, Memory Consistency Model: release patterns, base causality order, causality axiom).
No fence is removed.

    python3 tools/dl_ly_wsrel_patch.py            # writes cand/ly_wsrel.py (ship) and cand/ly_wsrel_ab.py (A/B arm)

ly_wsrel_ab.py adds a state switch for tools/ly_arms.py: {"state": {"wsrel0": True}} runs TINY / TINY2 / MID on
the unpatched ly_mpdl helper source (registered as @moe_helper0w0); the default arm runs the fixed source.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "muse_decode_layer/cand/ly_mpdl.py")
OUT = os.path.join(ROOT, "muse_decode_layer/cand/ly_wsrel.py")
OUT_AB = os.path.join(ROOT, "muse_decode_layer/cand/ly_wsrel_ab.py")
SRC_SHA = "a0ee638c"
SPLIT = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"

FENCE = "__threadfence();  // release after the relay: cumulative over the other threads' stores (see dl_ly_wsrel_patch)\n"

SITES = [
    ("skinny_ring SKDYN",
     "    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      if (nU > 0) {\n",
     "    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      " + FENCE + "      if (nU > 0) {\n"),
    ("skinny_ring",
     "    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;\n",
     "    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      " + FENCE
     + "      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;\n"),
    ("fat_stream",
     "      __threadfence();\n      CTA_SYNC();\n      if (tid == 0) atomicAdd(p.doneA + j, 1);\n",
     "      __threadfence();\n      CTA_SYNC();\n      if (tid == 0) {\n        " + FENCE
     + "        atomicAdd(p.doneA + j, 1);\n      }\n"),
    ("ws_helper",
     "        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n"
     "          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n",
     "        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n          " + FENCE
     + "          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n"),
]

AB_TAIL = '''

# A/B arm only (ly_wsrel_ab): {"state": {"wsrel0": True}} runs TINY / TINY2 / MID on the unpatched helper source
_WSREL_SITES = %r


def _wsrel_unpatch(s):
    for name, old, new in _WSREL_SITES:
        assert s.count(new) == 1, (name, s.count(new))
        s = s.replace(new, old)
    return s


_MA_ALT_SRCS['moe_helper0w0'] = _wsrel_unpatch(_MA_ALT_SRCS['moe_helper0'])
_wsrel_fork_native_moe = _fork_native_moe


def _fork_native_moe(M=None, state=None):
    if state is None or not state.get("wsrel0") or M is None or M > MOE_MID_MAX_M:
        return _wsrel_fork_native_moe(M, state)
    sfx = tuple(state["ma_sfx"])
    if MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        key, defines = "tiny", MOE_TINY_DEFINES
    elif MOE_TINY2_DEFINES and M <= MOE_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES)):
        key, defines = "tiny2", MOE_TINY2_DEFINES
    else:
        key, defines = "mid", MOE_MID_DEFINES
    defines = tuple("@moe_helper0w0" if d in ("@moe_helper0", "@moe_helper0df") else d for d in defines)
    assert "@moe_helper0w0" in defines, defines
    if state["ma_sc6"] and "-DOKC_WGS=1" in defines:
        defines = defines + ("-DOKC_SC6W=1",)
    key += "_w0"
    ext = state["ma_ext"].get(key)
    if ext is None:
        ext = _RouterExt(_ma_ext(defines + sfx))
        if MOE_HAS_PDL:
            ext.okc_set_pdl(0)
        state["ma_ext"][key] = ext
    return ext


_wsrel_prepare = prepare


def prepare(weights):
    st = _wsrel_prepare(weights)
    for T in (1, 8, 16):
        _moe(T, dict(st, wsrel0=True))
    print("wsrel arms", sorted(k for k in st["ma_ext"] if k.endswith("_w0")), flush=True)
    return st
'''


def main():
    text = open(SRC).read()
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert sha.startswith(SRC_SHA), sha
    assert text.count(SPLIT) == 1
    i = text.index(SPLIT)
    # the helper source ends at the first closing triple quote after the split marker
    j = text.index('\n"""\n', i + len(SPLIT))
    parts = [text[:i], text[i:j], text[j:]]
    assert all(old not in parts[2] for _, old, _ in SITES)
    helper0 = parts[1]
    for k in (0, 1):
        for name, old, new in SITES:
            n = parts[k].count(old)
            assert n == 1, (k, name, n)
            parts[k] = parts[k].replace(old, new)
    back = parts[1]
    for _, old, new in SITES:
        back = back.replace(new, old)
    assert back == helper0  # the ab arm's unpatch restores ly_mpdl's helper source exactly
    out = "".join(parts)
    assert out.count(FENCE) == 8
    open(OUT, "w").write(out)
    open(OUT_AB, "w").write(out + AB_TAIL % (SITES,))
    for f in (OUT, OUT_AB):
        print(os.path.relpath(f, ROOT), hashlib.sha256(open(f, "rb").read()).hexdigest())


if __name__ == "__main__":
    sys.exit(main())
