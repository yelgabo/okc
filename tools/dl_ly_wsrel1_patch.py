"""ly_wsrel1: ly_mpdl with the ws_helper doneA credit published by one post-relay release.

ly_mpdl's ws_helper credit (MOE_CUDA_SRC and the moe_helper0 copy used by TINY / TINY2 / MID) is: every lane stores
its p.inter rows and runs __threadfence(), then __syncwarp(), then lane 0 does relaxed atomicAdds. Lane 0's release
pattern starts before the relay, so the other 31 lanes' stores are not ordered before it. ly_wsrel1 drops the
per-lane pre-relay fence and has lane 0 fence after the relay:

    every lane: st p.inter;  __syncwarp();  lane 0: fence.acq_rel.gpu (ly_wsrel1) or fence.sc.gpu (ly_wsrel1_sc);
    atom.add doneA

Proof (PTX ISA, Memory Consistency Model) in docs/lanes/moe-wsrel2.md: the warp barrier orders each lane's stores
before lane 0's fence (bar.warp.sync: "guarantee memory ordering among threads participating in barrier"; base
causality order), fence plus strong atom is a release pattern ("Release and acquire patterns"), and the consumer's
ld.acquire.gpu poll synchronizes with it ("Synchronizes-with"; "Causality" axiom).

    python3 tools/dl_ly_wsrel1_patch.py   # writes cand/ly_wsrel1.py, cand/ly_wsrel1_sc.py, cand/ly_wsrel1_ab.py

ly_wsrel1_ab.py is ly_mpdl unchanged (the default arm is the base) plus a state switch for tools/ly_arms.py:
{"state": {"wsrel_h": "ar1"}} or {"state": {"wsrel_h": "sc1"}} runs TINY / TINY2 / MID on the patched helper.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "muse_decode_layer/cand/ly_mpdl.py")
OUTS = {"ar1": "muse_decode_layer/cand/ly_wsrel1.py", "sc1": "muse_decode_layer/cand/ly_wsrel1_sc.py"}
OUT_AB = "muse_decode_layer/cand/ly_wsrel1_ab.py"
SRC_SHA = "a0ee638c"
SPLIT = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"

FENCES = {
    "ar1": 'asm volatile("fence.acq_rel.gpu;" ::: "memory");\n',
    "sc1": 'asm volatile("fence.sc.gpu;" ::: "memory");\n',
}
OLD = ("        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n"
       "          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n")


def new(kind):
    return ("        __syncwarp();  // orders every lane's p.inter stores before lane 0's release (dl_ly_wsrel1_patch)\n"
            "        if (lane == 0) {\n          " + FENCES[kind]
            + "          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n")


AB_TAIL = '''

# A/B only (ly_wsrel1_ab): {"state": {"wsrel_h": "ar1" | "sc1"}} runs TINY / TINY2 / MID on a patched helper
_WSREL1_OLD, _WSREL1_NEW = %r, %r
for _h, _new in _WSREL1_NEW.items():
    assert _MA_ALT_SRCS['moe_helper0'].count(_WSREL1_OLD) == 1
    _MA_ALT_SRCS['moe_helper0' + _h] = _MA_ALT_SRCS['moe_helper0'].replace(_WSREL1_OLD, _new)
_wsrel1_fork_native_moe = _fork_native_moe


def _fork_native_moe(M=None, state=None):
    h = state.get("wsrel_h") if state is not None else None
    if not h or M is None or M > MOE_MID_MAX_M:
        return _wsrel1_fork_native_moe(M, state)
    sfx = tuple(state["ma_sfx"])
    if MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        key, defines = "tiny", MOE_TINY_DEFINES
    elif MOE_TINY2_DEFINES and M <= MOE_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES)):
        key, defines = "tiny2", MOE_TINY2_DEFINES
    else:
        key, defines = "mid", MOE_MID_DEFINES
    alt = "@moe_helper0" + h
    defines = tuple(alt if d in ("@moe_helper0", "@moe_helper0df") else d for d in defines)
    assert alt in defines, defines
    if state["ma_sc6"] and "-DOKC_WGS=1" in defines:
        defines = defines + ("-DOKC_SC6W=1",)
    key += "_" + h
    ext = state["ma_ext"].get(key)
    if ext is None:
        ext = _RouterExt(_ma_ext(defines + sfx))
        if MOE_HAS_PDL:
            ext.okc_set_pdl(0)
        state["ma_ext"][key] = ext
    return ext


_wsrel1_prepare = prepare


def prepare(weights):
    st = _wsrel1_prepare(weights)
    for h in _WSREL1_NEW:
        for T in (1, 8, 16):
            _moe(T, dict(st, wsrel_h=h))
    print("wsrel1 arms", sorted(k for k in st["ma_ext"] if k.endswith(tuple("_" + h for h in _WSREL1_NEW))), flush=True)
    return st
'''


def patch(text, kind):
    i = text.index(SPLIT)
    j = text.index('\n"""\n', i + len(SPLIT))  # end of the helper source
    parts = [text[:i], text[i:j], text[j:]]
    assert OLD not in parts[2]
    for k in (0, 1):
        assert parts[k].count(OLD) == 1, (k, parts[k].count(OLD))
        parts[k] = parts[k].replace(OLD, new(kind))
    out = "".join(parts)
    assert out.count(FENCES[kind]) == text.count(FENCES[kind]) + 2 and out.count(new(kind)) == 2
    assert out.replace(new(kind), OLD) == text
    return out


def main():
    text = open(SRC).read()
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert sha.startswith(SRC_SHA), sha
    assert text.count(SPLIT) == 1
    for kind, out in OUTS.items():
        open(os.path.join(ROOT, out), "w").write(patch(text, kind))
    open(os.path.join(ROOT, OUT_AB), "w").write(text + AB_TAIL % (OLD, {k: new(k) for k in OUTS}))
    for f in list(OUTS.values()) + [OUT_AB]:
        print(f, hashlib.sha256(open(os.path.join(ROOT, f), "rb").read()).hexdigest())


if __name__ == "__main__":
    sys.exit(main())
