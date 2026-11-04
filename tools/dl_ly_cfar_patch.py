#!/usr/bin/env python3
"""ly_cfar: the decode 1 MoE (TINY build) helper's doneA credit fence as fence.acq_rel.gpu instead of fence.sc.gpu.

Base: muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d..., checked). Target: the decode 1 A-to-B handoff
(docs/runs-decode-layer.md, "Decode 1 MoE ramp, handoff, tail"). The ly_mtl2 stamps put the helper's last phase A
epilogue plus its credit at 1.0 to 1.1 us from the moment the last unit's partials are in to the atomicAdd issue,
and the last CTA's credit gates every CTA's first phase B unit (B starts 0.14 to 0.18 us after the last credit).
In that window the only blocking instruction is the credit's __threadfence, which compiles to MEMBAR.SC.GPU;
fence.acq_rel.gpu compiles to MEMBAR.ALL.GPU (both verified with cuobjdump on the box, nvcc 13.0, sm_90a).

Change (TINY only, -DOKC_CFAR=1; TINY2 / MID compile the same source with OKC_CFAR 0): in ws_helper's credit block
(q == nA - 1) every helper lane executes fence.acq_rel.gpu where it executed __threadfence(); __syncwarp and lane
0's atomicAdd on doneA follow unchanged. Ordering argument (PTX ISA, Memory Consistency Model, "Release and acquire
patterns"): a release pattern is a release operation, or a strong write preceded in program order by fence.acq_rel
or fence.sc; lane 0's [fence; atomicAdd] is a release pattern with either fence, and the consumers' ld.acquire.gpu
poll is unchanged. The other lanes' inter stores reach lane 0's pattern through __syncwarp exactly as in the base
(the base's correctness for them already rests on the warp barrier plus lane 0's cumulative release, not on the
fence-SC total order, which no reader of doneA uses). No fence is removed, no proxy fence, barrier, acquire,
atomic, wait or trigger moves.

    python3 tools/dl_ly_cfar_patch.py [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfx.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_cfar.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

SRC_EDITS = [
    ("__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n",
     "#ifndef OKC_CFAR\n#define OKC_CFAR 0\n#endif\n"
     "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"),
    ("      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters\n"
     "        __threadfence();\n",
     "      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters\n"
     "#if OKC_CFAR\n"
     "        asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");  // release pattern with lane 0's atomicAdd\n"
     "#else\n"
     "        __threadfence();\n"
     "#endif\n"),
]
PY_EDITS = [
    ('MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n',
     'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
     'MOE_TINY_DEFINES += ("-DOKC_CFAR=1",)  # cfar: decode 1 doneA credit fence as fence.acq_rel.gpu\n'),
]
HEADER = ("# ly_cfar: ly_rfx with the decode 1 (TINY) MoE helper's doneA credit fence as fence.acq_rel.gpu instead of "
          "__threadfence (fence.sc.gpu) (tools/dl_ly_cfar_patch.py).\n")


def sub(text, old, new):
    if text.count(old) != 1:
        sys.exit(f"edit anchor found {text.count(old)} times, expected 1: {old!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a = text.index("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"")
    b = text.index('\n"""\n', a)
    seg = text[a:b]
    for old, new in SRC_EDITS:
        seg = sub(seg, old, new)
    text = text[:a] + seg + text[b:]
    for old, new in PY_EDITS:
        text = sub(text, old, new)
    if not text.startswith("# ly_rfx:"):
        sys.exit("base header line is not ly_rfx's")
    text = HEADER + text[text.index("\n") + 1:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
