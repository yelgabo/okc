#!/usr/bin/env python3
"""ly_finb1: the T <= 8 finish kernel loads b1's operands (s1 rows and the tail sum) before its griddepcontrol.wait.

Base: muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d..., checked). Target: the finish kernel at decode 1
(docs/runs-decode-layer.md, "Decode 1 finish and slice launch"). In the base SASS of `_sl_cluster_finish_cg` (T = 1,
8 warps) ptxas issues the slice output P right after ACQBULK and sinks the carry (s1) load below the first
reduction's shuffles and the tail / outcarry loads below its shared-memory store, so carry's L2 latency starts only
after P has landed: the two loads are serial although the PTX issues them together.

Change: in `_sl_cluster_finish_cg` only, the three loads of b1's outputs (carry = s1[t, :1024], tail[t],
outcarry = s1[t, tile]) move above the gdc_wait, still .cg; P (the slice output, the predecessor's) and NEXT stay
behind it. Per operand: s1 and tail are written only by `_pdl8_b1` (b1) in this call (allocated at the call's top,
alive to its return; the pooler, route, MoE and slice write h / hr, pooled / logits, xm / ids / lids / wts, m and P).
At T <= 8 the MoE is a cudaLaunchCooperativeKernel with no programmatic attribute (helper0 launcher), so it starts
only after all earlier work in the stream has completed, and route (every program waits) completes only after the
pooler, which completes only after b1. The slice launches only after the MoE has started and the finish only after
the slice has started, so b1 has completed before any finish CTA exists; the loads are ld.global.cg (L2, no L1
line). No fence, barrier, wait, trigger or launch attribute changes.

    python3 tools/dl_ly_finb1_patch.py [--base muse_decode_layer/cand/ly_rfx.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfx.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_finb1.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

OLD = """def _sl_cluster_finish_cg(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0,1024)
    beta = tl.load(BETA+c)
    gp = tl.load(POST+c).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    carry = tl.load(CARRY+t*8192+c, cache_modifier='.cg')
    tail = tl.load(TAIL+t, cache_modifier='.cg')
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o, cache_modifier='.cg')
    gn = tl.load(NEXT+o).to(tl.float32)
"""
NEW = """def _sl_cluster_finish_cg(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0,1024)
    beta = tl.load(BETA+c)
    gp = tl.load(POST+c).to(tl.float32)
    # s1 and tail are b1's: b1 completed before the cooperative MoE (no programmatic edge) began (dl_ly_finb1_patch.py)
    carry = tl.load(CARRY+t*8192+c, cache_modifier='.cg')
    tail = tl.load(TAIL+t, cache_modifier='.cg')
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o, cache_modifier='.cg')
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    gn = tl.load(NEXT+o).to(tl.float32)
"""
HEADER = ("# ly_finb1: ly_rfx with the T <= 8 finish kernel loading b1's s1 rows and tail sum before its "
          "griddepcontrol.wait (tools/dl_ly_finb1_patch.py).\n")


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    if text.count(OLD) != 1:
        sys.exit(f"finish anchor found {text.count(OLD)} times, expected 1")
    text = HEADER + text.replace(OLD, NEW)
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
