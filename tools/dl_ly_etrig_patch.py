#!/usr/bin/env python3
"""ly_etrig: the decode 1 MoE (TINY build) fires griddepcontrol.launch_dependents after its phase A credit.

Base: muse_decode_layer/cand/ly_rfx.py (SHA256 60ef465d..., checked). Target: the decode 1 post-MoE tail. The TINY
MoE triggers its PDL secondary (the unpooler-slice cgemv, 512 CTAs, 12 KB of codes each) only after each CTA's
phase B end and the helper's last epilogue (WS_JOIN), so the slice enters as the MoE CTAs leave and streams its
6.3 MB of codes after the MoE's phase B: at decode 1 full the codes land 1.6 us after the MoE's last exit, at
sliding they overlap the last 8 CTAs' final pass, which then takes 1.3 us after the barrier instead of 0.45
(docs/runs-decode-layer.md, "Decode 1 post-MoE tail", pmt stamps). Three slice CTAs fit beside a MoE CTA (the pmt
stamps show slice CTAs entering on the SMs whose MoE CTA still runs the final pass).

Change (TINY only, -DOKC_ETRIG=1; TINY2 / MID compile the same source with OKC_ETRIG 0, so their code is the
base's): the helper warp's lane 0 executes griddepcontrol.launch_dependents right after it credits the CTA's phase A
units to doneA. The PTX ISA (griddepcontrol) makes the first invocation in a CTA the CTA's signal and later ones
no-ops, so the existing trigger after WS_JOIN stays and is redundant on this build. The slice then launches during
the MoE's phase B, whose weights are mostly already in the 3-slot ring (DRAM has slack there), loads its codes
and blocks in its own griddepcontrol.wait, which still returns only after the whole MoE grid has completed and its
writes are visible ("all the memory operations from the prerequisite grids are performed and made visible to the
current grid"); the slice touches nothing the MoE writes before that wait. No fence, barrier, release, acquire,
wait or launch attribute changes; the PDL set per T is unchanged.

    python3 tools/dl_ly_etrig_patch.py [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_rfx.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_etrig.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "60ef465d47ff76cd83e236fc8d4768bc18eda7b77e2353e32f2c36491b704fda"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

REGION = ("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"", "\n\"\"\"\n")
SRC_EDITS = [
    ("__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n",
     "#ifndef OKC_ETRIG\n#define OKC_ETRIG 0\n#endif\n"
     "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n", 1),
    ("            atomicAdd(p.doneA + jj, b - a);\n          }\n",
     "            atomicAdd(p.doneA + jj, b - a);\n          }\n"
     "          if (OKC_ETRIG) asm volatile(\"griddepcontrol.launch_dependents;\\n\" ::: \"memory\");"
     "  // etrig: the slice may launch during phase B\n", 1),
]
PY_EDITS = [
    ('MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n',
     'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
     'MOE_TINY_DEFINES += ("-DOKC_ETRIG=1",)  # etrig: decode 1 MoE triggers the slice after its phase A credit\n', 1),
]
HEADER = ("# ly_etrig: ly_rfx with the decode 1 (TINY) MoE's griddepcontrol.launch_dependents fired after the "
          "helper's phase A credit instead of at the phase B end (tools/dl_ly_etrig_patch.py).\n")


def sub(text, old, new, n):
    if text.count(old) != n:
        sys.exit(f"edit anchor found {text.count(old)} times, expected {n}: {old!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a = text.index(REGION[0])
    b = text.index(REGION[1], a)
    seg = text[a:b]
    for old, new, n in SRC_EDITS:
        seg = sub(seg, old, new, n)
    text = text[:a] + seg + text[b:]
    for old, new, n in PY_EDITS:
        text = sub(text, old, new, n)
    if not text.startswith("# ly_rfx:"):
        sys.exit("base header line is not ly_rfx's")
    text = HEADER + text[text.index("\n") + 1:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
