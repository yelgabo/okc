#!/usr/bin/env python3
"""attn_port: the layer's ly_a16o2 ported into the standalone attention file. The q_len 1 plain per-request
build attn_one_kernel<1, 1> (mode 1: decode 16 and decode 32 sliding, 256 CTAs) reserves NB bytes of dynamic
shared memory, so at most two of its CTAs share an SM. Host-only: the attribute, the occupancy query and both
launch paths of Instance<1, 1> use LBYTES; every other instance keeps BYTES and the kernel is unchanged. Mode 1
is not cooperative, so the plan's residency table does not change the split count.

    python3 tools/dl_attp_patch.py [--base .build/repeats/attention_base.py] [--out muse_attention_decode/cand/attn_port.py] [--bytes 81920]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / ".build/repeats/attention_base.py"
DST = ROOT / "muse_attention_decode/cand/attn_port.py"
PIN = "aa3535d0f1fbb78c92da7e8628d264392c440a17ba756a2297d7566ee46e27e1"
NB = 81920
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
if "--bytes" in sys.argv:
    NB = int(sys.argv[sys.argv.index("--bytes") + 1])

EDITS = [
    ("# Repeat of check ca0f13d324ef48e8: this line is the only change.\n",
     f"# attn_port: decode 16 / 32 sliding (attn_one_kernel<1, 1>) at {NB} B dynamic smem, two CTAs per SM "
     "(layer ly_a16o2; tools/dl_attp_patch.py).\n"),
    ("template <int L, int MODE>\nstruct Instance {\n",
     f"#ifndef ONE_A16SM\n#define ONE_A16SM {NB}\n#endif\n\ntemplate <int L, int MODE>\nstruct Instance {{\n"),
    ("    static constexpr int BYTES = Smem<L, one_ns(L, MODE), one_pb(L, MODE)>::BYTES;\n",
     "    static constexpr int BYTES = Smem<L, one_ns(L, MODE), one_pb(L, MODE)>::BYTES;\n"
     "    static constexpr int LBYTES = (L == 1 && MODE == 1 && ONE_A16SM > BYTES) ? ONE_A16SM : BYTES;\n"),
    ("cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES);",
     "cudaFuncAttributeMaxDynamicSharedMemorySize, LBYTES);"),
    ("cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, NT, BYTES);",
     "cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, NT, LBYTES);"),
    ("NT, BYTES, stream>>>(p);", "NT, LBYTES, stream>>>(p);"),
    ("cfg.dynamicSmemBytes = BYTES;", "cfg.dynamicSmemBytes = LBYTES;"),
]


def sub(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for n, (old, new) in enumerate(EDITS):
        text = sub(text, old, new, f"edit {n}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, len(text.encode()), hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
