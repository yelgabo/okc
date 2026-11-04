#!/usr/bin/env python3
"""ly_a16o2: the decode 16 / 32 sliding attention launch (attn_kernel<1, false, 1>, the only q_len 1 launch of the
plain per-request build kn[1]) reserves 80 KB of dynamic shared memory, so at most two of its CTAs share an SM.

Base ly_stk2 (4bbf4010). Stamps (tools/ly_box_a16.py on ly_a16) show the 256 CTAs placed on 85 to 87 SMs, three per
SM: at launch (0.5 to 1 us before qkv's last CTA ends) only the SMs without a qkv CTA have room, and three CTAs fit
there, so the SMs that qkv frees last get none. With two per SM, 170-odd CTAs land at launch and the rest on the SMs
qkv frees, about 128 SMs in all. Host-only launch change (dynamic shared memory size and its attribute); the kernel,
its fences, wait, trigger and the PDL attribute are unchanged; every other launch keeps sm_n.

    python3 tools/dl_ly_a16o2_patch.py [--base muse_decode_layer/cand/ly_stk2.py] [--out muse_decode_layer/cand/ly_a16o2.py] [--bytes 81920]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk2.py"
DST = ROOT / "muse_decode_layer/cand/ly_a16o2.py"
PIN = "4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846"
NB = 81920
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
if "--bytes" in sys.argv:
    NB = int(sys.argv[sys.argv.index("--bytes") + 1])

EDITS = [
    ("#define OKC_CLSM 0 ",
     f"#define OKC_A16SM {NB}   // ly_a16o2: dynamic shared memory of the q_len 1 plain per-request launch (decode 16 / 32 sliding): two CTAs per SM\n"
     "#define OKC_CLSM 0 "),
    ("    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n",
     "    if (L == 1 && OKC_A16SM > sm_n) cudaFuncSetAttribute(kn[1], cudaFuncAttributeMaxDynamicSharedMemorySize, OKC_A16SM);\n"
     "    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);\n"),
    ("  if (!p.flat && cs > 1 && !clusters) kern = kn[0];   // (sm_80 emulation of clusters: run-time mode)\n",
     "  if (!p.flat && cs > 1 && !clusters) kern = kn[0];   // (sm_80 emulation of clusters: run-time mode)\n"
     "  const int smk = (L == 1 && kern == kn[1] && OKC_A16SM > sm) ? OKC_A16SM : sm;\n"),
    ("    cfg.dynamicSmemBytes = (cl && OKC_CLSM > sm) ? OKC_CLSM : sm;",
     "    cfg.dynamicSmemBytes = (cl && OKC_CLSM > sm) ? OKC_CLSM : smk;"),
    ("    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);",
     "    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, smk, stream>>>(q);"),
]
HEADER = (f"# ly_a16o2: ly_stk2 with the decode 16 / 32 sliding attention launch at {NB} B of dynamic shared memory (two CTAs "
          "per SM; tools/dl_ly_a16o2_patch.py).\n")


def sub(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected 1: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index('ATT7_CUDA_SRC = r"""\n')
    j = text.index('\n"""\n', i)
    at = text[i:j]
    for n, (old, new) in enumerate(EDITS):
        at = sub(at, old, new, f"att edit {n}")
    text = HEADER + text[:i] + at + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
