#!/usr/bin/env python3
"""Diagnostic copy of a layer file with the attention kernel's per-CTA stamps compiled in (OKC_DBGON, OKC_SEGDBG),
the SM id of each CTA in stamp slot 15, and _launch7 passing a module-level int64 tensor as the kernel's dbg
pointer. Not a candidate: the stamps cost cold instructions. Read with tools/attn_stamps_run.py.

    python3 tools/attn_stamps_make.py --base muse_decode_layer/cand/ly_fabq.py --out .build/attn/ly_fabq_stamps.py
"""
import argparse
import hashlib
import pathlib
import py_compile

ROOT = pathlib.Path(__file__).resolve().parents[1]

EDITS = [
    ("#define OKC_SEGDBG 0\n", "#define OKC_SEGDBG 1\n", 1),
    ("#define OKC_DBGON 0 ", "#define OKC_DBGON 1 ", 1),
    ("attn_kernel(const __grid_constant__ Params p) {\n"
     "  constexpr int NT = Cfg<L, FLAT>::NT;\n"
     "  constexpr int STAGES = Cfg<L, FLAT>::STAGES;\n"
     "  constexpr int IDX_SLOTS = 2 * STAGES;\n"
     "  extern __shared__ __align__(1024) uint8_t smem_raw[];\n",
     "attn_kernel(const __grid_constant__ Params p) {\n"
     "  constexpr int NT = Cfg<L, FLAT>::NT;\n"
     "  constexpr int STAGES = Cfg<L, FLAT>::STAGES;\n"
     "  constexpr int IDX_SLOTS = 2 * STAGES;\n"
     "  extern __shared__ __align__(1024) uint8_t smem_raw[];\n"
     "  if (OKC_DBGON && p.dbg && threadIdx.x == 0) {   // stamps diagnostic: the CTA's SM\n"
     "    unsigned _s; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(_s));\n"
     "    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = (long long)_s;\n"
     "  }\n", 1),
    ("def _launch7(data, dbg=None, _mod=None, pdl=None, attention_state=None, prefetch=(0,0), **override):\n",
     "_ATTN_DBG = None   # stamps diagnostic: int64 [ctas x DBGSTRIDE], passed as every attention launch's dbg\n\n\n"
     "def _launch7(data, dbg=None, _mod=None, pdl=None, attention_state=None, prefetch=(0,0), **override):\n"
     "    global _ATTN_DBG\n"
     "    if dbg is None:\n"
     "        if _ATTN_DBG is None:\n"
     "            _ATTN_DBG = torch.zeros(4096 * 64, dtype=torch.int64, device=data['qkv'].device)\n"
     "        dbg = _ATTN_DBG\n", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="muse_decode_layer/cand/ly_fabq.py")
    ap.add_argument("--out", default=".build/attn/ly_fabq_stamps.py")
    a = ap.parse_args()
    src = (ROOT / a.base).read_text()
    print("base", a.base, hashlib.sha256(src.encode()).hexdigest())
    for old, new, n in EDITS:
        assert src.count(old) == n, (src.count(old), old[:60])
        src = src.replace(old, new)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src)
    py_compile.compile(str(out), doraise=True)
    print(out, hashlib.sha256(src.encode()).hexdigest())


if __name__ == "__main__":
    main()
