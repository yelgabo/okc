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
    ("#define OKC_FPRO 0 ", "#define OKC_FPRO 2 ", 1),
    # the FPRO stamps store unconditionally (the stock first-launch-only form costs a global load per stamp)
    ("#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)",
     "#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); *_dp = gtime(); } } while (0)", 1),
    # producer thread 0: after the page-index barrier init (47) and after the first page fetches are issued (48)
    ("          asm volatile(\"bar.sync 2, %0;\\n\" ::\"n\"(NP) : \"memory\");\n        }\n      }\n      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation\n",
     "          asm volatile(\"bar.sync 2, %0;\\n\" ::\"n\"(NP) : \"memory\");\n        }\n      }\n      if (q == 0) FPRO_STAMP(47);\n      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation\n", 1),
    ("      if constexpr (TMAPG_ON) {\n        for (int x = 0; x < PD; ++x) fetch_pages(x);\n      }\n",
     "      if constexpr (TMAPG_ON) {\n        for (int x = 0; x < PD; ++x) fetch_pages(x);\n      }\n      if (q == 0) FPRO_STAMP(48);\n", 1),
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
    # entry-order stamps: producer thread (tid == NT) at kernel entry (50) and after the partition sync (51);
    # thread 0 after the partition sync (40 is the stock site) and after setmaxnreg.inc (53); producer after .dec (52)
    ("  if (threadIdx.x < 32) {\n    int run = 0;\n    int incl_l = 0, n_l = 0, seq_l = 0;\n",
     "  if (p.dbg && threadIdx.x == Cfg<L, FLAT>::NT) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 50] = gtime();\n"
     "  if (threadIdx.x < 32) {\n    int run = 0;\n    int incl_l = 0, n_l = 0, seq_l = 0;\n", 1),
    ("  __syncthreads();\n  if (threadIdx.x == 0) FPRO_STAMP(40);\n",
     "  __syncthreads();\n  if (threadIdx.x == 0) FPRO_STAMP(40);\n"
     "  if (p.dbg && threadIdx.x == Cfg<L, FLAT>::NT) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 51] = gtime();\n", 1),
    ("      if constexpr (OKC_SETMAXNREG) asm volatile(\"setmaxnreg.dec.sync.aligned.u32 %0;\\n\" ::\"n\"(Cfg<L, FLAT>::PREGS));\n      run_segments(std::integral_constant<int, 1>());\n",
     "      if constexpr (OKC_SETMAXNREG) asm volatile(\"setmaxnreg.dec.sync.aligned.u32 %0;\\n\" ::\"n\"(Cfg<L, FLAT>::PREGS));\n"
     "      if (p.dbg && threadIdx.x == Cfg<L, FLAT>::NT) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 52] = gtime();\n"
     "      run_segments(std::integral_constant<int, 1>());\n", 1),
    ("      if constexpr (OKC_SETMAXNREG) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 %0;\\n\" ::\"n\"(Cfg<L, FLAT>::CREGS));\n      run_segments(std::integral_constant<int, 2>());\n",
     "      if constexpr (OKC_SETMAXNREG) asm volatile(\"setmaxnreg.inc.sync.aligned.u32 %0;\\n\" ::\"n\"(Cfg<L, FLAT>::CREGS));\n"
     "      if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 53] = gtime();\n"
     "      run_segments(std::integral_constant<int, 2>());\n", 1),
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
        c = src.count(old)
        if c == 0 and "setmaxnreg.dec" in old:   # ly_apre2 moves the producer's dec into its pre-issue block
            continue
        assert c == n, (c, old[:60])
        src = src.replace(old, new)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src)
    py_compile.compile(str(out), doraise=True)
    print(out, hashlib.sha256(src.encode()).hexdigest())


if __name__ == "__main__":
    main()
