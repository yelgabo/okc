#!/usr/bin/env python3
"""ly_atst: per-CTA globaltimer stamps in the layer's attention (ATT7 attn_kernel) and in the qkv / o sk2 launches
at T = 8, on ly_v1un. Instrument only, never a candidate.

Attention (ATT7_CUDA_SRC): OKC_DBGON 1 and OKC_MRGDBG 1 (the kernel's own DBG stamps, thread 0, written only when
p.dbg is non-null), DBGSTRIDE 32, and five more slots. _pdl8_decode_attention7 passes the module global _ATST_DBG
(None outside the stamped graph) as _launch7's dbg. Slots per CTA (blockIdx.x for the per-request grid):
  0 entry   20 pre-wait work done (page-index copies issued)   16 griddepcontrol.wait returned   19 SM id
  1 page-index group landed + CTA barrier   8 K/V copies issued   17 q / new k,v / constants group landed
  9 RoPE table   10 q pass   11 new k,v pass (cache rows written)   2 new rows patched, consumer state next
  12 tile 0 landed   13 tile 0 consumed   3 merge start   7 reds issued (thread 0)   6 CTA barrier after the reds
  18 ticket atomic returned (thread 0)   4 grid barrier passed   5 finalize slice stored
sk2 (bfp_SK2_SRC, also built into the narrow qkv extension): stamps go to the buffer set by sk2_set_tlu(ptr) for
M = 8 only: qkv (K = 8192, N = 2304) at ptr, o (K = 1024) at ptr + 1024 * 32; 32 int64 per CTA:
  0 entry   24 SM id   2 griddepcontrol.wait returned (PDL launches only)   23 thread 0's output store done.
No fence, barrier, wait, trigger or launch attribute moves.

    python3 tools/dl_ly_atst_patch.py [--out muse_decode_layer/cand/ly_atst.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_atst.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

DBGW = "p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 19]"
ATT_EDITS = [
    ("#define OKC_DBGON 0", "#define OKC_DBGON 1"),
    ("#define OKC_MRGDBG 0", "#define OKC_MRGDBG 1"),
    ("#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)", "#define DBGSTRIDE (OKC_SEGDBG ? 64 : 32)"),
    ("\n\n  pdl_wait();   // qkv (q, new k / v, gate)", "\n  DBG(20);\n\n  pdl_wait();   // qkv (q, new k / v, gate)"),
    ("  if (tid == 0) FPRO_STAMP(44);\n",
     "  if (tid == 0) FPRO_STAMP(44);\n  DBG(16);\n"
     "  if (OKC_DBGON && p.dbg && threadIdx.x == 0) { unsigned smid_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(smid_)); "
     + DBGW + " = smid_; }\n"),
    ("  }\n  // ---- 3. while they load", "  }\n  DBG(17);\n  // ---- 3. while they load"),
    ("      else old = atomicAdd(c, mine);\n", "      else old = atomicAdd(c, mine);\n      DBG(18);\n"),
]

TLU = "if (dbg != nullptr) dbg[(size_t)blockIdx.x * 32 + "
SK2_EDITS = [
    ("void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n",
     "void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n"
     "static unsigned long long* g_sk2_tlu = nullptr;\n"
     "void sk2_set_tlu(int64_t p) { g_sk2_tlu = reinterpret_cast<unsigned long long*>(p); }\n"
     "#define SK2_TLU(k) do { " + TLU + "(k)] = gtime(); } while (0)\n"
     "#define SK2_TLV(k, v) do { " + TLU + "(k)] = (unsigned long long)(v); } while (0)\n"),
    ("  if (trace && threadIdx.x == 0) tr[0] = t_start;\n",
     "  if (trace && threadIdx.x == 0) tr[0] = t_start;\n"
     "  if (threadIdx.x == 0) { SK2_TLU(0); unsigned smid_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(smid_)); SK2_TLV(24, smid_); }\n"),
    ("  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n",
     "  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n"
     "  if (pdl && tid == 0) SK2_TLU(2);\n"),
    ("  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {",
     "  if (tid == 0) SK2_TLU(23);\n  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {"),
    ("  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n",
     "  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n"
     "  if (g_sk2_tlu != nullptr && M == 8 && K == 8192 && N == 2304) dbg = g_sk2_tlu;\n"
     "  else if (g_sk2_tlu != nullptr && M == 8 && K == 1024) dbg = g_sk2_tlu + 1024 * 32;\n"),
]
FILE_EDITS = [
    ("void sk2_set_pdl(int64_t n);\n\"\"\"\n\nbfp_w_rows", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_tlu(int64_t p);\n\"\"\"\n\nbfp_w_rows", 1),
    ('"sk2", "gq", "sk2f", "sk2_set_pdl"]', '"sk2", "gq", "sk2f", "sk2_set_pdl", "sk2_set_tlu"]', 1),
    ("+'void sk2_set_pdl(int64_t n);'", "+'void sk2_set_pdl(int64_t n);\\nvoid sk2_set_tlu(int64_t p);'", 1),
    ("functions=['sk2','sk2_set_pdl']", "functions=['sk2','sk2_set_pdl','sk2_set_tlu']", 1),
]
HEADER = ("# ly_atst: ly_v1un with per-CTA globaltimer stamps in the attention and the T = 8 qkv / o sk2 launches, off in "
          "the plain graph (tools/dl_ly_atst_patch.py). Instrument, not a candidate.\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def edit_block(text, start, edits, what):
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    blk = text[i:j]
    for old, new in edits:
        blk = sub(blk, old, new, 1, what)
    return text[:i] + blk + text[j:]


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    text = edit_block(text, 'ATT7_CUDA_SRC = r"""\n', ATT_EDITS, "att7")
    text = edit_block(text, 'bfp_SK2_SRC = r"""\n', SK2_EDITS, "sk2")
    for old, new, n in FILE_EDITS:
        text = sub(text, old, new, n, "file")
    k = text.index("def _pdl8_decode_attention7(")
    call = "attention_state=state, prefetch=prefetch, **pdl)"
    m = text.index(call, k)
    text = text[:m] + "attention_state=state, prefetch=prefetch, dbg=_ATST_DBG, **pdl)" + text[m + len(call):]
    text = text[:k] + "_ATST_DBG = None   # atst: the stamp tensor of the instrumented graph\n\n\n" + text[k:]
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
