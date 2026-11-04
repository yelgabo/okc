#!/usr/bin/env python3
"""ly_ast: per-CTA globaltimer stamps of the decode 1 attention kernel inside the layer, on ly_v1un.

Instrument only, never a candidate. Edits:

- ATT7_CUDA_SRC: OKC_DBGON 1 (the file's own DBG(k) slots, written only when the launch's dbg tensor is non-null),
  DBGSTRIDE 32, and four new slots: 15 after griddepcontrol.wait (+ launch_dependents), 16 the q / new k,v / constants
  group visible (IDXF wait), 17 thread 0's own K/V copies landed (cp.async.wait_group on its own groups only; --kvw),
  18 SM id. Existing slots: 0 entry, 1 page indices landed and the speculated length checked, 8 K/V copies issued,
  9 RoPE table, 10 q pass, 11 new k / v pass (cache row written), 2 new rows patched into the stage, 12 the tile's
  stage visible to the CTA, 13 first tile's math, 3 tile loop done, 4 ticket atomic returned (CTA barrier),
  14 last CTA: finalize start, 5 last CTA: finalize stores issued, 6 = 1 on the last CTA.
- _pdl8_decode_attention7 / _decode_attention7 pass dbg=_AST_DBG (module global, None in the plain graph).
- bfp_SK2_SRC: slots 0 entry, 2 after griddepcontrol.wait, 23 end (rank 0 store or peer push) for sk2 launches with
  M == 1 whose (N, K) matches one of two registered shapes (sk2_set_ast(i, N, K, ptr)); qkv and o at T = 1.

No fence, barrier, wait, trigger or launch attribute moves.

    python3 tools/dl_ly_ast_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_ast.py] [--kvw]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_ast.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
KVW = "--kvw" in sys.argv

IDXF_WAIT = ("  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread\n"
             "    cp_wait<STAGES - 1>();\n    csync<WS, NT>();\n  }\n")
ATT_EDITS = [
    ("#define OKC_DBGON 0", "#define OKC_DBGON 1"),
    ("#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)", "#define DBGSTRIDE (OKC_SEGDBG ? 64 : 32)"),
    ("  if (tid == 0) FPRO_STAMP(44);\n",
     "  if (tid == 0) FPRO_STAMP(44);\n  DBG(15);\n"
     "  if (p.dbg && tid == 0) { unsigned smid_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(smid_));"
     " p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 18] = smid_; }\n"),
    (IDXF_WAIT, IDXF_WAIT + "  DBG(16);\n" + (
        "  if (p.dbg && tid == 0) { cp_wait<0>(); DBG(17); }\n" if KVW else "")),
]
SK2_EDITS = [
    ("void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n",
     "void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n"
     "static unsigned long long* g_ast_p[2] = {nullptr, nullptr};\n"
     "static int64_t g_ast_n[2] = {0, 0}, g_ast_k[2] = {0, 0};\n"
     "void sk2_set_ast(int64_t i, int64_t n, int64_t k, int64_t p) { g_ast_p[i] = reinterpret_cast<unsigned long long*>(p);"
     " g_ast_n[i] = n; g_ast_k[i] = k; }\n"
     "#define AST_T(k) do { if (!FUSE && dbg != nullptr) dbg[(size_t)blockIdx.x * 32 + (k)] = gtime(); } while (0)\n"),
    ("  if (trace && threadIdx.x == 0) tr[0] = t_start;\n",
     "  if (trace && threadIdx.x == 0) tr[0] = t_start;\n  if (threadIdx.x == 0) AST_T(0);\n"),
    ("  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n",
     "  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n"
     "  if (tid == 0) AST_T(2);\n"),
    ("  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n",
     "  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n"
     "  for (int i_ = 0; i_ < 2; ++i_)\n"
     "    if (g_ast_p[i_] != nullptr && M == 1 && N == g_ast_n[i_] && K == g_ast_k[i_]) dbg = g_ast_p[i_];   // ast: T = 1 qkv / o\n"),
]
SK2_END = ("  if (trace && tid == 0) tr[15] = gtime();\n", "  if (tid == 0) AST_T(23);\n  if (trace && tid == 0) tr[15] = gtime();\n")
FILE_EDITS = [
    ("void sk2_set_pdl(int64_t n);\n\"\"\"\n\nbfp_w_rows", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_ast(int64_t i, int64_t n, int64_t k, int64_t p);\n\"\"\"\n\nbfp_w_rows", 1),
    ('"sk2", "gq", "sk2f", "sk2_set_pdl"]', '"sk2", "gq", "sk2f", "sk2_set_pdl", "sk2_set_ast"]', 1),
    ("attention_state=state, prefetch=prefetch, **pdl)", "attention_state=state, prefetch=prefetch, dbg=_AST_DBG, **pdl)", 2),
    ("ATT_HAS_PDL = True", "_AST_DBG = None   # ast: attention stamp buffer (instrumented graph only)\nATT_HAS_PDL = True", 1),
]
HEADER = ("# ly_ast: ly_v1un with per-CTA globaltimer stamps in the decode 1 attention kernel and the T = 1 qkv / o sk2 "
          "launches, off in the plain graph (tools/dl_ly_ast_patch.py). Instrument, not a candidate.\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def region(text, start):
    i = text.index(start) + len(start)
    return i, text.index('\n"""\n', i)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i, j = region(text, 'ATT7_CUDA_SRC = r"""\n')
    at = text[i:j]
    for old, new in ATT_EDITS:
        at = sub(at, old, new, 1, "att")
    text = text[:i] + at + text[j:]
    i, j = region(text, 'bfp_SK2_SRC = r"""\n')
    sk = text[i:j]
    for old, new in SK2_EDITS:
        sk = sub(sk, old, new, 1, "sk2")
    sk = sub(sk, *SK2_END, 2, "sk2 end")   # (the cluster peers' push end and rank 0's store end)
    text = text[:i] + sk + text[j:]
    for old, new, n in FILE_EDITS:
        text = sub(text, old, new, n, "file")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
