#!/usr/bin/env python3
"""ly_v1st: per-CTA globaltimer stamps in the verify 1 (T = 8) unpooler slice sk2 and its finish, on ly_v1un.

Instrument only, never a candidate. Input: muse_decode_layer/cand/ly_pmt_v1un.py, written by
`tools/dl_ly_pmt_patch.py --base muse_decode_layer/cand/ly_v1un.py --pin 88fbae86... --xld __ldca
--out muse_decode_layer/cand/ly_pmt_v1un.py` (the MoE TINY DBG stamps and the finish TLON stamps of the decode 1
tail). This tool adds:

- the TINY2 MoE build (T = 8) with -DOKC_DEBUG=1, so the same MoE stamps exist at verify 1 (taken only with the
  fourth scratch tensor, as in ly_pmt);
- stamps in bfp_SK2_SRC's sk2_kernel, non-fused instantiations only, written only when the kernel's dbg argument is
  non-null. The host `sk2` passes the stamp buffer set by the new export `sk2_set_tlu(ptr)` as dbg, and only for the
  T = 8 slice's shape (M = 8, N = 1024); with the buffer unset (0) dbg stays what the base passes (null: the base's
  dummy tensor is below the trace size), so the plain graph takes no stamp. SK2_DIAG stays 0, so dbg's diagnostic
  uses stay compiled out.

Slots (32 int64 per CTA, CTA = blockIdx.x):
  0 entry (tid 0)   1 prologue done: pre-wait weight stages and the escape table issued, barriers initialised (tid 0)
  2 after griddepcontrol.wait (tid 0)   3 x tiles of the pre-issued stages issued (tid 0)
  4 producer lane: every refill stage issued (before its launch_dependents)   5 escape warp 0: table landed
  6 + s consumer warp 0 saw stage s full (s < 4)   10 + s consumer warp NW - 1 saw stage s full
  14 / 15 consumer warp 0 / NW - 1 left the stage loop   16 + e escape warp e done (after its cbar arrive)
  20 epilogue: tid 0 after the cbar wait   21 after the first named barrier   22 after the second (partials in smem)
  23 tid 0's output store done   24 SM id   25 escape count   26 + s escape warp 0 saw stage s full (s < 4)
The finish program's slot 5 (unused by ly_pmt) gets its SM id.
No fence, barrier, wait, trigger or launch attribute moves.

    python3 tools/dl_ly_v1st_patch.py [--src muse_decode_layer/cand/ly_pmt_v1un.py] [--out muse_decode_layer/cand/ly_v1st.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_pmt_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_v1st.py"
PIN = "1041126b57bbae37749227c0929b21391b6b14846368137bc60af85cdbb34890"
if "--src" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
DIAG = "--diag" in sys.argv   # also honour sk2's diagnostic bits (dg) in the stamped launch only, set by sk2_set_tdg(n)
START = 'bfp_SK2_SRC = r"""\n'

TLU = "if (!FUSE && dbg != nullptr) dbg[(size_t)blockIdx.x * 32 + "
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
    ("      mbar_init(qbar, 1);\n      asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");\n    } else {\n",
     "      mbar_init(qbar, 1);\n      asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");\n      SK2_TLU(1);\n    } else {\n"),
    ("  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n",
     "  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n"
     "  if (tid == 0) SK2_TLU(2);\n"),
    ("      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n  }\n",
     "      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n"
     "    if (tid == 0) SK2_TLU(3);\n  }\n"),
    ("      if (pdl) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");  // last weight stage issued\n",
     "      SK2_TLU(4);\n"
     "      if (pdl) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");  // last weight stage issued\n"),
    ("      mbar_wait_sleep(pbar, 0);\n      const int cnt = ptab_s[0].x;\n",
     "      mbar_wait_sleep(pbar, 0);\n      const int cnt = ptab_s[0].x;\n"
     "      if (lane == 0 && ew == 0) { SK2_TLU(5); SK2_TLV(25, cnt); }\n"),
    ("        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if ((st % NESC) != ew) {",
     "        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if (lane == 0 && ew == 0 && st < 4) SK2_TLU(26 + st);\n        if ((st % NESC) != ew) {"),
    ("    mbar_arrive(cbar);\n    __syncwarp();\n",
     "    mbar_arrive(cbar);\n    if (COMP && lane == 0 && ew < 4) SK2_TLU(16 + ew);\n    __syncwarp();\n"),
    ("    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();\n",
     "    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();\n"
     "    if (lane == 0 && st < 4 && (warp == 0 || warp == NW - 1)) SK2_TLU((warp == 0 ? 6 : 10) + st);\n"),
    ("  if (pdl) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");   // the next kernel may start its prologue\n",
     "  if (lane == 0 && (warp == 0 || warp == NW - 1)) SK2_TLU(warp == 0 ? 14 : 15);\n"
     "  if (pdl) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");   // the next kernel may start its prologue\n"),
    ("  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done\n",
     "  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done\n"
     "  if (tid == 0) SK2_TLU(20);\n"),
    ("  if (trace && tid == 0) tr[9] = gtime();\n", "  if (trace && tid == 0) tr[9] = gtime();\n  if (tid == 0) SK2_TLU(21);\n"),
    ("  if (trace && tid == 0) tr[10] = gtime();\n", "  if (trace && tid == 0) tr[10] = gtime();\n  if (tid == 0) SK2_TLU(22);\n"),
    ("  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {",
     "  if (tid == 0) SK2_TLU(23);\n  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {"),
    ("  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n",
     "  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n"
     "  if (g_sk2_tlu != nullptr && M == 8 && N == 1024) dbg = g_sk2_tlu;   // v1st: the T = 8 slice only\n"),
]
# --diag: the stamped slice launch (and only it) passes dg = g_sk2_tdg and the kernel honours dg when dbg is set, so
# the bits 2 (consumers skip their math) and 4 (no x: the stages carry weights only) can time parts of the slice; the
# output is then wrong by design. The plain graph (dbg null) keeps dgv = 0.
DIAG_EDITS = [
    ("  const int dgv = SK2_DIAG ? dg : 0;", "  const int dgv = (SK2_DIAG || (!FUSE && dbg != nullptr)) ? (dg & ~32) : 0;"),
    ("void sk2_set_tlu(int64_t p) { g_sk2_tlu = reinterpret_cast<unsigned long long*>(p); }\n",
     "void sk2_set_tlu(int64_t p) { g_sk2_tlu = reinterpret_cast<unsigned long long*>(p); }\n"
     "static int g_sk2_tdg = 0;\nvoid sk2_set_tdg(int64_t n) { g_sk2_tdg = (int)n; }\n"),
    ("  if (g_sk2_tlu != nullptr && M == 8 && N == 1024) dbg = g_sk2_tlu;   // v1st: the T = 8 slice only\n",
     "  if (g_sk2_tlu != nullptr && M == 8 && N == 1024) { dbg = g_sk2_tlu; dg = g_sk2_tdg; }   // v1st: the T = 8 slice only\n"),
]
FILE_EDITS = [
    ('def _pmt_gt(v):\n    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [v], dtype=tl.int64, is_pure=False, pack=1)\n',
     'def _pmt_gt(v):\n    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [v], dtype=tl.int64, is_pure=False, pack=1)\n\n\n'
     '@triton.jit\ndef _v1st_smid(v):\n    return tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)\n', 1),
    ("        tl.store(tlp + 0, _pmt_gt(t))\n",
     "        tl.store(tlp + 0, _pmt_gt(t))\n        tl.store(tlp + 5, _v1st_smid(t).to(tl.int64))\n", 1),
    ("void sk2_set_pdl(int64_t n);\n\"\"\"\n\nbfp_w_rows", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_tlu(int64_t p);\n\"\"\"\n\nbfp_w_rows", 1),
    ('"sk2", "gq", "sk2f", "sk2_set_pdl"]', '"sk2", "gq", "sk2f", "sk2_set_pdl", "sk2_set_tlu"]', 1),
    ('MOE_TINY2_DEFINES += ("@moe_helper0",)\n',
     'MOE_TINY2_DEFINES += ("@moe_helper0",)\nMOE_TINY2_DEFINES += ("-DOKC_DEBUG=1",)   # v1st: DBG stamps at T = 8, taken only with a fourth scratch tensor\n', 1),
]
HEADER = ("# ly_v1st: ly_v1un with per-CTA globaltimer stamps in the T = 8 MoE, unpooler slice sk2 and finish, off in the "
          "plain graph (tools/dl_ly_pmt_patch.py + tools/dl_ly_v1st_patch.py). Instrument, not a candidate.\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index(START) + len(START)
    j = text.index('\n"""\n', i)
    sk = text[i:j]
    for old, new in SK2_EDITS:
        sk = sub(sk, old, new, 1, "sk2")
    if DIAG:
        for old, new in DIAG_EDITS:
            sk = sub(sk, old, new, 1, "diag")
    text = text[:i] + sk + text[j:]
    for old, new, n in FILE_EDITS:
        text = sub(text, old, new, n, "file")
    if DIAG:
        text = sub(text, 'void sk2_set_tlu(int64_t p);\n', 'void sk2_set_tlu(int64_t p);\nvoid sk2_set_tdg(int64_t n);\n', 1, "diag decl")
        text = sub(text, '"sk2_set_pdl", "sk2_set_tlu"]', '"sk2_set_pdl", "sk2_set_tlu", "sk2_set_tdg"]', 1, "diag fn")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
