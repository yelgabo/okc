#!/usr/bin/env python3
"""Pooler + router timeline: writes muse_decode_layer/cand/ly_prtl.py from ly_micro2.py (0bc2bd8b, checked).

Per-CTA globaltimer stamps in the fused pooler + router sk2 kernel (FUSE = true): sk2_kernel_gle at T <= 8 (BF16
pooler + interleaved FP32 router, gle extension) and sk2 FUSE at T = 16 / 32 (BF16 hi / lo router, bf16_proj
extension). The stamps sit in bfp_SK2_SRC under `if (FUSE && dbg != nullptr)`; the fused launches pass the host
pointer g_sk2_tlb (set by the new export sk2_set_tlb, null by default) as the kernel's dbg argument, which the fused
dispatch passed as nullptr before. Non-fused sk2 instantiations compile the stamps out. Nothing is stamped inside
the consumers' stage loop; fences, barriers, griddepcontrol, the ring protocol and the router counter are unchanged.

Slots (64 int64 per CTA, CTA = blockIdx.x):
  0 entry (tid 0)   1 first pre-wait weight issue   2 pre-wait issues done   3 after griddepcontrol.wait (tid 0)
  4 x tiles of the pre-issued stages issued (tid 0)   5 router counter: before the atom (part stored)
  6 after the atom's barrier   7 last-arriver router output stored (tid 0, last quarter only)
  8 epilogue start (after the cbar wait and barrier)   9 reduction buffer written (after the barrier)
  10 final store done (tid 0)   11 SM id   12 producer: last weight stage issued
  16 + s (s < 20): producer issue of stage s (pre-wait stages in the prologue)
  36 + s (s < 20): escape warp 0 saw stage s land (its full-barrier wait returned; never earlier than the landing)
  56 + w (w < 8): consumer warp w left the stage loop (pooler stages and, at T <= 8, the interleaved router stages)

    python3 tools/dl_ly_prtl_patch.py [--src FILE] [--out muse_decode_layer/cand/ly_prtl.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_micro2.py"
DST = ROOT / "muse_decode_layer/cand/ly_prtl.py"
PIN = "0bc2bd8bdeecd7514ab259bd8f2b5b6b66cef88734f583c5ccaf466f1075cacd"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
if "--src" in sys.argv:  # instrument a candidate built on ly_micro2 (its SHA is printed, not pinned)
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
START = 'bfp_SK2_SRC = r"""\n'
END = '\n"""\n'

K_EDITS = [
    ("static int g_sk2_pdl = 0;\nvoid sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n",
     "static int g_sk2_pdl = 0;\nvoid sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n"
     "// prtl: the fused pooler + router launches pass this as dbg (timeline stamps); null = no stamps\n"
     "static unsigned long long* g_sk2_tlb = nullptr;\n"
     "void sk2_set_tlb(int64_t p) { g_sk2_tlb = reinterpret_cast<unsigned long long*>(p); }\n"
     "__device__ __forceinline__ unsigned long long sk2_gt() { unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }\n"
     "#define SK2_TLS(k) do { if (FUSE && dbg != nullptr) dbg[(size_t)blockIdx.x * 64 + (k)] = sk2_gt(); } while (0)\n"),
    ("  if (trace && threadIdx.x == 0) tr[0] = t_start;\n",
     "  if (trace && threadIdx.x == 0) tr[0] = t_start;\n  if (threadIdx.x == 0) SK2_TLS(0);\n"),
    ("    const long long pcx_t0 = clock64();\n    for (int st = 0; st < P; ++st) {\n",
     "    const long long pcx_t0 = clock64();\n    SK2_TLS(1);\n    for (int st = 0; st < P; ++st) {\n"),
    ("      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }\n",
     "      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }\n      SK2_TLS(16 + st);\n"),
    ("    if constexpr (CL == 1) {\n      // PCX hoist: every barrier init",
     "    SK2_TLS(2);\n    if constexpr (CL == 1) {\n      // PCX hoist: every barrier init"),
    ("  if constexpr (CL == 1) {   // PCX hoist: the x tiles of the pre-issued stages, right after the wait\n"
     "    if (tid == 0 && !(dgv & 4))\n"
     "      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n"
     "  }\n",
     "  if (tid == 0) SK2_TLS(3);\n"
     "  if constexpr (CL == 1) {   // PCX hoist: the x tiles of the pre-issued stages, right after the wait\n"
     "    if (tid == 0 && !(dgv & 4))\n"
     "      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n"
     "  }\n  if (tid == 0) SK2_TLS(4);\n"),
    ("        if (trace) tr[16 + 3 * st] = gtime();\n",
     "        if (trace) tr[16 + 3 * st] = gtime();\n        if (st < 20) SK2_TLS(16 + st);\n"),
    ('      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued\n',
     '      SK2_TLS(12);\n'
     '      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued\n'),
    ("          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n          __syncwarp();\n",
     "          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n          if (ew == 0 && lane == 0 && st < 20) SK2_TLS(36 + st);\n          __syncwarp();\n"),
    ("        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if ((st % NESC) != ew) {",
     "        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if (ew == 0 && lane == 0 && st < 20) SK2_TLS(36 + st);\n        if ((st % NESC) != ew) {"),
    ('  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue\n',
     '  if (lane == 0) SK2_TLS(56 + warp);\n'
     '  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue\n'),
    ("    __shared__ int s_rlast;\n    if (tid == 0) {\n",
     "    __shared__ int s_rlast;\n    if (tid == 0) SK2_TLS(5);\n    if (tid == 0) {\n"),
    ("    named_bar(1, NC);\n    if (s_rlast) {\n",
     "    named_bar(1, NC);\n    if (tid == 0) SK2_TLS(6);\n    if (s_rlast) {\n"),
    ("        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;\n      }\n",
     "        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;\n      }\n      if (tid == 0) SK2_TLS(7);\n"),
    ("  if (trace && tid == 0) tr[9] = gtime();\n", "  if (trace && tid == 0) tr[9] = gtime();\n  if (tid == 0) SK2_TLS(8);\n"),
    ("  if (trace && tid == 0) tr[10] = gtime();\n", "  if (trace && tid == 0) tr[10] = gtime();\n  if (tid == 0) SK2_TLS(9);\n"),
    ("  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32)",
     "  if (trace && tid == 0) tr[15] = gtime();\n"
     "  if (tid == 0) SK2_TLS(10);\n"
     "  if (FUSE && dbg != nullptr && tid == 0) { unsigned smid; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(smid)); dbg[(size_t)blockIdx.x * 64 + 11] = smid; }\n"
     "  if ((dgv & 32)"),
]

# whole-file edits: the fused dispatch passes g_sk2_tlb, and both extensions export sk2_set_tlb
F_EDITS = [
    ("M, 4096, 8192, 0, nullptr, ", "M, 4096, 8192, 0, g_sk2_tlb, ", 2),
    ("void sk2_set_pdl(int64_t n);\n\"\"\"\n", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_tlb(int64_t p);\n\"\"\"\n", 1),
    ('functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],',
     'functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl", "sk2_set_tlb"],', 1),
    (' torch::Tensor x32, torch::Tensor w32, int64_t wpf);\\nvoid sk2_set_pdl(int64_t n);\\n")',
     ' torch::Tensor x32, torch::Tensor w32, int64_t wpf);\\nvoid sk2_set_pdl(int64_t n);\\nvoid sk2_set_tlb(int64_t p);\\n")', 1),
    ('functions=["gle_o", "gle_pr", "sk2_set_pdl"],', 'functions=["gle_o", "gle_pr", "sk2_set_pdl", "sk2_set_tlb"],', 1),
]


def patch(whole, k_edits=K_EDITS, f_edits=F_EDITS):
    assert whole.count(START) == 1
    head, rest = whole.split(START, 1)
    k = rest.index(END)
    ks, tail = rest[:k], rest[k:]
    for old, new in k_edits:
        n = ks.count(old)
        if n != 1:
            sys.exit(f"kernel anchor found {n} times, 1 expected: {old[:100]!r}")
        ks = ks.replace(old, new)
    out = head + START + ks + tail
    for old, new, cnt in f_edits:
        n = out.count(old)
        if n != cnt:
            sys.exit(f"file anchor found {n} times, {cnt} expected: {old[:100]!r}")
        out = out.replace(old, new)
    return out


def main():
    raw = SRC.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"{SRC.name} is {sha}, not {PIN}")
    print("base", SRC.name, sha)
    out = patch(raw.decode())
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
