#!/usr/bin/env python3
"""Decode 1 qkv_gate timeline: writes muse_decode_layer/cand/ly_qtl.py from ly_v1un.py (88fbae86, checked).

Per-CTA globaltimer stamps in the qkv sk2 kernel at T = 1 (`qkv8_r176`: 24 rows per CTA, 96 CTAs, 4 K-warps x 2 units,
3 n8 tiles, 6-slot ring). The bf16_proj extension's host `sk2()` passes the buffer set by the new export
`sk2_set_qtl` as the kernel's dbg argument only when N == 2304 and M == 1; the stamps are compiled only into the
qkv geometry (NT 3, WSK 4, UPW 2, not FUSE). Nothing is stamped inside the consumers' stage loop. Fences, barriers,
griddepcontrol and the ring protocol are unchanged. A diagnostic, never a candidate.

Slots (128 int64 per CTA, CTA = blockIdx.x):
  0 entry (tid 0)   1 pre-wait weight issues done (tid 0)   2 barrier inits + escape table issued (tid 0)
  3 after the wait and the pre-issued stages' x tiles (tid 0)   4 SM id   5 producer: last weight stage issued
  6 epilogue start (after the cbar wait and the named barrier)   7 final store done (tid 0)
  16 + s (s < 16): weight issue of stage s (prologue for s < P, the producer lane after)
  32 + s (s < 16): escape warp 0 saw stage s land (its full-barrier wait returned; never earlier than the landing)
  48 + w (w < 4): consumer warp w left the stage loop
  64 + s (s < 16): consumer warp 0's full-barrier wait for stage s returned (inside the stage loop: diagnostic SASS)
  80 + s (s < 16): consumer warp 0 done with stage s (after its slot hand-back)
  96 + e (e < 4): escape warp e arrives on cbar (all its stages walked)
  100: consumers past the cbar wait (tid 0, before the named barrier)

    python3 tools/dl_ly_qtl_patch.py [--src FILE] [--out muse_decode_layer/cand/ly_qtl.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_qtl.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
if "--src" in sys.argv:  # instrument a candidate built on ly_v1un (its SHA is printed, not pinned)
    SRC = ROOT / sys.argv[sys.argv.index("--src") + 1]
    PIN = None
START = 'bfp_SK2_SRC = r"""\n'
END = '\n"""\n'

QTL_COND = "(!FUSE && NT == 3 && WSK == 4 && UPW == 2 && dbg != nullptr)"
K_EDITS = [
    ("static int g_sk2_pdl = 0;\nvoid sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n",
     "static int g_sk2_pdl = 0;\nvoid sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n"
     "// qtl: the qkv_gate launch at M = 1 passes this as dbg (timeline stamps); null = no stamps\n"
     "static unsigned long long* g_sk2_qtl = nullptr;\n"
     "void sk2_set_qtl(int64_t p) { g_sk2_qtl = reinterpret_cast<unsigned long long*>(p); }\n"
     "__device__ __forceinline__ unsigned long long sk2_qgt() { unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }\n"
     f"#define SK2_QTL(k) do {{ if {QTL_COND} dbg[(size_t)blockIdx.x * 128 + (k)] = sk2_qgt(); }} while (0)\n"),
    ("  if (trace && threadIdx.x == 0) tr[0] = t_start;\n",
     "  if (trace && threadIdx.x == 0) tr[0] = t_start;\n  if (threadIdx.x == 0) SK2_QTL(0);\n"),
    ("      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }\n",
     "      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }\n      SK2_QTL(16 + st);\n"),
    ("    if constexpr (CL == 1) {\n      // PCX hoist: every barrier init",
     "    SK2_QTL(1);\n    if constexpr (CL == 1) {\n      // PCX hoist: every barrier init"),
    ("  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;\n",
     "  if (tid == 0) SK2_QTL(2);\n  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;\n"),
    ("  if constexpr (CL > 1) asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\" ::: \"memory\");   // inits fenced above\n",
     "  if (tid == 0) SK2_QTL(3);\n"
     "  if constexpr (CL > 1) asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\" ::: \"memory\");   // inits fenced above\n"),
    ("        if (trace) tr[16 + 3 * st] = gtime();\n",
     "        if (trace) tr[16 + 3 * st] = gtime();\n        if (st < 16) SK2_QTL(16 + st);\n"),
    ('      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued\n',
     '      SK2_QTL(5);\n'
     '      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued\n'),
    ("        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if ((st % NESC) != ew) {",
     "        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);\n        if (ew == 0 && lane == 0 && st < 16) SK2_QTL(32 + st);\n        if ((st % NESC) != ew) {"),
    ('  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue\n',
     '  if (lane == 0 && warp < 4) SK2_QTL(48 + warp);\n'
     '  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue\n'),
    ("    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();\n",
     "    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();\n"
     "    if (warp == 0 && lane == 0 && st < 16) SK2_QTL(64 + st);\n"),
    ("    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();\n",
     "    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();\n    if (warp == 0 && lane == 0 && st < 16) SK2_QTL(80 + st);\n"),
    ("    mbar_arrive(cbar);\n    __syncwarp();\n",
     "    if (lane == 0) SK2_QTL(96 + (warp - (NW + 1)));\n    mbar_arrive(cbar);\n    __syncwarp();\n"),
    ("  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done\n",
     "  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done\n  if (tid == 0) SK2_QTL(100);\n"),
    ("  if (trace && tid == 0) tr[9] = gtime();\n", "  if (trace && tid == 0) tr[9] = gtime();\n  if (tid == 0) SK2_QTL(6);\n"),
    ("  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32)",
     "  if (trace && tid == 0) tr[15] = gtime();\n"
     "  if (tid == 0) SK2_QTL(7);\n"
     f"  if ({QTL_COND} && tid == 0) {{ unsigned smid; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(smid)); dbg[(size_t)blockIdx.x * 128 + 4] = smid; }}\n"
     "  if ((dgv & 32)"),
    ("  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n",
     "  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n"
     "  if (g_sk2_qtl != nullptr && N == 2304 && M == 1) dbg = g_sk2_qtl;\n"),
]

F_EDITS = [
    ("void sk2_set_pdl(int64_t n);\n\"\"\"\n", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_qtl(int64_t p);\n\"\"\"\n", 1),
    ('functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],',
     'functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl", "sk2_set_qtl"],', 1),
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
