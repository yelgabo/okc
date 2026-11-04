#!/usr/bin/env python3
"""ly_rsl: at decode 1 the unpooler slice holds its codes in registers beside the MoE, launched from the phase A end.

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). Target: the MoE to slice edge at decode 1
(docs/runs-decode-layer.md, "Decode 1 finish and slice launch" and "Decode 1 register slice"). In the base the slice
(cgemv40_kernel<128, 2>, 512 CTAs of 5 warps, 12 KB of codes each) launches 1.3 to 1.6 us after the MoE's trigger at
its phase B end, and at decode 1 full every code byte is read from DRAM after the MoE has ended.

Residency (tools/ly_rslc_occ.cu, measured): a TINY MoE CTA (9 warps at 152 allocated registers) puts 3 warps on one
SM sub-partition, which leaves 1,792 registers there; the warps of a CTA are spread over the four sub-partitions,
so beside it fit one 5-warp CTA at 40 or 32 registers (the base slice), no CTA of 8 or more warps at 40 registers,
and one CTA of 7 warps at 56 registers with up to 15 KB of shared memory.

Change, two parts, both reached only at T = 1:
- MoE (helper0 source, the TINY build only, MOE_TINY_DEFINES is used for T <= 4): the compute threads'
  griddepcontrol.launch_dependents moves from after the phase B join to right after run_phase<false> (DBG(2)) under
  OKC_TRIGA, the ly_trga edit. The slice's griddepcontrol.wait still waits for the whole MoE grid.
- slice (CG_CUDA_SRC; cgemv_fwd runs only for the unpooler slice at m = 1): cgemv_launch<128, 2> with half_n 512
  launches cgrs_kernel, 128 CTAs of 7 warps at 56 registers, one per SM. CTA b owns row pairs 4b..4b+3. A (row,
  quarter) task is cgemv_body's work of one compute warp for one row: lane l takes chunks 32q + l and 32q + l + 128.
  Warp w owns tasks w, w + 7, ...; it loads its first three tasks' codes into registers before
  griddepcontrol.wait and prefetches the other tasks' lines into L2. After the wait x goes to shared memory, then
  every task runs chunk_fma in cgemv_body's order, the same butterfly, and the four quarters and the escape walk
  (warps 3..6, one row pair each, cgemv_body's loop) are summed in cgemv_body's order, so the output is bitwise equal
  to cgemv40_kernel<128, 2>. Nothing the MoE writes is read before the wait (pre-wait reads: the prepare-time stream
  and escape table); trigger at the CTA's end as before.

    python3 tools/dl_ly_rsl_patch.py [--base F] [--pin <sha256>] [--out F] [--tl]

--tl writes the stamped variant (cgrs_kernel<true> launched while the pmt stamp buffer is set; for
tools/dl_ly_pmt_patch.py on top of it), never a candidate.
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SRC = ROOT / arg("--base", "muse_decode_layer/cand/ly_v1un.py")
PIN = arg("--pin", "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145")
TL = "--tl" in sys.argv
HELD = 3   # the task schedule in kernel_src assumes three held tasks per warp
DST = ROOT / arg("--out", "muse_decode_layer/cand/" + ("ly_rsl_tl.py" if TL else "ly_rsl.py"))


def kernel_src(tl):
    """cgrs_kernel source; tl adds globaltimer stamps (slots of tools/ly_box_pmt.py) behind `if (TL)`."""
    S = (lambda s: s) if tl else (lambda s: "")
    head = "template <bool TL>\n__global__ void __maxnreg__(56)\n" if tl else "__global__ void __maxnreg__(56)\n"
    return (
        "// rsl: the m = 1 unpooler slice as 128 CTAs of 7 warps at 56 registers, one per SM beside the TINY MoE CTA\n"
        "// (a TINY CTA leaves 1,792 registers on one SM sub-partition; tools/ly_rslc_occ.cu). CTA b owns row pairs\n"
        "// 4b..4b+3; task (row, quarter q) is one cgemv_body compute warp's work for one row. Warp w takes tasks w, w + 7,\n"
        f"// ..., holds the first {HELD} tasks' codes in registers from before griddepcontrol.wait and prefetches the rest into\n"
        "// L2. FMA order, butterfly, quarter order and escape walk are cgemv_body's: bitwise equal to cgemv40_kernel<128, 2>.\n"
        f"constexpr int RS_CTAS = 128, RS_WARPS = 7, RS_HELD = {HELD}, RS_KK = 4096;\n\n"
        "// task tk = (row tk & 7 of the CTA, quarter tk >> 3): lane l's chunks 32q + l and 32q + l + 128, cgemv_body's order\n"
        "__device__ __forceinline__ void rs_task(uint4 l0, uint4 l1, uint2 c0, uint2 c1, const uint4* xs, int tk, int ln,\n"
        "                                        unsigned base4, float (*part)[5]) {\n"
        "    const int ch = 32 * (tk >> 3) + ln;\n"
        "    float a = 0.f;\n"
        "    chunk_fma(l0, c0.x, c0.y, xs[2 * ch], xs[2 * ch + 1], base4, a);\n"
        "    chunk_fma(l1, c1.x, c1.y, xs[2 * ch + 256], xs[2 * ch + 257], base4, a);\n"
        "    a = warp_sum(a);\n"
        "    if (ln == 0) part[tk & 7][tk >> 3] = a;\n"
        "}\n\n"
        "__device__ __forceinline__ void rs_load(const uint8_t* lo_plane, const uint8_t* cd_plane, int b, int tk, int ln,\n"
        "                                        uint4& l0, uint4& l1, uint2& c0, uint2& c1) {\n"
        "    const int row = 8 * b + (tk & 7), ch = 32 * (tk >> 3) + ln;\n"
        "    const uint4* lr = reinterpret_cast<const uint4*>(lo_plane + (size_t)row * RS_KK);\n"
        "    const uint2* cr = reinterpret_cast<const uint2*>(cd_plane + (size_t)row * (RS_KK / 2));\n"
        "    l0 = __ldcg(lr + ch);\n"
        "    l1 = __ldcg(lr + ch + 128);\n"
        "    c0 = __ldcg(cr + ch);\n"
        "    c1 = __ldcg(cr + ch + 128);\n"
        "}\n\n"
        + head +
        "cgrs_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,\n"
        "            unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {\n"
        "    __shared__ uint4 xs[RS_KK / 8];\n"
        "    __shared__ float part[8][5];\n"
        "    const int b = blockIdx.x, tid = threadIdx.x, ln = tid & 31, wi = tid >> 5;\n"
        + S("    unsigned long long* tl = TL ? g_cg_tlp + (size_t)b * 16 : nullptr;\n"
            "    if (TL && tid == 0) { tl[0] = cg_gt(); unsigned sm; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm)); tl[10] = sm; }\n") +
        "    const uint8_t* lo_plane = stream;\n"
        "    const uint8_t* cd_plane = stream + (size_t)half_n * 2 * RS_KK;\n"
        "    uint4 hl[RS_HELD][2];\n"
        "    uint2 hc[RS_HELD][2];\n"
        "#pragma unroll\n"
        "    for (int j = 0; j < RS_HELD; ++j) {\n"
        "        const int tk = wi + RS_WARPS * j, row = 8 * b + (tk & 7), ch = 32 * (tk >> 3) + ln;\n"
        "        const uint4* lr = reinterpret_cast<const uint4*>(lo_plane + (size_t)row * RS_KK);\n"
        "        const uint2* cr = reinterpret_cast<const uint2*>(cd_plane + (size_t)row * (RS_KK / 2));\n"
        "        hl[j][0] = ld_stream(lr + ch);\n"
        "        hl[j][1] = ld_stream(lr + ch + 128);\n"
        "        hc[j][0] = __ldg(cr + ch);\n"
        "        hc[j][1] = __ldg(cr + ch + 128);\n"
        "    }\n"
        "    // a task's lines: 4 + 4 of low bytes (chunks 32q.., 32q + 128..), 2 + 2 of codes\n"
        "    for (int tk = wi + RS_WARPS * RS_HELD; tk < 32; tk += RS_WARPS) {\n"
        "        if (ln < 12) {\n"
        "            const int row = 8 * b + (tk & 7), q = tk >> 3;\n"
        "            const uint8_t* p = ln < 8\n"
        "                ? lo_plane + (size_t)row * RS_KK + q * 512 + (ln >> 2) * 2048 + (ln & 3) * 128\n"
        "                : cd_plane + (size_t)row * (RS_KK / 2) + q * 256 + ((ln - 8) >> 1) * 1024 + (ln & 1) * 128;\n"
        "            asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(p));\n"
        "        }\n"
        "    }\n"
        "    // warps 3..6 walk the escape tables of row pairs 0..3 after the wait (lane i takes entry i, entry 0 the\n"
        "    // count); before it their lines only go to L2, so no register is held for them across the wait\n"
        "    const int pr = wi - 3;\n"
        "    if (pr >= 0 && ln * 16 < pmax)\n"
        "        asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(table + (size_t)(4 * b + pr) * pmax + ln * 16));\n"
        + S("    if (TL) {\n"
            "        if (tid == 0) tl[1] = cg_gt();\n"
            "#pragma unroll\n"
            "        for (int j = 0; j < RS_HELD; ++j) {\n"
            "            cg_use(hl[j][0].x); cg_use(hl[j][0].y); cg_use(hl[j][0].z); cg_use(hl[j][0].w);\n"
            "            cg_use(hl[j][1].x); cg_use(hl[j][1].y); cg_use(hl[j][1].z); cg_use(hl[j][1].w);\n"
            "            cg_use(hc[j][0].x); cg_use(hc[j][0].y); cg_use(hc[j][1].x); cg_use(hc[j][1].y);\n"
            "        }\n"
            "        __syncthreads();\n"
            "        if (tid == 0) tl[13] = cg_gt();\n"
            "        if (pr == 0 && ln == 0) tl[5] = cg_gt();\n"
            "    }\n") +
        "    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
        + S("    if (TL && tid == 0) tl[2] = cg_gt();\n"
            "    if (TL && pr == 0 && ln == 0) tl[6] = cg_gt();\n") +
        "    const uint4* x4 = reinterpret_cast<const uint4*>(x);\n"
        "    const uint4 xv0 = __ldca(x4 + tid), xv1 = __ldca(x4 + tid + 224);\n"
        "    if (tid < RS_KK / 8 - 448) xs[tid + 448] = __ldca(x4 + tid + 448);\n"
        "    xs[tid] = xv0;\n"
        "    xs[tid + 224] = xv1;\n"
        "    __syncthreads();\n"
        + S("    if (TL && tid == 0) { const unsigned long long g = cg_gt(); tl[3] = g; tl[12] = g; }\n") +
        "    // warp w's tasks: w, w + 7, w + 14 held; w + 21 and (w < 4) w + 28 from L2, each loaded after a held task's\n"
        "    // math has freed its registers, so the L2 latency hides under the next held task\n"
        "    const int2* tb = table + (size_t)(4 * b + (pr < 0 ? 0 : pr)) * pmax;\n"
        "    int2 en = make_int2(0, 0);\n"
        "    if (pr >= 0) en = __ldg(tb + min(ln, pmax - 1));\n"
        "    rs_task(hl[0][0], hl[0][1], hc[0][0], hc[0][1], xs, wi, ln, base4, part);\n"
        "    uint4 al0, al1, bl0, bl1;\n"
        "    uint2 ac0, ac1, bc0, bc1;\n"
        "    rs_load(lo_plane, cd_plane, b, wi + 21, ln, al0, al1, ac0, ac1);\n"
        "    rs_task(hl[1][0], hl[1][1], hc[1][0], hc[1][1], xs, wi + 7, ln, base4, part);\n"
        "    if (wi < 4) rs_load(lo_plane, cd_plane, b, wi + 28, ln, bl0, bl1, bc0, bc1);\n"
        "    rs_task(hl[2][0], hl[2][1], hc[2][0], hc[2][1], xs, wi + 14, ln, base4, part);\n"
        "    rs_task(al0, al1, ac0, ac1, xs, wi + 21, ln, base4, part);\n"
        "    if (wi < 4) rs_task(bl0, bl1, bc0, bc1, xs, wi + 28, ln, base4, part);\n"
        + S("    if (TL && tid == 0) tl[4] = cg_gt();\n") +
        "    if (pr >= 0) {\n"
        "        const int cnt = __shfl_sync(0xFFFFFFFFu, en.x, 0);\n"
        + S("        if (TL && pr == 0 && ln == 0) tl[11] = (unsigned long long)cnt;\n") +
        "        const unsigned ebase = base4 & 0xFFu;\n"
        "        const unsigned short* xh = reinterpret_cast<const unsigned short*>(xs);\n"
        "        float s0 = 0.f, s1 = 0.f;\n"
        "        for (int e = ln; e <= cnt; e += 32) {\n"
        "            const int rsel = en.x >> 16, kk = en.x & 0xFFFF;\n"
        "            const unsigned eb = (unsigned)en.y;\n"
        "            const bool live = e >= 1;\n"
        "            en = __ldg(tb + min(e + 32, pmax - 1));\n"
        "            if (live) {\n"
        "                const int off = min(max((int)((eb >> 8) & 0x7Fu) - (int)ebase, 0), EXP_WINDOW - 1);\n"
        "                const unsigned dec = (eb & 0x80FFu) | ((ebase + off) << 8);\n"
        "                const float delta = __fsub_rn(__uint_as_float(eb << 16), __uint_as_float(dec << 16));\n"
        "                const float d = __fmul_rn(delta, __uint_as_float((unsigned)xh[kk] << 16));\n"
        "                s0 = __fadd_rn(s0, rsel ? 0.f : d);\n"
        "                s1 = __fadd_rn(s1, rsel ? d : 0.f);\n"
        "            }\n"
        "        }\n"
        "        s0 = warp_sum(s0);\n"
        "        s1 = warp_sum(s1);\n"
        "        if (ln == 0) {\n"
        "            part[2 * pr][4] = s0;\n"
        "            part[2 * pr + 1][4] = s1;\n"
        "        }\n"
        + S("        if (TL && pr == 0 && ln == 0) { const unsigned long long g = cg_gt(); tl[7] = g; tl[15] = g; }\n") +
        "    }\n"
        "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
        "    __syncthreads();\n"
        + S("    if (TL && tid == 0) tl[8] = cg_gt();\n") +
        "    if (tid < 4) {\n"
        "        float r0 = 0.f, r1 = 0.f;\n"
        "#pragma unroll\n"
        "        for (int q = 0; q <= 4; ++q) {\n"
        "            r0 += part[2 * tid][q];\n"
        "            r1 += part[2 * tid + 1][q];\n"
        "        }\n"
        "        *reinterpret_cast<unsigned*>(out + 2 * (4 * b + tid)) = pack_bf16x2(r0, r1);\n"
        + S("        if (TL && tid == 0) tl[9] = cg_gt();\n") +
        "    }\n"
        "}\n\n"
    )


CG40_GO = "        cg_go(cgemv40_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);"
RS_ARGS = "s, stream, half_n, table, pmax, base4, x, out);"
RS_GO = ("        if constexpr (NTH == 128 && CPT == 2) {\n"
         "            if (half_n == 4 * RS_CTAS) {\n"
         + ("                if (g_cg_tl) cg_go(cgrs_kernel<true>, RS_CTAS, RS_WARPS * 32, " + RS_ARGS + "\n"
            "                else cg_go(cgrs_kernel<false>, RS_CTAS, RS_WARPS * 32, " + RS_ARGS + "\n" if TL else
            "                cg_go(cgrs_kernel, RS_CTAS, RS_WARPS * 32, " + RS_ARGS + "\n")
         + "                return;\n"
         "            }\n"
         "        }\n")

HELPER_START = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\""
HELPER_EDITS = [
    ('  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n',
     '  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n'
     '#if OKC_TRIGA\n'
     '  // rsl: the slice may launch now; its griddepcontrol.wait still waits for this whole grid\n'
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n'
     '#endif\n'
     '  run_phase<true, BIG>(p, S, xbuf);\n', 1),
    ('  // waits for this whole grid to finish and its writes to be visible)\n'
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n',
     '  // waits for this whole grid to finish and its writes to be visible)\n'
     '#if !OKC_TRIGA\n'
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n'
     '#endif\n', 1),
]
FILE_EDITS = [
    ('MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n',
     'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
     'MOE_TINY_DEFINES += ("-DOKC_TRIGA=1",)  # rsl: T <= 4 triggers the slice at the phase A end\n', 1),
    # the kernel goes into the raw source, in front of cgemv_launch
    ("template <int NTH, int CPT>\nvoid cgemv_launch(const uint8_t* stream,",
     kernel_src(TL) + "template <int NTH, int CPT>\nvoid cgemv_launch(const uint8_t* stream,", 1),
    # its launch rides on the pdl8 edit that turns the cgemv40 launch into cg_go (one-line list literal)
    (repr(CG40_GO)[1:-1], repr(RS_GO + CG40_GO)[1:-1], 1),
]
HEADER = ("# ly_rsl: ly_v1un with the decode 1 unpooler slice as 128 CTAs of 7 warps that hold their codes from the MoE's "
          "phase A end (TINY trigger moved there), bitwise equal output (tools/dl_ly_rsl_patch.py).\n")
if TL:
    HEADER = ("# ly_rsl_tl: ly_rsl with globaltimer stamps in cgrs_kernel<true> (needs tools/dl_ly_pmt_patch.py on top). "
              "Instrument, not a candidate.\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:100]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a = text.index(HELPER_START)
    b = text.index('"""', a + len(HELPER_START))
    blk = text[a:b]
    for old, new, n in HELPER_EDITS:
        blk = sub(blk, old, new, n, "helper0")
    text = text[:a] + blk + text[b:]
    for old, new, n in FILE_EDITS:
        text = sub(text, old, new, n, "file")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    # replay the import-time pdl8 edits on the new CG_CUDA_SRC: every anchor once, launch present
    ns = {}
    i = text.index('CG_CUDA_SRC = r"""\n') + len('CG_CUDA_SRC = r"""\n')
    cg = text[i:text.index('\n"""\n', i) + 1]
    j = text.index("_PDL8_CG_EDITS = [")
    exec(text[j:text.index("\n", j)], ns)  # noqa: S102  (one-line list literal)
    for old, new, n in ns["_PDL8_CG_EDITS"]:
        cg = sub(cg, old, new, n, "pdl8 replay")
    assert cg.count("cgrs_kernel") >= 2 and cg.count(CG40_GO) == 1
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
