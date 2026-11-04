#!/usr/bin/env python3
"""ly_pmt: ly_fabq with per-CTA globaltimer stamps in the decode 1 post-MoE tail (MoE exit, slice cgemv, finish).

Base: muse_decode_layer/cand/ly_fabq.py (SHA256 37654700..., checked). Instrument only, never a candidate. Every
stamp is behind a switch that the timed graph of the plain run leaves off, so the plain graph runs the base's code:

- slice cgemv (CG_CUDA_SRC, after the pdl8 edits): cgemv_body / cgemv_kernel / cgemv40_kernel get a `bool TL`
  template parameter (default false, the launched kernel is unchanged); cgemv_launch picks the TL instantiation only
  while cg_set_tlb(ptr) holds a non-null stamp buffer (a __device__ pointer). Slots (16 int64 per CTA, blockIdx.x):
    0 entry (t 0)   1 weight loads issued (t 0, nothing consumed before the wait)   2 after griddepcontrol.wait (t 0)
    3 lane 0's x landed   12 warp 0: every lane's x landed   13 warp 1: every lane's weights landed (consumed, then
    __syncwarp)   4 warp 0's FMAs and warp sums done   5 escape warp: count landed (before its wait)   6 escape warp
    after its wait   14 escape lane 1: entry 1's x element landed   15 escape lane 0 after its loop   7 escape walk
    done (after the warp sums)   8 after the CTA barrier (t 0)   9 output stored (t 0)   10 SM id   11 escape count
- MoE TINY build (helper0): -DOKC_DEBUG=1 enables the source's own DBG(k) stamps (p.dbg, 8 per CTA: 0 entry,
  1 prologue, 7 gather, 6 WS join, 2 phase A end, 3 phase B end = launch_dependents, 4 after the grid barrier,
  5 exit after the final pass); they fire only when a launch passes a fourth scratch tensor (_PMT_MOE_DBG), which
  the plain graph does not.
- finish (_sl_cluster_finish_cg, Triton): a stamp pointer argument and TLON constexpr (8 int64 per program:
  0 entry, 1 after gdc_wait, 2 the slice output P consumed (inv formed), 3 carry / tail consumed (invnext), 4 stores
  issued); off, the kernel gets a dummy pointer and TLON = False (no code).
No fence, barrier, wait, trigger or launch attribute moves. tools/ly_box_pmt.py captures a plain and an instrumented
graph per cell and reads the stamps per fresh input.

    python3 tools/dl_ly_pmt_patch.py [--base muse_decode_layer/cand/ly_fabq.py] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_fabq.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_pmt.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "3765470032deb505c20e7573fcd0f01aac88a8474added7f2546579f74dde555"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
XLD = "__ldcg"   # the base's x load intrinsic after the pdl8 edits (--xld __ldca for a candidate that changed it)
if "--xld" in sys.argv:
    XLD = sys.argv[sys.argv.index("--xld") + 1]

# edits on CG_CUDA_SRC after the pdl8 edits (applied at import under `if PDL_CHAIN:`), as (old, new, count)
CG_EDITS = [
    ("namespace {\n\nconstexpr int EXP_WINDOW = 8;\n",
     "namespace {\n\n"
     "// pmt: per-CTA timeline stamps (globaltimer ns, 16 per CTA) in the TL instantiation only (cg_set_tlb)\n"
     "__device__ unsigned long long* g_cg_tlp = nullptr;\n"
     "__device__ __forceinline__ unsigned long long cg_gt() { unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }\n"
     "__device__ __forceinline__ void cg_use(unsigned v) { asm volatile(\"\" :: \"r\"(v)); }\n\n"
     "constexpr int EXP_WINDOW = 8;\n", 1),
    ("template <int NTH, int CPT>\n__device__ __forceinline__ void cgemv_body(",
     "template <int NTH, int CPT, bool TL = false>\n__device__ __forceinline__ void cgemv_body(", 1),
    ("    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;\n    if (warp < NW) {\n        const uint8_t* lp = stream + (size_t)c * 2 * K;\n",
     "    const int c = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;\n"
     "    unsigned long long* tl = nullptr;\n"
     "    if (TL) {\n"
     "        tl = g_cg_tlp + (size_t)c * 16;\n"
     "        if (t == 0) { tl[0] = cg_gt(); unsigned sm; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm)); tl[10] = sm; }\n"
     "    }\n"
     "    if (warp < NW) {\n        const uint8_t* lp = stream + (size_t)c * 2 * K;\n", 1),
    ("            q1[i] = __ldg(reinterpret_cast<const uint2*>(cp + K / 2) + j);\n        }\n"
     "        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n#pragma unroll\n        for (int i = 0; i < CPT; ++i) {\n            xa[i] = XLD(xp + i * NTH * 2);\n            xb[i] = XLD(xp + i * NTH * 2 + 1);\n        }\n        float acc0 = 0.f, acc1 = 0.f;\n",
     "            q1[i] = __ldg(reinterpret_cast<const uint2*>(cp + K / 2) + j);\n        }\n"
     "        if (TL && t == 0) tl[1] = cg_gt();\n"
     "        if (TL && warp == 1) {\n#pragma unroll\n            for (int i = 0; i < CPT; ++i) {\n"
     "                cg_use(l0[i].x); cg_use(l0[i].y); cg_use(l0[i].z); cg_use(l0[i].w);\n"
     "                cg_use(l1[i].x); cg_use(l1[i].y); cg_use(l1[i].z); cg_use(l1[i].w);\n"
     "                cg_use(q0[i].x); cg_use(q0[i].y); cg_use(q1[i].x); cg_use(q1[i].y);\n            }\n"
     "            __syncwarp();\n            if (lane == 0) tl[13] = cg_gt();\n        }\n"
     "        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
     "        if (TL && t == 0) tl[2] = cg_gt();\n"
     "#pragma unroll\n        for (int i = 0; i < CPT; ++i) {\n            xa[i] = XLD(xp + i * NTH * 2);\n            xb[i] = XLD(xp + i * NTH * 2 + 1);\n        }\n"
     "        if (TL && warp == 0) {\n#pragma unroll\n            for (int i = 0; i < CPT; ++i) {\n"
     "                cg_use(xa[i].x); cg_use(xa[i].y); cg_use(xa[i].z); cg_use(xa[i].w);\n"
     "                cg_use(xb[i].x); cg_use(xb[i].y); cg_use(xb[i].z); cg_use(xb[i].w);\n            }\n"
     "            if (lane == 0) tl[3] = cg_gt();\n"
     "            __syncwarp();\n            if (lane == 0) tl[12] = cg_gt();\n        }\n"
     "        float acc0 = 0.f, acc1 = 0.f;\n", 1),
    ("        acc0 = warp_sum(acc0);\n        acc1 = warp_sum(acc1);\n        if (lane == 0) {\n            red[0][warp] = acc0;\n",
     "        acc0 = warp_sum(acc0);\n        acc1 = warp_sum(acc1);\n        if (TL && t == 0) tl[4] = cg_gt();\n        if (lane == 0) {\n            red[0][warp] = acc0;\n", 1),
    ("        const int count = __shfl_sync(0xFFFFFFFFu, ent.x, 0);\n",
     "        const int count = __shfl_sync(0xFFFFFFFFu, ent.x, 0);\n        if (TL && lane == 0) { tl[5] = cg_gt(); tl[11] = (unsigned long long)count; }\n", 1),
    ("        float corr0 = 0.f, corr1 = 0.f;\n        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n        for (int e = lane; e <= count; e += 32) {",
     "        float corr0 = 0.f, corr1 = 0.f;\n        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n        if (TL && lane == 0) tl[6] = cg_gt();\n        for (int e = lane; e <= count; e += 32) {", 1),
    ("                const float xv = __uint_as_float(\n                    (unsigned)XLD(reinterpret_cast<const unsigned short*>(x) + kk) << 16);\n",
     "                const float xv = __uint_as_float(\n                    (unsigned)XLD(reinterpret_cast<const unsigned short*>(x) + kk) << 16);\n"
     "                if (TL && lane == 1 && e == 1) { cg_use(__float_as_uint(xv)); tl[14] = cg_gt(); }\n", 1),
    ("        corr0 = warp_sum(corr0);\n        corr1 = warp_sum(corr1);\n        if (lane == 0) {\n            red[0][NW] = corr0;\n",
     "        if (TL && lane == 0) tl[15] = cg_gt();\n        corr0 = warp_sum(corr0);\n        corr1 = warp_sum(corr1);\n        if (TL && lane == 0) tl[7] = cg_gt();\n        if (lane == 0) {\n            red[0][NW] = corr0;\n", 1),
    ("    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n    __syncthreads();\n    if (t == 0) {\n        float r0 = 0.f, r1 = 0.f;",
     "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n    __syncthreads();\n    if (t == 0) {\n        if (TL) tl[8] = cg_gt();\n        float r0 = 0.f, r1 = 0.f;", 1),
    ("        *reinterpret_cast<unsigned*>(out + 2 * c) = pack_bf16x2(r0, r1);\n    }\n}\n",
     "        *reinterpret_cast<unsigned*>(out + 2 * c) = pack_bf16x2(r0, r1);\n        if (TL) tl[9] = cg_gt();\n    }\n}\n", 1),
    ("template <int NTH, int CPT>\n__global__ void __maxnreg__(32)\ncgemv_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,\n             unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {\n    cgemv_body<NTH, CPT>(stream, half_n, table, pmax, base4, x, out);\n}\n",
     "template <int NTH, int CPT, bool TL = false>\n__global__ void __maxnreg__(32)\ncgemv_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,\n             unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {\n    cgemv_body<NTH, CPT, TL>(stream, half_n, table, pmax, base4, x, out);\n}\n", 1),
    ("template <int NTH, int CPT>\n__global__ void __maxnreg__(40)\ncgemv40_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,\n               unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {\n    cgemv_body<NTH, CPT>(stream, half_n, table, pmax, base4, x, out);\n}\n",
     "template <int NTH, int CPT, bool TL = false>\n__global__ void __maxnreg__(40)\ncgemv40_kernel(const uint8_t* __restrict__ stream, int half_n, const int2* __restrict__ table, int pmax,\n               unsigned base4, const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out) {\n    cgemv_body<NTH, CPT, TL>(stream, half_n, table, pmax, base4, x, out);\n}\n", 1),
    ("static int g_cg_pdl = 0;\n", "static int g_cg_pdl = 0;\nstatic int g_cg_tl = 0;\n", 1),
    ("        cg_go(cgemv_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);\n",
     "        if (g_cg_tl) cg_go(cgemv_kernel<NTH, CPT, true>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);\n"
     "        else cg_go(cgemv_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);\n", 1),
    ("        cg_go(cgemv40_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);\n",
     "        if (g_cg_tl) cg_go(cgemv40_kernel<NTH, CPT, true>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);\n"
     "        else cg_go(cgemv40_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);\n", 1),
    ("void cg_set_pdl(int64_t on) { g_cg_pdl = on ? 1 : 0; }\n",
     "void cg_set_pdl(int64_t on) { g_cg_pdl = on ? 1 : 0; }\n"
     "void cg_set_tlb(int64_t p) {\n"
     "    unsigned long long* q = reinterpret_cast<unsigned long long*>(p);\n"
     "    if (q) (void)cudaMemcpyToSymbol(g_cg_tlp, &q, sizeof(q));   // the captured TL kernel reads it at replay; 0 only stops new TL launches\n"
     "    g_cg_tl = p != 0;\n"
     "}\n", 1),
]

FILE_EDITS = [
    # the cg edits are applied to CG_CUDA_SRC after the pdl8 edits, at import
    ("if PDL_CHAIN:\n    for _old, _new, _n in _PDL8_CG_EDITS:\n        CG_CUDA_SRC = _pdl8_sub(CG_CUDA_SRC, _old, _new, _n)\n    CG_CPP_DECLS = CG_CPP_DECLS + \"void cg_set_pdl(int64_t on);\\n\"\n",
     "if PDL_CHAIN:\n    for _old, _new, _n in _PDL8_CG_EDITS:\n        CG_CUDA_SRC = _pdl8_sub(CG_CUDA_SRC, _old, _new, _n)\n    CG_CPP_DECLS = CG_CPP_DECLS + \"void cg_set_pdl(int64_t on);\\n\"\n"
     "    for _old, _new, _n in _PMT_CG_EDITS:\n        CG_CUDA_SRC = _pdl8_sub(CG_CUDA_SRC, _old, _new, _n)\n    CG_CPP_DECLS = CG_CPP_DECLS + \"void cg_set_tlb(int64_t p);\\n\"\n", 1),
    ('                  functions=["cgemv_fwd", "cg_set_pdl"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],',
     '                  functions=["cgemv_fwd", "cg_set_pdl", "cg_set_tlb"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],', 1),
    # MoE TINY build: the helper0 source's own DBG stamps on, taken only with a fourth scratch tensor
    ("#define OKC_DEBUG 0\n", "#ifndef OKC_DEBUG\n#define OKC_DEBUG 0\n#endif\n", 2),
    ('MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n',
     'MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1", "-DOKC_NOSCL=1")\n'
     'MOE_TINY_DEFINES += ("-DOKC_DEBUG=1",)   # pmt: DBG stamps compiled in, taken only with a fourth scratch tensor\n', 1),
    ('ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1]', 'ext.okc_run(state["ma_w"], _pmt_scratch(state), xm[m0:m1]', 6),
    ("def _ma_make_scratch(dev):\n",
     "_PMT_MOE_DBG = None   # pmt: int64 [grid * 8] stamp tensor passed as the MoE's fourth scratch (None: off)\n"
     "_PMT_FIN_TLB = None   # pmt: int64 [programs * 8] stamp tensor for the finish kernel (None: off)\n"
     "_PMT_FIN_DUMMY = {}\n\n\n"
     "def _pmt_scratch(state):\n"
     "    return list(state[\"ma_scratch\"]) + [_PMT_MOE_DBG] if _PMT_MOE_DBG is not None else state[\"ma_scratch\"]\n\n\n"
     "def _pmt_fin_tlb(dev):\n"
     "    if _PMT_FIN_TLB is not None:\n        return _PMT_FIN_TLB\n"
     "    if dev not in _PMT_FIN_DUMMY:\n        _PMT_FIN_DUMMY[dev] = torch.zeros(8, dtype=torch.int64, device=dev)\n"
     "    return _PMT_FIN_DUMMY[dev]\n\n\n"
     "def _ma_make_scratch(dev):\n", 1),
    # finish kernel stamps
    ("@triton.jit\ndef _sl_cluster_finish_cg(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,\n                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):\n    t = tl.program_id(0)\n    tile = tl.program_id(1)\n    c = tl.arange(0,1024)\n    beta = tl.load(BETA+c)\n    gp = tl.load(POST+c).to(tl.float32)\n    if USE_PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    carry = tl.load(CARRY+t*8192+c, cache_modifier='.cg')\n    tail = tl.load(TAIL+t, cache_modifier='.cg')\n    o = tile*BC+tl.arange(0,BC)\n    outcarry = tl.load(CARRY+t*8192+o, cache_modifier='.cg')\n    gn = tl.load(NEXT+o).to(tl.float32)\n    f = (tl.load(P+t*1024+c, cache_modifier='.cg').to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)\n    inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)\n    n = (f*inv).to(tl.bfloat16).to(tl.float32)\n    head = carry+beta*n\n    invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)\n    result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)\n    tl.store(SOUT+t*8192+o,result)\n    tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))\n",
     '@triton.jit\ndef _pmt_gt(v):\n    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [v], dtype=tl.int64, is_pure=False, pack=1)\n\n\n@triton.jit\ndef _sl_cluster_finish_cg(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT, TLB,\n                       BC: tl.constexpr, USE_PDL: tl.constexpr = True, TLON: tl.constexpr = False):\n    t = tl.program_id(0)\n    tile = tl.program_id(1)\n    tlp = TLB + (t * tl.num_programs(1) + tile) * 8\n    if TLON:\n        tl.store(tlp + 0, _pmt_gt(t))\n    c = tl.arange(0,1024)\n    beta = tl.load(BETA+c)\n    gp = tl.load(POST+c).to(tl.float32)\n    if USE_PDL:\n        tl.extra.cuda.gdc_wait()\n        tl.extra.cuda.gdc_launch_dependents()\n    if TLON:\n        tl.store(tlp + 1, _pmt_gt(t))\n    carry = tl.load(CARRY+t*8192+c, cache_modifier=\'.cg\')\n    tail = tl.load(TAIL+t, cache_modifier=\'.cg\')\n    o = tile*BC+tl.arange(0,BC)\n    outcarry = tl.load(CARRY+t*8192+o, cache_modifier=\'.cg\')\n    gn = tl.load(NEXT+o).to(tl.float32)\n    f = (tl.load(P+t*1024+c, cache_modifier=\'.cg\').to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)\n    inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)\n    if TLON:\n        tl.store(tlp + 2, _pmt_gt(inv.to(tl.int32, bitcast=True)))\n    n = (f*inv).to(tl.bfloat16).to(tl.float32)\n    head = carry+beta*n\n    invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)\n    if TLON:\n        tl.store(tlp + 3, _pmt_gt(invnext.to(tl.int32, bitcast=True)))\n    result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)\n    tl.store(SOUT+t*8192+o,result)\n    tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))\n    if TLON:\n        tl.store(tlp + 4, _pmt_gt(t))\n', 1),
    ("    _sl_cluster_finish_cg[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=_pdl8_edge(\"fin\"))\n",
     "    _sl_cluster_finish_cg[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,_pmt_fin_tlb(out.device),BC=bc,num_warps=nw,TLON=_PMT_FIN_TLB is not None,launch_pdl=_pdl8_edge(\"fin\"))\n", 1),
]
HEADER = ("# ly_pmt: ly_fabq with per-CTA globaltimer stamps in the decode 1 post-MoE tail (MoE exit, slice cgemv, "
          "finish), off in the plain graph (tools/dl_ly_pmt_patch.py). Instrument, not a candidate.\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    cg_edits = [(o.replace("XLD", XLD), nw.replace("XLD", XLD), n) for o, nw, n in CG_EDITS]
    for old, new, n in FILE_EDITS:
        text = sub(text, old, new, n, "file")
    # the cg edits as a module-level list, applied to the post-pdl8 CG_CUDA_SRC at import
    anchor = "# pdl8e: the three m = 1 cgemv kernels as PDL participants at T = 1"
    text = sub(text, anchor, "_PMT_CG_EDITS = " + repr(cg_edits) + "\n\n" + anchor, 1, "cg edits")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    # the cg anchors must exist in the base's post-pdl8 cgemv source: replay the pdl8 edits here and check
    ns = {}
    src = text
    i = src.index('CG_CUDA_SRC = r"""\n') + len('CG_CUDA_SRC = r"""\n')
    cg = src[i:src.index('\n"""\n', i) + 1]
    j = src.index("_PDL8_CG_EDITS = [")
    exec(src[j:src.index("\n", j)], ns)  # noqa: S102  (the list is one line)
    for old, new, n in ns["_PDL8_CG_EDITS"]:
        cg = sub(cg, old, new, n, "pdl8 replay")
    for old, new, n in cg_edits:
        cg = sub(cg, old, new, n, "cg")
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
