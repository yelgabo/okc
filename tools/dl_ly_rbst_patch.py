#!/usr/bin/env python3
"""ly_rbst: per-program globaltimer stamps in the decode 16 / 32 row boundaries and their producers, on ly_stk3.

Instrument only, never a candidate. Every stamp is off in the plain graph: each switch is read on the host when a
kernel is launched (so during graph capture) and is set only for the instrumented capture.

- sk2 (bfp_SK2_SRC, main bfp extension only): new export `sk2_set_rbt(ptr)`. While set, the host `sk2` passes
  ptr + region * 1024 * 32 as dbg for N = 8192 and M in (16, 32): region 0 = o (K = 1024), region 1 = unpooler
  (K = 4096). Per CTA (linear id y * gridDim.x + x), 32 int64 slots: 0 entry (tid 0), 2 after griddepcontrol.wait
  (tid 0), 23 tid 0 after the output stores (both return paths). dbg stays null otherwise (the base passes a dummy
  tensor below the trace size), and SK2_DIAG stays 0.
- Gluon row boundary `_graft_row_boundary_pre_cg` (b1 at T = 32, b2 at T = 16 / 32): while `_RBST_BUF` is set, the
  two `_pdl8_graft_boundary` call sites launch `_rbst_row_boundary`, the same body plus stamps of thread 0 / element
  0, 16 slots per row, region 0 = b1, 1 = b2 (64 rows each): 0 entry, 1 pre-wait weights landed, 2 after
  gdc_wait + gdc_launch, 3 branch landed, 4 first reduction done, 5 stream landed, 6 after the NewStream store,
  7 second reduction done, 8 after the Out store, 9 SM id. Data-dependent stamps take the value as an asm input.
- PDL8 graft glue `glue_cluster_boundary` (b1 at T = 16, 8-CTA clusters of 128 threads): GlueArgs gains a stamp
  pointer, set in `b1` from `glue_set_ts(ptr)`; tid 0 of each CTA, 16 slots per CTA: 0 entry, 1 after the
  pre-wait loads and cluster.sync, 2 after griddepcontrol.wait, 3 branch landed, 4 r1, 5 r2, 6 after the stores,
  7 SM id.
No fence, barrier, wait, trigger, launch attribute or operation order moves.

    python3 tools/dl_ly_rbst_patch.py [--out muse_decode_layer/cand/ly_rbst.py]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"
DST = ROOT / "muse_decode_layer/cand/ly_rbst.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

SK2_START = 'bfp_SK2_SRC = r"""\n'
RB = "if (!FUSE && dbg != nullptr) dbg[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * 32 + "
SK2_EDITS = [
    ("void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n",
     "void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }\n"
     "static unsigned long long* g_sk2_rbt = nullptr;\n"
     "void sk2_set_rbt(int64_t p) { g_sk2_rbt = reinterpret_cast<unsigned long long*>(p); }\n"
     "#define SK2_RB(k) do { " + RB + "(k)] = gtime(); } while (0)\n"),
    ("  if (trace && threadIdx.x == 0) tr[0] = t_start;\n",
     "  if (trace && threadIdx.x == 0) tr[0] = t_start;\n  if (threadIdx.x == 0) SK2_RB(0);\n"),
    ("  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n",
     "  if (pdl) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");   // every thread: x and the outputs are ours from here\n"
     "  if (tid == 0) SK2_RB(2);\n"),
    ("    if (trace && tid == 0) tr[15] = gtime();\n    return;\n",
     "    if (tid == 0) SK2_RB(23);\n    if (trace && tid == 0) tr[15] = gtime();\n    return;\n"),
    ("  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {",
     "  if (tid == 0) SK2_RB(23);\n  if (trace && tid == 0) tr[15] = gtime();\n  if ((dgv & 32) && dbg != nullptr && tid == 0) {"),
    ("  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n",
     "  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;\n"
     "  if (g_sk2_rbt != nullptr && N == 8192 && (M == 16 || M == 32)) dbg = g_sk2_rbt + (K == 4096 ? 1 : 0) * 1024 * 32;\n"),
]

GLUE_EDITS = [
    ("  float *next, *router;\n  int rows;\n};\n",
     "  float *next, *router;\n  int rows;\n  unsigned long long* ts;\n};\n"
     "__device__ __forceinline__ unsigned long long rbg_t() {\n"
     "  unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t;\n}\n"
     "__device__ __forceinline__ unsigned long long rbg_tf(float v) {\n"
     "  unsigned long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t) : \"f\"(v)); return t;\n}\n"
     "#define RBG(k) do { if (a.ts && threadIdx.x == 0) a.ts[blockIdx.x * 16 + (k)] = rbg_t(); } while (0)\n"
     "#define RBGD(k, v) do { if (a.ts && threadIdx.x == 0) a.ts[blockIdx.x * 16 + (k)] = rbg_tf(v); } while (0)\n"
     "static unsigned long long* g_glue_ts = nullptr;\n"
     "void glue_set_ts(int64_t p) { g_glue_ts = reinterpret_cast<unsigned long long*>(p); }\n"),
    ("void glue_cluster_boundary(GlueArgs a) {\n", "void glue_cluster_boundary(GlueArgs a) {\n  RBG(0);\n"),
    ("  cluster.sync();\n  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n",
     "  cluster.sync();\n  RBG(1);\n  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n  RBG(2);\n"),
    ("  asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n  float r1=",
     "  RBGD(3, b[N-1]);\n  asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n  float r1="),
    ("  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);\n",
     "  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);\n  RBGD(4, r1);\n"),
    ("  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);\n",
     "  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);\n  RBGD(5, r2);\n"),
    ("      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);\n    }\n  }\n}\n",
     "      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);\n    }\n  }\n  RBG(6);\n"
     "  if (a.ts && threadIdx.x == 0) { unsigned sm_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); "
     "a.ts[blockIdx.x * 16 + 7] = sm_; }\n}\n"),
    ("             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0)};\n"
     "  if(a.rows<1 || a.rows>16) throw std::runtime_error(\"graft b1: rows must be 1..16\");\n",
     "             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0)};\n"
     "  a.ts = g_glue_ts;\n"
     "  if(a.rows<1 || a.rows>16) throw std::runtime_error(\"graft b1: rows must be 1..16\");\n"),
]

FILE_EDITS = [
    ("void sk2_set_pdl(int64_t n);\n\"\"\"\n\nbfp_w_rows", "void sk2_set_pdl(int64_t n);\nvoid sk2_set_rbt(int64_t p);\n\"\"\"\n\nbfp_w_rows", 1),
    ('"sk2", "gq", "sk2f", "sk2_set_pdl"]', '"sk2", "gq", "sk2f", "sk2_set_pdl", "sk2_set_rbt"]', 1),
    ("        for old, new, n in _PDL8_GRAFT_GLUE_EDITS:\n            src = _pdl8_sub(src, old, new, n)\n",
     "        for old, new, n in _PDL8_GRAFT_GLUE_EDITS:\n            src = _pdl8_sub(src, old, new, n)\n"
     "        for old, new in _RBST_GLUE_EDITS:\n            src = _pdl8_sub(src, old, new, 1)\n", 1),
    ("cuda_sources=src, functions=['b1', 'b2c', 'glue_set_pdl'],",
     "cuda_sources=src, functions=['b1', 'b2c', 'glue_set_pdl', 'glue_set_ts'],", 1),
    ("                _graft_row_boundary_pre_cg[(T,)](br, stream, g[\"post2\"], g[\"pre2\"], g[\"alpha2\"], g[\"beta2\"], h_out, s_out,\n"
     "                                                 HIDDEN, width, True, 4, 32, num_warps=32, launch_pdl=True)\n",
     "                _rbst_launch(True, T, br, stream, g[\"post2\"], g[\"pre2\"], g[\"alpha2\"], g[\"beta2\"], h_out, s_out,\n"
     "                             HIDDEN, width, True, 4, 32, num_warps=32, launch_pdl=True)\n", 1),
    ("        _graft_row_boundary_pre_cg[(T,)](br, stream, g[\"post2\"], g[\"pre2\" if post else \"pre1\"], g[\"alpha\" + pfx],\n"
     "                                         g[\"beta\" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,\n"
     "                                         num_warps=32, launch_pdl=True)\n",
     "        _rbst_launch(post, T, br, stream, g[\"post2\"], g[\"pre2\" if post else \"pre1\"], g[\"alpha\" + pfx],\n"
     "                     g[\"beta\" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,\n"
     "                     num_warps=32, launch_pdl=True)\n", 1),
]

GLUON = '''

@_glx.jit
def _rbst_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream, TS,
                       H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    tsp = TS + row * 16 + c * 0
    m0 = c == 0
    _gl.store(tsp + 0, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [c], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gl.store(tsp + 1, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,f,f", [pw, beta], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    _gdc_wait()
    _gdc_launch()
    _gl.store(tsp + 2, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [c], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg').to(_gl.bfloat16).to(_gl.float32)
    _gl.store(tsp + 3, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,f", [b], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    _gl.store(tsp + 4, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r,f", [c, r], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c, cache_modifier='.cg')
    _gl.store(tsp + 5, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,f", [s], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    _gl.store(tsp + 6, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,f", [ns], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    _gl.store(tsp + 7, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r,f", [c, r], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    h = ns * r * pw
    _gl.store(Out + row * H + c, h)
    _gl.store(tsp + 8, _gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,f", [h], dtype=_gl.int64, is_pure=False, pack=1), mask=m0)
    _gl.store(tsp + 9, _gl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [c], dtype=_gl.int32, is_pure=False, pack=1).to(_gl.int64), mask=m0)


_RBST_BUF = None   # int64 [2 * 64 * 16]: region 0 = b1, 1 = b2; set only while the instrumented graph is captured
_RBST_GLUE_EDITS = GLUE_EDITS_PLACEHOLDER


def _rbst_launch(post, T, *args, **kw):
    if _RBST_BUF is None:
        return _graft_row_boundary_pre_cg[(T,)](*args, **kw)
    ts = _RBST_BUF[(1 if post else 0) * 64 * 16:]
    return _rbst_row_boundary[(T,)](*args[:8], ts, *args[8:], **kw)
'''

HEADER = ("# ly_rbst: ly_stk3 with per-program globaltimer stamps in the decode 16 / 32 row boundaries (Gluon b1 / b2, "
          "glue b1 at T = 16) and the o / unpooler sk2 producers, off in the plain graph (tools/dl_ly_rbst_patch.py). "
          "Instrument, not a candidate.\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    i = text.index(SK2_START) + len(SK2_START)
    j = text.index('\n"""\n', i)
    sk = text[i:j]
    for old, new in SK2_EDITS:
        sk = sub(sk, old, new, 1, "sk2")
    text = text[:i] + sk + text[j:]
    for old, new, n in FILE_EDITS:
        text = sub(text, old, new, n, "file")
    # the glue anchors are checked against the PDL8 graft glue source as the module builds it
    g0 = text.index("_GRAFT_GLUE_CUDA = r'''") + len("_GRAFT_GLUE_CUDA = r'''")
    glue = text[g0:text.index("'''", g0)]
    line = next(x for x in text.splitlines() if x.startswith("_PDL8_GRAFT_GLUE_EDITS = "))
    for old, new, n in ast.literal_eval(line.split(" = ", 1)[1]):
        glue = sub(glue, old, new, n, "pdl8 glue")
    for old, new in GLUE_EDITS:
        glue = sub(glue, old, new, 1, "rbst glue")
    block = GLUON.replace("GLUE_EDITS_PLACEHOLDER", repr(GLUE_EDITS))
    anchor = "\n\n_V1FS_SMEM = 118784"
    text = sub(text, anchor, block + anchor, 1, "gluon block")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
