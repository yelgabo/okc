#!/usr/bin/env python3
"""PDL on for the non-cooperative edges of the T <= 8 chain of the public layer file (g1b4_stage_release.py,
ba9e327c). Writes muse_decode_layer/cand/g1b4_pdl8.py: the release bytes plus one appended block.

With PDL_CHAIN = False the block edits no source string and rebinds no name, so the file builds and runs exactly as
the release. With PDL_CHAIN = True (the generated default):

  edges made programmatic (consumer launched with programmaticStreamSerializationAllowed = 1):
    qkv sk2 -> attention, attention -> o sk2, o sk2 -> b1 boundary, b1 -> fused pooler / router sk2,
    pooler / router -> route (Triton), unpooler sk2 -> finish (Triton)
  edges left serialized: route -> MoE and MoE -> unpooler (the MoE is a cooperative launch)

  consumers read predecessor outputs only after griddepcontrol.wait and only through L1-bypassing loads:
    attention gate rows __ldcg (q / k / v were cp.async.cg already), b1 branch ld.global.cg, route and finish
    tl.load(..., cache_modifier='.cg'); sk2 consumers read x through TMA behind fence.proxy.async.global
  producer triggers after the last weight / input issue: sk2 (already), attention at the start of each segment's
    merge (all K/V tiles of the segment issued and consumed), b1 after its branch reads
  qkv and unpooler run the sk2 PDL prologue and trigger but launch without the attribute (pdl word bit 29)

prepare() times PDL on against PDL off of the same build at decode 1 and verify 1, sliding and full (cold-L2 graph
replays, paired rounds), runs the replay stress (2000 replays per arm of decode 1 sliding and verify 1 full on fixed
inputs, every replay compared with the PDL-off output) and turns PDL on per T only if the stress passes.

    python3 tools/dl_pdl8_patch.py
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "muse_decode_layer/cand/g1b4_stage_release.py"
DST = ROOT / "muse_decode_layer/cand/g1b4_pdl8.py"
PIN = "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"

GATE = ("*reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + "
        "(16 * pair + hh) * 64 + part * 8)")
GATE_CG = ("__ldcg(reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + "
           "(16 * pair + hh) * 64 + part * 8))")
GLBCG = r'''template<int V> __device__ __forceinline__ void glbcg(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.cg.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.cg.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
'''
SK2_ATTR = ("      if (pdl > 0) {   // programmatic dependent launch: may start while the previous kernel in the stream finishes\n"
            "        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
            "        at_[na].val.programmaticStreamSerializationAllowed = 0;")
EDITS = {
    "ATT7_CUDA_SRC": [
        ("#define OKC_PDLKV 0",
         "#define OKC_PDLTRIG 1   // pdl8: trigger at each segment's merge (its K/V tiles issued and consumed)\n#define OKC_PDLKV 0", 1),
        ("  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it\n"
         "  pdl_launch_dependents();\n",
         "  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it\n"
         "  if (!OKC_PDLTRIG) pdl_launch_dependents();\n", 1),
        ("  DBG(3);\n  // ---- merge", "  DBG(3);\n  if (OKC_PDLTRIG) pdl_launch_dependents();\n  // ---- merge", 1),
        (GATE, GATE_CG, 4),
        ("      attr[na].val.programmaticStreamSerializationAllowed = 0;",
         "      attr[na].val.programmaticStreamSerializationAllowed = 1;", 1),
    ],
    "bfp_SK2_SRC": [
        (SK2_ATTR, SK2_ATTR.replace("if (pdl > 0) {", "if (pdl > 0 && !(pdl & (1 << 29))) {")
                           .replace("Allowed = 0;", "Allowed = 1;"), 1),
    ],
    "_SL_GLUE_CUDA": [
        ("template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {",
         "static int g_glue_pdl=0;\nvoid glue_set_pdl(int64_t on){g_glue_pdl=on?1:0;}\n"
         "template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {", 1),
        ("  attrs[1].val.programmaticStreamSerializationAllowed=0;",
         "  attrs[1].val.programmaticStreamSerializationAllowed=g_glue_pdl;", 1),
        ("void glue_cluster_boundary(GlueArgs a) {\n  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");",
         "void glue_cluster_boundary(GlueArgs a) {\n  if constexpr (SLICED) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");", 1),
        ("  asm volatile(\"griddepcontrol.wait; griddepcontrol.launch_dependents;\" ::: \"memory\");",
         "  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");", 1),
        ("      glb<V>(a.branch+row*8192+col,b+r*V);\n", "      glbcg<V>(a.branch+row*8192+col,b+r*V);\n", 1),
        ("template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {\n",
         GLBCG + "template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {\n", 1),
        ("  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);",
         "  asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
         "  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);", 1),
    ],
}
ROUTE_CG = [
    ("def _sl_tile_route(", "def _sl_tile_route_cg(", 1),
    ("tl.load(POOLED+t*4096+c)", "tl.load(POOLED+t*4096+c, cache_modifier='.cg')", 1),
    ("tl.load(ALL+t*4608+c)", "tl.load(ALL+t*4608+c, cache_modifier='.cg')", 1),
    ("tl.load(POOLED+t*4096+oc)", "tl.load(POOLED+t*4096+oc, cache_modifier='.cg')", 1),
    ("tl.load(ALL+t*4608+oc)", "tl.load(ALL+t*4608+oc, cache_modifier='.cg')", 1),
    ("tl.load(LOGITS + t * 256 + e)", "tl.load(LOGITS + t * 256 + e, cache_modifier='.cg')", 1),
    ("tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)",
     "tl.load(ALL + t * 4608 + 4096 + e, cache_modifier='.cg') + tl.load(ALL + t * 4608 + 4352 + e, cache_modifier='.cg')", 1),
]
FINISH_CG = [
    ("def _sl_cluster_finish(", "def _sl_cluster_finish_cg(", 1),
    ("tl.load(CARRY+t*8192+c)", "tl.load(CARRY+t*8192+c, cache_modifier='.cg')", 1),
    ("tl.load(TAIL+t)", "tl.load(TAIL+t, cache_modifier='.cg')", 1),
    ("tl.load(CARRY+t*8192+o)", "tl.load(CARRY+t*8192+o, cache_modifier='.cg')", 1),
    ("tl.load(P+t*1024+c)", "tl.load(P+t*1024+c, cache_modifier='.cg')", 1),
]
AFTER = [
    ("def _sl_tile_after_attention(", "def _pdl8_sl_tile_after_attention(", 1),
    ("_sl_cluster_module().b1(", "_pdl8_b1(", 1),
    ("_sl_tile_route[", "_sl_tile_route_cg[", 2),
    ("_sl_cluster_finish[", "_sl_cluster_finish_cg[", 1),
    ("launch_pdl=False)", "launch_pdl=_PDL8)", 3),
]
ATTN = [
    ("def _decode_attention7(", "def _pdl8_decode_attention7(", 1),
    ('pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}',
     'pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"]) or _PDL8} if ATT_HAS_PDL else {}', 1),
]

BLOCK_HEAD = r'''

# Added by yelgabo, November 3, 2026 (tools/dl_pdl8_patch.py): PDL on for the non-cooperative edges of the T <= 8 chain.
PDL_CHAIN = True
PDL8_T = (1, 8)
PDL8_BOUND = 1e-4
PDL8_STRESS = 2000
_PDL8 = False


def _pdl8_sub(s, old, new, n):
    if s.count(old) != n:
        raise RuntimeError(f"pdl8 anchor {old[:60]!r}: {s.count(old)} found, {n} expected")
    return s.replace(old, new)


_PDL8_EDITS = @@EDITS@@
'''

BLOCK_TAIL = r'''

_PDL8_GLUE = None
def _pdl8_cluster_module():
    global _PDL8_GLUE
    if _PDL8_GLUE is None:
        from tvm_ffi.cpp import load_inline as _tvm_load_inline
        _PDL8_GLUE = _tvm_load_inline(name='sl_early_tail_pdl8_' + _hashlib.md5(_SL_GLUE_CUDA.encode()).hexdigest()[:12],
                                      cuda_sources=_SL_GLUE_CUDA, functions=['b1', 'glue_set_pdl'],
                                      extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])
    return _PDL8_GLUE


def _pdl8_b1(*args):
    mod = _sl_cluster_module()
    mod.glue_set_pdl(1 if _PDL8 else 0)
    try:
        mod.b1(*args)
    finally:
        mod.glue_set_pdl(0)


_pdl8_prior_bfp_pdl = _bfp_pdl
def _pdl8_bfp_pdl(T, role=None):
    global bfp_SK2_PDL
    if _PDL8 and role in ("qkv", "o", "pool", "un"):
        bfp_SK2_PDL = _pcx_val(role, T) | ((1 << 29) if role in ("qkv", "un") else 0)
    else:
        _pdl8_prior_bfp_pdl(T, role)


def _pdl8_data(T, layer, gen, dev, ctx=80000):
    q_len = 1 if T == 1 else 8
    seq = torch.full((1,), ctx, dtype=torch.int32, device=dev)
    pages = ctx + q_len
    pt = torch.randperm(pages, generator=gen, device=dev).to(torch.int32).view(1, pages)
    pos = (seq[:, None].long() + torch.arange(q_len, device=dev)).view(1, q_len)
    loc = torch.gather(pt.long(), 1, pos).reshape(-1)
    kc = torch.randn((pages, 1, NUM_KV_HEADS, HEAD_DIM), generator=gen, device=dev).to(torch.float8_e4m3fn)
    vc = torch.randn((pages, 1, NUM_KV_HEADS, HEAD_DIM), generator=gen, device=dev).to(torch.float8_e4m3fn)
    stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
    xin = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    sliding = layer == "sliding"
    cfg = dict(batch=1, q_len=q_len, tokens=T, hidden=HIDDEN, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
               head_dim=HEAD_DIM, page_size=1, window=2048 if sliding else 0, rope_theta=ROPE_THETA if sliding else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0, n_experts=256, top_k=8, local_experts=32)
    one = torch.ones((), dtype=torch.float32, device=dev)
    return {"stream": stream, "attn_input": xin, "positions": pos.reshape(-1), "cache_loc": loc,
            "cache_seqlens": (seq + q_len).to(torch.int32),
            "cu_seqlens_q": torch.tensor([0, q_len], dtype=torch.int32, device=dev), "scheduler_metadata": None,
            "k_cache": kc, "v_cache": vc, "k_scale": one, "v_scale": one.clone(), "page_table": pt, "seq_lens": seq,
            "local_start": torch.zeros(1, dtype=torch.int32, device=dev), "config": cfg}


def _pdl8_flat(r):
    return [t for t in (r if isinstance(r, (tuple, list)) else (r,)) if torch.is_tensor(t)]


def _pdl8_graph(fn):
    fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        outs = _pdl8_flat(fn())
    return gr, outs


def _pdl8_time(graphs, rounds=9, reps=3):
    flush = torch.ones((65536, 1024), device='cuda', dtype=torch.float32)
    sink = torch.empty(1024, device='cuda', dtype=torch.float32)
    per = [[[] for _ in graphs] for _ in range(rounds)]
    for r in range(rounds):
        for k in range(reps):
            order = range(len(graphs)) if (r + k) % 2 == 0 else reversed(range(len(graphs)))
            for i in order:
                torch.sum(flush, dim=0, out=sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); graphs[i].replay(); e1.record()
                per[r][i].append((e0, e1))
    torch.cuda.synchronize()
    rows = []
    for r in range(rounds):
        row = []
        for i in range(len(graphs)):
            v = sorted(a.elapsed_time(b) * 1000 for a, b in per[r][i])
            row.append(v[len(v) // 2])
        rows.append(row)
    del flush
    return rows


def _pdl8_drift(graph, outs, ref, n):
    dev = ref[0].device
    acc = torch.zeros(len(ref), device=dev)
    bad = torch.zeros((), dtype=torch.int64, device=dev)
    for _ in range(n):
        graph.replay()
        for k, (a, b) in enumerate(zip(outs, ref)):
            if a.is_floating_point():
                d = (a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)
                acc[k] = torch.maximum(acc[k], torch.nan_to_num(d, nan=1e30))
            else:
                bad += (a != b).sum()
    torch.cuda.synchronize()
    return [float(x) for x in acc.tolist()], int(bad)


def _pdl8_setup(st):
    dev = st["qkv"].device
    gen = torch.Generator(device=dev).manual_seed(31415)
    cells = {}
    for T, layer in ((1, "sliding"), (1, "full"), (8, "sliding"), (8, "full")):
        data = _pdl8_data(T, layer, gen, dev)
        off = dict(st, pdl8_on={})
        on = dict(st, pdl8_on={T: True})
        ids = _pdl8_flat(kernel_fn(data, off))[-1]
        if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 8 * T:
            counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
            data["local_start"].fill_(32 * int(counts.argmax()))
        g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
        g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
        rows = _pdl8_time([g_off, g_on])
        d = sorted(r[0] - r[1] for r in rows)
        moff = sorted(r[0] for r in rows)[len(rows) // 2]
        mon = sorted(r[1] for r in rows)[len(rows) // 2]
        cells[(T, layer)] = (moff, mon, d[len(d) // 2])
        print(f"PDL8_AB T={T} {layer}: off {moff:.2f} on {mon:.2f} us, median paired saving {d[len(d) // 2]:.2f}, "
              f"wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
        if (T, layer) in ((1, "sliding"), (8, "full")):
            g_off.replay()
            torch.cuda.synchronize()
            ref = [t.clone() for t in o_off]
            off_max, off_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
            on_max, on_bad = _pdl8_drift(g_on, o_on, ref, PDL8_STRESS)
            strict = on_bad == 0 and all(v <= PDL8_BOUND for v in on_max)
            within = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))
            st["pdl8_on"][T] = within
            st.setdefault("pdl8_log", []).append((T, layer, off_max, off_bad, on_max, on_bad, strict, within))
            print(f"PDL8_STRESS T={T} {layer}: {PDL8_STRESS} replays per arm, max rel diff to the PDL-off output "
                  f"off {['%.2e' % v for v in off_max]} (int mismatches {off_bad}), on {['%.2e' % v for v in on_max]} "
                  f"(int mismatches {on_bad}); bound {PDL8_BOUND:.0e}: strict {'pass' if strict else 'FAIL'}, "
                  f"within the off arm's own drift {'pass' if within else 'FAIL'}", flush=True)
        del g_off, g_on
    w = {}
    for T in PDL8_T:
        s, f = cells[(T, "sliding")][2], cells[(T, "full")][2]
        w[T] = (46 * s + 16 * f) / 62
    est = 0.212 * w[1] + 0.144 * w[8]
    print(f"PDL8 weighted paired saving d1 {w[1]:.2f} us, v1 {w[8]:.2f} us; six-load estimate {est:+.2f}% "
          f"(other loads untouched); enabled {st['pdl8_on']}", flush=True)


if PDL_CHAIN:
    for _name, _edits in _PDL8_EDITS.items():
        _s = globals()[_name]
        for _old, _new, _n in _edits:
            _s = _pdl8_sub(_s, _old, _new, _n)
        globals()[_name] = _s
    _sl_cluster_module = _pdl8_cluster_module
    _bfp_pdl = _pdl8_bfp_pdl
    _decode_attention7 = _pdl8_decode_attention7
    _sl_tile_after_attention = _pdl8_sl_tile_after_attention

    _pdl8_parent_kernel = kernel_fn
    def kernel_fn(data, state):
        global _PDL8, bfp_SK2_PDL
        T = data["config"]["tokens"]
        _PDL8 = T in PDL8_T and bool(state.get("pdl8_on", {}).get(T, False))
        try:
            return _pdl8_parent_kernel(data, state)
        finally:
            _PDL8 = False
            bfp_SK2_PDL = 0

    _pdl8_prior_prepare = prepare
    def prepare(weights):
        st = _pdl8_prior_prepare(weights)
        st["pdl8_on"] = {}
        try:
            _pdl8_setup(st)
        except Exception as e:
            print("PDL8 off:", repr(e)[:400], flush=True)
            st["pdl8_on"] = {}
            torch.cuda.synchronize()
        return st
'''


def sub(s, old, new, n):
    if s.count(old) != n:
        sys.exit(f"expected {n} of {old[:70]!r}, found {s.count(old)}")
    return s.replace(old, new)


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("source is not the pinned public layer file")
    s = raw.decode()
    tree = ast.parse(s)
    lines = s.split("\n")
    assigns = {ast.unparse(n.targets[0]): n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1}
    for name, edits in EDITS.items():
        lit = ast.literal_eval(assigns[name].value)
        for old, new, n in edits:
            lit = sub(lit, old, new, n)
    funcs = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            funcs.setdefault(n.name, []).append("\n".join(lines[start - 1:n.end_lineno]))

    def copy(name, edits):
        if len(funcs[name]) != 1:
            sys.exit(f"{name} defined {len(funcs[name])} times")
        t = funcs[name][0]
        for old, new, n in edits:
            t = sub(t, old, new, n)
        return t

    body = (BLOCK_HEAD.replace("@@EDITS@@", repr(EDITS)) + "\n\n" + copy("_sl_tile_route", ROUTE_CG) + "\n\n\n"
            + copy("_sl_cluster_finish", FINISH_CG) + "\n\n\n" + copy("_sl_tile_after_attention", AFTER) + "\n\n\n"
            + copy("_decode_attention7", ATTN) + BLOCK_TAIL)
    DST.write_text(s.rstrip("\n") + "\n" + body)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
