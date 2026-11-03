#!/usr/bin/env python3
"""Check-2 variant of tools/dl_pdl8_patch.py: writes muse_decode_layer/cand/g1b4_pdl8b.py from the pinned release.

Against g1b4_pdl8.py: PDL per T is fixed at build time (T = 1 on, T = 8 off: its stress failed in 8f6acddaca2e4c50),
and the same non-cooperative edges are made programmatic at T = 16 (graft glue b1 / b2c) and T = 32 (Gluon row
boundaries). The T = 16 / 32 consumers with changed loads (a separate graft glue module, a .cg copy of the Gluon row
boundary, the .cg route) run only while PDL is on, so the PDL-off arm runs the release's own kernels there. Each
enabled T gets the in-prepare A/B and a 2000-replay stress (decode 1 sliding, decode 16 sliding, decode 32 full);
a stress whose PDL-on drift exceeds the off arm's own drift turns that T off.

    python3 tools/dl_pdl8b_patch.py
"""
import ast
import hashlib
import importlib.util
import py_compile
import sys

spec = importlib.util.spec_from_file_location("pdl8", __file__.replace("dl_pdl8b_patch.py", "dl_pdl8_patch.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

DST = A.ROOT / "muse_decode_layer/cand/g1b4_pdl8b.py"

GLFCG = r'''template<int V> __device__ __forceinline__ void glfcg(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.cg.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
'''
GRAFT_GLUE_EDITS = [
    ("template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {",
     "static int g_glue_pdl=0;\nvoid glue_set_pdl(int64_t on){g_glue_pdl=on?1:0;}\n"
     "template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {", 1),
    ("  attrs[1].val.programmaticStreamSerializationAllowed=0;",
     "  attrs[1].val.programmaticStreamSerializationAllowed=g_glue_pdl;", 1),
    ("void glue_cluster_boundary(GlueArgs a) {\n  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");",
     "void glue_cluster_boundary(GlueArgs a) {\n  if constexpr (SLICED || POST) asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");", 1),
    ("  asm volatile(\"griddepcontrol.wait; griddepcontrol.launch_dependents;\" ::: \"memory\");",
     "  asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");", 1),
    ("      glb<V>(a.branch+row*8192+col,b+r*V);\n", "      glbcg<V>(a.branch+row*8192+col,b+r*V);\n", 1),
    ("    glf<V>(a.stream+row*8192+col,s+r*V);\n", "    glfcg<V>(a.stream+row*8192+col,s+r*V);\n", 1),
    ("template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {\n",
     A.GLBCG + GLFCG + "template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {\n", 1),
    ("  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);",
     "  asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
     "  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);", 1),
]
ROW_CG = [
    ("def _graft_row_boundary_pre(", "def _graft_row_boundary_pre_cg(", 1),
    ("b = _gl.load(Branch + row * F + c, c < F, 0)", "b = _gl.load(Branch + row * F + c, c < F, 0, cache_modifier='.cg')", 1),
    ("s = _gl.load(Stream + row * H + c)", "s = _gl.load(Stream + row * H + c, cache_modifier='.cg')", 1),
]
GRAFT_AFTER = [
    ("def _graft_after_attention(", "def _pdl8_graft_after_attention(", 1),
    ("_graft_boundary(", "_pdl8_graft_boundary(", 2),
    ("_sl_tile_route[(T, parts + 1)]", "(_sl_tile_route_cg if _PDL8 else _sl_tile_route)[(T, parts + 1)]", 2),
    ("                num_warps=nw, launch_pdl=False)", "                num_warps=nw, launch_pdl=_PDL8)", 2),
]

BLOCK_B = r'''

_PDL8_GRAFT_GLUE = None
def _pdl8_graft_glue_module():
    global _PDL8_GRAFT_GLUE
    if _PDL8_GRAFT_GLUE is None:
        src = _GRAFT_GLUE_CUDA
        for old, new, n in _PDL8_GRAFT_GLUE_EDITS:
            src = _pdl8_sub(src, old, new, n)
        from tvm_ffi.cpp import load_inline as _tvm_load_inline
        _PDL8_GRAFT_GLUE = _tvm_load_inline(name='graft_glue_pdl8_' + _hashlib.md5(src.encode()).hexdigest()[:12],
                                            cuda_sources=src, functions=['b1', 'b2c', 'glue_set_pdl'],
                                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])
    return _PDL8_GRAFT_GLUE


def _pdl8_glue_call(fn, *args):
    mod = _pdl8_graft_glue_module()
    mod.glue_set_pdl(1)
    try:
        getattr(mod, fn)(*args)
    finally:
        mod.glue_set_pdl(0)


def _pdl8_graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):
    if not _PDL8 or S != 1 or T > 32:
        return _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=nu, router_out=router_out)
    g = st["graft"]
    br = P.view(T, -1)
    width = br.shape[1]
    pfx = "2" if post else "1"
    if T <= 16 and br.dtype == torch.bfloat16:
        _graft_al(br, stream, s_out, h_out)
        if not post and width == HIDDEN:
            _pdl8_glue_call('b1', br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out,
                            router_out if router_out is not None else s_out, router_out is not None)
            return True
        if post and width == 1024 and nu == 1024:
            _graft_b2_module().b2s(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        if post and width == HIDDEN and nu == HIDDEN:
            _pdl8_glue_call('b2c', br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        return False
    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary_pre_cg[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                         g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,
                                         num_warps=32, launch_pdl=True)
        return True
    return _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=nu, router_out=router_out)
'''

DATA_B = r'''def _pdl8_data(B, q_len, layer, gen, dev, ctx=80000, spread=0.1):
    T = B * q_len
    seq = (ctx * (1 + spread * (2 * torch.rand((B,), generator=gen, device=dev) - 1))).long().clamp_min(1)
    if B == 1:
        seq.fill_(ctx)
    max_pages = int(ctx * (1 + spread)) + 1 + q_len
    total = B * max_pages
    perm = torch.randperm(total, generator=gen, device=dev).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=dev)
    per = (seq + q_len).tolist()
    at = 0
    for b in range(B):
        pt[b, :per[b]] = perm[at:at + per[b]]
        at += per[b]
    pos = (seq[:, None] + torch.arange(q_len, device=dev)).view(B, q_len)
    loc = torch.gather(pt.long(), 1, pos).reshape(-1)
    kc = torch.empty((total, 1, NUM_KV_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=dev)
    vc = torch.empty_like(kc)
    for dst in (kc, vc):
        flat = dst.view(-1, NUM_KV_HEADS, HEAD_DIM)
        for i in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - i)
            flat[i:i + n] = torch.randn((n, NUM_KV_HEADS, HEAD_DIM), generator=gen, device=dev).to(torch.float8_e4m3fn)
    stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
    xin = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    sliding = layer == "sliding"
    cfg = dict(batch=B, q_len=q_len, tokens=T, hidden=HIDDEN, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
               head_dim=HEAD_DIM, page_size=1, window=2048 if sliding else 0, rope_theta=ROPE_THETA if sliding else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0, n_experts=256, top_k=8, local_experts=32)
    one = torch.ones((), dtype=torch.float32, device=dev)
    return {"stream": stream, "attn_input": xin, "positions": pos.reshape(-1), "cache_loc": loc,
            "cache_seqlens": (seq + q_len).to(torch.int32),
            "cu_seqlens_q": torch.arange(0, T + 1, q_len, dtype=torch.int32, device=dev), "scheduler_metadata": None,
            "k_cache": kc, "v_cache": vc, "k_scale": one, "v_scale": one.clone(), "page_table": pt,
            "seq_lens": seq.to(torch.int32), "local_start": torch.zeros(1, dtype=torch.int32, device=dev), "config": cfg}


'''

SETUP_B = r'''def _pdl8_setup(st):
    dev = st["qkv"].device
    gen = torch.Generator(device=dev).manual_seed(31415)
    st["pdl8_on"] = dict(PDL8_HARD)
    cells = {}
    plan = ((1, "sliding", True), (1, "full", False), (16, "sliding", True), (16, "full", False),
            (32, "sliding", False), (32, "full", True))
    for B, layer, stress in plan:
        T = B
        if not st["pdl8_on"].get(T):
            continue
        try:
            data = _pdl8_data(B, 1, layer, gen, dev)
            off = dict(st, pdl8_on={})
            on = dict(st, pdl8_on={T: True})
            ids = _pdl8_flat(kernel_fn(data, off))[-1]
            if ids.dtype in (torch.int32, torch.int64) and ids.numel() == 8 * T:
                counts = torch.bincount(ids.flatten().long().clamp_min(0) // 32, minlength=8)
                data["local_start"].fill_(32 * int(counts.argmax()))
            kernel_fn(data, on)
            torch.cuda.synchronize()
            g_off, o_off = _pdl8_graph(lambda: kernel_fn(data, off))
            g_on, o_on = _pdl8_graph(lambda: kernel_fn(data, on))
        except Exception as e:
            st["pdl8_on"][T] = False
            print(f"PDL8 T={T} off: {repr(e)[:400]}", flush=True)
            torch.cuda.synchronize()
            continue
        rows = _pdl8_time([g_off, g_on])
        d = sorted(r[0] - r[1] for r in rows)
        moff = sorted(r[0] for r in rows)[len(rows) // 2]
        mon = sorted(r[1] for r in rows)[len(rows) // 2]
        cells[(T, layer)] = d[len(d) // 2]
        print(f"PDL8_AB T={T} {layer}: off {moff:.2f} on {mon:.2f} us, median paired saving {d[len(d) // 2]:.2f}, "
              f"wins {sum(1 for v in d if v > 0)}/{len(d)}", flush=True)
        if stress:
            g_off.replay()
            torch.cuda.synchronize()
            ref = [t.clone() for t in o_off]
            off_max, off_bad = _pdl8_drift(g_off, o_off, ref, PDL8_STRESS)
            on_max, on_bad = _pdl8_drift(g_on, o_on, ref, PDL8_STRESS)
            strict = on_bad == 0 and all(v <= PDL8_BOUND for v in on_max)
            within = on_bad == 0 and all(v <= max(PDL8_BOUND, c) for v, c in zip(on_max, off_max))
            if not within:
                st["pdl8_on"][T] = False
            st.setdefault("pdl8_log", []).append((T, layer, off_max, off_bad, on_max, on_bad, strict, within))
            print(f"PDL8_STRESS T={T} {layer}: {PDL8_STRESS} replays per arm, max rel diff to the PDL-off output "
                  f"off {['%.2e' % v for v in off_max]} (int mismatches {off_bad}), on {['%.2e' % v for v in on_max]} "
                  f"(int mismatches {on_bad}); bound {PDL8_BOUND:.0e}: strict {'pass' if strict else 'FAIL'}, "
                  f"within the off arm's own drift {'pass' if within else 'FAIL, T off'}", flush=True)
        del g_off, g_on, o_off, o_on, data
        torch.cuda.empty_cache()
    value = {1: 0.212, 16: 0.105, 32: 0.087}
    est = 0.0
    for T in (1, 16, 32):
        if (T, "sliding") in cells and (T, "full") in cells and st["pdl8_on"].get(T):
            w = (46 * cells[(T, "sliding")] + 16 * cells[(T, "full")]) / 62
            est += value[T] * w
            print(f"PDL8 weighted paired saving T={T}: {w:.2f} us", flush=True)
    print(f"PDL8 six-load estimate {est:+.2f}% (enabled loads only); enabled {st['pdl8_on']}", flush=True)


'''


def cut(s, start, end, new):
    a = s.index(start)
    b = s.index(end, a)
    return s[:a] + new + s[b:]


def main():
    raw = A.SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != A.PIN:
        sys.exit("source is not the pinned public layer file")
    s = raw.decode()
    tree = ast.parse(s)
    lines = s.split("\n")
    assigns = {ast.unparse(n.targets[0]): n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1}
    edits = dict(A.EDITS)
    for name, ed in list(edits.items()) + [("_GRAFT_GLUE_CUDA", GRAFT_GLUE_EDITS)]:
        lit = ast.literal_eval(assigns[name].value)
        for old, new, n in ed:
            lit = A.sub(lit, old, new, n)
    funcs = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            funcs.setdefault(n.name, []).append("\n".join(lines[start - 1:n.end_lineno]))

    def copy(name, ed):
        if len(funcs[name]) != 1:
            sys.exit(f"{name} defined {len(funcs[name])} times")
        t = funcs[name][0]
        for old, new, n in ed:
            t = A.sub(t, old, new, n)
        return t

    head = A.sub(A.BLOCK_HEAD.replace("@@EDITS@@", repr(A.EDITS)), "PDL8_T = (1, 8)\n",
                 "PDL8_T = (1, 16, 32)\nPDL8_HARD = {1: True, 8: False, 16: True, 32: True}\n", 1)
    head += "_PDL8_GRAFT_GLUE_EDITS = " + repr(GRAFT_GLUE_EDITS) + "\n"
    tail = A.BLOCK_TAIL
    tail = cut(tail, "def _pdl8_data(", "def _pdl8_flat(", DATA_B)
    tail = cut(tail, "def _pdl8_setup(", "\nif PDL_CHAIN:", SETUP_B + BLOCK_B.lstrip("\n") + "\n")
    tail = A.sub(tail, "    _sl_tile_after_attention = _pdl8_sl_tile_after_attention\n",
                 "    _sl_tile_after_attention = _pdl8_sl_tile_after_attention\n"
                 "    _graft_after_attention = _pdl8_graft_after_attention\n", 1)
    head = A.sub(head, "PDL on for the non-cooperative edges of the T <= 8 chain.",
                 "PDL on for the non-cooperative edges of the T = 1, 16 and 32 chains (T = 8 off).", 1)
    body = (head + "\n\n" + copy("_sl_tile_route", A.ROUTE_CG) + "\n\n\n"
            + copy("_sl_cluster_finish", A.FINISH_CG) + "\n\n\n" + copy("_graft_row_boundary_pre", ROW_CG) + "\n\n\n"
            + copy("_sl_tile_after_attention", A.AFTER) + "\n\n\n" + copy("_graft_after_attention", GRAFT_AFTER) + "\n\n\n"
            + copy("_decode_attention7", A.ATTN) + tail)
    DST.write_text(s.rstrip("\n") + "\n" + body)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(DST.read_bytes()).hexdigest(), len(DST.read_bytes()))


if __name__ == "__main__":
    main()
