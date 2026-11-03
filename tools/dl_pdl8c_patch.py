#!/usr/bin/env python3
"""Composed candidate: the public layer release (ba9e327c) + the parity lane's m = 1 cgemv block
(tools/dl_cg_patch.py, its check-2 bytes 440397ab) [+ its T = 1 split-front block with --front] + the pdl8b block
(tools/dl_pdl8b_patch.py, public fefd7f7b8e834bd7). Writes muse_decode_layer/cand/g1b4_pdl8c.py (g1b4_pdl8cf.py with
--front).

    python3 tools/dl_pdl8c_patch.py CG_TOOL C2 [--front] [--cgpdl]
      CG_TOOL  tools/dl_cg_patch.py from branch worktree-agent-a51254a802fef2de0
      C2       git show 3044ee7:muse_bf16_proj/submission.py (b2c4c096)

Order: the cgemv block comes first, so its prepare (tuner picks re-applied after the pinned-pick prepare) runs
inside the pdl8b prepare, and the PDL A/B and stress time the composed T = 1 chain. The cgemv kernels are plain
launches with no griddepcontrol, so the edges into them stay serialized and the edges out of them start at their
exit; the PDL enable map is set by the pdl8b prepare after every cgemv pick is final.
"""
import ast
import hashlib
import importlib.util
import pathlib
import py_compile
import sys

HERE = pathlib.Path(__file__).resolve().parent
CG_BYTES_SHA = "440397ab59440dd053c93d152f2110f3548ad97694615783a582fd6ced042e5a"


CG_PDL_EDITS = [
    ("#pragma unroll\n        for (int i = 0; i < CPT; ++i) {\n            xa[i] = __ldg(xp + i * NTH * 2);\n"
     "            xb[i] = __ldg(xp + i * NTH * 2 + 1);\n        }",
     "        asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
     "        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
     "#pragma unroll\n        for (int i = 0; i < CPT; ++i) {\n            xa[i] = __ldcg(xp + i * NTH * 2);\n"
     "            xb[i] = __ldcg(xp + i * NTH * 2 + 1);\n        }", 1),
    ("        float corr0 = 0.f, corr1 = 0.f;\n        for (int e = lane; e <= count; e += 32) {",
     "        float corr0 = 0.f, corr1 = 0.f;\n        asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
     "        for (int e = lane; e <= count; e += 32) {", 1),
    ("(unsigned)__ldg(reinterpret_cast<const unsigned short*>(x) + kk) << 16);",
     "(unsigned)__ldcg(reinterpret_cast<const unsigned short*>(x) + kk) << 16);", 1),
    ("        cgemv_kernel<NTH, CPT><<<half_n, NTH + 32, 0, s>>>(stream, half_n, table, pmax, base4, x, out);",
     "        cg_go(cgemv_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);", 1),
    ("        cgemv40_kernel<NTH, CPT><<<half_n, NTH + 32, 0, s>>>(stream, half_n, table, pmax, base4, x, out);",
     "        cg_go(cgemv40_kernel<NTH, CPT>, half_n, NTH + 32, s, stream, half_n, table, pmax, base4, x, out);", 1),
    ("template <int NTH, int CPT>\nvoid cgemv_launch(",
     "static int g_cg_pdl = 0;\n"
     "template <typename... P, typename... A>\n"
     "void cg_go(void (*kern)(P...), int grid, int block, cudaStream_t s, A... a) {\n"
     "    cudaLaunchConfig_t cfg = {};\n"
     "    cfg.gridDim = dim3(grid);\n"
     "    cfg.blockDim = dim3(block);\n"
     "    cfg.dynamicSmemBytes = 0;\n"
     "    cfg.stream = s;\n"
     "    cudaLaunchAttribute at[1];\n"
     "    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
     "    at[0].val.programmaticStreamSerializationAllowed = 1;\n"
     "    cfg.attrs = at;\n"
     "    cfg.numAttrs = g_cg_pdl ? 1 : 0;\n"
     "    (void)cudaLaunchKernelEx(&cfg, kern, a...);\n"
     "}\n\n"
     "template <int NTH, int CPT>\nvoid cgemv_launch(", 1),
    ("    C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n#endif\n",
     "    C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\nvoid cg_set_pdl(int64_t on) { g_cg_pdl = on ? 1 : 0; }\n#endif\n", 1),
]

CG_PDL_BLOCK = r'''

# pdl8c2: the three m = 1 cgemv kernels as PDL participants at T = 1 (wait before x, x through __ldcg, trigger
# after the weight issue; programmatic launch only while T = 1 runs with PDL on).
_PDL8_CG_EDITS = @@CGEDITS@@


def _pdl8_cg_build():
    global _CG_EXT
    try:
        kw = dict(name="okc_dl_cgemv_" + _cg_hashlib.md5(CG_CUDA_SRC.encode()).hexdigest()[:12],
                  cpp_sources="#include <torch/extension.h>\n" + CG_CPP_DECLS, cuda_sources=CG_CUDA_SRC,
                  functions=["cgemv_fwd", "cg_set_pdl"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                  verbose=False)
        try:
            _CG_EXT = load_inline(no_implicit_headers=True, **kw)
        except TypeError:
            kw["cuda_sources"] = CG_CUDA_SRC.replace("#include <ATen/core/Tensor.h>", "#include <torch/types.h>")
            _CG_EXT = load_inline(**kw)
    except Exception as ex:  # noqa: BLE001
        _CG_ERR.append(ex)


_pdl8_prior_cg_call = _cg_call
def _pdl8_cg_call(e, x, k):
    if not _PDL8:
        return _pdl8_prior_cg_call(e, x, k)
    _CG_EXT.cg_set_pdl(1)
    try:
        return _pdl8_prior_cg_call(e, x, k)
    finally:
        _CG_EXT.cg_set_pdl(0)


if PDL_CHAIN:
    for _old, _new, _n in _PDL8_CG_EDITS:
        CG_CUDA_SRC = _pdl8_sub(CG_CUDA_SRC, _old, _new, _n)
    CG_CPP_DECLS = CG_CPP_DECLS + "void cg_set_pdl(int64_t on);\n"
    _cg_build = _pdl8_cg_build
    _cg_call = _pdl8_cg_call
'''


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    front = "--front" in sys.argv[1:]
    cgpdl = "--cgpdl" in sys.argv[1:]
    cg_tool, c2 = args
    CG = load("dl_cg_patch", cg_tool)
    B = load("dl_pdl8b_patch", HERE / "dl_pdl8b_patch.py")
    release, block = B.make_block()
    rel_bytes = release.encode()
    c2_bytes = pathlib.Path(c2).read_bytes()
    plain = CG.build(rel_bytes, c2_bytes, front=False)
    if hashlib.sha256(plain).hexdigest() != CG_BYTES_SHA:
        sys.exit("the cgemv tool no longer reproduces the parity lane's check-2 bytes 440397ab")
    composed = CG.build(rel_bytes, c2_bytes, front=front).decode()
    for name in ("_pdl8", "PDL8_", "PDL_CHAIN"):
        if name in composed:
            sys.exit(f"{name} already present before the pdl8b block")
    head = "# okc: problem=muse_decode_layer\n"
    composed = composed.replace(head, head + "# pdl8c: release + cgemv m = 1 port" + (" + split T = 1 front" if front else "")
                                + " (tools/dl_cg_patch.py) + pdl8b (tools/dl_pdl8b_patch.py), composed by"
                                " tools/dl_pdl8c_patch.py.\n", 1)
    out = composed.rstrip("\n") + "\n" + block
    name = "g1b4_pdl8c" + ("f" if front else "")
    if cgpdl:
        tree = ast.parse(composed)
        lit = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "CG_CUDA_SRC")
        for old, new, n in CG_PDL_EDITS:
            if lit.count(old) != n:
                sys.exit(f"cgemv anchor {old[:60]!r}: {lit.count(old)} found, {n} expected")
            lit = lit.replace(old, new)
        out = out.rstrip("\n") + "\n" + CG_PDL_BLOCK.replace("@@CGEDITS@@", repr(CG_PDL_EDITS))
        name = name + "2"
    dst = B.A.ROOT / f"muse_decode_layer/cand/{name}.py"
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst, hashlib.sha256(dst.read_bytes()).hexdigest(), len(dst.read_bytes()))


if __name__ == "__main__":
    main()
