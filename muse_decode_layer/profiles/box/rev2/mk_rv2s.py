"""Builds ly_rv2s.py: ly_stk4 plus launch-time screen knobs that default to the base behaviour (revival round 2).
Also adds MOE_X: {"tiny" | "tiny2": defines} replaces that build's defines (cached under its own ma_ext key)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:60], s.count(old))
    s = s.replace(old, new)


sub("#define OKC_A16SM 81920", "#include <cstdlib>\n#define OKC_A16SM 81920")
sub("  const int smk = (L == 1 && kern == kn[1] && OKC_A16SM > sm) ? OKC_A16SM : sm;\n",
    "  int smk = (L == 1 && kern == kn[1] && OKC_A16SM > sm) ? OKC_A16SM : sm;\n"
    "  if (!p.flat && !cl) {\n"
    "    const char* ev = getenv(\"OKC_ASMEM\");\n"
    "    const int rs = ev ? atoi(ev) : 0;\n"
    "    if (rs > smk) {\n"
    "      cudaFuncAttributes fa;\n"
    "      cudaFuncGetAttributes(&fa, kern);\n"
    "      if (fa.maxDynamicSharedSizeBytes < rs) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, rs);\n"
    "      int o = 0;\n"
    "      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kern, Cfg<L, false>::NTX, rs);\n"
    "      // a cooperative grid must stay resident at the larger reservation or its grid barrier never returns\n"
    "      if (o > 0 && (!q.coop || (long long)grid.x * grid.y * grid.z <= (long long)o * nsm)) smk = rs;\n"
    "    }\n"
    "  }\n")
sub("    (_mod or _att7()).run(qkv, data[\"seq_lens\"],",
    "    os.environ[\"OKC_ASMEM\"] = str(ASMEM_CFG.get((B, L, bool(window)), 0))\n"
    "    (_mod or _att7()).run(qkv, data[\"seq_lens\"],")
sub("def _att7():", "ASMEM_CFG = {}   # (B, L, sliding) -> dynamic shared memory bytes of the attention launch; 0 keeps the base\n\n\ndef _att7():")
sub("    _CG_EXT.cgemv_fwd(e[\"stream\"], e[\"table\"], e[\"pmax\"], e[\"base4\"], x.view(1, k), out, e[\"nth\"])",
    "    _CG_EXT.cgemv_fwd(e[\"stream\"], e[\"table\"], e[\"pmax\"], e[\"base4\"], x.view(1, k), out,\n"
    "                      CG_NTH_OVR.get((e[\"n\"], k), e[\"nth\"]))")
sub("def _cg_call(e, x, k):", "CG_NTH_OVR = {}\n\n\ndef _cg_call(e, x, k):")
sub("    if state[\"ma_sc6\"] and \"-DOKC_WGS=1\" in defines:\n",
    "    if key in MOE_X:\n"
    "        defines = tuple(MOE_X[key])\n"
    "        key = key + repr(defines)\n"
    "    if state[\"ma_sc6\"] and \"-DOKC_WGS=1\" in defines:\n")
sub("def _fork_native_moe(M=None, state=None):", "MOE_X = {}\n\n\ndef _fork_native_moe(M=None, state=None):")
i = s.index("def _v1fs_reserve():")
j = s.index("def _pdl8_sl_tile_after_attention")
s = s[:i] + '''V1FS_T = {8: _V1FS_SMEM}   # T -> finish dynamic shared memory; kernels below it are raised, others restored
_V1FS_ORIG = {}
_V1FS_SET = {}


def _v1fs_reserve(T=8):
    global _V1FS_CU
    cache = _sl_cluster_finish_cg.device_caches[triton.runtime.driver.active.get_current_device()][0]
    want = V1FS_T.get(T, 0)
    for k in cache.values():
        if k.metadata.num_warps != SL_CLUSTER_CONFIG[T][0]:
            continue
        orig = _V1FS_ORIG.setdefault(id(k), k.packed_metadata[2])
        tgt = max(orig, want)
        if k.packed_metadata[2] == tgt:
            continue
        if tgt > _V1FS_SET.get(id(k), orig):
            k._init_handles()
            if _V1FS_CU is None:
                _V1FS_CU = _C.CDLL("libcuda.so.1")
            rc = _V1FS_CU.cuFuncSetAttribute(_C.c_void_p(k.function), _C.c_int(8), _C.c_int(tgt))
            if rc != 0:
                raise RuntimeError(f"v1fs: cuFuncSetAttribute returned {rc}")
            _V1FS_SET[id(k)] = tgt
        k.packed_metadata = (k.packed_metadata[0], k.packed_metadata[1], tgt)


''' + s[j:]
sub("    if T == 8:\n        _v1fs_reserve()\n", "    _v1fs_reserve(T)\n", 2)
open(dst, "w").write(s)
