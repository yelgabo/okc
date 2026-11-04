#!/usr/bin/env python3
"""ly_acl: decode 1 sliding attention in the layer as the ONE_CL cluster kernel (attn_one.cu mode 136), PDL secondary
of qkv, its window's K/V issued before griddepcontrol.wait.

Base ly_v1un (88fbae86). Stamps (docs/runs-decode-layer.md "Decode 1 sliding attention in the layer (box)") put
2.6 us of the 5.2 us between qkv's end and the o projection's wait return into the workspace merge (reds, the
acq_rel ticket, the last CTA's finalize) and 0.42 us into attention's own completion flush. The ONE_CL kernel
(`muse_attention_decode/attn_one.cu`, `attn_cl_kernel<1>`; one CTA per KV head and 4-tile range, the 8 CTAs of a KV
head in one cluster, DSMEM sum after one cluster barrier, rank r finalizes head r; no workspace, red, ticket or
counter) lost +1.31 us standalone (c1becf5a66274d37), where its cluster launch and the window's pull sit on the
critical path. Inside the layer both can sit before the wait: the grid launches during qkv (PDL) and every operand
of the K/V copies is a layer input (page_table, seq_lens, the cached rows below seq; the new token's row is never
read from the cache).

Edits:
- the file embeds `muse_attention_decode/attn_one.cu` (built with -DONE_CL=1 as its own extension) with two edits in
  attn_cl_kernel: (1) order: page indices (group 0) waited, the speculated length checked, the K/V copies of every
  tile issued, then `griddepcontrol.wait` and `griddepcontrol.launch_dependents`, then the q / new k,v / constants
  group (the qkv reads) and the gate L2 prefetch, waited with wait_group 0; (2) Instance<L, 136>::launch adds the
  programmatic-serialization attribute when attn_cl_set_pdl(1) was called. Everything after (q pass, new-row pass
  and cache write, tile math, cluster.sync, DSMEM gather, finalize, second cluster.sync) is the standalone code.
- the T = 1 attention launcher (`_pdl8_decode_attention7`, `_decode_attention7`) takes this kernel for batch 1,
  q_len 1, a window of at most 2048 + 1 - q_len keys, page size 1, contiguous qkv rows of 2304 elements; anything else
  runs the base path. q_len 8 and every other shape are unchanged.

    python3 tools/dl_ly_acl_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_acl.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_acl.py"
ONE = ROOT / "muse_attention_decode/attn_one.cu"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

KV_END = "#pragma unroll\n    for (int t = 0; t < NT; ++t) {\n        if (t < ntiles) {\n            issue_kv(t);\n        }\n        cp_commit();\n    }\n    cp_wait<NT>();\n    __syncthreads();\n"

LAUNCH_OLD = """            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim.x = CL_SIZE;
            attr[0].val.clusterDim.y = 1;
            attr[0].val.clusterDim.z = 1;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
            const cudaError_t err = cudaLaunchKernelEx(&cfg, attn_cl_kernel<L>, p);"""
LAUNCH_NEW = """            cudaLaunchAttribute attr[2];
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim.x = CL_SIZE;
            attr[0].val.clusterDim.y = 1;
            attr[0].val.clusterDim.z = 1;
            attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[1].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = attr;
            cfg.numAttrs = g_cl_pdl ? 2 : 1;
            const cudaError_t err = cudaLaunchKernelEx(&cfg, attn_cl_kernel<L>, p);"""
PDL_GLOBAL_OLD = "constexpr int M_CL = 128;\n"
PDL_GLOBAL_NEW = ("constexpr int M_CL = 128;\n"
                  "// ly_acl: the cluster launch carries the programmatic attribute (a PDL secondary of qkv)\n"
                  "static int g_cl_pdl = 0;\n")
SET_PDL = """
void attn_cl_set_pdl(int64_t on) { g_cl_pdl = on ? 1 : 0; }
"""
CPP = r"""
#include <torch/extension.h>
torch::Tensor attn_one(torch::Tensor qkv, torch::Tensor page_table, torch::Tensor seq_lens,
                       torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor k_scale,
                       torch::Tensor v_scale, torch::Tensor k_cache, torch::Tensor v_cache,
                       torch::Tensor ws_acc, torch::Tensor ws_l, torch::Tensor ws_cnt,
                       int64_t batch, int64_t q_len, int64_t page_size, int64_t window, bool rope,
                       double sl2, int64_t n_splits, int64_t nc, int64_t mode, bool coop_launch,
                       torch::Tensor dbg, int64_t phase_limit);
void attn_cl_set_pdl(int64_t on);
"""

PY_BLOCK = r'''

ACL_CPP_SRC = r"""@@CPP@@"""
_ACL = None
_ACL_DUMMY = {}


def _acl():
    global _ACL
    if _ACL is None:
        import hashlib as _hl
        from torch.utils.cpp_extension import load_inline
        _ACL = load_inline(name="muse_acl_" + _hl.md5(ACL_CUDA_SRC.encode()).hexdigest()[:12], cpp_sources=ACL_CPP_SRC,
                           cuda_sources=ACL_CUDA_SRC, functions=["attn_one", "attn_cl_set_pdl"],
                           extra_cuda_cflags=["-O3", "-DONE_CL=1", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _ACL


def _acl_fits(data, cfg, qkv):
    return (cfg["batch"] == 1 and cfg["q_len"] == 1 and cfg["page_size"] == 1 and cfg["window"]
            and (cfg["window"] + 63) // 64 <= 32 and qkv.dim() == 2 and qkv.shape[1] == 2304
            and qkv.stride(1) == 1 and qkv.stride(0) == 2304 and torch.cuda.get_device_capability()[0] >= 9)


def _acl_attention(data, cfg, qkv, pdl):
    dev = qkv.device
    d = _ACL_DUMMY.get(str(dev))
    if d is None:
        d = _ACL_DUMMY[str(dev)] = (torch.zeros(4, dtype=torch.float32, device=dev),
                                    torch.zeros(4, dtype=torch.int32, device=dev),
                                    torch.empty(0, dtype=torch.int32, device=dev))
    m = _acl()
    m.attn_cl_set_pdl(1 if pdl else 0)
    theta = cfg["rope_theta"]
    out = m.attn_one(qkv, data["page_table"], data["seq_lens"], data["cache_loc"],
                     _inv_freq(theta if theta else 1.0, dev), data["k_scale"], data["v_scale"],
                     data["k_cache"], data["v_cache"], d[0], d[0], d[1], 1, 1, 1, cfg["window"], bool(theta),
                     cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, 8, 1, 136, False, d[2], 0)
    return out.view(1, Q_DIM)
'''

DISPATCH_OLD = """    out = _launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, attention_state=state, prefetch=prefetch, **pdl)
    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)"""
DISPATCH_NEW = """    if _acl_fits(data, cfg, qkv):   # (ly_acl: decode 1 sliding as the ONE_CL cluster kernel)
        return _acl_attention(data, cfg, qkv, pdl.get("pdl", False))
""" + DISPATCH_OLD
HEADER = ("# ly_acl: ly_v1un with decode 1 sliding attention as attn_one.cu's ONE_CL cluster kernel, PDL secondary of "
          "qkv, K/V before the wait (tools/dl_ly_acl_patch.py).\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def cl_source():
    s = ONE.read_text()
    g1_start = "    // Group 1: this head's query rows"
    g1_stop = "    cp_commit();\n    cp_wait<1>();\n    __syncthreads();\n    if (seq_real != seq) {\n"
    i = s.index(g1_start)
    j = s.index(g1_stop)
    assert s.count(g1_start) == 1 and s.count(g1_stop) == 1 and i < j
    g1 = s[i:j]   # comment, "{", the q / new k,v / constants copies and the gate prefetch, "}"
    s = s[:i] + s[j:]
    s = sub(s, "    cp_commit();\n    cp_commit();\n    cp_wait<1>();\n    __syncthreads();\n    if (seq_real != seq) {\n"
               "        cp_wait<0>();\n        __syncthreads();\n",
            "    cp_commit();\n    cp_wait<0>();   // (ly_acl: group 0 only; the qkv reads come after the wait)\n"
            "    __syncthreads();\n    if (seq_real != seq) {\n", 1, "group order")
    kv_new = ("#pragma unroll\n    for (int t = 0; t < NT; ++t) {\n        if (t < ntiles) {\n            issue_kv(t);\n"
              "        }\n        cp_commit();\n    }\n"
              "    // ly_acl: page_table, seq_lens and the cached rows above are layer inputs; qkv (q, the new k / v, the\n"
              "    // gate) is the preceding grid's output: read only after the wait\n"
              "    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
              "    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n"
              + g1 + "    cp_commit();\n    cp_wait<0>();\n    __syncthreads();\n")
    s = sub(s, KV_END, kv_new, 1, "kv issue")
    s = sub(s, LAUNCH_OLD, LAUNCH_NEW, 1, "launch")
    s = sub(s, PDL_GLOBAL_OLD, PDL_GLOBAL_NEW, 1, "pdl global")
    k = s.index("#define ONE_INSTANCES(X) \\\n    X(1, 1) X(1, 3)")
    tail = "ONE_CL_INSTANCES(X)\n#endif"
    e = s.index(tail, k)
    s = s[:k] + "#define ONE_INSTANCES(X) X(1, 136)   // (ly_acl: the one instance the layer launches)\n#endif" + s[e + len(tail):]
    return s + SET_PDL


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    cl = cl_source()
    assert '"""' not in cl
    block = 'ACL_CUDA_SRC = r"""' + cl + '"""' + PY_BLOCK.replace("@@CPP@@", CPP)
    text = sub(text, DISPATCH_OLD, DISPATCH_NEW, 2, "dispatch")
    anchor = "ATT_HAS_PDL = True"
    text = sub(text, anchor, block + "\n\n" + anchor, 1, "module block")
    text = HEADER + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
