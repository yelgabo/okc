"""Debug copy of the public MoE file's helper0 extension: the same kernel source with three dumps added.

The public file is only read. Its `_MA_ALT_SRCS['moe_helper0']` string gets:
  1. per-unit phase B sums of squares: next to each of the two WS-build atomicAdds into ss (the helper warp's
     epilogue in ws_helper and the compute warps' last-unit epilogue in skinny_ring_ws), the unit's partial is also
     stored to dbg_sq[entry position * 256 + rb];
  2. the routing tables after route(): njF, njS, (jE, jS, jC) per job and pos[entry] (entry = token * 8 + slot,
     pos = row in inter / ybuf or -1).
  3. each unit's raw FP32 cross-warp sums (the float4 `sm` before the epilogue), phase A and phase B, in both
     WS epilogues.
All three write through Params::dbg (scratch[3], unused by the plain builds: OKC_DEBUG is 0). None changes an
arithmetic expression. The flags are the public `_ma_ext` flags. The written source is helper0_debug.cu next to
this file.
"""

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EMAX = 384 * 8
JMAX = 32 + EMAX // 64 + 8
RB_B = 256
DBG_SQ_FLOATS = EMAX * RB_B
DBG_ROUTE_OFF = DBG_SQ_FLOATS             # int offset of the routing dump
DBG_POS_OFF = DBG_ROUTE_OFF + 2 + 3 * JMAX
DBG_ACC_A_OFF = DBG_POS_OFF + EMAX      # float [EMAX][512 * 16]: phase A raw FP32 unit sums, col 16 rb + g (+ 8)
DBG_ACC_B_OFF = DBG_ACC_A_OFF + EMAX * 8192   # float [EMAX][4096]: phase B raw FP32 unit sums, col 16 rb + g (+ 8)
DBG_WORDS = DBG_ACC_B_OFF + EMAX * 4096

SQ_OLD = "if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);"
SQ_NEW = ("if (gg == 0 && tk < cnt) { atomicAdd(&ss_cur()[st + tk], sq); "
          "if (p.dbg) reinterpret_cast<float*>(p.dbg)[(size_t)(st + tk) * RB_B + rb] = sq; }")
ACC_OLD = "const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];"
ACC_NEW = ACC_OLD + f"""
    if (p.dbg) {{  // CL-0.5 debug: the unit's raw FP32 cross-warp sums before any epilogue rounding
      float* da = reinterpret_cast<float*>(p.dbg) + (PB ? {DBG_ACC_B_OFF} : {DBG_ACC_A_OFF});
      const size_t W = PB ? 4096 : 8192;
      for (int h = 0; h < 2; ++h) {{
        const int tk = 2 * tt + h;
        if (tk < cnt) {{ da[(size_t)(st + tk) * W + rb * 16 + gg] = h ? sm.y : sm.x; da[(size_t)(st + tk) * W + rb * 16 + 8 + gg] = h ? sm.w : sm.z; }}
      }}
    }}"""
ROUTE_OLD = "  route(p, S);\n"
ROUTE_NEW = ROUTE_OLD + f"""  if (p.dbg && blockIdx.x == 0 && tid == 0) {{  // CL-0.5 debug: routing tables
    int* d = reinterpret_cast<int*>(p.dbg) + {DBG_ROUTE_OFF};
    d[0] = S.njF; d[1] = S.njS;
    for (int j = 0; j < S.njF + S.njS; ++j) {{ d[2 + 3 * j] = S.jE[j]; d[3 + 3 * j] = S.jS[j]; d[4 + 3 * j] = S.jC[j]; }}
    for (int i = 0; i < p.M * 8; ++i) d[{DBG_POS_OFF - DBG_ROUTE_OFF} + i] = S.pos[i];
  }}
"""


def _replace_in(src, start_marker, old, new):
    a = src.index(start_marker)
    b = src.index("\n}\n", a)
    region = src[a:b]
    assert region.count(old) == 1, (start_marker, region.count(old))
    return src[:a] + region.replace(old, new) + src[b:]


def debug_source(pub):
    src = pub._MA_ALT_SRCS["moe_helper0"]
    assert src.count(ROUTE_OLD) == 1
    src = _replace_in(src, "__device__ void skinny_ring_ws(", SQ_OLD, SQ_NEW)
    src = _replace_in(src, "__device__ void ws_helper(", SQ_OLD, SQ_NEW)
    src = _replace_in(src, "__device__ void skinny_ring_ws(", ACC_OLD, ACC_NEW)
    src = _replace_in(src, "__device__ void ws_helper(", ACC_OLD, ACC_NEW)
    src = src.replace(ROUTE_OLD, ROUTE_NEW)
    return src


def write_source(pub):
    src = debug_source(pub)
    path = os.path.join(HERE, "helper0_debug.cu")
    with open(path, "w") as f:
        f.write(src)
    return path, hashlib.sha256(src.encode()).hexdigest()


def full_defines(pub, defines, sfx):
    """The tuple the public _ma_ext builds for (defines + sfx), with its DQ7 addition."""
    d = tuple(defines) + tuple(sfx)
    if pub.DQ7 and "-DOKC_DQ7=1" not in d:
        d = d + ("-DOKC_DQ7=1",)
    return d


def cflags(pub, defines):
    """The public _ma_ext's extra_cuda_cflags for this define tuple (sm_90a)."""
    import torch
    defs = [d for d in defines if not d.startswith("@")]
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    return ["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
            *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={pub._MA_MAIN_NS if major >= 9 else 1}"]),
            f"-DOKC_FNST={4 if major >= 9 else 3}",
            *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs]


def build(pub, defines, sfx):
    from torch.utils.cpp_extension import load_inline
    d = full_defines(pub, defines, sfx)
    assert any(x in ("@moe_helper0", "@moe_helper0df") for x in d), d
    src = debug_source(pub)
    tag = hashlib.md5((pub.MOE_CPP_SRC + src + repr((d, pub._MA_MAIN_NS))).encode()).hexdigest()[:16]
    return load_inline(name=f"cl05_h0dbg_{tag}", cpp_sources=pub.MOE_CPP_SRC, cuda_sources=src,
                       functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=cflags(pub, d), verbose=False)
