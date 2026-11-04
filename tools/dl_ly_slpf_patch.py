#!/usr/bin/env python3
"""ly_slpf: at decode 1 the MoE's 132 CTAs prefetch the unpooler slice's 6.3 MB of codes into L2.

Base: muse_decode_layer/cand/ly_v1un.py (SHA256 88fbae86..., checked). Target: the decode 1 slice cgemv
(cgemv40_kernel<128, 2>, 512 CTAs, 12,288 B of prepare-time codes each, e["stream"] of the unpooler cg entry), whose
code loads run from DRAM after the MoE's exit at full (about 1.5 us) and under the MoE's final pass at sliding
(docs/runs-decode-layer.md, "Decode 1 finish and slice launch"). Each MoE CTA issues
cp.async.bulk.prefetch.L2.global for its 1/132 share of that tensor; no data enters shared memory or registers, so no
value the kernel or the slice computes can change, and nothing is ordered by it.

--at a (candidate ly_slpf): the helper warp's lane 0 issues it right after the CTA's doneA credit (the CTA's phase A
end, at most one handoff before phase B starts); the helper warp then only waits for the first phase B partials.
--at b (candidate ly_slpfb): the same lane issues it after the helper's last phase B epilogue, before the CTA's
final WS_JOIN (the CTA's phase B end, DRAM otherwise idle until the final pass).

Changes, helper0 source (TINY / TINY2 / MID share it; OKC_SLPF is defined only in MOE_TINY_DEFINES, M <= 4, so the
TINY2 and MID builds preprocess to the base's device and host code): an `OKC_SLPF` default, two Params fields and
`slpf_issue` under `#if OKC_SLPF`, one call site, a host setter `okc_set_slpf` and the Params fill in okc_run. File:
the setter's declaration in MOE_CPP_SRC, `_ma_ext` binds it when the source defines it, `MOE_TINY_DEFINES` gains
-DOKC_SLPF=1, and a `_moe` wrapper hands the TINY extension the slice's stream tensor (or an empty one) on every
call, so every T = 1 launch, captured or eager, carries the pointer.

    python3 tools/dl_ly_slpf_patch.py [--at a|b] [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
AT = sys.argv[sys.argv.index("--at") + 1] if "--at" in sys.argv else "a"
DST = ROOT / ("muse_decode_layer/cand/ly_slpf.py" if AT == "a" else "muse_decode_layer/cand/ly_slpfb.py")
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]

HELPER_START = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\""

SLPF_FN = (
    '#if OKC_SLPF\n'
    '// slpf: this CTA\'s share of the decode 1 unpooler slice\'s codes (prepare-time constants the slice cgemv reads\n'
    '// after this kernel) is pulled into L2; nothing lands in this CTA and nothing is ordered. The share is computed on\n'
    '// the host: an integer divide here would run from a cold instruction cache.\n'
    '__device__ __forceinline__ void slpf_issue(const Params& p) {\n'
    '  const unsigned o0 = blockIdx.x * p.slpf_share, o1 = min(p.slpf_bytes, o0 + p.slpf_share);\n'
    '  for (unsigned o = o0; o < o1; o += 32768u)\n'
    '    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\\n" ::"l"(p.slpf + o), "r"(min(32768u, o1 - o)) : "memory");\n'
    '}\n'
    '#endif\n'
    '\n'
    '__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n')

HELPER_EDITS = [
    ('#ifndef OKC_EARLYN\n#define OKC_EARLYN 1\n#endif\n',
     '#ifndef OKC_EARLYN\n#define OKC_EARLYN 1\n#endif\n'
     '#ifndef OKC_SLPF\n#define OKC_SLPF 0   // slpf: L2 prefetch of the decode 1 unpooler slice\'s codes (okc_set_slpf)\n#endif\n', 1),
    ('  __nv_bfloat16* out;\n  unsigned long long* dbg;\n};\n',
     '  __nv_bfloat16* out;\n  unsigned long long* dbg;\n'
     '#if OKC_SLPF\n'
     '  const unsigned char* slpf; unsigned slpf_bytes, slpf_share;  // slice codes to pull into L2 (bytes 0: none)\n'
     '#endif\n'
     '};\n', 1),
    ('__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n', SLPF_FN, 1),
    ('static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n'
     '\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n',
     'static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n'
     '\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n'
     '\n// slpf: the tensor stays alive in the caller\'s state; only its address and size are kept\n'
     'static const void* g_slpf_ptr = nullptr;\nstatic int64_t g_slpf_bytes = 0;\n'
     'void okc_set_slpf(torch::Tensor t) {\n'
     '  TORCH_CHECK(t.numel() == 0 || (t.is_cuda() && t.is_contiguous() && (t.numel() * t.element_size()) % 16 == 0\n'
     '              && ((uintptr_t)t.data_ptr() & 15) == 0 && t.numel() * t.element_size() < (1ll << 31)), "okc_set_slpf");\n'
     '  g_slpf_ptr = t.numel() ? t.data_ptr() : nullptr;\n'
     '  g_slpf_bytes = t.numel() ? t.numel() * t.element_size() : 0;\n'
     '}\n', 1),
    ('  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n',
     '  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n'
     '#if OKC_SLPF\n'
     '  p.slpf = (const unsigned char*)g_slpf_ptr;\n'
     '  p.slpf_bytes = (unsigned)g_slpf_bytes;\n'
     '  p.slpf_share = grid > 0 ? (unsigned)(((g_slpf_bytes + grid - 1) / grid + 15) & ~(int64_t)15) : 0u;\n'
     '#endif\n', 1),
]
CALL_A = (
    ('            atomicAdd(p.doneA + jj, b - a);\n          }\n        }\n      }\n    } else {\n',
     '            atomicAdd(p.doneA + jj, b - a);\n          }\n'
     '#if OKC_SLPF\n'
     '          if (p.slpf_bytes) slpf_issue(p);  // after the credit: the helper next waits for phase B partials\n'
     '#endif\n'
     '        }\n      }\n    } else {\n', 1),
)
CALL_B = (
    ('        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n  }\n'
     '  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n',
     '        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n  }\n'
     '#if OKC_SLPF\n'
     '  if (lane == 0 && p.slpf_bytes) slpf_issue(p);  // the helper\'s last epilogue is issued: phase B end\n'
     '#endif\n'
     '  __threadfence();\n  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n}\n', 1),
)
FILE_EDITS = [
    ('void okc_set_pdl(int64_t on);\nstd::vector<int64_t> okc_attrs();\n"""\n',
     'void okc_set_pdl(int64_t on);\nstd::vector<int64_t> okc_attrs();\nvoid okc_set_slpf(torch::Tensor t);\n"""\n', 1),
    ('            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),\n',
     '            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else [])\n'
     '            + (["okc_set_slpf"] if "okc_set_slpf" in src else []),\n', 1),
]
TAIL = '''

# slpf (tools/dl_ly_slpf_patch.py): the TINY build (M <= 4, decode 1) prefetches the m = 1 unpooler slice's codes into
# L2; every call hands the TINY extension the cgemv stream it will read, or an empty tensor when the cgemv is not set up
MOE_TINY_DEFINES += ("-DOKC_SLPF=1",)
_SLPF_NONE = {}
_slpf_prior_moe = _moe
def _moe(M=None, state=None):
    ext = _slpf_prior_moe(M, state)
    if ext is not None and state is not None and M is not None and M <= MOE_TINY_MAX_M and hasattr(ext, "okc_set_slpf"):
        e = ((state.get("bfp") or {}).get("cg") or {}).get("unpooler")
        if e is not None:
            ext.okc_set_slpf(e["stream"])
        else:
            dev = torch.cuda.current_device()
            if dev not in _SLPF_NONE:
                _SLPF_NONE[dev] = torch.empty(0, dtype=torch.uint8, device=f"cuda:{dev}")
            ext.okc_set_slpf(_SLPF_NONE[dev])
    return ext
'''
HEADER = {
    "a": ("# ly_slpf: ly_v1un with the decode 1 MoE (TINY build) prefetching the unpooler slice's codes into L2, each CTA "
          "its 1/132 share after its doneA credit (tools/dl_ly_slpf_patch.py --at a).\n"),
    "b": ("# ly_slpfb: ly_v1un with the decode 1 MoE (TINY build) prefetching the unpooler slice's codes into L2, each "
          "CTA its 1/132 share after its last phase B epilogue (tools/dl_ly_slpf_patch.py --at b).\n"),
}


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    a = text.index(HELPER_START)
    b = text.index('"""', a + len(HELPER_START))
    blk = text[a:b]
    for old, new, n in HELPER_EDITS + list(CALL_A if AT == "a" else CALL_B):
        if blk.count(old) != n:
            sys.exit(f"helper0 anchor found {blk.count(old)} times, expected {n}: {old!r}")
        blk = blk.replace(old, new)
    text = text[:a] + blk + text[b:]
    for old, new, n in FILE_EDITS:
        if text.count(old) != n:
            sys.exit(f"file anchor found {text.count(old)} times, expected {n}: {old!r}")
        text = text.replace(old, new)
    text = HEADER[AT] + text + TAIL
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
