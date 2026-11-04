#!/usr/bin/env python3
"""ly_unpf: at decode 16 / 32 each MoE CTA pulls its share of the unpooler sk2's first weight stages into L2 at its
phase B end, so the unpooler's first stages read L2 instead of DRAM.

Base: muse_decode_layer/cand/ly_stk1.py (SHA256 d7db12a1..., checked). At T = 16 / 32 the unpooler is sk2 over all
8192 rows (50.3 MB of 12-bit codes in a stage-major stream: 16 stages of K, each one contiguous 3.1 MB region over
all row blocks), a PDL secondary of the MoE (ly_d16un). Its CTAs do not fit beside a MoE CTA, so they start as MoE
CTAs exit, after the grid barrier and the final pass; its critical path after the MoE is 13.25 / 14.37 us against
an 11.36 us stream (docs/runs-decode-layer.md, "Decode 16 / 32 sliding timeline (box)"). Meanwhile the MoE CTAs
finish phase B over a spread of 3.9 us (MID, decode 16) and 7.7 us (B4, decode 32) and then wait at the grid
barrier with their weight stream done, so DRAM is partly idle in the MoE's tail.

Change: right after the CTA's `griddepcontrol.launch_dependents` (phase B end, every weight copy of the CTA issued
and consumed), thread 0 issues `cp.async.bulk.prefetch.L2.global` over its 1/grid share of the first
UNPF_STAGES[T] / 16 of the unpooler stream, in 32 KB pieces. The prefetch writes nothing (no shared memory, no
register, no global store) and orders nothing; no fence, barrier, acquire, release, atomic, wait, trigger or launch
attribute is added, removed or moved. Compiled only into the MID (T = 16) and B4 (T = 32) builds (`-DOKC_UNPF=1`);
every other build preprocesses to the base's device code (the host setter is added to both MoE sources).

Edits: in MOE_CUDA_SRC (B4) and the helper0 source (MID): an OKC_UNPF default, three Params fields under
`#if OKC_UNPF`, the prefetch after the trigger, a host setter `okc_set_unpf` and the Params fill in okc_run. File:
the setter's declaration in MOE_CPP_SRC, `_ma_ext` binds it when the source defines it, the fork's B4 defines gain
-DOKC_UNPF=1, MOE_MID_DEFINES gains it, and a `_moe` wrapper hands the extension the stream prefix on every
M = 16 / 32 call (an empty tensor when UNPF_STAGES[M] is 0 or the sk2 entry is missing).

    python3 tools/dl_ly_unpf_patch.py [--stages 16:4,32:4] [--base F] [--pin <sha256>] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk1.py"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
DST = ROOT / "muse_decode_layer/cand/ly_unpf.py"
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]
PIN = "d7db12a14807eeb57df914dac468f4f2567b6bfc4e6771a549c167e92fc39878"
if "--pin" in sys.argv:
    PIN = sys.argv[sys.argv.index("--pin") + 1]
STAGES = {16: 4, 32: 4}
if "--stages" in sys.argv:
    STAGES = {int(k): int(v) for k, v in (kv.split(":") for kv in sys.argv[sys.argv.index("--stages") + 1].split(","))}

BLOCKS = [("MOE_CUDA_SRC = r'''", "'''"), ("_MA_ALT_SRCS['moe_helper0'] = r\"\"\"", '"""')]

BLOCK_EDITS = [
    ('#ifndef OKC_EARLYN\n',
     '#ifndef OKC_UNPF\n#define OKC_UNPF 0   // unpf: L2 prefetch of the T = 16 / 32 unpooler stream prefix (okc_set_unpf)\n#endif\n'
     '#ifndef OKC_EARLYN\n', 1),
    ('  unsigned long long* dbg;\n',
     '  unsigned long long* dbg;\n'
     '#if OKC_UNPF\n'
     '  const unsigned char* unpf; unsigned unpf_bytes, unpf_share;  // unpooler stream prefix to pull into L2 (0: none)\n'
     '#endif\n', 1),
    ('  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n',
     '  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n'
     '#if OKC_UNPF\n'
     '  // unpf: this CTA\'s weight stream is done; pull its share of the unpooler\'s first stages (prepare-time\n'
     '  // constants) into L2. Nothing lands in this CTA and nothing is ordered by it.\n'
     '  if (tid == 0 && p.unpf_bytes) {\n'
     '    const unsigned o0 = blockIdx.x * p.unpf_share, o1 = min(p.unpf_bytes, o0 + p.unpf_share);\n'
     '    for (unsigned o = o0; o < o1; o += 32768u)\n'
     '      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\\n" ::"l"(p.unpf + o), "r"(min(32768u, o1 - o)) : "memory");\n'
     '  }\n'
     '#endif\n', 1),
    ('void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n',
     'void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n'
     '\n// unpf: the tensor stays alive in the caller\'s state; only its address and size are kept\n'
     'static const void* g_unpf_ptr = nullptr;\nstatic int64_t g_unpf_bytes = 0;\n'
     'void okc_set_unpf(torch::Tensor t) {\n'
     '  TORCH_CHECK(t.numel() == 0 || (t.is_cuda() && t.is_contiguous() && (t.numel() * t.element_size()) % 16 == 0\n'
     '              && ((uintptr_t)t.data_ptr() & 15) == 0 && t.numel() * t.element_size() < (1ll << 31)), "okc_set_unpf");\n'
     '  g_unpf_ptr = t.numel() ? t.data_ptr() : nullptr;\n'
     '  g_unpf_bytes = t.numel() ? t.numel() * t.element_size() : 0;\n'
     '}\n', 1),
    ('  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n',
     '  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n'
     '#if OKC_UNPF\n'
     '  p.unpf = (const unsigned char*)g_unpf_ptr;\n'
     '  p.unpf_bytes = (unsigned)g_unpf_bytes;\n'
     '  p.unpf_share = grid > 0 ? (unsigned)(((g_unpf_bytes + grid - 1) / grid + 15) & ~(int64_t)15) : 0u;\n'
     '#endif\n', 1),
]
FILE_EDITS = [
    ('void okc_set_pdl(int64_t on);\nstd::vector<int64_t> okc_attrs();\n"""\n',
     'void okc_set_pdl(int64_t on);\nstd::vector<int64_t> okc_attrs();\nvoid okc_set_unpf(torch::Tensor t);\n"""\n', 1),
    ('            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),\n',
     '            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else [])\n'
     '            + (["okc_set_unpf"] if "okc_set_unpf" in src else []),\n', 1),
    ('        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)\n',
     '        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1", "-DOKC_UNPF=1")\n', 1),
]
TAIL = '''

# unpf (tools/dl_ly_unpf_patch.py): the MID (T = 16) and B4 (T = 32) builds prefetch the first UNPF_STAGES[T] of the
# unpooler sk2's 16 stage-major stages into L2 at each CTA's phase B end
MOE_MID_DEFINES += ("-DOKC_UNPF=1",)
UNPF_STAGES = __STAGES__
_UNPF_NONE = {}
_UNPF_SEEN = set()


def _unpf_view(state, M):
    k = UNPF_STAGES.get(M, 0)
    try:
        stream = state["bfp"]["sk2"][bfp__sk2_name("unpooler", M)][1]
    except (KeyError, TypeError):
        stream = None
    if k <= 0 or stream is None or not stream.is_contiguous():
        dev = torch.cuda.current_device()
        if dev not in _UNPF_NONE:
            _UNPF_NONE[dev] = torch.empty(0, dtype=torch.uint8, device=f"cuda:{dev}")
        return _UNPF_NONE[dev]
    flat = stream.view(-1).view(torch.uint8)
    nb = (flat.numel() * min(k, 16) // 16) & ~15
    if (M, k) not in _UNPF_SEEN:
        _UNPF_SEEN.add((M, k))
        print(f"UNPF T={M} {bfp__sk2_name('unpooler', M)}: stream {flat.numel()} B, prefetch {k} / 16 stages = {nb} B",
              flush=True)
    return flat[:nb]


_unpf_prior_moe = _moe
def _moe(M=None, state=None):
    ext = _unpf_prior_moe(M, state)
    if ext is not None and state is not None and M in (16, 32) and hasattr(ext, "okc_set_unpf"):
        ext.okc_set_unpf(_unpf_view(state, M))
    return ext
'''


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    for start, end in BLOCKS:
        a = text.index(start)
        b = text.index(end, a + len(start))
        blk = text[a:b]
        for old, new, n in BLOCK_EDITS:
            if blk.count(old) != n:
                sys.exit(f"{start[:20]} anchor found {blk.count(old)} times, expected {n}: {old[:70]!r}")
            blk = blk.replace(old, new)
        text = text[:a] + blk + text[b:]
    for old, new, n in FILE_EDITS:
        if text.count(old) != n:
            sys.exit(f"file anchor found {text.count(old)} times, expected {n}: {old[:70]!r}")
        text = text.replace(old, new)
    text = (f"# {DST.stem}: ly_stk1 with the decode 16 / 32 MoE (MID / B4) prefetching the unpooler's first "
            f"stages {STAGES} into L2 at each CTA's phase B end (tools/dl_ly_unpf_patch.py).\n"
            + text + TAIL.replace("__STAGES__", repr(STAGES)))
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
