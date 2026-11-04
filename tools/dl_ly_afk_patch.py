#!/usr/bin/env python3
"""ly_afk: the decode 1 sliding attention merge without the in-kernel ticket; the finalize runs as its own PDL kernel.

Base ly_v1un (88fbae86). Only launches of attn_kernel<1, false, 13> (q_len 1, one request, page size 1, one tile per
split, plain split-KV, one workspace copy) with PDL change. Stamps (ly_ast, 20 inputs) put 1.8 us per CTA between
its tile loop end and the return of its ticket (atom.acq_rel.gpu after the reds), then 0.5 us of last-CTA finalize,
then 0.43 us until the o kernel's griddepcontrol.wait returns.

Change: with Params.afk set, each CTA issues its reds into the workspace as today and exits without the ticket
atomic. A second kernel, attn_fin_kernel<1> (one CTA of 128 threads per (request, KV pair)), launched right after
it on the same stream with the programmatic attribute, triggers its dependents at entry, waits with
griddepcontrol.wait for the attention grid's completion (its reds performed and visible), and runs the same
fin_nc1 finalize the last CTA ran (sum, normalize, rmsnorm * sigmoid(gate), store, zero the workspace). The o
kernel, the next launch, is now this kernel's PDL secondary; its wait covers this grid, which completed only after
the attention grid did. The ordering the ticket gave (every CTA's reds before the finalize's reads) now comes from
the grid boundary. No pre-wait read of any predecessor output; no fence removed from any other path; every other
attention instance and q_len launches the base code (afk = 0).

    python3 tools/dl_ly_afk_patch.py [--base muse_decode_layer/cand/ly_v1un.py] [--out muse_decode_layer/cand/ly_afk.py]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_v1un.py"
DST = ROOT / "muse_decode_layer/cand/ly_afk.py"
PIN = "88fbae8644f946af46b87ab89be17409e133ebab26d20cf61e9c3f3941788145"
if "--base" in sys.argv:
    SRC = ROOT / sys.argv[sys.argv.index("--base") + 1]
    PIN = None
if "--out" in sys.argv:
    DST = ROOT / sys.argv[sys.argv.index("--out") + 1]

FIN_KERNEL = r"""// ly_afk: the last-CTA finalize of the q_len 1 one-request plain split-KV build as its own grid (one CTA per
// request and KV pair). The attention grid's reds are ordered before these reads by the grid boundary
// (griddepcontrol.wait returns after the attention grid has completed and its memory operations are visible).
template <int L>
__global__ void __launch_bounds__(Cfg<L, false>::NT) attn_fin_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, false>::NT;
  pdl_launch_dependents();
  pdl_wait();
  const int b = blockIdx.y, pair = blockIdx.z;
  const int bp = b * (p.Hkv >> 1) + pair;
  fin_nc1<L, NT>(p, b, pair, bp, __ldg(p.v_scale), 0);
}

}  // namespace

template <int L>
void launch("""

FIN_LAUNCH = r"""    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
  if (q.afk) {
    cudaLaunchConfig_t fc = {};
    fc.gridDim = dim3(1, grid.y, grid.z);
    fc.blockDim = dim3(Cfg<L, false>::NT, 1, 1);
    fc.dynamicSmemBytes = 0;
    fc.stream = stream;
    cudaLaunchAttribute fa[1];
    fa[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    fa[0].val.programmaticStreamSerializationAllowed = 1;
    fc.attrs = fa;
    fc.numAttrs = 1;
    cudaLaunchKernelEx(&fc, attn_fin_kernel<L>, q);
  }
}
"""

EDITS = [
    ("#define OKC_PDLKV 0", "#define OKC_AFK 1       // ly_afk: q_len 1 one-request ONE builds finalize in attn_fin_kernel (no ticket)\n#define OKC_PDLKV 0", 1),
    ("  unsigned pf_chunk;    // bytes per CTA\n};", "  unsigned pf_chunk;    // bytes per CTA\n  int afk;              // ly_afk: no ticket, attn_fin_kernel finalizes\n};", 1),
    ("  } else {\n  if (tid == 0) {\n    const int mine = end - start;\n",
     "  } else {\n  if (OKC_AFK && L == 1 && ONE && p.afk) return true;   // (ly_afk: the reds are this CTA's last global access)\n"
     "  if (tid == 0) {\n    const int mine = end - start;\n", 1),
    ("}  // namespace\n\ntemplate <int L>\nvoid launch(", FIN_KERNEL, 1),
    ("  if (!p.flat && cs > 1 && !clusters) kern = kn[0];   // (sm_80 emulation of clusters: run-time mode)\n",
     "  if (!p.flat && cs > 1 && !clusters) kern = kn[0];   // (sm_80 emulation of clusters: run-time mode)\n"
     "  q.afk = (OKC_AFK && L == 1 && pdl && kern == ksp1[1] && p.nc == 1) ? 1 : 0;\n", 1),
    ("    cudaLaunchKernelEx(&cfg, kern, q);\n  } else {\n    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);\n  }\n}\n",
     FIN_LAUNCH, 1),
]
HEADER = ("# ly_afk: ly_v1un with the decode 1 sliding attention's ticket and last-CTA finalize replaced by a one-CTA "
          "PDL finalize kernel (tools/dl_ly_afk_patch.py).\n")


def sub(text, old, new, n, what):
    if text.count(old) != n:
        sys.exit(f"{what}: anchor found {text.count(old)} times, expected {n}: {old[:90]!r}")
    return text.replace(old, new)


def main():
    text = SRC.read_text()
    sha = hashlib.sha256(text.encode()).hexdigest()
    if PIN and sha != PIN:
        sys.exit(f"base {SRC} is {sha}, expected {PIN}")
    start = 'ATT7_CUDA_SRC = r"""\n'
    i = text.index(start) + len(start)
    j = text.index('\n"""\n', i)
    at = text[i:j]
    for old, new, n in EDITS:
        at = sub(at, old, new, n, "att")
    text = HEADER + text[:i] + at + text[j:]
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
