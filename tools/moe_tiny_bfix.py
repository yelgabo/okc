"""Build a Tiny / Tiny2 candidate from current_control_release.py with two short-load fixes, each a compile-time
option on the tiny and tiny2 defines only (the helper0 source is shared with MID, which keeps both off):

  --bpf K   OKC_BPF=K: at a phase-B job change whose doneA counter is not yet complete, warp 0 lane 0 issues
            cp.async.bulk.prefetch.L2 for the CTA's next K phase-B units beyond the ring window, so the HBM time
            the CTA would idle through fills L2 with weights it streams right after the wait. A hint only: no
            completion, no fence, no data visibility depends on it.
  --bal     OKC_BAL=1: phase-B unit ranges chosen so that each CTA's A + B unit count is balanced (11 or 12 at
            T = 1 instead of 10 to 12); phase-A ranges and the FAST1 preissue are unchanged.

    python3 tools/moe_tiny_bfix.py <release.py> <out.py> [--bpf K] [--bal]
"""
import sys
from pathlib import Path

args = sys.argv[1:]
src_path, dst = Path(args[0]), Path(args[1])
bpf = int(args[args.index('--bpf') + 1]) if '--bpf' in args else 0
bal = '--bal' in args
whole = src_path.read_text()
MARK = "_MA_ALT_SRCS = {'moe_helper0': r'''"
assert whole.count(MARK) == 1
head, src = whole.split(MARK)
head += MARK


def rep(old, new, count=1):
    """Edit the helper0 source only; MOE_CUDA_SRC (B4 / MAIN) sits before the marker and is untouched."""
    global src
    assert src.count(old) == count, (src.count(old), old[:120])
    src = src.replace(old, new)


rep('''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
''', '''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#ifndef OKC_BPF
#define OKC_BPF 0  // WS phase B: L2-prefetch this many units beyond the ring while the CTA waits for a job's phase A
#endif
#ifndef OKC_BAL
#define OKC_BAL 0  // WS: phase-B ranges balance each CTA's A + B unit count (phase-A ranges unchanged)
#endif
''')

# ---------------------------------------------------------------- OKC_BPF: prefetch across the doneA wait
rep('''  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
''', '''  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
#if OKC_BPF
  int pfn = 0;  // next ring position whose unit this CTA has not L2-prefetched yet
#endif
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
#if OKC_BPF
        // the ring holds positions seq0 + i .. + NS - 1; while this CTA waits for the job's phase A, HBM has time
        // for the units after them, so pull the next OKC_BPF into L2 (a hint: the bulk copies still move the bytes)
        if (warp == 0 && lane == 0 && ld_acquire(p.doneA + j) < RB_A) {
          const int nAw = S.wsr[1] - S.wsr[0], Nw = nAw + (S.wsr[3] - S.wsr[2]);
          int qq = max(pfn, seq0 + i + OKC_NS);
          const int qe = min(Nw, seq0 + i + OKC_NS + OKC_BPF);
          for (; qq < qe; ++qq) {
            const int vb = S.wsr[2] + (qq - nAw);
            const int jb = njF + vb / RB_B, rbb = vb - (vb / RB_B) * RB_B;
            const unsigned char* srcb = p.wBu + ((size_t)S.jE[jb] * RB_B + rbb) * (size_t)UNIT_BYTES;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(srcb), "r"(UNIT_BYTES) : "memory");
          }
          if (qe > pfn) pfn = qe;
        }
#endif
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
''')

# ---------------------------------------------------------------- OKC_BAL: balanced phase-B ranges
# thread 0's ranges for the helper warp
rep('''    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
''', '''    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
#if OKC_BAL
    S.wsr[2] = muldiv_g(nA + nB, c, G, rG) - S.wsr[0]; S.wsr[3] = muldiv_g(nA + nB, c + 1, G, rG) - S.wsr[1];
#else
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
#endif
  }
  WS_JOIN();
''')
# run_phase's compute-warp ranges (the same split; no fat jobs in the tiny builds, no run-time pool)
rep('''  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
''', '''  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
#if OKC_BAL
  static_assert(!OKC_BAL || (OKC_WS && OKC_TINY && !SKDYN), "OKC_BAL: WS tiny build without a run-time pool");
  if (PB) { v0 = S.wsr[2]; v1 = S.wsr[3]; }  // thread 0's balanced split, published before WS_JOIN
#endif
''')

flags = []
if bpf:
    flags.append("'-DOKC_BPF=%d'" % bpf)
if bal:
    flags.append("'-DOKC_BAL=1'")
tail = ''.join(', ' + f for f in flags)
for old, new in (("EXPERT_MOE_IDENTITIES = ('moe-control-release', ",
                  "EXPERT_MOE_IDENTITIES = ('moe-tiny-bfix-%s', " % ('-'.join(['bpf%d' % bpf] * bool(bpf) + ['bal'] * bal) or 'none')),
                 ("'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1')", "'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1'%s)" % tail),
                 ("'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0')", "'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0'%s)" % tail)):
    assert head.count(old) == 1, old
    head = head.replace(old, new)

dst.write_text(head + src)
print('wrote', dst, len(head + src), 'bytes', 'bpf', bpf, 'bal', bal)
