"""Build muse_moe_w4a16/cand/tiny_timeline.py from current_control_release.py: per-CTA globaltimer timeline in the
Tiny / Tiny2 helper builds (OKC_TLX=1), gated at run time by a fourth scratch tensor, printed from prepare() on
synthetic routing and once per M in {1, 8} from kernel_fn on the judge's inputs."""
import sys
from pathlib import Path

whole = Path(sys.argv[1]).read_text()
dst = Path(sys.argv[2])
MARK = "_MA_ALT_SRCS = {'moe_helper0': r'''"
assert whole.count(MARK) == 1
head, src = whole.split(MARK)
head += MARK


def rep(old, new, count=1):
    """Edit the helper0 source and the host code after it; the B4/MAIN source (MOE_CUDA_SRC, before the marker) is untouched."""
    global src
    assert src.count(old) == count, (src.count(old), old[:120])
    src = src.replace(old, new)


# ---------------------------------------------------------------- CUDA (helper0 source, shared by tiny / tiny2 / mid)
rep('''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
''', '''#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#ifndef OKC_TLX
#define OKC_TLX 0
#endif
// OKC_TLX: per-CTA globaltimer timeline (64 stamps per CTA in p.dbg) when the host passes a fourth scratch tensor;
// without it (the timed path) every stamp is a uniform not-taken branch
#define TLX_ON (OKC_TLX && p.dbg != nullptr)
#define TLS(k, v) do { if (TLX_ON) p.dbg[blockIdx.x * 64 + (k)] = (v); } while (0)
''')

# compute loop (skinny_ring_ws): weights waits, buffer waits, doneA waits, unit retires
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
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
''', '''  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
#if OKC_TLX
  const bool tl = TLX_ON && warp == 0;
  unsigned long long tl_wait = 0ull, tl_buf = 0ull, tl_dep = 0ull, tl_ndep = 0ull, tl_t0 = 0ull;
#define TLW(k, v) do { if (tl && lane == 0) p.dbg[blockIdx.x * 64 + (k)] = (v); } while (0)
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
#if OKC_TLX
        if (tl) tl_t0 = gtimer();
#endif
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
#if OKC_TLX
        if (tl) {
          const unsigned long long t1 = gtimer();
          tl_dep += t1 - tl_t0;
          if (tl_ndep == 0ull) { TLW(4, tl_t0); TLW(5, t1); }
          ++tl_ndep;
        }
#endif
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
#if OKC_TLX
    if (tl) tl_t0 = gtimer();
#endif
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
#if OKC_TLX
    if (tl) {
      const unsigned long long t1 = gtimer();
      tl_wait += t1 - tl_t0;
      if (!PB && i == 0) TLW(2, t1);
    }
#endif
    const unsigned char* slot = ring + sl * UNIT_BYTES;
''')

rep('''    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
  }
''', '''    const int bb = q % NRB;
#if OKC_TLX
    if (tl) tl_t0 = gtimer();
#endif
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
#if OKC_TLX
    if (tl) tl_buf += gtimer() - tl_t0;
#endif
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
#if OKC_TLX
    if (tl) {
      const unsigned long long t = gtimer();
      if (q < 46) TLW(18 + q, t);
      if (i == nU - 1) TLW(PB ? 6 : 3, t);
    }
#endif
  }
#if OKC_TLX
  if (tl) {
    TLW(PB ? 13 : 12, tl_wait);
    TLW(PB ? 15 : 14, tl_buf);
    if (PB) {
      TLW(16, tl_dep);
      const unsigned long long nA = (unsigned long long)(S.wsr[1] - S.wsr[0]), nB = (unsigned long long)(S.wsr[3] - S.wsr[2]);
      TLW(17, nA | (nB << 16) | ((unsigned long long)S.preissued << 32) | ((unsigned long long)S.njS << 40) | (tl_ndep << 48));
    }
  }
#undef TLW
#endif
''')

# helper warp: the doneA credit and the last epilogue
rep('''        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
''', '''        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
          TLS(7, gtimer());
        }
''')
rep('''  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
''', '''  if (lane == 0) TLS(8, gtimer());
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
''')

# kernel body: entry, routing done, grid barrier, exit
rep('''  asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
''', '''  asm volatile("griddepcontrol.wait;\\n" ::: "memory");
  if (tid == 0) TLS(0, gtimer());
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
''')
rep('''    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
''', '''    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
  if (tid == 0) TLS(1, gtimer());
#endif
''')
rep('''  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
''', '''  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    TLS(9, gtimer());
    __threadfence();
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
    TLS(10, gtimer());
  }
  CTA_SYNC();
  DBG(4);
''')
rep('''  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}
''', '''  if (tid == 0) TLS(11, gtimer());
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}
''')

# ---------------------------------------------------------------- host (the defines sit before the marker)
for old, new in (("EXPERT_MOE_IDENTITIES = ('moe-control-release', ", "EXPERT_MOE_IDENTITIES = ('moe-tiny-timeline-01', "),
                 ("'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1')", "'-DOKC_WSBLK=2', '@moe_helper0', '-DOKC_FINPF=1', '-DOKC_TLX=1')"),
                 ("'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0')", "'-DOKC_WSBLK=1', '-DOKC_FINPF=1', '@moe_helper0', '-DOKC_TLX=1')")):
    assert head.count(old) == 1, old
    head = head.replace(old, new)

rep('''def prepare(weights):
    return _expert_moe_prepare(weights, False)
''', '''def prepare(weights):
    state = _expert_moe_prepare(weights, False)
    state['_tl_done'] = {}
    for tokens in (1, 8):
        try:
            _tl_run(state, tokens, *_tl_synthetic(tokens, state['device']), 'prep')
        except Exception as exc:  # the timeline is a diagnostic: never fail prepare for it
            print('MOE_TL prep T=%d error %r' % (tokens, exc), flush=True)
    return state


TL_SLOTS = 64
TL_UNIT0 = 18


def _tl_synthetic(tokens, device):
    """Judge-like routing: skewed, this GPU the busiest. T=1 hits 2 local experts; T=8 hits 7 local experts over 20 routes, distinct per token."""
    g = torch.Generator(device='cpu').manual_seed(20261102 + tokens)
    x = (torch.randn(tokens, 4096, generator=g) * 0.5).to(torch.bfloat16).to(device)
    ids = torch.empty(tokens, 8, dtype=torch.int32)
    if tokens == 1:
        ids[0] = torch.tensor([3, 17, 40, 77, 101, 150, 200, 240], dtype=torch.int32)
    else:
        local = ([1, 4, 9], [1, 13], [1, 9, 20], [4, 13], [1, 25, 30], [13, 9], [1, 13, 25], [30, 4])
        rows = []
        for t in range(tokens):
            mine = local[t]
            rest = [int(v) for v in (32 + torch.randperm(224, generator=g)[:8 - len(mine)])]
            rows.append(sorted(mine + rest))
        ids = torch.tensor(rows, dtype=torch.int32)
    w = torch.softmax(torch.rand(tokens, 8, generator=g), 1).float()
    return x.contiguous(), ids.to(device).contiguous(), w.to(device).contiguous()


def _tl_run(state, tokens, x, ids, w, label, ref=None):
    """One extra launch of the same build with the timeline buffer as scratch[3]; prints a compact summary."""
    device = state['device']
    tl = torch.zeros(136 * TL_SLOTS, dtype=torch.int64, device=device)
    out = torch.empty(tokens, 4096, dtype=torch.bfloat16, device=device)
    ext = _fork_native_moe(tokens, state)
    torch.cuda.synchronize(device)
    ext.okc_run(state['ma_w'], state['ma_scratch'] + [tl], x, ids, w, out, 0, 32, state['nsm'])
    torch.cuda.synchronize(device)
    same = ''
    if ref is not None:
        same = ' same=%d maxdiff=%.3g' % (int(torch.equal(out, ref)), float((out.float() - ref.float()).abs().max()))
    _tl_summary(tl[: state['nsm'] * TL_SLOTS].view(state['nsm'], TL_SLOTS).cpu(), tokens, label + same)


def _tl_summary(s, tokens, label):
    import math
    n = s.shape[0]
    t0 = int(s[:, 0].min())
    us = lambda v: (v - t0) / 1000.0 if v > 0 else float('nan')
    cnt = s[:, 17]
    nA = (cnt & 0xFFFF).tolist()
    nB = ((cnt >> 16) & 0xFFFF).tolist()
    pre = int((cnt[0] >> 32) & 0xFF)
    njS = int((cnt[0] >> 40) & 0xFF)
    ndep = ((cnt >> 48) & 0xFFFF).tolist()
    col = lambda k: [us(int(v)) for v in s[:, k]]
    dur = lambda k: [int(v) / 1000.0 for v in s[:, k]]
    def st(vals):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return 'n/a'
        vals.sort()
        return '%.1f/%.1f/%.1f' % (vals[0], sum(vals) / len(vals), vals[-1])
    entry, route, first, aend = col(0), col(1), col(2), col(3)
    dep0, dep1, bend, credit, hdone, barr, brel, fin = col(4), col(5), col(6), col(7), col(8), col(9), col(10), col(11)
    # per-unit retire intervals (warp 0, after the unit barrier), slots 18.. for q < 46
    dts, dtsb = [], []
    for c in range(n):
        prev = None
        for q in range(46):
            v = int(s[c, TL_UNIT0 + q])
            if v == 0:
                break
            if prev is not None and q < nA[c]:
                dts.append((v - prev) / 1000.0)
            elif prev is not None and q > nA[c]:
                dtsb.append((v - prev) / 1000.0)
            prev = v
    dts.sort()
    rez = min((d for d in dts if d > 0), default=float('nan'))
    lines = []
    lines.append('MOE_TL %s T=%d grid=%d njS=%d pre=%d nA=%d..%d nB=%d..%d depwaits=%d..%d  (us min/mean/max over CTAs, from the first entry)'
                 % (label, tokens, n, njS, pre, min(nA), max(nA), min(nB), max(nB), min(ndep), max(ndep)))
    lines.append(' entry %s | route+ranges %s | first slot full %s' % (st(entry), st(route), st(first)))
    lines.append(' A end %s | A unit %s us (n=%d, min dt %.3f) | A wwait %s | A bufwait %s'
                 % (st(aend), st(dts) if dts else 'n/a', len(dts), rez, st(dur(12)), st(dur(14))))
    lines.append(' credit %s | dep wait start %s | dep wait end %s | dep wait sum %s'
                 % (st(credit), st(dep0), st(dep1), st(dur(16))))
    lines.append(' B end %s | B unit %s us (n=%d) | B wwait %s | B bufwait %s | helper done %s'
                 % (st(bend), st(dtsb) if dtsb else 'n/a', len(dtsb), st(dur(13)), st(dur(15)), st(hdone)))
    bidle = [b - a for a, b in zip(barr, brel)]
    lines.append(' bar arrive %s | bar release %s | bar idle %s | final %s | exit %s'
                 % (st(barr), st(brel), st(bidle), st([b - a for a, b in zip(brel, fin)]), st(fin)))
    # exposure of the mean CTA: ramp, A stream, A->B bubble, B stream, barrier, final
    m = lambda vals: sum(v for v in vals if not math.isnan(v)) / max(1, sum(1 for v in vals if not math.isnan(v)))
    lines.append(' mean CTA: ramp %.1f | A %.1f | A-end->depend %.1f (of which wait %.1f) | B %.1f | B-end->bar %.1f | bar %.1f | final %.1f'
                 % (m(first), m(aend) - m(first), m(dep1) - m(aend), m(dur(16)), m(bend) - m(dep1), m(barr) - m(bend),
                    m(bidle), m(fin) - m(brel)))
    print('\\n'.join(lines), flush=True)
''')

rep('''    output = torch.empty_like(data['hidden_states'])
    _fork_native_moe(tokens, state).okc_run(
        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],
        data['topk_weights'], output, 0, 32, state['nsm'])
    return output
''', '''    output = torch.empty_like(data['hidden_states'])
    _fork_native_moe(tokens, state).okc_run(
        state['ma_w'], state['ma_scratch'], data['hidden_states'], data['topk_ids'],
        data['topk_weights'], output, 0, 32, state['nsm'])
    if tokens in (1, 8) and not state.get('_tl_done', {}).get(tokens) and not torch.cuda.is_current_stream_capturing():
        state.setdefault('_tl_done', {})[tokens] = True
        try:
            _tl_run(state, tokens, data['hidden_states'], data['topk_ids'], data['topk_weights'], 'judge', output)
        except Exception as exc:
            print('MOE_TL judge T=%d error %r' % (tokens, exc), flush=True)
    return output
''')

dst.write_text(head + src)
print('wrote', dst, len(head + src), 'bytes')
