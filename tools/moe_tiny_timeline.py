"""Bounded WS-only timestamps on the selected Tiny and Tiny2 helper."""
import hashlib

QMAX = 192
HEADER = 32
STRIDE = HEADER + QMAX * 6
HELPER_PIN = 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'
HELPERS = r'''
__device__ __forceinline__ void tiny_stamp(const Params& p,int field) {
  asm volatile("{ .reg .u64 t; mov.u64 t, %%globaltimer; st.global.u64 [%0], t; }"
               ::"l"(p.dbg + blockIdx.x * 1184 + field):"memory");
}
__device__ __forceinline__ void tiny_unit(const Params& p,int q,int event) {
  if (q >= 192) asm volatile("trap;");
  tiny_stamp(p,32 + q * 6 + event);
}
'''


def transform(native):
    assert hashlib.sha256(native.encode()).hexdigest() == HELPER_PIN
    edits = []

    def edit(old, new):
        nonlocal native
        count = native.count(old)
        if count != 1:
            start = native.index('__device__ void skinny_ring_ws(')
            end = native.index('__device__ __forceinline__ void ws_issue(', start)
            body = native[start:end]
            assert body.count(old) == 1, (old[:80], count)
            native = native[:start] + body.replace(old, new) + native[end:]
        else:
            native = native.replace(old, new)
        edits.append((old, new, 1))

    edit('template <bool PB>\n__device__ void skinny_ring_ws(',
         HELPERS + '\ntemplate <bool PB>\n__device__ void skinny_ring_ws(')
    edit('    if (j != curj) {\n      curj = j;\n      if (PB) {',
         '    if (j != curj) {\n      curj = j;\n      if (PB) {\n'
         '        if (tid == 0) tiny_unit(p,seq0+i,0);')
    edit('        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);',
         '        __syncwarp();\n        if (tid == 0) tiny_unit(p,seq0+i,1);\n'
         '      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);')
    edit('    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    const unsigned char* slot',
         '    if (tid == 0) tiny_unit(p,q,2);\n'
         '    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n'
         '    if (tid == 0) tiny_unit(p,q,3);\n    const unsigned char* slot')
    edit('    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);',
         '    if (tid == 0) {\n      tiny_unit(p,q,4);\n'
         '      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);')
    edit('  const int njF = S.njF;\n  if (lane == 0)\n    for (int q = S.preissued;',
         '  const int njF = S.njF;\n  if (lane == 0) {\n'
         '    p.dbg[blockIdx.x*1184+17]=a0; p.dbg[blockIdx.x*1184+18]=a1;\n'
         '    p.dbg[blockIdx.x*1184+19]=b0; p.dbg[blockIdx.x*1184+20]=b1;\n'
         '    p.dbg[blockIdx.x*1184+21]=OKC_WSBLK;\n'
         '    p.dbg[blockIdx.x*1184+22]=OKC_WSB && OKC_WSBL;\n'
         '    tiny_stamp(p,1);\n  }\n  if (lane == 0)\n    for (int q = S.preissued;')
    edit('  }\n  __threadfence();\n  WS_JOIN();  // every epilogue store',
         '    if (lane == 0) tiny_unit(p,q,5);\n  }\n'
         '  if (lane == 0) tiny_stamp(p,16);\n'
         '  __threadfence();\n  WS_JOIN();  // every epilogue store')
    edit('  asm volatile("griddepcontrol.wait;\\n" ::: "memory");',
         '  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n'
         '  if (tid == 0) tiny_stamp(p,0);')
    edit('  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);',
         '  if (tid == 0) tiny_stamp(p,2);\n'
         '  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n'
         '  if (tid == 0) tiny_stamp(p,3);\n'
         '  if (tid == 0) tiny_stamp(p,4);\n'
         '  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n'
         '  if (tid == 0) tiny_stamp(p,5);')
    edit('  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);',
         '  CTA_SYNC();\n  if (tid == 0) {\n    tiny_stamp(p,6);\n'
         '    int* bc = p.ctr + 4 + (S.gen & 1);')
    edit('  DBG(4);\n  // phases A and B',
         '  DBG(4);\n  if (tid == 0) tiny_stamp(p,7);\n  // phases A and B')
    edit('  DBG(5);  // nothing to reset at exit:',
         '  if (tid == 0) tiny_stamp(p,8);\n  DBG(5);  // nothing to reset at exit:')
    return native, edits


def trace(values, grid):
    if grid != 132 or len(values) != 136 * STRIDE:
        raise ValueError('trace grid/buffer')
    rows = [values[c * STRIDE:(c + 1) * STRIDE] for c in range(grid)]
    origin = min(r[0] for r in rows)
    packed = []
    units = []
    for c, row in enumerate(rows):
        times = [row[i] for i in (0, 2, 3, 4, 5, 6, 7, 8)]
        if not all(x > 0 for x in times) or times != sorted(times):
            raise ValueError('stale or unordered CTA boundaries')
        a0, a1, b0, b1, tail, retained = row[17:23]
        nA, nB = a1 - a0, b1 - b0
        if not 0 <= nA + nB <= QMAX or not row[1] or not row[16]:
            raise ValueError('unit count or helper boundary')
        waits = dep = compute = 0
        detail = []
        previous = 0
        for q in range(nA + nB):
            t = row[HEADER + q * 6:HEADER + (q + 1) * 6]
            if not (times[0] <= t[2] <= t[3] <= t[4] <= times[4]):
                raise ValueError('unit boundary order')
            if bool(t[0]) != bool(t[1]) or t[0] and not (times[3] <= t[0] <= t[1] <= t[2]):
                raise ValueError('phase B activation wait')
            skipped = retained and q >= nA + nB - min(tail, nB)
            if not skipped and not (t[4] <= t[5] <= row[16]):
                raise ValueError('helper retirement')
            waits += t[3] - t[2]
            dep += t[1] - t[0]
            compute += t[4] - t[3]
            rounded = [int((x - origin) // 10) if x else None for x in t]
            base = rounded[2]
            detail.append([base-previous, rounded[3]-base, rounded[4]-rounded[3],
                           rounded[5]-rounded[4] if t[5] else None,
                           rounded[0]-base if t[0] else None,
                           rounded[1]-rounded[0] if t[0] else None])
            previous = base
        packed.append([c, *[int((x - origin) // 10) for x in times], nA, nB,
                       waits // 10, dep // 10, compute // 10])
        units.append(detail)
    for pb, first, last, width in ((False, 17, 18, 512), (True, 19, 20, 256)):
        ranges = [(r[first], r[last]) for r in rows]
        if ranges[0][0] != 0 or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('unit partition gap')
        if ranges[-1][1] % width:
            raise ValueError('job partition extent')
    critical = sorted(range(grid), key=lambda c: rows[c][8], reverse=True)[:2]
    return dict(tick_ns=10, unit_encoding='delta_ring_entry,wait,compute,helper_lag,dep_offset,dep_wait',
                ctas=packed, critical_units=[[c, units[c]] for c in critical],
                span_us=(max(r[8] for r in rows) - origin) / 1000,
                entry_spread_us=(max(r[0] for r in rows) - origin) / 1000,
                last_A_us=(max(r[3] for r in rows) - origin) / 1000,
                last_B_us=(max(r[5] for r in rows) - origin) / 1000,
                barrier_release_us=(max(r[7] for r in rows) - origin) / 1000,
                final_tail_us=(max(r[8] for r in rows) - max(r[7] for r in rows)) / 1000)
