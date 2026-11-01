"""Off-board instrumentation for the repaired WGS expert finalization schedule.

The samples precede a real writer's epilogue. They are an optimistic dependency
time, never a publication or an ordering primitive. All production joins remain.
"""
import hashlib


LAYER = 'c5ececf50cb8893ca6189cda42a051228b4a8576faf71142cccfa0b5917c3fa4'
STANDALONE = '80d0b608091b70465fc14e254a10b7fcd81a8aa18d30180a3a4174cdb1c05c42'
WARPS = 16
EXPERTS = 32


def patch_device(source):
    if hashlib.sha256(source.encode()).hexdigest() not in (LAYER, STANDALONE):
        raise ValueError('unreviewed WGS source')

    def insert(anchor, addition, before=False):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError(f'ambiguous frontier marker: {anchor[:60]}')
        source = source.replace(anchor, addition + anchor if before else anchor + addition)

    insert('struct Params {\n', '''#if OKC_FRONTIER
#if OKC_B4FIN
#error frontier excludes the segmented B4 finalizer
#endif
  static_assert(OKC_WGS && OKC_WGSFINSS && !OKC_MFIN && !OKC_DEBUG && NW == 16,
                "frontier only supports the repaired whole-token WGS finalizer");
  unsigned long long* frontier;
#endif
''')
    insert('#define DBG(k)', '''#if OKC_FRONTIER
__device__ __forceinline__ unsigned long long frontier_tick() {
  unsigned long long tick;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(tick) :: "memory");
  return tick;
}
#endif
''', before=True)
    # Distinct addresses for each writer warp/expert. Repeated stores to the
    # same address are from the same lane and retain its last sample. The first
    # lane has a live tk at every emitted nt (nt*8 < cnt).
    insert('    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;\n', '''#if OKC_FRONTIER
    if (lane == 0 && nt * 8 < cnt) {
      p.frontier[((size_t)blockIdx.x * NW + (threadIdx.x >> 5)) * 32 + e] = frontier_tick();
    }
#endif
''')
    # The grid is cooperative. Sample the representative lane after the
    # preexisting CTA join and before the inter-CTA fence/counter/wait.
    insert('  DBG(0);\n', '''#if OKC_FRONTIER
  unsigned long long* fi = p.frontier + (size_t)gridDim.x * NW * 32;
  if (tid == 0) fi[blockIdx.x] = frontier_tick();
#endif
''')
    insert('    int* bc = p.ctr + 4 + (S.gen & 1);\n', '''#if OKC_FRONTIER
    fi[gridDim.x + blockIdx.x] = frontier_tick();
#endif
''')
    insert('  DBG(4);\n  // phases A and B', '''#if OKC_FRONTIER
  if (tid == 0) fi[2 * gridDim.x + blockIdx.x] = frontier_tick();
#endif
''', before=True)
    insert('    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n', '''#if OKC_FRONTIER
      if (tid == 0) fi[3 * gridDim.x + tk] = frontier_tick();
#endif
''')
    insert('      CTA_SYNC();  // red is reused by the next token\n', '''#if OKC_FRONTIER
      if (tid == 0) fi[3 * gridDim.x + p.M + tk] = frontier_tick();
#endif
''')
    insert('  p.out = (__nv_bfloat16*)out.data_ptr();\n', '''#if OKC_FRONTIER
  TORCH_CHECK(scratch.size() == 5 && scratch[4].device() == x.device()
              && scratch[4].scalar_type() == at::kLong && scratch[4].is_contiguous()
              && scratch[4].numel() == grid * (okc::NW * 32 + 3) + 2 * p.M,
              "FORK_FRONTIER private trace extent");
  p.frontier = reinterpret_cast<unsigned long long*>(scratch[4].data_ptr<int64_t>());
#endif
''')
    return source


def trace_words(grid, tokens):
    return grid * (WARPS * EXPERTS + 3) + 2 * tokens


def analyze(stamps, ids, local_start, grid):
    """Evaluate an ideal fixed-owner schedule against this single stamped run."""
    tokens = len(ids)
    if tokens < 1 or not 1 <= grid <= 136 or len(stamps) != trace_words(grid, tokens):
        raise ValueError('bad trace extent')
    if any(len(row) != 8 for row in ids):
        raise ValueError('expected eight route slots per token')
    end_matrix = grid * WARPS * EXPERTS
    entry = stamps[end_matrix:end_matrix + grid]
    owner = stamps[end_matrix + grid:end_matrix + 2 * grid]
    released = stamps[end_matrix + 2 * grid:end_matrix + 3 * grid]
    start = stamps[end_matrix + 3 * grid:end_matrix + 3 * grid + tokens]
    end = stamps[-tokens:]
    if any(not 0 < a <= b <= c for a,b,c in zip(entry,owner,released)):
        raise ValueError('incomplete or nonmonotone CTA timestamps')
    if any(not released[t % grid] <= start[t] <= end[t] for t in range(tokens)):
        raise ValueError('invalid token timing')
    demand = [{eid - local_start for eid in row if local_start <= eid < local_start + EXPERTS}
              for row in ids]
    writers = [0] * EXPERTS
    for i in range(end_matrix):
        stamp = stamps[i]
        if stamp:
            cta = i // (WARPS * EXPERTS)
            if not entry[cta] <= stamp <= owner[cta]:
                raise ValueError('writer timestamp outside its CTA compute interval')
            expert = i % EXPERTS
            writers[expert] = max(writers[expert], stamp)
    if {e for e,tick in enumerate(writers) if tick} != set().union(*demand):
        raise ValueError('sampled experts disagree with current input routes')
    # Each CTA still does its tokens in the original order. Simply subtracting
    # the wait of every token would count concurrent waits more than once.
    ideal_end = list(owner)
    for t in range(tokens):
        cta = t % grid
        if t >= grid and start[t] < end[t-grid]:
            raise ValueError('overlapping tokens on one finalizer')
        earliest = max((writers[e] for e in demand[t]), default=0)
        ideal_end[cta] = max(ideal_end[cta], earliest) + end[t] - start[t]
    finish = max(max(released), max(end))
    if max(ideal_end) > finish:
        raise ValueError('inconsistent timestamp schedule')
    return {'measured_section_us': (finish-min(entry))/1000,
            'last_owner_us': (max(owner)-min(entry))/1000,
            'last_writer_issue_us': (max(writers, default=0)-min(entry))/1000 if any(writers) else None,
            'max_join_wait_us': max((b-a)/1000 for a,b in zip(owner,released)),
            'ideal_saving_us': (finish-max(ideal_end))/1000,
            'unique_experts': sum(bool(t) for t in writers), 'tokens': tokens}


def encode_trace(stamps, ids, local_start, grid):
    """Compact, data-only receipt for one prescribed case. Preserve nanoseconds."""
    import base64
    import struct
    import zlib
    tokens = len(ids)
    count = grid * WARPS * EXPERTS
    origin = min(stamps[count:count + grid])
    writers = [max(stamps[e:count:EXPERTS]) for e in range(EXPERTS)]
    # Encode zero (no writer) distinctly from a timestamp equal to the origin.
    fields = [t-origin+1 if t else 0 for t in writers]
    fields.extend(t-origin for t in stamps[count:])
    for row in ids:
        mask = 0
        for eid in row:
            if local_start <= eid < local_start + EXPERTS:
                mask |= 1 << (eid-local_start)
        fields.append(mask)
    raw = struct.pack('<4sHH', b'WGF1', grid, tokens) + struct.pack(f'<{len(fields)}I', *fields)
    packed = base64.b64encode(zlib.compress(raw,9)).decode('ascii')
    if len(packed) > 7500:
        raise ValueError('frontier trace exceeds log-tail budget')
    return packed
