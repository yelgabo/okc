# Porting the one-launch attention kernel (attn_one.cu) into muse_decode_layer

Source: `muse_attention_decode/attn_one.cu`, inlined into `submission.py` as `_ONE_SRC` by
`tools/attn_inline_one.py`. Design: the public 2.58x entry's memo (fixed-frame merge, one
launch per call); numbers and run history in `docs/runs-attention.md`.

## What one call does

`out = attn_one(...)` is one kernel launch. Each CTA takes one request, both KV heads of its
128-byte cache rows (so all 16 query heads), and one key range. It

1. loads the page indices of its first tiles, then the query rows, the block's new k/v rows,
   the constants (inv_freq, scales, the cooperative epoch, cache_loc) in two cp.async groups;
2. issues the K/V tiles as soon as the indices land (the query group still in flight);
3. normalizes and rotates the queries (q_len 1: f16 pre-scaled into log2 units; q_len 8:
   E4M3 of the unit-RMS value), normalizes, rotates, rounds and writes the block's new keys
   and values (the range that holds the position writes the cache row) and patches them into
   its tiles;
4. runs the tiles: q_len 1 on f16 `mma.sync` with K and V converted from E4M3 in registers;
   q_len 8 on `wgmma` e4m3 (Q.K from the 128B-swizzled stage, P rounded to E4M3 with a lazy
   offset that keeps P below 2^8, P.V against V transposed per tile into a 64B-swizzled
   K-major buffer whose rows 64..71 are ones, so the same wgmma yields the row sums);
5. adds `o * 2^m` and `l * 2^m` (m = its own offset, log2 units) into the workspace with
   `red.global.add.v4.f32`: no split ever sees another's max. QK-norm bounds every logit
   (|s| <= ~67 in log2 units), so the sums stay inside FP32 range;
6. finalizes: the last CTA of the request (key-count ticket, `atom.acq_rel.gpu`) or every
   CTA of a cooperative launch (parity counters, a slice of rows each) reads the rows with
   `ld.global.cg`, applies 1/l, v_scale, the per-head RMS norm and sigmoid(gate), writes
   bf16, and zeroes exactly what it read; the counters reset themselves.

Flat mode (full context, batch > 1): the grid (up to 264 CTAs, two per SM) shares a virtual
key axis holding every request's keys plus 400 (q_len 1) or 600 (q_len 8) keys per request
boundary; each CTA maps its share back to whole-tile ranges per request, runs them as
segments, and the finalize is per request (q_len 1, last CTA) or grid-wide (q_len 8).
With |32 a producer warpgroup gathers the tiles into an mbarrier ring (3 stages at q_len 1,
4 at q_len 8): producers wait on the stage's empty barrier, issue 8 x 16-byte cp.async per
128-byte row (page entries three tiles ahead in registers), and publish the stage with
`cp.async.mbarrier.arrive.noinc`; consumers wait on the full barrier, fence the async proxy,
patch the block's own rows if the tile holds them, compute, and every consumer thread
arrives on the empty barrier. Registers: `setmaxnreg.dec 56 / .inc 200` (q_len 1) and
`32 / 104` (q_len 8) at two CTAs per SM. The producer arm must end in PTX `exit`, not
`return`: a `return` sends both roles to one shared exit block, ptxas then wraps them in one
convergence region and caps the consumers at the launch-bounds count (80 at q_len 8, 304 bytes
of spills). With `exit` the q_len 8 consumers use registers up to R95 with no spills
(docs/runs-attention.md, 10-29). |64 is the one-CTA-per-SM variant without the split (113
registers, no spills).

## Host interface

```
attn_one(qkv, page_table, seq_lens, cache_loc, inv_freq, k_scale, v_scale, k_cache, v_cache,
         ws_acc, ws_l, ws_cnt, batch, q_len, page_size, window, rope, sl2, n_splits, nc, mode)
attn_one_prepare() -> [L, mode, resident CTAs, smem bytes] x instances
```

- `qkv` bf16 `[B*L, 2304]` (q | k | v | gate as SGLang's projection writes it), `page_table`
  int32 `[B, max_pages]`, `seq_lens` int32 `[B]`, `cache_loc` int64 `[B*L]`, `inv_freq` fp32
  `[32]` (`_inv_freq`, SGLang's formula), scales fp32 scalars, caches e4m3 viewed as
  `[rows, 128 bytes]`. Output bf16 `[B, L, 1024]`.
- `sl2 = sm_scale * qk_scale_factor * log2(e)`; the key scale is applied on the device.
- `mode`: 1 last-CTA finalize, 3 cooperative, |4 one tile per CTA, |8 sequence length
  speculated as `page_table.shape[1] - L` and verified on the device (batch 1, page size 1),
  |16 flat, |32 producer warpgroup, |64 one CTA per SM. Built instances: q_len 1: 1, 3, 11,
  13, 17, 49; q_len 8: 1, 3, 11, 15, 19, 51, 115.
- Per-request modes launch `(n_splits, batch)` CTAs; flat modes launch `n_splits` CTAs.
- `nc` copies of each workspace row spread same-address atomics (8 at batch 1 full).

## Buffers the layer must own

Persistent, allocated once per device, zeroed once (`_one_state` in submission.py):

- `ws_acc` fp32 `[65536, 64]` (16 MB), `ws_l` fp32 `[65536]`, `ws_cnt` int32 `[32768]`.
- Row `((b * nc + copy) * L + j) * 16 + head`; the plan clamps `nc` so `B * nc * L * 16`
  fits. Counters: `[b]` tickets, `[4096 + 2b + parity]` cooperative counters, `[8192 + b]`
  epochs, `[16384 ..]` the flat kernel's.
- Invariant: after every completed launch every row that was touched is zero again and
  the tickets are zero; the parity counter of the next launch is zero and the epoch is the
  launch count. Nothing is memset inside the graph. A launch that faults leaves the
  workspace dirty (the process is dead anyway).

## Launch plans (`_one_plan`, a pure function of shapes and device constants)

| shape | splits x batch (CTAs) | mode |
|---|---|---|
| decode 1 sliding, page 1 | 32 x 1 | 13 (plain, one tile, speculated length) |
| decode 16 / 32 sliding | 16 x 16 / 8 x 32 | 1 |
| verify 1 sliding, page 1 | 33 x 1 | 15 (cooperative, one tile, speculated) |
| verify 16 / 32 sliding | 8 x 16 / 4 x 32 | 3 |
| decode / verify 1 full, page 1 | 252 x 1, nc 8 | 11 (cooperative, speculated) |
| decode 16 / 32 full | 264 flat | 49 (producer warpgroup; 17 without) |
| verify 16 / 32 full | 264 flat | 51 (producer warpgroup, grid-wide cooperative; 19 when 51's residency is short; 115 one CTA per SM). The layer builds 51 and routes it the same way since 075525d |

Cooperative modes need every active CTA resident: `attn_one_prepare()` reports the limit
per instance (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` x SMs, an idle-device figure)
and the plan falls back to the plain instance, or shrinks the flat grid, when a grid would
not fit. Those launches carry `cudaLaunchAttributeCooperative` (via `cudaLaunchKernelEx`), so
the driver guarantees co-residency or refuses the launch; `attn_one_probe_coop()` checks once
that the attribute captures into a graph on the driver at hand, and when it does not, the plan
keeps one SM of headroom below the limit instead. The device traps after 2^24 polls of a
cooperative counter instead of hanging the run; the self-test reads the longest spin back
through `_ONE_DBG`.

## What the layer must pass and keep

- `(num_q_heads, num_kv_heads, head_dim) == (16, 2, 64)` only; other head counts fall back
  (`_one_ok`).
- Page size a power of two (since 10-29 the kernel indexes pages by shift and mask,
  `ONE_DIET`); `_one_ok` sends any other page size to the fallback, and the host wrapper
  refuses it.
- Build switches (defaults are what the judge ran in 0d05f9b2df1e4834): `ONE_DIET=1` (the
  once-path diet; the q_len 8 producer builds keep the plain path), `ONE_DIAG=0` (no stamps or
  phase exits; a bench that wants `phase > 0` or the stamps builds with `-DONE_DIAG=1`),
  `ONE_FIN8=1` (the batch-1 full plan's finalize loads all eight workspace copies before
  zeroing any; mode 11 only; judged -1.4 / -1.7 us at decode / verify 1 full, ordering argument
  in docs/runs-attention.md, 10-29), `ONE_KVFAST=0` (branch-free in-loop copy, not promoted),
  `ONE_QKOV=1 ONE_EREL=1 ONE_PCHK=1 ONE_UNR2=1` (the public flat tile order: V^T under the
  in-flight Q.K, stage released after Q.K, packed-P offset check, loop unrolled twice; modes
  51 and 115 only, every other instance SASS-identical; judged -8.3 us at verify 32 full in
  cb85d590c301410e / 6f75235c0d364a7a). A layer that routes verify 16 / 32 full to mode 51 gets
  it by re-inlining; nothing it passes or owns changes.
  `ONE_PFAST=1` (producer builds 49, 51, 115: a tile whose rows are all cached is gathered
  without per-row branches, 32-bit offsets, page index by shift; the host wrapper sets
  `off32` itself from the cache sizes and the page size; judged -5.8 / -12.3 us at verify
  16 / 32 full and -2.6 / -10.4 at decode 16 / 32 full in ca0f13d324ef48e8 /
  0d05f9b2df1e4834). The layer gets it by re-inlining plus one host line: its wrapper sets
  `Params::off32` by the same rule (`Params p` is not zero-initialized, so a wrapper without
  that line reads an indeterminate value).
  `ONE_DCARRY=1` (modes 51 and 115: tiles below the block's own positions tested with one
  compare, Q and K descriptors carried in uniform registers; judged -3.6 / -7.9 us at verify 16 /
  32 full in f64e8680b3e441be / bf49eceb15a64f81). The layer gets it by re-inlining; nothing it
  passes or owns changes. `ONE_PK16=1` (every instance: `e4m3x4` joins its halves as
  `lo | hi << 16`, which ptxas folds into one `F2FP ... MERGE_C`; same bits) and `ONE_VHOIST=1`
  (modes 51 and 115: V^T transpose addresses per segment, V^T buffer and P.V descriptor toggled
  by XOR per tile); bench 025ccaf9c57a4ba2 had them -9.6 / -5.4 us cold at verify 32 / 16 full.
  The layer gets both by re-inlining; nothing it passes or owns changes. `ONE_WAVE=0` (a per-SM pair split of the flat partition; lost, off; when
  on, the host probes `%smid` once and the kernel uses `ws_cnt[CNT_FLAT + 4096 + smid]`).
  `ONE_STAMP=0 ONE_PROBE=0`: bench-only attribution builds (per-CTA stamps into `dbg`, timing
  probes with wrong output); never set them in a judged file.
  `ONE_TRIG=0` (not judged): with 1, the q_len 1 and batch-1 q_len 8 instances execute
  `griddepcontrol.launch_dependents` at entry so a PDL-launched sk o projection (`SK_PDL`, see
  docs/proj-port-guide.md) can stream its weights during attention; at 0 every instance is
  SASS-identical (docs/steering/2026-10-29-v1-d32-breakdown.md).
  A layer copy that inlines `attn_one.cu` gets FIN8 by re-inlining; it changes nothing the
  layer passes or owns.
  `ONE_HMRG=1` (q_len 1 modes 1, 3, 11, 13: the key-half merge before the reds) and `ONE_PRQ=31`
  (q_len 8 per-request modes 1, 3, 11, 15: the tile in the public per-request order: Q.K first,
  V^T under it, branch-free 32-bit in-loop copies, P.V per half, offset vote on each thread's
  partials; docs/runs-attention.md, 10-30 rounds 2 and 3). The layer gets both by re-inlining;
  PRQ's copy bit reads `Params::off32`, which the layer's wrapper already sets for PFAST.
  The standalone file's `_ONE_WS_PICK = 3` (time three candidate workspaces on synthetic batch-1
  full inputs at state creation, keep the fastest) is host Python in submission.py's
  `_one_state`; a layer that wants it copies `_one_ws_alloc` / `_one_ws_pick` into its own state
  setup. It was measured on the standalone problem only. `tools/dl_one_port_patch.py` does
  that for the layer (candidate layer_hp, 10-30), fail-safe: an exception or a dirty workspace
  after the pick falls back to a fresh zeroed allocation.
- Fresh inputs every call; the plan reads only shapes (`page_table.shape[1]` for the
  speculated length) and never input values on the host.
- The first call on a device allocates the workspace and queries residency: do it outside
  graph capture (the judge's tests do).
- Two launches of this kernel on different streams would share the workspace and corrupt
  it: one stream, or one workspace per stream.
- The protocol relies on stream order between launches: a launch starts after the previous
  one has zeroed its rows and reset its counters. Under PDL in the layer (not yet in the
  file) everything before the query group (page indices, K/V gather) is independent of the
  qkv projection, but the epoch reads (the flat `__ldcg` at the kernel start and the
  per-request `cp.async` in the constants group), the ticket and cooperative atomics and the
  reds must all follow `griddepcontrol.wait`, with `launch_dependents` after the finalize;
  otherwise a dependent could add into the parity the previous launch still spins on.

## Self-test and bench

`python3 tools/attn_make_bench_one.py` writes `muse_attention_decode/bench.py`: the
submission plus, at import, SELFTEST lines for every judge test shape and the scored shapes
(the judge's own check recomputed in FP32: output bound, cache rows, worst row, and a second
call on the same inputs to prove the workspace was left zero), STRESS lines (2,000
back-to-back graph replays on six shapes, the replayed output against an eager run, a NaN
check, and the workspace rows and counters read back as zero), then BENCH lines (graph
replays after a 256 MB flush) of `_one` and its mode variants against the earlier paths. Local checks:
`tools/attn_ws_quick.sh -s muse_attention_decode/attn_one.cu` (ptxas, SASS),
`tools/attn_host_check.sh muse_attention_decode/attn_one.cu`,
`tools/attn_compile_check.sh muse_attention_decode/submission.py`,
`uv run --with numpy python tools/attn_emulate_one.py` (fragment layouts and numerics).

Residency on an H200 (idle device): q_len 1 instances 58.9 KB of smem plus the 1 KB
reservation, three CTAs per SM (396 resident); q_len 8 instances 90.6 KB (non-producer) or
107 KB (producer, 4 stages), two per SM (264). The self-test prints what
`attn_one_prepare()` returns on the judge's device.
