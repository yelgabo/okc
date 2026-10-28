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
  |16 flat. Built instances: q_len 1: 1, 3, 11, 13, 17; q_len 8: 1, 3, 11, 15, 19.
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
| decode 16 / 32 full | 264 flat | 17 |
| verify 16 / 32 full | 264 flat | 19 (grid-wide cooperative) |

Cooperative modes need every active CTA resident: `attn_one_prepare()` reports the limit
per instance (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` x SMs) and the plan falls back
to the plain instance, or shrinks the flat grid, when a grid would not fit. The device
traps after 2^24 polls of a cooperative counter instead of hanging the run.

## What the layer must pass and keep

- `(num_q_heads, num_kv_heads, head_dim) == (16, 2, 64)` only; other head counts fall back
  (`_one_ok`).
- Fresh inputs every call; the plan reads only shapes (`page_table.shape[1]` for the
  speculated length) and never input values on the host.
- The first call on a device allocates the workspace and queries residency: do it outside
  graph capture (the judge's tests do).
- Under PDL in the layer: everything before the query group (page indices, K/V gather) is
  independent of the qkv projection; a `griddepcontrol.wait` before the query group and
  `launch_dependents` after the tile loop are the hooks (not yet in the file).

## Self-test and bench

`python3 tools/attn_make_bench_one.py` writes `muse_attention_decode/bench.py`: the
submission plus, at import, SELFTEST lines for every judge test shape and the scored shapes
(the judge's own check recomputed in FP32: output bound, cache rows, worst row, and a second
call on the same inputs to prove the workspace was left zero), then BENCH lines (graph
replays after a 256 MB flush) of `_one` against the earlier paths. Local checks:
`tools/attn_ws_quick.sh -s muse_attention_decode/attn_one.cu` (ptxas, SASS),
`tools/attn_host_check.sh muse_attention_decode/attn_one.cu`,
`tools/attn_compile_check.sh muse_attention_decode/submission.py`,
`uv run --with numpy python tools/attn_emulate_one.py` (fragment layouts and numerics).
