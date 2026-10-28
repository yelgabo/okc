# Porting the projection kernels into muse_decode_layer

What `muse_bf16_proj/submission.py` builds, how the layer calls it, and what it must pass.
The design follows the public 1.50x entry's memo (skinny kernel, 12-bit weights, escape warps,
hi/lo router on the same kernel); every kernel here is our own implementation of that design.

## Entry points (one `load_inline` extension, `no_implicit_headers`)

```
sk_fwd(cfg, ws, x, out, ptab, pmax, base4, n, k, diag)
```

- `cfg`: index into `SK_CONFIGS` (name -> index in `SK_INDEX`). Each config fixes the kernel
  template: `(MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, out dtype)`.
- `ws`: the stage-major weight stream for that config (uint8, see below).
- `x`: `[M, K]` bfloat16, contiguous, row stride K; `1 <= M <= TOKR` of the config.
- `out`: preallocated `[M, n]` bfloat16 (float32 for the router configs).
- `ptab`, `pmax`, `base4`: the escape table, its entry count per CTA and the window base
  replicated into four bytes; only compressed configs (`COMP`) read them (pass any tensor and
  0, 0 otherwise).
- `n`: rows computed (1024 for the sliced unpooler, the full N otherwise), `k`: K.
- `diag`: int32[4] on the device; a wait that times out ORs a code into `diag[0]`.

The tensor map for x is encoded on the host at every call from `x.data_ptr()` and passed as a
`__grid_constant__` kernel parameter, so the call is CUDA-graph capturable (replays reuse the
captured buffer). `cudaFuncSetAttribute` runs on the first call of each config: run every
config once (the self-test does) before any graph capture.

Python wrappers in the file: `_sk_runner(ext, name, stream, tab, pmax, base4, n, k, dtype, diag)`
returns `run(x) -> out`; `prepare()` builds `state["pick"][(layer, m)]` for every m <= 32.

## Weight formats (all built by torch ops in `prepare()`)

Tile unit: an 8-row x 64-k tile of W in mma.m16n8k16 B-fragment order. Lane `4g + t` holds row
`g`; its value `4j + 2h + p` is column `16j + 8h + 2t + p` (`tile_values`).

- Raw BF16 unit, 1024 B: 32 B per lane, the 16 values in order.
- Hi/lo unit (router), 2048 B: the hi tile then the lo tile, hi = bf16(w), lo = bf16(w - hi).
- 12-bit unit, 768 B: bytes 0-511 the raw low byte (exponent bit 0 and the mantissa) of each
  lane's 16 values; bytes 512-767 eight code bytes per lane, byte `b` of word `w` holds the
  4-bit codes of values `8w + b` (low nibble) and `8w + 4 + b` (high nibble). Code =
  `sign << 3 | clamp(e71 - base, 0, 7)` where `e71` is the top seven exponent bits and `base`
  (0..120) starts the densest 8-value window of the matrix (`window_base`). The kernel decodes
  high bytes with `((c & 0x07070707) + base4) | ((c << 4) & 0x80808080)` and the matching form
  for the high nibbles, then `prmt` interleaves low and high bytes into bf16 pairs.

Escapes (e71 outside the window, about 1.5e-4 of Gaussian weights) decode with the clamped
exponent. Table per CTA, int32 `[ncta, 2*pmax + R]`: ints 0-1 = (count, 0); entries 1..count at
ints `2e, 2e+1` = `(row_local << 16 | k, FP32 bits of w - w_decoded)` sorted by k; ints
`2*pmax ..` = the row map (original row of each local row; identity unless the matrix was
balanced). `pmax` is even so each CTA's table is a 16-byte multiple for the bulk copy. A matrix
with more than N*K/200 escapes is left raw.

Stage-major stream for a config with `R` rows per CTA, `NW = NWR*WSK` consumer warps and
`UPS = WSK*UPW` units per stage: `[stage][cta][warp][unit in warp][tile]` units, so a CTA's
stage is one contiguous bulk copy of `WBY = NW*UPW*NT*TU` bytes and all CTAs read one
contiguous region at any moment (`stage_stream`). One stream per geometry; one table per
(R, rows).

qkv_gate rows are permuted so escapes balance across its 96 row blocks (`balance_perm`:
4-row groups sorted by escape count, snake-dealt); the epilogue stores through the row map,
4 rows at a time, so stores stay 8-byte vectors.

## Kernel structure

Warps: `NW` consumers, 1 producer (lane 0 refills the ring), `NESC = min(NSTAGE, 4)` escape
warps (compressed configs only). Shared memory: `NSTAGE` slots of `[x tile: UPS units x M rows
x 128 B, 128B-swizzled][weights: WBY]`, then the mbarriers (full/empty per slot, pbar for the
table, cbar for the escape warps, qbar for the cluster), the per-escape-warp corrections
`[TOKS][R + 1]`, the cluster peer buffer, the table, a 128-byte zero row for token rows >= M.

Per stage the producer waits on the slot's empty barrier (suspend hint), arms the full barrier
with `expect_tx = UPS*M*128 + WBY`, issues one `cp.async.bulk` of the weights and one 3-D TMA
box of x (`dims {64, M, K/64}`, box `{64, M, UPS}`). Consumers spin on the full barrier, read
every shared-memory operand of their units (16 B low bytes + 8 B codes per tile per lane,
`ldmatrix.x4` for x), then `fence.proxy.async`, arrive on the empty barrier, decode and run the
mma. Escape warps take turns on stages (`st % NESC`), read x from the swizzled tile, and
accumulate `delta * x` into private accumulators; every escape warp arrives on every empty
barrier (count = NW + NESC) so the ring cannot run ahead of it.

Epilogue: consumers wait cbar, a named barrier, float2 partials into the dead ring
`[wk][token][RP]` (`RP = R mod 32 == 8`), a second barrier, then one thread per (token, 4 rows)
sums WSK partials, NESC corrections and (router) the two peer CTAs' partials, and stores 8 B of
bf16 (16 B of f32).

Router (`CL = 3`): three CTAs per 8-row block split the stages unevenly (5, 5, 6 of 16); ranks
1 and 2 push their per-warp partials into rank 0's peer buffer with
`st.async ... mbarrier::complete_tx`, rank 0's qbar counts the bytes.

Every mbarrier wait gives up after 2^28 cycles, marks the thread dead and ORs a code into
`diag[0]`, so a broken pipeline fails the self-test instead of hanging the run.

## Self-test and autotune

`prepare()` self-tests every candidate (compressed and raw sk configs, GEMVs, router configs)
against an FP64 reference on inputs with probe rows at m in (1, 2, 5, 7, 8), (9, 12, 16),
(17, 31, 32) by bucket, tolerance 5e-3 (bf16 layers), 5e-5 (router), probe rows 5e-4. Failing
candidates are dropped with a printed reason. Then every scored (layer, m) with more than one
survivor is timed by CUDA-graph replay from a cold L2 (a 256 MB read before each replay),
24 interleaved repetitions, median; the winner becomes `state["pick"][(layer, m)]`; other m
share the bucket's pick. Picks are printed as `tune <layer> m=<m>: ...`.

## What the layer must pass

- x contiguous bf16 `[T, K]`; T <= 32 (above that, cuBLAS). The unpooler at T <= 8 uses the
  1024-row slice stream (`n = 1024`).
- The packed state from `prepare()`: streams and tables per config, `base4` per layer.
- Fresh `out` tensors (allocate with `torch.empty`; no zeroing needed).
- Launch order: nothing in the kernel reads global memory other than the stream, the table and
  x; PDL (`griddepcontrol.wait`) could go in before the first x read if the layer adopts it.

## Fused pooler + router

Planned entry `skf_fwd` (not yet built): after its pooler stages each CTA streams one router
piece (8 rows x K/pieces, hi and lo) through the same ring; quarter partials go to a small
global buffer; the last CTA of a row block (release/acquire counter, monotonic) writes the
FP32 router output.
