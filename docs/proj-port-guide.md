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

## m = 1 GEMVs (`cgemv_fwd`, `rgemv_fwd`, `sgemv_fwd`)

```
cgemv_fwd(stream, table, pmax, base4, x, out, nthreads)
```

- `stream`: `pack_cgemv(w)["stream"]`, uint8 `[N/2, K/16, 48]`; the shape only fixes the byte
  count. The bytes are two row-major planes: every row's low bytes `[N, K]`, then every row's
  codes `[N, K/2]`, byte b of a row's 16-k chunk j holding the codes of k = 16j+b (low nibble)
  and 16j+8+b (high nibble). Thread t of CTA c loads, per row and chunk j = i*nth + t, 16 B of
  low bytes and 8 B of codes, so each warp load is 512 or 256 contiguous bytes. `nthreads` must
  be `cgemv_threads(N, K)`.
- `table`: int32 `[N/2, pmax, 2]`, entry 0 = (count, 0), then ((row_local << 16) | k, original
  BF16 bits); one escape warp forms the corrections, lane i loading entry i, so the count comes
  with the first batch in one load (worth about 1 us on the pooler, 5ff76a62b0044378).
- One CTA per row pair, `nth + 32` threads. One chunk per thread (pooler, o): `__maxnreg__(32)`,
  because registers are allocated per warp from one SM sub-partition's 16384, so 32 registers
  keep 16 warps per sub-partition (3 pooler CTAs per SM) where 33-40 keep 12. Two chunks per
  thread (qkv_gate, the slice): `__maxnreg__(40)` (`cgemv40_kernel`), 0.2-0.3 us faster there.

`rgemv_fwd` is the FP32 router (one row per CTA), `sgemv_fwd` the raw BF16 unpooler slice (four
rows per CTA).

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

## Fused pooler + router (`skf_fwd`)

```
skf_fwd(cfg, ws, rws, x, out, rout, ptab, pmax, base4, n, k, rpart, rcount, diag)
```

`cfg` is a fused pooler config (`pool8f` for T <= 8, `pool16f` for 9 <= T <= 16): the pooler
geometry with a `FUSE` flag. After its pooler stages each CTA streams one router piece through
the same ring: row block `blockIdx % 32` (8 of the 256 router rows), K piece `blockIdx / 32`
of `pieces = ncta / 32` (8 pieces of 2 stages at T <= 8, 4 of 4 at T <= 16), hi and lo tiles
(2048 B per 64-k unit, `rws` = `stage_stream(hilo_units, _fused_router_geom(name), 256)`),
every consumer warp taking `UPS / NW` units. The piece's per-warp partials go through the ring,
one float4 per (token, 4 rows) is stored to `rpart[piece][T][256]` (f32 scratch, at least
`[8, 16, 256]`), `__threadfence`, then one `atomicAdd` on `rcount[row block]` (int32[32]; the last arrival,
`old + 1 == pieces`, zeroes the counter for the next launch, so configs with different piece
counts may alternate; scratch is allocated per config); the last
CTA of a row block sums the pieces through L2 (`__ldcg`) and writes `rout[T][256]` f32. `out`
is the pooler result as in `sk_fwd`. x is the same `[T, 8192]` tensor for both products.

`prepare()` self-tests both fused configs against FP64 (pooler and router outputs, probe rows)
and stores `state["fused"][bucket]` = `run(x) -> (out, rout)` for the layer; the standalone
problem's `kernel_fn` never uses them.

## Wide kernel (`wd_fwd`, m = 33 to 256)

```
wd_fwd(cfg, w, x, out, n, k, diag)
```

- `cfg`: index into `WD_NAMES` (`WD_GEOM` name -> `(WG, BN, CLK, CLR, HL, NSTAGE)`).
- `w`: the weight matrix as given, bf16 `[N, K]` contiguous (no repack); for the router (`HL`)
  the stacked `[hi; lo]` bf16 `[2N, K]` (`state["router_hilo"]`).
- `x`: bf16 `[M, K]` contiguous, any M >= 1 (the token tail reads TMA zero fill).
- `out`: preallocated `[M, n]`, bf16 (f32 for the router).

64 * WG rows per CTA, one consumer warpgroup per 64 rows plus a producer warp; cluster of
CLK * CLR CTAs (K split across CLK, x multicast across CLR row blocks); grid
`(N / (64 * WG * CLR) * CLK * CLR, ceil(M / BN))`. Tensor maps are encoded per call from the
data pointers, so the call is graph-capturable as long as the captured buffers are replayed.
`cudaFuncSetAttribute` runs on the first call of each config (the self-test does it before any
capture). prepare() picks per (layer, m) for m in 33..256 from buckets 64 / 128 / 256.

## CUTLASS pooler GEMM (`cut_fwd`, pooler m = 129 to 256)

```
cut_fwd(W, x, out, ws)
```

- `W`: the pooler weights as given, bf16 `[4096, 8192]` contiguous (no repack).
- `x`: bf16 `[M, 8192]` contiguous, any M (TMA zero fill and epilogue predication cover the tail).
- `out`: preallocated bf16 `[M, 4096]`.
- `ws`: uint8 workspace on the device; 1 MB (`state["cut_ws"]`), checked against
  `get_workspace_size` on every call.

A CUTLASS 3.x sm90 GEMM instantiated from the headers in the image's FlashInfer package
(`<flashinfer>/data/cutlass/include` and `tools/util/include`; 4.5.0 on the judge): swap-AB
(A = W row-major, B = x as a K x M column-major matrix, D column-major [N, M] = out), pingpong,
persistent scheduler, tile 128 x 64 x 64, cluster 1, FP32 accumulation, BF16 out. At M = 256 the
grid is 128 tiles, one wave on 132 SMs; that is why it wins there and nowhere else
(docs/runs.md, CUTLASS sweep 2026-10-29). The CUTLASS translation unit (`CUT_SRC`) holds no torch
headers. `CUT_CPP`, compiled by the host compiler as `load_inline`'s `cpp_sources`, wraps it
and takes the stream from `at::cuda::getCurrentCUDAStream()`, so the call is graph-capturable.
Do not add torch headers or a `-std` flag to the CUTLASS unit, because torch on the judge
requires C++20 and that broke the first build. The build takes about 49 s on the judge
(76051953a96a4168), inside prepare.

For the layer: not ported. The layer's T = 128 / 256 front computes the pooler inside concat
(pooler + router hi + lo, one cuBLAS GEMM with FP32 output), and no CUTLASS config beats cuBLAS
on that shape (best 42.3 vs 33.3 us at T = 256). A separate CUTLASS pooler holds 128 of 132
SMs and leaves the router no room to hide (docs/ruled-out.md, declined proposals).

## `SK_PDL`: sk weight prefill before `griddepcontrol.wait` (2026-10-29, not judged)

Off by default (`#define SK_PDL 0` in `_SK_SRC`); with 0 the device code is SASS-identical to the
judged build. With 1, single-CTA sk configs stream the first ring's worth of weight stages (and
fused router pieces) before `griddepcontrol.wait`, and x after it; `sk_fwd` / `skf_fwd` launch
with the programmatic-serialization attribute when bit 16 of `cfg` is set. For the layer:
re-inline with `tools/dl_proj_inline.py`, build the skp extension with `-DSK_PDL=1`, set the bit
only on the o (and front) calls at T <= 32, and pair it with `ONE_TRIG=1` in `attn_one.cu` so
attention triggers its dependents early. Rationale, compile gate and stop rule:
[docs/steering/2026-10-29-v1-d32-breakdown.md](steering/2026-10-29-v1-d32-breakdown.md).

## Direct cuBLASLt plans (`LT_CPP`, m = 128 / 256 dense, 2026-10-30)

One `load_inline` host translation unit (`okc_lt_v2_<i>`, torch headers plus `cublasLt.h`,
linked `-l:libcublasLt.so.13`, then `-lcublasLt`, then `.so.12`; the judge has cuBLASLt 130101
under torch CUDA 13.0). Plans live in a C++ table and are addressed by integer:

```
lt_plan_new(m, n, k, layout, ws_cap) -> plan     # layout 0/1 column order, 2/3 row order;
                                                 # odd layouts take Wt = W^T contiguous [k, n]
lt_heuristic(plan, ws_bytes, req)                # heuristic answers under a workspace cap
lt_sweep(plan, idx, clusters, max_add)           # every tile of idx's algorithm x cluster ids
lt_expand(plan, idx, max_add)                    # stages, FP32-reduced split-K, swizzle, custom, cluster
lt_add_config(plan, [9 config fields]) -> idx    # restore one config by value
lt_info(plan, idx) -> 9 fields + workspace + 1000 x waves
lt_run(plan, idx, w_or_wt, x, out, ws) -> cuBLAS status   # current stream, graph-capturable
```

Every accepted algorithm has FP32 accumulation, no TF32 input, and either no split-K or the
FP32 (compute-type) reduction; `cublasLtMatmulAlgoCheck` must pass with workspace <= 32 MB.
`prepare()` keeps `state["pick_base"]` (the pick before this stage), `state["lt_pick"]` (layout +
fields of each cuBLASLt pick) and replaces `state["pick"][(layer, m)]` only for m = 128 / 256.
A plan is built for one m; the runner falls back to the old pick for any other m or an operand
not 256-byte aligned. Preparation grows from about 70 s to 105 s (6cbd8b981a5d4c3f): an 18 s
build and about 16 s of search.

For the layer: not ported. The layer's T = 128 / 256 front is the concat GEMM, and its qkv, o and
unpooler run through its own dispatch; porting means creating plans for the layer's exact
shapes (and concat with FP32 output, which needs a CUDA_R_32F D layout and is not covered here),
then a complete-layer A/B. Standalone and layer promotion are separate decisions.

Router geometries added the same day: `r64c8s8` (picked at m = 128), `r128c8s6` (picked at
m = 256), `r64c4s8` (never picked); same `wd_kernel` template, only NST / BN / CLK differ.
SASS (nvcc 12.8, sm_90a) against the NST 4 router instances: MEMBAR 5, FENCE 2, BAR 4, UCGABAR 4
in all; SYNCS 18 -> 22 (NST 6) and 26 (NST 8), two mbarrier inits per extra stage; no spills,
60 / 58 / 90 registers.

## Round 2 router changes (2026-10-30)

### `rt_fwd`: small-token router (m = 1 to 32)

```
rt_fwd(cfg, wp, x, out, diag)      # cfg = RT_NAMES index; MT (1 / 2 / 4) from x.size(0)
```

- `wp`: `rt_pack(hi, lo)` of the router's BF16 hi / lo split (8 MB, `state["router_rt"]`).
- `x`: bf16 `[M, 8192]`, 1 <= M <= 32. `out`: f32 `[M, 256]`.
- `RT_GEOM` name -> (S cluster ranks, WK warps per CTA, P owner ranks, exit barrier). Grid
  16 * S CTAs of WK warps, clusters of S (16 is a non-portable size). Each warp loads its whole
  share of weight fragments and x straight into registers (one round trip), runs mma.sync for
  hi and lo, the CTA sums its warps in shared memory and pushes one float4 per item with
  `st.async` to one of P owner ranks, which add the S pieces in rank order.
- Design from NathenFernandes/muse-bf16-proj v18 `routertail_small` (MIT License, Copyright
  (c) 2026 Nathen Fernandes); the packing, indexing and synchronization are our own and are
  checked on the CPU by `tools/proj_router_emulate.py` (rt_pack against FP64, XA slot mapping).
- Picks in the judged runs: rt8w8 at m = 8, rt16w8 at 16, rt16w8p8 at 32; rgemv stays at m = 1.

### wd router VAR bits (`WD_VAR`, `WV_*`)

A wd geometry name carries VAR bits; 0 is the build judged through 6cbd8b981a5d4c3f. The
production picks use KLUE = PACK | PROMO64 | PUSH | EARLY | TAILSYNC (`r64c8s8k`, `r128c8s6k`)
and `r64c4s8a` = KLUE | NOSLEEP | XA; TAILSYNC is the 2026-10-30 conforming decision below.

- PACK: weights from `router_pack(hi, lo)`, `[N/64 * K/64 * 128, 64]`, so one 128-row TMA box
  carries the hi and lo tiles of a (64-row block, k tile). Pass `state["router_pack"]`.
- PROMO64: 64-byte L2 promotion on both tensor maps.
- PUSH: each rank owns BN / CLK tokens; partials go by `st.async` to the owner, which waits for
  (CLK - 1) peers' bytes on a receive mbarrier and adds the CLK pieces in rank order (the same
  order as the pull reduction). No pull and no exit cluster barrier.
- EARLY: the producer issues the prologue loads before the CTA barrier.
- NOSLEEP: the producer's empty-barrier waits drop the suspend-time hint.
- XA: x tile as the wgmma A operand (64 tokens) and the packed box as one 128-wide B operand,
  so one m64n128k16 replaces two m64n64k16 per k16 step; hi and lo accumulators are summed once
  at the end. Needs PACK, PUSH, WG 1, BN 64; owners of 8, 16 or 32 tokens (CLK 8, 4, 2).
- Bench-only bits: NOPROD (no producer warp), SPIN (no watchdog), TAILSYNC, DUP (null copy),
  NOEMPTY (no-producer refill without the empty barrier; prepare() refuses it).

Synchronization argument for PUSH and rt (independent read-only review on 2026-10-30 found no
blocking defect):

1. Receive-barrier publication: one thread runs `mbarrier.init`, `fence.mbarrier_init.release.cluster`,
   `mbarrier.arrive.expect_tx`; every thread then `barrier.cluster.arrive.relaxed`; senders run
   `barrier.cluster.wait` (acquire) before any `st.async` (the CUTLASS cluster-init order). A
   complete_tx may still land before the expect; the tx count then goes negative, which the
   mbarrier allows, and the phase cannot complete before the expect's arrive (pending count 1).
2. Data publication: `st.async`'s complete_tx releases at cluster scope; readers return from
   `mbarrier.try_wait.parity.acquire.cluster` before reading. Local slots are read only by the
   thread that wrote them (wd) or after a CTA barrier.
3. Lifetime: only receive slots are accessed remotely (never the ring); a receiving CTA's
   readers wait for every expected byte before reading and before exiting, and rt ranks >= P are
   never targets.
4. Reuse: no global scratch, counters or self-resetting state; barriers and slots live in the
   launch's shared memory, every slot has one writer per launch, so CUDA-graph replay is safe
   (each bench arm was replayed twice on fresh inputs and checked).
5. Two points the documentation leaves open: whether a CTA may exit while its own outgoing
   `st.async` is still in flight (the CUDA guide's DSMEM rule is about the target), and whether
   `st.async` may target the issuing CTA (the ISA neither allows nor forbids it). The reference
   kernel relies on both.

**Decision (coordinator, 2026-10-30): ship the conforming form.** Where the PTX documentation
is ambiguous we do not rely on the undocumented behaviour (repository rule: no ordering
shortcut without a citation), even though the reference does. In the shipped configs:

- Every push path ends with a cluster barrier (wd `WV_TAILSYNC` in `_WD_KLUE`; rt TAIL true in
  every `RT_GEOM` entry). Every receiver arrives there only after its receive wait saw all of its
  bytes, so passing the barrier implies all of the CTA's own pushes completed: no CTA exits with
  an outgoing `st.async` in flight.
- rt writes its own-rank share with a plain shared store, followed by a CTA barrier before the
  owner threads read it; the receive barrier expects (S - 1) * OWN * 16 bytes. wd already wrote
  its own share locally.
- SASS against 1e12506 (nvcc 12.8, sm_90a), shipped instances only; all 69 other kernels are
  instruction-identical. rt: +1 STS, +2 BAR.SYNC, +2 UCGABAR. wd: +2 MEMBAR, +1 BAR.SYNC,
  +2 UCGABAR, +1 ERRBAR, which is the exit sequence of the production pull path. No spills.
- Cost measured in a91f2152d8974458: +0.10 to +0.21 us on rt (rt16w8t against rt16w8 at
  8 / 16 / 32 tokens) and about +0.3 us on wd (NKLUET against NKLUE at 128 / 256). It is
  confirmed on the judged file in the checks logged in docs/runs.md.

#### Split exit barrier (candidate, 2026-10-30 evening)

The exit barrier above costs +0.58 us at m = 128 on the judge (and gave back the router gain
at m = 32 and 256). In ce25f67 the wd form is `barrier.cluster.arrive.release` +
`wait.acquire` after the output stores, so its release (SASS `MEMBAR.ALL.CTA`,
`MEMBAR.ALL.GPU`, `ERRBAR`, `CGAERRBAR`) waits for those global stores to drain, and every CTA
then waits for the slowest CTA of its cluster to finish its own stores. The rt form is a relaxed
arrive and an acquire wait after the stores.

The split form (wd `WV_TAILSPLIT`, rt `RT_GEOM` mode 2 or 3) keeps the same barrier and moves
only the arrive: each thread that ran the receive wait arrives right after that wait returned
(`__syncwarp` first, since `.aligned` needs the whole warp), every other thread arrives at the
same program point (rt) or just before exit (the wd producer warp), and every thread runs
`barrier.cluster.wait.acquire` just before exit. The reduction and the output stores then run
between the arrive and the wait. `WV_TAILRLX` / rt mode 3 make the arrive `.relaxed` (no
membar); `WV_TAILSPLIT` alone / rt mode 2 keep `.release`. Each thread still alternates arrive and
wait (the PUSH init pair is the previous phase) and arrives exactly once per phase, including
after a watchdog timeout.

Argument that no CTA exits with an outgoing `st.async` in flight (PTX ISA: `st.async`,
`mbarrier` phase completion and `mbarrier.try_wait`, `barrier.cluster`):

1. A sender A pushes to rank B only into B's receive slots, with complete_tx on B's receive
   mbarrier. `st.async` performs its write and then a complete_tx on that mbarrier.
2. B's receive phase completes only when its pending count (the expect's arrive) and its tx
   count are both zero, and B expects exactly the bytes all (CLK - 1) or (S - 1) peers push to
   it. So when a B thread's `mbarrier.try_wait.parity.acquire.cluster` returns true, every
   `st.async` into B, A's included, has been performed.
3. B's waiting threads execute their cluster arrive after the wait loop exited on that true
   result (program order; the arrive is control-dependent on the loop's exit).
4. `barrier.cluster.wait` returns only after every non-exited thread of the cluster performed
   its `barrier.cluster.arrive` for the phase. No thread exits before its own wait, so B's
   threads cannot drop out of the count by exiting.
5. Hence each A thread passes its wait, and A exits, only after every `st.async` it issued was
   performed at its target. This is an execution-order property of the barrier; `.relaxed`
   removes only the memory-ordering guarantee for B's accesses before its arrive, which nothing
   here relies on: A reads nothing B wrote, and B reads only its own receive slots after its
   own acquire wait (step 2), unchanged from ce25f67.
6. With the `.release` arrive (mode 2 / `WV_TAILSPLIT` alone) the same steps also give a
   memory-model chain: complete_tx releases at cluster scope (`st.async`), B's try_wait acquires
   at cluster scope, B's arrive releases and A's wait acquires, so A's `st.async` write
   happens-before everything A does after its wait.
7. Lifetimes and reuse are unchanged: remote accesses into B are only the pushes, all of which
   completed before B's wait returned (step 2); after that point nothing touches B's shared
   memory from outside, so B finishing its reduction after A exited is safe. The receive
   barrier and slots are not reused within a launch.

The rt kernel in ce25f67 already used the relaxed arrive with argument steps 2 to 5; the split
moves it ahead of the reduction and stores without changing which wait it follows.

Scope of the relaxed form: the PTX memory model has no notion of an operation in flight, so
with `.relaxed` the guarantee is execution order only (steps 2 to 5), and it assumes the arrive
is not issued before the wait loop that precedes it exits (both are `asm volatile` with a
memory clobber, the arrive follows the loop exit and a `__syncwarp`; GPUs do not issue barriers
speculatively, but the ISA does not state it). Only the `.release` form gives the step 6 chain.
A's code after its wait depends on nothing B wrote, so nothing needs that chain.

Independent read-only review (2026-10-30, of db0f8ad): sound with notes on all four questions
(guarantee, arrive/wait well-formedness on every path including partial-warp rt owners, the wd
producer warp, XA and the watchdog path; lifetimes; the bool-to-int rt template change). Notes
taken: the paragraph above, and static_asserts that `WV_TAILRLX` needs `WV_TAILSPLIT` and that
the rt mode is 0 to 3. Note kept: with CLR > 1 and PUSH (not built) the multicast and remote
empty-barrier arrives would need the same re-check.

For the layer: not ported. The layer's T <= 32 router is fused into its pooler / concat front,
and its T = 128 / 256 router is the concat GEMM. Porting rt means a separate router launch beside
the pooler; that is a layer decision with its own A/B.

### Stable cuBLASLt choice (`LT_PREF`, `_stable_pick`)

The final cuBLASLt decision per (layer, m) times the pre-Lt pick, the pinned `LT_PREF` config
(if the library accepts it and it passes the FP64 self-test) and the search's finalists in
STABLE_ROUNDS = 9 rounds with the order rotated each round, STABLE_REPS = 3 replays per round.
The preferred candidate (LT_PREF, else the pre-Lt pick) stays unless another saves more than
max(0.3 us, 2 x the preferred's round MAD) in median paired difference and is faster in at least
8 of 9 rounds. Same rule as `tools/dl_stable_tuner_patch.py`. The `ltune` line prints every
candidate's median, each paired saving with its win count, the margin and the pick.
