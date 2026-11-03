# muse_bf16_proj: what it computes, five ways to do it, and what the code leaves on the table

Written 2026-11-03 10:40 PST from a full read of `muse_bf16_proj/submission.py` (4,837 lines), the
public receipt `muse_bf16_proj/profiles/scored.9eba96894ee1402e.json` (1.5278x, rank 385 of 1700 at
09:39 PST) and the bf16_proj rows of docs/runs.md and docs/ruled-out.md.

## The component

Five dense projections of one decoder layer, each judged alone: out = x @ W.T with x [m, k] BF16 and
W [n, k]. qkv_gate 2304x8192, o 8192x1024, pooler 4096x8192, unpooler 8192x4096 (only its first 1024
rows at m <= 8, this GPU's slice), router 256x8192 in FP32 (FP32 output, 1e-4 tolerance, allowed as a
BF16 hi + lo pair with FP32 accumulation). m is 1, 16, 32 (decode loads) and 8, 128, 256 (verify
loads). Weights arrive once in prepare() and may be repacked losslessly. Every timed replay starts
with a cold L2, each shape is its own CUDA graph, and the judge subtracts about 4 us of graph cost.
The score is the geometric mean over the six loads of SGLang's load time over ours, so a microsecond
is worth 0.47% at decode 1, 0.44% at verify 1, 0.36% at decode 16, 0.32% at decode 32, 0.25% at
verify 16 and 0.20% at verify 32. Rank 100 needs +3.6%, rank 50 +4.3%.

At m <= 32 the work is pure weight streaming: 50 MB (pooler, unpooler above 8 tokens), 28 MB (qkv),
12.6 MB (o), 6.3 MB (slice), 8 MB (router) at 12 bits per BF16 weight. At m = 128 the four BF16
layers are still byte-bound (floors 8.4 / 4.0 / 14.6 / 14.6 us raw, 25% less at 12 bits); at m = 256
pooler and unpooler reach the tensor-core floor (17.4 us).

## Where the public entry's time goes (judged, us)

| load | qkv_gate | o | pooler | unpooler | router | total | SGLang | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 (m 1) | 9.06 cgemv | 5.04 cgemv | 13.52 rd | 3.73 cgemv | 3.84 rgemv | 35.2 | 59.9 | 1.704 |
| verify 1 (m 8) | 9.77 sk | 5.48 rd | 13.42 rd | 4.53 sk | 4.47 rt | 37.7 | 75.8 | 2.013 |
| decode 16 | 9.74 sk | 5.37 sk | 13.80 sk | 13.56 sk | 4.45 rt | 46.9 | 74.5 | 1.588 |
| decode 32 | 10.56 sk | 5.87 sk | 15.08 sk | 14.72 sk | 5.49 rt | 51.7 | 77.0 | 1.490 |
| verify 16 (m 128) | 14.66 wd | 6.81 cuBLASLt | 19.99 cuBLASLt | 18.99 cuBLASLt | 6.64 wd | 67.1 | 85.9 | 1.280 |
| verify 32 (m 256) | 16.94 cuBLASLt | 8.34 cuBLASLt | 24.43 cuBLASLt | 23.44 cuBLASLt | 9.26 wd | 82.4 | 101.0 | 1.225 |

Slack against (floor at the bytes the kernel actually reads + 2 us of fixed launch cost, the figure
docs/runs.md:2326 established for the short cells):

| load | slack us | score if recovered |
|---|---:|---:|
| decode 1 | about 3 | 1.4% (already at 4.3 to 4.5 TB/s; only bytes or the fixed cost are left) |
| verify 1 | about 4 | 1.8% |
| decode 16 | about 6 | 2.0% |
| decode 32 | about 10 | 3.4% |
| verify 16 | about 11 raw, 20 at 12 bits | 2.8% / 5% |
| verify 32 | about 20 | 4.1% |

The absolute slack sits at m = 32 and above; the per-microsecond value sits at m = 1 and 8, where
the kernels are already at the byte floor plus the fixed cost.

## Five ways to compute it

1. **Library GEMMs with an algorithm search** (cuBLAS, cuBLASLt plans, CUTLASS pingpong). What
   SGLang does; our prepare() screens about 1,000 cuBLASLt configs per shape and pins the stable
   winners (`LT_PREF`), plus one CUTLASS build for pooler m = 256. Status: the incumbent for every
   BF16 layer at m >= 64 except qkv_gate m = 128. Ceiling reached: 1.04 to 1.27x per shape, and
   cuBLASLt split-K (8b9e8b7c) and larger tiles lost. Nothing left here.

2. **Shared-memory staged streaming** (TMA / bulk-copy ring into smem, mma.sync at m <= 32 or wgmma
   above, weights 12-bit with escape warps, x as the A operand). The `sk` and `wd` families.
   Status: the incumbent at m = 16 / 32 and for qkv m = 128 and the router at m >= 64. Every wide
   in-house geometry at m >= 64 lost 10 to 15 us to cuBLASLt; the memory-only ablation of the
   128-row x 128-token K-split-4 twin (4922f366) streamed 64 MB in 29 us, so the ring itself runs at
   2.2 TB/s there. Untested inside this family: x multicast at m = 16 / 32 (way 5 below).

3. **Register-direct streaming** (no shared memory: weights go global to registers, decode in
   registers, mma.sync A operand or CUDA-core FMA; `rd`, `rt`, `cgemv`, `rgemv`). Status: the
   incumbent at pooler and o m <= 8, every m = 1 cell, and the router at m <= 32. Extensions to qkv
   m <= 16, pooler m = 16 and the slice lost (+0.7 to +2.9 us, ruled-out.md:189) because the
   register ring holds only PF + 1 units and the x re-reads per warp grow with the token tiles; the
   cp.async x ring for rd16 also lost (+0.5 us). Exhausted at m >= 16 as built.

4. **Register-A wgmma** (untested here; the reopen condition written in ruled-out.md:182 and the
   mechanism a public MoE kernel uses, ruled-out.md:386). The consumer warpgroup decodes the 12-bit
   stream straight into wgmma A fragments in registers (the rd kernel's `av[4]` per warp is exactly
   the 64 x 16 A fragment of `wgmma.m64nNk16` for that warp's 16-row slice) and x is the B operand
   through a smem descriptor that a TMA ring fills once per 64-k unit and all four warps share.
   One asynchronous instruction per 64 rows x N tokens x 16 k replaces MT x NT x 4 mma.sync plus
   eight ldmatrix per unit per warp, and the CUDA-core decode overlaps the tensor pipe instead of
   serializing with it (the P3 kernel's decoder warps cost +7.5 us because they reconstructed BF16
   tiles in smem and the consumers waited on them). It is the only design that brings 12-bit bytes
   to verify 16 / 32 without the failed smem reconstruction, and it also covers m = 8..32 with
   N = 8..32. Hypothesis if it reaches the 12-bit floor plus 2 us: pooler and unpooler m = 128
   from 20 / 19 to about 14 us (-11 us, +2.8% at verify 16), qkv / pooler / unpooler m = 256 -12 us
   (+2.5% at verify 32), and 1 to 2 us per big layer at m = 32. That is the only mechanism on the
   list whose upside reaches rank 50. Risk: every in-house kernel at m >= 64 has lost so far;
   accumulators at N = 128 are 64 registers per thread, so the m = 128 / 256 tiles need two
   warpgroups or a K split with the existing DSMEM push. Cheapest decisive test: one bench build
   for pooler m = 128 (128 rows x 128 tokens per CTA, K split 4, the wd push reduction) paired
   against the cuBLASLt pick with three ablations (decode off, wgmma off, x ring only), one check.

5. **Cluster-shared activations at m = 16 / 32**: TMA multicast of the x stage across a 2- or 4-CTA
   cluster whose CTAs own different row blocks, so per-SM L2-to-SM traffic drops from W + x to
   W + x / CL. At pool32 the x box per stage (UPS x TOKR x 128 B = 32 KB) exceeds the weight bytes
   per stage (24 KB), and all 128 CTAs fetch the whole 512 KB x: 64 MB of x traffic for 50 MB of
   weights. Status: untested in sk (its cluster code only splits K for the router); the wd
   multicast that lost (80b89e11) was at m >= 64 with 8-CTA clusters. Bound on the gain: the m = 16
   to m = 32 time difference is 1.3 us on the pooler, so at most about 0.5 us per big layer and
   load, under 1% in all.

A sixth lever, the denser code (11 or 11.5 bits), is measured and closed: the pair7 model showed
+0.85% before decode cost and the knock-in lost 2.5 to 6.3 us (ruled-out.md:121). Reopen only
inside way 4, where the decode has slack to hide in.

## Code-level findings from the read

- **Fixed cost dominates the two highest-value loads.** Every m <= 8 kernel carries about 2 to 3 us
  over its byte floor (o 5.04 vs 2.6, slice 3.73 vs 1.3, router 3.84 vs 1.75); that is 13 to 15 us of
  the 35 to 38 us load. The ramp lane (793602020fd64ce9) showed CTA count and threads do not move it
  and PDL is not available inside a one-kernel graph. The one untested piece of it: TLB walks. The
  stage-major stream has all CTAs cross each 2 MB page together, one walk per 0.45 us of streaming
  at 4.4 TB/s. A one-line warm-up (lane 0 of CTA i touches the first line of page i of the weight
  stream before the ring starts, as `prefetch.tensormap` already does for the descriptor) would
  overlap the walks at t = 0. Zero cost to try inside any next check, paired.
- **The router at m = 128 / 256 is the cell furthest from its floor** (6.64 and 9.26 us for 8 MB,
  3 to 3.5x the floor) on the lowest-sensitivity loads; docs/runs.md:1730 still lists 1.1 to 1.6 us
  unattributed against the reference. Worth 0.3% at most.
- **prepare() builds and self-tests the fused pooler + router kernels (`_build_fused`) that this
  problem never dispatches.** They exist for the layer port; harmless (about 1 s) but they also mean
  the 106 s prepare is 35% CUTLASS and cuBLASLt builds for one shape each.
- **`SK_NACC` keeps one accumulator chain per (mt, nt).** With UPW 4 that is 16 dependent mma.sync
  per stage per chain; NACC 2 was measured as -0.08 us (null 0.03), so the tensor pipe is not the
  limiter at m <= 32. Consistent with the stage path (wait, loads, fence, arrive) being the cost.
- **The m = 1 pooler runs the m = 8 rd kernel** (rdp8 17.6 vs cgemv 17.8 in the tuner): seven of its
  eight B columns are copies of row 0. Harmless, and it says the m = 1 cell is at the stream's
  speed, not the math's.
- **No correctness or graph-safety problems found**: every wait fails closed through `diag`, every
  pick is self-tested at the judge's odd token counts (1, 2, 5, 7, 8, 9, 12, 16, 17, 31, 32, 33, 47,
  64, 65, 100, 128, 129, 130, 200, 256) and the escape deltas are exact (0 inexact in the pack log).

## Recommendation

The quota today is 11 checks and 2 public runs on this board, and the deadline is 2026-11-04 23:59
PT. Ways 1 to 3 are at their measured ceilings; way 5 and the TLB warm-up are sub-1% each and cheap;
way 4 is the only route with rank-50 upside and it is a one-day build with a real chance of losing
like every in-house wide kernel before it. If a worker is free, run way 4's single-shape bench with
the three ablations first; stop if the x-ring-plus-decode arm is not within 2 us of the cuBLASLt
pooler m = 128 pick, because then the wgmma cannot close the gap.


## November3 18:36UTC release follow-up: selected null twins are a separate lead

TLB/XMC00005928is a valid treatmentnegative,but official3128.993210us is belowpublicC23161.918773us and actualdispatcherselects9null-prefetch twins (oM1rdo8n versuspubliccgemv). “Null” adds explicit L2self-lineprefetch;original94/94identity claim does not qualify selectedtwins. Privatewrapperb42257c3/244253LF is not a publicartifact. Existing TLB/XMCowner/lead should pin/clean the selecteddispatcher/source/packing/contracts,then reach one matchedcomplete-call/all30release decision under a separately explicit bounded owning allocation. Preserve originalwarm/XMCstops;no automaticrenewal,GPU/publicauthority,newworker or earlypermissionquestion. No trackedcleanrelease or matchedcausalgain yet. [Source inventory,centraldisposition and nextgates](2026-11-03-driver-monitor-1836.md#incoming-bf16-terminals-and-a-separate-selected-code-release-lead--8e3484ed).
