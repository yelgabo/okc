# Grouped engine scheduled-work decision: no new broad reduction identified

November2,2026. Allocation **c8538a57**, current unchanged layer control **ba9e327c**,
own fresh control **cc474883117947c0**. This20-active-minute task reuses the
completed P1 intake and P2 semantics, not a new library survey or implementation.
No third-party code execution, download, compiler/install, child, kernel/harness,
GPU/Driver/challenge POST, public or default action. Captain retains the index.

## Finite decision

**NO SUCH WORK REDUCTION IDENTIFIED for the proposed grouped-prefix/compact
mechanism across MID/B4/MAIN.** Old GROUPED_MASKED already excludes empty
allocated capacity from scheduled token tiles. New grouped-contiguous prefix/
raster changes traversal/addressing, not that fundamental count. The native
control already uses count-dependent8-token matrix widths and reuses each WGS
weight stream across up to64tokens. At equal8-row padding, the grouped engine
does not eliminate broad padded MMA work against ba9. Wider grouped token tiles
can instead trade fewer repeated weight requests for more padded math; native
WGS already captures the64-token reuse without rounding all jobs up to64 math
rows. Compact producer/resource scheduling could still affect runtime, but this
census identifies no substantial removed main-computation work to price it.

This is a source-work conclusion, **not measured0%**, not a proof new Humming
cannot be faster, and not a repetition of the old version's measured loss.
Neither another team's identical Muse benchmark nor a new version number is the
deciding gate here: the proposed main-work elimination does not survive the
control comparison. No check or further readiness/implementation is requested.

Operational correction: the latest **20:36:22UTC** receipt reports **10layer
private checks remaining**, with20private/10scored daily caps and no live jobs.
Zero quota/reset waiting is not a blocker. A source-backed substantial mechanism
could earn an informative private test under a later finite allocation. Public
release still lacks a qualified source/current authorization.

## Exact source identity and missing old runtime configuration

Current Humming **948e6fd547e703065be51529ac1db5569256774e**, old installed
**636ba85648c30ae2c2bfb335c9399593a67ecc1d**, complete native
**ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**.
Old own measured GROUPED_MASKED job **ae8993f4e3e1435c** retains mathematical
success and a large loss, but actual BM/BN/BK/warp/stage configs and padded counts
fell outside its retained log tail. **Old actual configuration is UNIDENTIFIED.**
No tile shape is reconstructed from timing, kernel names or a current heuristic.

The source-only old/new scheduler equations can nevertheless be compared
parametrically at the same explicitly stated geometry. Our CPU results label
this distinction in every row. The8/16/32/64 token geometries are algebraic
cross-sections of the existing policy domain, **not a tuning sweep**, compiled
configurations, predictions of selected configs or new candidates.

Source witnesses in existing ignored scratch (hash-pinned by census.json):

- Old scheduler.cuh:calc_m_blocks sums CEIL_DIV(expert_tokens,BlockShape::M)
  for grouped masked as well as grouped contiguous; fetch_moe_group_block skips
  exhausted/empty experts, selecting different row bases for the two layouts.
- New scheduler.cuh:calc_m_blocks builds a prefix of the same per-expert ceil
  counts for kUseGroupedRaster. Binary lookup maps a live token-tile index to
  its expert; it does not newly skip32*CAP allocated rows of GEMM work.
- Current humming_ws.cuh consumes all num_slice_iters and issues/runs MMA per
  fragment. Output-store masks do not guard away the whole corresponding MMA.
  Current non-WS compact scheduling uses the same Scheduler task counts.
- SM90 policy's MoE-A16 seed caps token block-M at64 and constructs candidate
  sizes in8-row increments. Compaction preserves an eligible fitted tile/warp/
  residency/pipeline; replacing producers/TMA is not a count reduction.
- Native setup splits MID at8tokens/job; B4/MAIN at64. WGS computes
  ntt=(job_count+7)>>3 and dispatches wgs_seg<PB,1..8>, issuing m64n(8*NTT)k16
  forms. It does **not** unconditionally compute64token rows per small job.

Pin/use the final `_fork_native_moe` and helper aliases as in P2, not an earlier
obsolete function. MID uses helper0df's unpatched helper0, DF off; it must not
inherit WGS scheduling. Current ba9 and all implementation files remain untouched.
Foreign bytes remain only in `.build/worker-broad-expert-architecture`; committed
evidence contains our census, source hashes and assumptions, no donor code.

## Own CPU census and legal fixtures

[census.py](evidence/2026-11-02-worker-grouped-engine-work-decision/census.py)
checks source anchors and constructs **24legal fixtures**: each T16/32/128/256
crossed with empty, one-local/expert-balanced, eight-local/balanced, one-hot-local,
eight-hot-local and irregular-two-expert routes. Every token has8distinct IDs in
0..255; remote IDs stay outside0..31. Local counts never exceedT. Counts are
checked against generated IDs. These are reproducible analytical fixtures,
**not recovered judge histograms**. Both sliding/full contexts can legally
produce each histogram; no context-specific routing frequencies are inferred.

For counts c_e, output width N (G1=8192, G2=4096), K=4096:

- Native padded math rows P=8*sum_e ceil(c_e/8), including count-dependent WGS
  tails. Native weight streams J=sum_e ceil(c_e/8) for MID, or
  sum_e ceil(c_e/64) for B4/MAIN. Zero-count experts contribute0.
- Grouped at token tile b schedules M_b=sum_e ceil(c_e/b). Old masked and new
  contiguous have the **same** token-tile count at equal geometry. Math rows
  P_b=b*M_b. Masked capacity32*ceil(T/b)*b is allocation capacity, not P_b.
- With the explicitly illustrative Ntile128/Ktile64, output tiles=M_b*N/128,
  full-K tile iterations=that count*64, padded mathematical FLOPs=2*P_b*N*4096.
  Source admissibility/resource fit of a proposed config is not established by
  this arithmetic. Shapes may use different native MMA instructions; normalized
  FLOPs are not instruction-count or cycle equality.
- Logical weight elements requested=M_b*N*4096 versus J*N*4096 for native.
  Native MID's16-output-row complete-K units and WGS's256-row/K256 pieces are
  separately recorded. Fragment request counts across those unlike units cannot
  be compared without their byte size and coverage. Split-K partitions distribute
  the required reduction; for K4096/group16/Ktile64 there is no geometric K tail
  to remove. Exact runtime Stream-K fixup/duplicated partial overhead is unknown.

Every fixture includes both GEMMs, useful and padded work, weight requests,
metadata slots8T, valid SiLU and norm elements4096R. All IDs/histograms and
per-engine counts are in [census.json](evidence/2026-11-02-worker-grouped-engine-work-decision/census.json).
The model does not claim physical cache/HBM requests, actual grid scheduling,
runtime work frequency, occupancy, complete launch overhead or measured savings.

### Concrete comparisons

Here grouped b16 is a **fixed algebraic example**, not the missing old config or
a prescribed replacement. Ratios apply to each GEMM and therefore their sum.

|T/fixture|Native padded math rows|Grouped b16 padded rows|Grouped/native weight-stream ratio|
|---|---:|---:|---:|
|16/one-local-balanced|128|256|1|
|32/one-local-balanced|256|512|1|
|128/one-local-balanced|256|512|1|
|256/one-local-balanced|256|512|1|
|16/one-hot|16|16|0.5|
|32/one-hot|32|32|2|
|128/one-hot|128|128|4|
|256/one-hot|256|256|4|
|128/eight-local-balanced|1024|1024|2|
|256/eight-local-balanced|2048|2048|4|
|16/irregular counts9,7|24|32|2/3|
|Any/empty|0|0|undefined (both0)|

Example of the capacity fallacy: T256 one-local-balanced has8rows/expert.
Masked capacity at b16 is8192rows, yet **only512rows** are scheduled by both old
and new formulas; native uses256. The8192allocated-row story cannot be
credited as matrix work removed by grouped-contiguous. The CPU result explicitly
records skipped capacity separate from scheduled padded rows.

At b8 the grouped padded math equals native in all24fixtures. For every larger
8-multiple b used here, grouped P_b>=native P: no larger math elimination appears.
For highly skewed WGS counts128/256, b64 matches native job reuse/math; b8 repeats
weight requests8x relative to native job64. For small counts b64 can pad up to8x
more matrix rows. This is a real tradeoff, not evidence native attains optimal
hardware time. The new engine could have different dequant/issue/occupancy cost
per unit of work, but the source must name and size that separate mechanism.

The only favorable broad-format **request** example here is MID count9..16:
one b16 tile can replace two native8-token weight streams. It does not remove
the underlying useful dot products; a9-token tail can increase padded math.
The same change is not a win for B4/MAIN, which already use larger count-dependent
jobs. Hence this is not the hypothesized large multi-load engine work reduction,
and it does not authorize another MID tile/helper experiment. Actual scored
frequency and exposed repeated-stream cost are missing. Logical repeated
requests may hit L2; no HBM time is assigned to them.

## Complete callable costs and six-load scope

Even an engine mainloop survivor must pay: current FP32 router/top8 and shard
mapping (not removed), runtime counts/prefix/rank/gather; G1 with correct FP32
postscale and separate BF16 gate/up boundaries; FP32 SiLU/product and BF16 store;
G2/global postscale/BF16 boundary; layer effective BF16 gamma, per-expert4096-wide
RMS with epsilon1e-8, FP32 route-weighted combine in the required order and final
BF16; empty-output definition, private scratch/reset/publication, graph replays
and capture-compatible preparation. Per-tile GEMM scaling/bias does not implement
the norm before mixing. Remote slots and normalized top8 weights remain intact.

New grouped-contiguous metadata may shrink allocated buffers and remove masked
tail zeroing or address gaps. Those changes can matter, but the old loss profile
was dominated by its GEMMs; skipped tail stores cannot be counted as skipped MMA.
Current native already gathers, stores inter/y and normalizes; proposed81,920R
gross array traffic is **not all incremental**. Conversely storing full G1 output
and replacing native fusion with optimistically five callable boundaries is real
replacement work that cannot be declared free. Net timing/resources are unknown.

Scope remains D16/D32/V16/V32 in both contexts,46:16 weighting and equal-six log
mean. Using P1's frozen fresh endpoints/activities as a **sensitivity model**,
if all assumed gross reductions reach endpoints:

|Gross expert-activity reduction assumed|Net extra cost assumed /load|Conditional layer gain|
|---|---:|---:|
|0%|0us|0% (algebra only)|
|0%|5us|-1.508%|
|10%|0us|3.428%|
|10%|5us|1.792%|
|10%|10us|0.223%|
|20%|5us|5.388%|
|20%|10us|3.681%|
|30%|5us|9.324%|

These are cost scenarios, **not forecasts** or a measured runtime0%. There is no
source-backed reason to populate the10/20/30% rows from reduced masked capacity
or prefix traversal. P2 previously computed that10% layer needs26.993% activity
reduction with no extra cost,31.625% at5us or36.235% at10us/load. The sampled
cutoff is roughly6.6% above the incumbent; prior6.5763% exact sizing is dated,
not a newly queried threshold. A10x reduction in expert-index lookup would not
make this unchanged main computation10x cheaper. A10x reduction of actual
dominant dots/weight processing would matter, but this count comparison does
not identify it.

## Stop boundary and what would change it

The finite hypothesis—new prefix/contiguous/compact scheduling skips substantial
padded/duplicated matrix work absent from native—is not supported. STOP this
mechanism as a reason for a broad replacement measurement. Exact old runtime
config remains unidentified, and no missing identity is silently filled in.

This does not require a foreign exact-Muse benchmark before novel work can earn
a test. A specific new mainloop mechanism could earn one by showing a substantial
change in necessary work **against ba9**, or a source-backed utilization/overlap
case with complete replacement costs and a decisive matched discriminator. A
renamed scheduler, allocated-capacity reduction or unpriced producer compaction
alone supplies neither. If such a case later appears, compare one exact complete
callable on matched legal routes/private states/fresh replay outputs and both
GEMMs/nonlinear/norm glue; freeze null/regression/whole-layer criteria before
dispatch. No such harness/config is built or automatically allocated here.

Captain receives this completed20-minute decision and owns the next finite
main-computation action. Do not reopen attention/P1 expansion/T1/pre-gamma or
spend restored quota just to fill it. Reproduce the own CPU census from existing
ignored P1 source; no download or foreign execution is needed. README and other
sessions' edits were left untouched.
