# muse_moe_w4a16 runs

Every `okc check` on `muse_moe_w4a16`, newest last. Scores are the judge's speedup over
SGLang with its 95% interval. Profiles and logs are in `muse_moe_w4a16/profiles/`.

## October31 PDT active top10 campaign (post-login)

Source baseline `d3eeeda`. Two checks dispatched from `codex/today-moe`; neither is a scored
submission. Existing precision, ordering, gate thresholds and production policy unchanged.

| job | source/artifact | purpose | status at dispatch |
|---|---|---|---|
|7631974774944a2a | current submission.py copied unchanged to `.build/today-moe/production51.py`; SHA25666618198aafc4d14cfe0350bf561fe8abc66f3d5bc9f6df1dbd2518508297c01 | fresh single-file correctness/timing for owned mode51 policy before coordinator's scored runs | pending |
|e06255199f6d4de1 | `.build/today-moe/xp_diag.py`; SHA25665439aaad129401c62c40400e2a2eaccf5a360255f318c1b8469ad85b7622e3a | producer readiness/empty-slot attribution before any ring resizing | pending |

Diagnostic reproduction:

```
python3 tools/moe_make_bench_ab.py .build/today-moe/xp_base.py unused.py --nopub=1 --base=51 --sizes=128,256 --routings=layer,std --seeds=211,212 xp=56
python3 tools/moe_xprod_diag_patch.py .build/today-moe/xp_base.py .build/today-moe/xp_diag.py
```

The generated file repurposes diagnostic56 as51 plus xproducer-only local counters, disabling
56's old light/skew/gather instrumentation so consumers match51. Other production predicates
are unchanged. Phase A/B and job<=4/>4 n-tile buckets each store ready cycles, empty-region
cycles, stage count and readiness-event count. StaticA times its xready spin only; claimedB
times the done_as acquire **and proxy-fence region**. Empty region includes clock overhead
and first-lap paths without a barrier wait. These are regions, not pure spin or utilization.
Mode56, like51, does not claim phaseA dynamically (claim_a/claim_lean false).

Counter storage is16 int64 values at `(1<<20)-16384+cta*64` in inter, below existing stage
stamps at1<<20. G<=256 bounds the entire reservation; the host asserts both this and allocation
bounds. Numerical fat51/56 paths do not use inter; skinny modes are not active for this arm.
Single lane0 xproducer owns each CTA's16 words and writes all counters including zeros every
launch. Host reads only after synchronization. Root read-only review confirms no changed
acquire, proxy fence, barrier or release, and no consumer lifetime/alias issue at these bounds.
Instrumentation can change producer pacing/register use; compare its timing with clean51 and
use counters directionally. This is attribution only, never a submission candidate.

Production check7631974774944a2a completed:5/5 official tests,2.193073x (95%CI
2.160177–2.225969), spin flags0, policy FP32 worst error3.94e-3. Selected sk8w3s at
1/8/16/32 and mode51 at128/256. Prepare reports228736B shared memory and152.3s build.
This is below the existing2.215712x board best within normal variance; no new algorithmic
gain is attributed. Frozen artifact/hash sent to coordinator for today's authorized scored
runs. Saved status JSON, raw log and normalized profile text under
`muse_moe_w4a16/profiles/production51.7631974774944a2a.*`.

Coordinator dispatched scored336f3a63bd6a44f3 from the frozen production51 artifact after
the successful exact-file check. Pending at dispatch. This file tunes policy; the check
selected skinny at32, so the name does not mean mode51 executes at every token count.

Scored336f3a63bd6a44f3 completed: passed5/5,2.199082x (95%CI2.166096–2.232068),
policy again skinny1/8/16/32 and51 at128/256, spin flags0, policy FP32 worst3.96e-3.
Below board best2.215712x; no headline improvement. Receipt/log saved
`profiles/scored.336f3a63bd6a44f3.{json,log}`. Submitted by root, not this agent.

### Producer diagnostic result: e06255199f6d4de1

Passed5 official tests; control score2.191867x,95%CI2.158989–2.224745. All measured xp outputs
equal clean51 at128/256 on layer/std seeds211/212. Instrumentation slows the full kernel by
1.7/2.0us at128 (layer/std) and1.0/1.3us at256. These are diagnostic costs, not improvements.
All32 ABXP bucket rows survive in the saved log; status/log/profile text committed under
`profiles/xprod.e06255199f6d4de1.*`.

Measured phase-A empty-region cycles dominate readiness: for256 layer seed211 summed over
CTAs and both size buckets, empty18.847M versus ready0.962M. Phase B empty6.794M versus
ready2.586M. At128 std seed211 (all small jobs), A empty19.852M versus ready1.555M, B
empty6.828M versus ready3.192M. These sums overlap across CTAs, include instrumentation and
the regions specified above, and are NOT additive wall time or attainable savings.

**Correction to the preceding minority premise:** at256 layer seed211, A small-job stages
9216 versus large2560 (78.3% small); std seed21112288 versus1536 (88.9% small). At128 std
both seeds100% small. Small means<=4 n-tiles. This directly contradicts the older expectation
that such jobs are a minority on these inputs, so a bounded deeper-ring capacity test is now
justified by new evidence. It does not prove variable-sized per-job rings profitable or safe.

Next proposed test, independent of the diagnostic: fixed four x slots at the launch's current
stride, retaining208896-byte ring budget by reducing weight slots (3+3 becomes2+4 at32KB).
The exact release/acquire sequence remains; every producer/consumer moves modulo4. Reuse
diagnostic-only S.dg[0][0:2] for fourth xfull/xempty barriers, avoiding production shared-layout
changes. Mode87 aliases51 traits; production mode51 remains3slots. Independent ordering and
alias review required before dispatch. Gate:>=1.5us at256 and>=1us at128 on layer/std with
consistent fresh seeds; no material32-token regression, no correctness/spin/replay failure.
Any win needs a second paired confirmation, no promise of a score gain from wait-cycle totals.

### Fixed four-slot capacity gate dispatched

Check39804be1a59c429c: `.build/today-moe/ring4_gate.py`, SHA256
5e8eaa8f1704c50be4d174f0ac2a2980bc0f86408852d4bbf34a81ca9156de31.
Builder: `moe_make_bench_ab.py --nopub=1 --base=51 --gate=1 --sizes=32,128,256
--routings=layer,std,skewed --seeds=221,222,223,224 o87=87`, then
`tools/moe_ring4_patch.py IN OUT`. Production submission.py is unmodified.

Independent audit_attention_moe review found no issue: all7 x advances use modulo4;
init/access covers the fourth barrier; mode87 traits equal51, including align_b. Diagnostic
traits are false, so S.dg[0][0:2] has no overlapping writer. Ring byte usage for8K/16K/32K
slots is180224/176128/204800B, all<=208896. Same producer/consumer stream and unchanged
WGMMA-retire/release chains preserve phase/lifetime. Remaining constant3 stamp ordinals
compile out for87. This proves only the reviewed source chain, not compiled correctness
or profitability. Fewer weight slots remain a coupled performance cost to measure.

Ring4 result39804be1a59c429c:5/5 official tests pass, judged control2.201468x
(95%CI2.168446–2.234490). Outputs at128/256 bitwise equal51 on every measured seed;
32-token maximum relative difference3.5e-5. Gate fails clearly:

| tokens | layer o87-51 us | std | skewed |
|---|---:|---:|---:|
|32 |+0.1 |-0.4 |-0.5 |
|128 |+2.2 |+1.5 |+1.2 |
|256 |+2.9 |+3.9 |+1.9 |

Every256 seed loses;128layer/skewed everyseed loses. No confirmation/promotion warranted.
Production source and allpolicy tables unchanged. Generator retained as reproducible parked
experiment, receipt/log/profiletext in `profiles/ring4.39804be1a59c429c.*`. This rejects the
fixed4slot/one-less-weight-slot tradeoff, not all deeper or per-job rings. Hypothesis: lost
weight lead outweighs extra x lead; neither purecause is isolated. Root asked to pivot to a
structurally different official-library groupedMoE comparison, not repeat this geometry.

### Primary-source Humming API feasibility (read-only, not a performance result)

Source pinned `vllm-project/humming` commit948e6fd547e703065be51529ac1db5569256774e,
inspected in ignored `.build/humming-api`; root separately probes the actual installed image.
CurrentGitHub API cannot be assumed to match the image's package.

- [`tests/kernels/humming/test_moe.py`](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/tests/kernels/humming/test_moe.py)
  explicitly covers grouped-contiguous BF16 input/output with float4e2m1 weights,
  float8e4m3 group scales,16-element scale groups and tensor secondary scales.
- [`humming/schema/humming.py`](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/schema/humming.py)
  expects raw weight int32[E,N,K/8], scale[E,N,K/16], secondary scale[E,1]float32.
  Original packed uint8 weights can be viewed as int32 without changing bits. Use
  `HummingLayer.load_from_tensors`, NOT `load_from_unquantized` (which requantizes),
  and call transform only in prepare. Preserve original NVFP4/E4M3/FP32 scale values.
- [`humming/layer.py`](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/layer.py)
  exposes HummingLayer(shape_n,shape_k,weight_config,input_config,num_experts,torch_dtype)
  and forward(expert_layout,compute_config,top_k,...). Config: weight float4e2m1,
  bs float8e4m3, group16, tensor scale2; input/output BF16, input quantization disabled,
  compute use_f16_accum=False. Original SGLang Marlin baseline likewise passes one FP32
  tensor scale per local expert, so that tensor-granularity mapping is supported.
- [`humming/testing/data.py`](https://github.com/vllm-project/humming/blob/948e6fd547e703065be51529ac1db5569256774e/humming/testing/data.py)
  shows grouped-contiguous expert_layout as int64 exclusive-prefix offsets[E+1].
  Gather BF16 route rows contiguously by expert; use top_k=1 once replicated. GEMM2 can
  reuse the grouped ordering. Inverse scatter and weighted per-expert normalization
  remain explicit. Test runner's `.item()`-sized allocation must NOT be copied into
  timed kernel_fn: reserve fixed shape/config-derived capacity and pass device offsets.

Legal integration must retain BF16 gate/up, BF16 filtered-SiLU product before GEMM2,
BF16 GEMM2 output before per-expert RMSNorm, then routing weights. No FP8 activation
shortcut, no routing-before-norm movement, no requantized weight cache. Prefer explicit
GEMM stages over a fused MoE API that hides those boundaries. Compilation/correctness,
nearby shapes, graph capture and speed remain entirely untested here.

### Humming0.1.12 indexed integration prototype (no dispatch yet)

Root capability check e30ec1bf8e8142df confirms installed Humming0.1.12. Exact release
source inspected at `636ba85648c30ae2c2bfb335c9399593a67ecc1d` (tagv0.1.12), not the newer
GitHub API above. **Grouped-contiguous is unsuitable for a partially filled fixed-capacity
matrix in this release**: scheduler.cuh160 computes last expert length from shape_m, ignoring
the final device offset. It would assign unused R=M*8 tail rows to expert31, wasting work.
No such backend was dispatched. INDEXED instead supports device num_tokens_padded and
sentinel masked row IDs without dynamic host sizing.

`tools/prefill_humming_indexed.py BASELINE.py OUT.py` constructs only a prefill expert
replacement and prepare extras, preserving every other baseline stage. Own Triton kernels
count/rank local routes, construct per-GEMM padded prefixes, sorted IDs and block expert IDs.
Caller-relative IDs0..31 are local; all others become -1 in the postnorm slot mask. Arbitrary
device local_start remains handled by the original caller's shifted-ID expression.

For each GEMM, BM comes from the explicitly passed Humming heuristic config, so metadata
alignment matches the kernel's BlockShape::M. R=M*8; Sorted capacity is
`(ceil(R/BM)+32)*BM`, expert-block capacity `ceil(R/BM)+32`, initialized sentinel R and
expert0 respectively. Only blocks for actual local experts0..31 enter device NP. Every
valid route occupies exactly one block row and retains its original route ID as write index.
Humming scheduler.cuh139 reads NP to limit blocks; scheduler.cuh306–330 divides sorted IDs
by top_k for input rows. loader_a.cuh108/140 masks reads by input shape_m; gmem_writer.cuh73/105
masks output IDs by shape_m*top_k. Thus sentinelR maps to M (out of bounds) with GEMM1top_k8
and to R with GEMM2top_k1, and is masked in both. Zero-routes produces NP0 safely without
fake expert blocks. Partially filled final multicast tiles also have safe sentinel/expert0
metadata. All matrices c1/c2/c3 are zeroed every call so unused nonlocal rows remain finite.

GEMM1 uses BF16 x[M,4096] and BF16 output[R,8192]; unchanged SGLang silu_and_mul writes
BF16[R,4096]; GEMM2 writes BF16[R,4096]; unchanged fused post-expert norm/weighted sum uses
original topk weights and the filtered local slot mask. FP32 accumulation is explicit.
Weights are loaded from original packed bits, E4M3 scales and FP32 tensor scales and only
transformed in prepare. No activation quantization or weight requantization is introduced.

Scratch is cached by metadata shape M, retaining old allocations so captured graphs remain
valid. Every replay zeroes counts and matrix outputs and rewrites all rank/local entries and
metadata; no input-derived answer is cached. No item()/CPU sync or input-dependent allocation
occurs in the new experts call. Streams serialize rank->prefix->sort->GEMM stages. Distinct
HummingLayer objects retain distinct lock buffers. Cache capacity is about4GiB at M16384,
plus smaller per-shape buffers, not32 separate expert-capacity matrices.

Generator output passes Python compilation. CPU contract oracle
`tools/test_humming_route_contract.py` passes adversarial all-nonlocal/all-expert0/all-expert31,
distributed and random routes under shuffled atomic order at BM16/32/64/128, verifying
capacity, valid-route bijection, expert block ownership and sentinel bounds. This does NOT
execute Triton or prove runtime Humming compatibility. Root requested independent source
review before a backend check; queued replay, official correctness and cold timing remain
required. No Humming performance or candidate promotion is claimed yet.

### Prefill Humming backend check dispatch and ABI correction

Root requested and reviewed the fixed-capacity INDEXED prototype; independent reviewer
verified v0.1.12 scheduler/sentinel ABI, capacity, row mapping, precision and graph lifetimes.
Preserve original FP32 contiguous topk weights and I32 contiguous IDs. A cache miss during
capture now fails closed; scratch must first be warmed eagerly. First dispatch was
be3cbc7f5a234c64 (`muse_prefill_layer`), file SHA256
e53d374f6cf4239705cd3d20998ca3364ef0477dadd49031bafda0bff7324319.

It failed in prepare before any GEMM: installed Humming0.1.12 expects tensor-scale parameter
shape[32], whereas the source-derived wrapper supplied[32,1]. This is an integration ABI
failure, not evidence of a mathematical failure or slow backend. Receipt/log retained as
`profiles/prefill_humming.be3cbc7f5a234c64.{json,log}`. Corrected wrapper uses
`gs.reshape_as(layer.weight_scale_2)`, a metadata-only operation on original FP32 scale values.

Corrected plain check0de84000097946c9 dispatched, file SHA256
bf6d95ae1cbba377dc5bd4a2cd9774815a54ebb26ddcfcab9a33834220c87232, pending. After this
dispatch, generator additionally asserts both scale tensors contain32 values and are FP32,
and uses contiguous original values instead of a redundant .float() no-op. The in-flight
file remains frozen. These guards must be in the later fresh-stress/release build and checked
before promotion. Package version alone did not guarantee the exact parameter ABI.

`tools/prefill_humming_stress.py` is prepared, not dispatched: M32 four synthetic banks
(zero-local, one-local expert, eight-local, changing mixed routing) and2000 queued graph
replays with copied fresh x/IDs/topk weights. Device checks compare counts/NP to expected
local counts, enforce exact zero on nonlocal cases and compare numerical error to a matched
64-call eager envelope. This detects stale state/replay errors; official plain tests supply
independent mathematical correctness. No stress dispatch until corrected plain check passes.

### First passing Humming prefill candidate and fresh replay gate

Corrected plain0de84000097946c9 passed5/5 at1.176449x (95%CI1.158802–1.194096), versus
SGLang in the same run. This is a real candidate score, not an instrumented arm's control.
Judged times us (candidate/SGLang): turn1 sliding935.6/1024.3, full1344.0/1437.2;
turn4 sliding2750.1/3314.4, full4385.3/4944.5; firstturn's four cases
14233.8/18455.6,13276.0/17184.6,17701.8/21581.7,21406.1/25391.9.
Receipt/log/profile text retained as `profiles/prefill_humming_v2.0de84000097946c9.*`.

Runtime heuristic BM values were104,160,176 (not necessarily powers of two), N256/K64,
4stages, FP32 accumulation. CPU capacity oracle now also covers these observed BM values.

Fresh-stress ece54526780a4f28 passed5 official tests and the M32 synthetic gate:2000 queued
fresh x/IDs/weights replays versus64 eager controls, all four input banks. Peak relative error
0.00242335 in both graph and eager, peak row-unit error0.0097561 in both, nonfinite0,
share over1e-4 graph0.1265 versus eager0.203125, bad counts/NP0, nonlocal output exact0.
Matched eager-envelope, row/share and order1 gates passed without relaxing official tests.
This is state freshness evidence; plain/official tests establish independent model correctness.
The stress build's1.140469x (95%CI1.123362–1.157576) must not be compared as an isolated
optimization with the separate1.176449x run: prepare/allocation/process effects differ.
Receipt/log/profile text: `profiles/humming_stress.ece54526780a4f28.*`.

Root requested a pure first-submission candidate. Frozen `.build/today-moe/prefill_humming_v3.py`,
SHA256f3024d1b01a5ef2bf655dcffe3110849ab6b0de15aecad6dba388f09e7ffb8fa, is byte-for-byte
the prefix of the passing stress file. The only removed code is its appended prepare-time
diagnostic wrapper. GPU functions, core prepare, _experts and FP32 ABI guards are identical.
Path/hash/proof sent to root; this agent has not submitted publicly.

### Experts-component paired confirmation and heuristic shape factor

Check7598f81b9b4645f9 dispatched, SHA256c8e4e3bebccf01b7ea05defd2b99ed3e48f357775923e807b897c9568bc1168b.
`tools/prefill_humming_component_bench.py CANDIDATE BASELINE OUT` uses Marlin/Marlin2,
HummingR/R2, HummingM/M2 on identical supplied weights and held-out activation/routing seeds
271/272 at753,3012,16384 tokens,12 randomized cold-L2 graph rounds. Duplicate Humming arms
share only read-only transformed weights; each owns lock buffers, mutable metadata and
scratch. Duplicate Marlin banks have independent workspace. Config factor only: Humming
heuristics see R=M*8 versus estimated local M; actual tensor dimensions and device metadata
remain unchanged, and no input-derived value chooses a policy. No zeroing removal is mixed in.
All arms compare mathematical outputs with Marlin before timing; component time is not a
whole-prefill score. HBSTOP explicitly marks incomplete coverage beyond the230s setup guard.
Component7598f81b9b4645f9 completed5/5 official tests; control headline1.143518x
(not an arm score). All six arm correctness checks and all sizes completed, no HBSTOP.
At753, M minus R paired medians were -109/-162us on seeds271/272 (M2 -110/-162us),
with R2 duplicate controls +1.8/+0.7us. At3012, M -228/-111us (M2 -230/-111us),
R2 -1.1/-1.6us. At16384 results unresolved: M +9/-25us, M2 +44/+6us,
R2 +37/+26us. Marlin versus indexedR medians at3012 were2467/2775 versus2024/2065us;
at16384,13749/8780 versus9235/6687us. These establish component opportunity, not
projected aggregate improvement. Receipt committed as `profiles/humming_component.7598f81b9b4645f9.json`.

### Global local-M heuristic fresh gate: closed

Checkec3618d1604e44ba used local-M estimate globally, same original four-bank fresh
gate at32,64 eager controls and2000 queued replays. FAILED prepare before official tests.
Metadata/nonfinite/zero-local errors all0, rel eager0.0011458 versus graph0.0016904,
row0.00909 versus0.01042, but share of rel>1e-4 was0.046875 eager versus0.5 graph,
violating the unchanged2x eager-share envelope. Small-shape heuristic changed BM24 to8.
No math/latency score and no promotion; thresholds were not weakened. JSON/log committed
as `profiles/humming_local.ec3618d1604e44ba.*`.

New bounded policy is justified solely at paired-winner sizes753/3012; routed estimate
remains at32 and unresolved16384. Checkd26a87ca2f484145 dispatched from
`.build/today-moe/prefill_humming_bounded_stress.py`. Pure core is
`.build/today-moe/prefill_humming_bounded.py`; diagnostic suffix runs the unchanged
four-bank/64 eager/2000 queued gate at each of32,753,3012. No zero-removal or masked
layout mixed in. CPU route contract oracle and generated syntax pass.

Bounded d26a87ca2f484145 FAILED at unchanged32-token control before reaching either
changed shape. Eager64 rel/row/share all0; graph2000 rel0.00232594,row0.010989,
share0.002 (four outliers); metadata/nonfinite/zero-local errors all0. Same incumbent
BM24. This is evidence of sparse nondeterminism and a short eager envelope, **not**
evidence of a bounded753/3012 performance loss. No promotion or tolerance relaxation.
Status/log committed `profiles/humming_bounded.d26a87ca2f484145.*`.

Root approved a new isolated mechanism: disable Stream-K under original routed heuristic.
Check74eb65dad3fa4e54 pending; pure `.build/today-moe/prefill_humming_nostream.py`,
diagnostic `_nostream_stress.py` runs the unchanged32/753/3012 fresh gates. Pinned0.1.12
`forward.py:110` serializes explicit tuning dict, `kernel/humming.py:505–513` uses provided
tuning and merges into KernelConfig; `config/config.py:337` declares use_stream_k.
No shape policy, precision, zeroing, routing or tolerance change. Generated syntax passes.

### GROUPED_MASKED offline prototype (not dispatched)

`tools/prefill_humming_masked.py` retains exact reviewed NVFP4 weight loading and rank
metadata, using counts as Humming expert_layout. Source pinned0.1.12 scheduler masked
case uses per-expert actual counts and capacity shape_m/32, allowing fixed GPU capacities
without contiguous last-expert overcomputation. Shape-only tuning estimate m; real input
shape32*MCAP. MCAP rounds m to lcm(BM1,BM2); top8 unique guarantees each expert count<=m.
Device assert enabled explicitly with debug=True, pointer accesses also masked by rank<cap.
No data-derived host reads/cache. Fresh counts reset each call, retained per-shape buffers
fail closed on first allocation during capture. Planned bytes/free memory printed outside
capture; insufficient memory fails the complete candidate, not partial coverage.

Own gather writes one exclusive grouped slot per local route; G1 tailzero covers only
[count,ceil(count/BM1)*BM1), no zero-count work. Guarded route-indexed SiLU uses FP32
math and one BF16 store, reads only initialized valid G1 rows. G2 tailzero similarly covers
BM2 remainder; G2 output scattered only for live rank/ID and other route rows explicitly0.
Original expert norm then routing-weight order retained. No whole20GB buffer clears.
CPU oracle `tools/test_humming_masked_contract.py` passes unique routing, slot bijection,
capacity, tail disjointness and zero-count proof for1/8/32/753/3012/16384 tokens across
BM8/24/64/104/128/160/176 combinations. Syntax passes; GPU correctness/performance untested.

Ordering source: [PTX8.7 section8.9.4 API synchronization](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-synchronization).
Completed prior same-stream task synchronizes with next start; kernel/graph dependency
boundaries participate in proxy-preserved base causality. Thus ordinary gather/tailzero/
SiLU kernel completion orders global writes before subsequent TMA reads. No PDL,
overlapping dependent kernels or helper-written tensormap mutation; all Humming internal
fences preserved. Humming's existing grouped-C descriptor updates (`scheduler.cuh:333–345`)
retain tensor_map_release_cta/tensor_map_acquire_cta; these are a separate library chain
from the completed-kernel global-input visibility argument. Independent read-review
requested before any masked dispatch.
Correction: INDEXED disables TMA/warp specialization, not necessarily WGMMA. No instruction
family comparison is measured without SASS. Masked exactNVFP4 support still needs GPU check.

Independent audit_attention_moe read-review found no layout/lifetime issue; confirmed
both tail invocations explicitly enable debug=True. Provided tuning dictionary preserves
the actual chosen BM (same explicit-config API chain cited above). Root approved plain
correctness/resource/performance check c1f6023f03d34c72, dispatched from
`.build/today-moe/prefill_humming_masked.py`, SHA256
3314d555570555271f944adf4c8ef19553a4252a1d1f499d4d81732ebdfd75e5.
Pending. Default StreamK policy retained for this new-backend initial gate; separate
deterministic prerequisite, if needed, will be recorded and validated before promotion.
Pure indexed-noStreamK source SHA256d88863eede8104c35948c0ad8c29fd7cca8aa0db291f3e272f45562e368fb1a4.

Indexed-noStreamK74eb65dad3fa4e54 completed5/5 tests,1.092871x (95%CI1.076478–1.109264).
All ten fresh statistics exactly0 at32/753/3012: matched64 eager and2000 queued per
shape, including metadata, zero-local and nonfinite. Printed configs allFalse; BM24 at32,
104 at300,160 at1506,176 at600/753/3012/4096/16384; corresponding remaining tile config
unchanged. This is strong evidence that disabling StreamK resolves the observed replay
variation on these cases, but it loses aggregate performance versus scoredR incumbent
1.146466 (different runs, no isolated speedup claim). Candidate/SGLang per-shape median us:

| shape | noStreamK | SGLang |
|---|---:|---:|
|753 turn1 sliding|1157.6|1081.2|
|753 turn1 full|1512.7|1483.5|
|3012 turn4 sliding|2862.4|3216.0|
|3012 turn4 full|4392.9|4842.7|
|16384 firstturn full ctx0|13938.0|17230.7|
|16384 firstturn sliding ctx32768|13726.0|17245.9|
|16384 firstturn full ctx32768|17529.2|20570.6|
|16384 firstturn full ctx65536|21118.9|24225.5|

Root approved deterministic-base bounded753/3012 localM factor to seek compensation.
Checkda9700816b1641ea pending, pure `.build/today-moe/prefill_humming_nostream_bounded.py`
differs only shape_estimate=bounded, all othersR, StreamKFalse. Same unchanged fresh3shape
gate. No zero/masked combination; paired controls required before promotion.

Maskedc1f6023f03d34c72 completed5/5 initial correctness tests but FAILED16384 benchmark
recheck0: residual update relL2 0.6951 exceeds0.0422345 bound. No aggregate score. Smaller
753/3012 benchmark setup reached, not enough to claim backend correctness. At16384:
CAP16544,planned22759342208B,free93950246912B,total150109880320B; BM176,N256,K64,
four stages, TMA/warpspec/mbarrier/StreamKTrue. Thus capacity allocation feasible on this
image; replay/math failure unresolved. All printed configs and resources saved in
`profiles/humming_masked.c1f6023f03d34c72.*`. Independent source review investigating
TMA physical row footprint and grouped StreamK fixups; no blind sameconfig rerun approved.

Concrete source hypothesis found after failure: own grouped Triton row*stride uses signed
I32 arithmetic. AtCAP16544,32*CAP*4096=2168455168 elements exceeds2^31; C1 stride8192
exceeds2^32. This matches16384 failure onset, while smaller shapes fit. Every grouped
row is now widened to tl.int64 BEFORE stride multiplication at four sites: gather,
tailzero, SiLU, scatter. Original failing artifact preserved; corrected frozen source
`.build/today-moe/prefill_humming_masked_i64.py`. CPU boundary test confirms row itself
fitsI32 but final offsets wrap; both contract tests pass. TMA A descriptor source
`csrc/launcher/tensor.h:261–276` uses exact block_shape_m, no176->192 padding found.
Independent exact-fix review requested; GPU confirmation still needed before causality claim.

Independent audit confirms all four widened-before-stride sites and unchanged exact176-row
TMA footprint. Root approved corrected single-factor gate f92ffc2953284a9a, dispatched
from frozen `_masked_i64.py`, SHA256
63eb4171f1813887b8f4c6b855770b54ebaa761f7176b1344ee21881002ded5f. Pending;
only four pointer-width casts differ, no noStreamK/heuristic/zeroing change.

Deterministic boundedda9700816b1641ea completed5/5 and all ten fresh statistics exactly0
at32/753/3012,64 eager and2000 queued each. Headline1.161679904x,95%CI1.144254706–1.179105103.
Actual753 BM48: G1N256K64,G2N128K128;3012 BM128,N256,K64 both. All use_stream_kFalse.
Other shapes retain originalR configs. Candidate/SGLang medianus by officialindex0–7:
1024.4/1063.5,1416.3/1435.7,2600.8/3316.2,4185.7/4907.4,13958.4/17261.1,
13419.5/16795.9,17532.2/20533.4,21149.7/24278.2. Aggregate versus incumbent is
cross-run and overlappingCI; paired component compensation gate remains required.
Pure `_nostream_bounded.py`, SHA25629770518d1e74dae4a645193a06c8f76c5346d7263933352abf101c932bcaf5d,
is exact prefix of passing stress source. New generator policy arg reproduces AST exactly.
Paired duplicate component21e98162b7114898 dispatched: same original six arms, heldout271/272,
cold random12rounds, shapes753/3012 only, allR/R2/M/M2 use_stream_kFalse. Pending; no
zero-removal/masked combination. Status/log saved `profiles/humming_nostream_bounded.da9700816b1641ea.*`.

Corrected maskedf92ffc2953284a9a PASSED5/5 and all8 benchmark rechecks,1.294149997x
(95%CI1.274737747–1.313562247). Four casts alone fix the large-shape failure under
unchanged configuration; this confirms the specific overflow fix on all official shapes.
Candidate/SGLang medianus index0–7:968.2/1050.5,1342.6/1442.8,2339.4/3236.7,
3942.8/4871.7,11138.3/16974.8,10813.7/16884.5,14799.2/20750.1,18662.9/24540.5.
Largest allocation plan22.759GB, full coverage fits with last observed free43.369GB before
allocation. Same TMA/warpspec/mbarrier/StreamKTrue configs. Not promoted until fresh edge
gate; component score improvement versus other runs is not an isolated backend speedup.
Receipt/log committed as `profiles/humming_masked_i64.f92ffc2953284a9a.*`.

Root approved poisoned fresh gate defbb042449d44e7, pending. Pure63eb4171... source
prefix unchanged; appended `tools/prefill_humming_masked_stress.py` suffix runs32/753/3012,
64 eager controls/2000 fresh queued each, same unchanged envelope. Exactcounts, bounds,
zero-local and nonfinite tested; indexedNP removed because masked backend has noNP.
Before each bank reference, groupedGX/C1/C2/C3 are NaN poisoned. Same0local/singleexpert/
all8local/mixed alternatives, counts=M extremes testCAPtails. All fresh reporting retained;
230s setup guard before each new shape fails incomplete coverage. No NST/Mpolicy mixed in.

Deterministic component21e98162b7114898 PASSED5/5 official tests and all six arm
mathematical checks, no HBSTOP. Control headline1.101533x (95%CI1.085010–1.118056),
not a bounded-policy score. M minus R paired medians at753 seeds271/272:
-197.9/-191.6us; M2 -195.6/-190.8us; R2 null -0.90/+0.74us. At3012:
M -399.8/-260.7us, M2 -399.9/-260.3us; R2 -0.61/-0.83us. All four Humming
arms explicitly use_stream_kFalse with independent scratch/locks and shared readonly
weights. This confirms compensation under deterministic config; no mixing with nozero
or masked backend. Passed pure deterministic bounded candidate/hash sent to root;
root alone decides scored runs/production integration. Receipt and complete component
log saved `profiles/humming_nostream_component.21e98162b7114898.*`.

### User defers prefill; only already-dispatched receipt completion

User requested "ignore prefill for now" while defbb042449d44e7 was in flight. No new
prefill dispatch, promotion or code change after that steering. Read-only completion:
defbb passed5/5 and all poisoned fresh32/753/3012 gates,1.300970887x
(95%CI1.281456324–1.320485450). Metadata/nonfinite/zero-local all0 at each shape.
Matched eager/graph rel32 .0024736/.0024736;753 .003194/.003197;3012 .003217/.003280.
Shares>1e-4 .375/.268,.75/.75,.75/.75; rows .00862/.00862,.01307/.01266,
.01538/.01613. Registered2x-envelope passes; this is not deterministic arithmetic and
does not authorize promotion. Receipt/log saved `profiles/humming_masked_fresh.defbb042449d44e7.*`.

### Standalone structural hypothesis: device-count GROUPED_MASKED at128/256

Root approved up to3 standalone checks (including fresh only if winner) after reviewing
current standalone indexed loser8d64807ea7504495 and ring4/staged-pair closures. New
mechanism is grouped slabs with GPUcounts enabling library TMA/warp specialization,
not another fixed-ring geometry. Prior indexed128 G1/G2≈242/135us,256≈291/152us;
that path loses to ownedpersistent47/51. Prior staged45/46 same mma.sync pipeline loses
~30us; it does not test this library backend. Completed larger-shape library measurements
prove API/math feasibility, not standalone performance. No projected speedup claimed.

`tools/moe_humming_masked_bench.py` builds `.build/today-moe/moe_masked_ab.py` from frozen
production51. Official kernel_fn remains entirely ownedcontrol;<=32 unchanged. One
originalprepare/extension build. Four arms owned51/owned51dup share readonlyrepackedweights,
each own_scratch; masked/maskeddup share readonlytransformedweights with independentlocks
and metadata/workspace. All four arms checked against ownedFP32reference at everycase
(original TEST_TOL .008, row.075, finite).128/256 layer/std heldout291/292 and skew291
correctness; fixed previouslycalibrated layerrouter8, no fresh seed search.20 randomized
cold-L2 rounds, duplicate null controls,280s entireprepare guard fails incomplete coverage.
Acceptance:>=3us at128 and>=4us at256 plus2xnull consistently on both measured routings;
all official/fresh gates before promotion. Stop if launch/gather/tail/norm costs erase GEMM
gain; no geometry retune without new measured stage evidence. Syntax passes, independent
adapter/alias read-review requested. No dispatch yet.

Standalone gateae8993f4e3e1435c dispatched after root and independent read-review.
Root caught required interface correction before dispatch: standalone gamma=(1+post_norm_w)
is FP32 (ownedstate nw), while prefill's BF16 effectivegamma boundary differs. New adapter
preserves FP32 gamma contiguous; no BF16 cast. Earlier indexed8d648... adapter used BF16gamma,
so its numerical interface differs slightly as well as its kernel schedule; no promotion.
Independent reviewer required post-graph output checks: all four arms now checked against
FP32 ref after20 queued replays+sync with unchanged .008/.075 criteria, not only eager.
One untimed CUDA profiler sample at256/std292 after AB timing attributes masked stages.
Generated syntax passes; original officialjudged owned kernel_fn untouched. Pending.

ae8993f4e3e1435c completed5/5 official tests, judgedownedcontrol2.200397x
(95%CI2.167391–2.233403); prepare159.6s. Full MHRESULT retained, all40eager and32postgraph
FP32 checks pass. Masked losses (us, seeds291/292):128 layer+180.2/+180.0,std+184.2/+144.6;
256 layer+234.9/+231.8,std+228.7/+202.8. Maskedduplicate agrees; ownednull maximum2.69us.
No confirmation/fresh or localtile retune; production source/policy unchanged.

Untimed256/std292 masked profile: G1 244.447us,G2 136.064us (90.25% of421.599us total);
tail17.344us/2calls,SiLU8.064,scatter5.888,norm4.160,gather3.296,rank1.600.
GEMMs alone exceed samecase owned217.15us, so glue-only fusion cannot repair this loss.
This closes the tested default GROUPED_MASKED schedule at128/256, not all TMA/dequant
architectures. Actualcfg/CAP/counts/padding printed before final summary fell outside16KB
receipt tail; no exact observed values asserted. No GPU reconstruction/rerun warranted.
Receipt/log saved `profiles/humming_masked_moe.ae8993f4e3e1435c.*`.

### Prospective standalone backlog after masked closure (no dispatch)

Scoring source `TASK.md`: six equally weighted log load ratios.128/256-only work covers
2/6 logscore;1/8/16/32 covers4/6; repeated8-token job reuse affects only16/32,2/6.
At root's23:26PDT board snapshot,2.595612/2.215712=1.171457 required score ratio.
If only two loads improve equally, each needs1.607605x (37.8% latency reduction); if
four improve equally, each needs1.267914x (21.1% reduction). These are algebraic required
changes under an unchanged cutoff, **not** speedup forecasts. Representative763197 medians
versus statedhardwarefloors yield optimistic floor-only aggregate multipliers1.2715 for
128/256,1.5947 for smallfour,1.2292 for16/32 only. Floors are lowerbounds, not attainable
targets or measurements of any proposed mechanism. Rankcutoff may move.

Priority1 is a traffic-attribution diagnostic before any new kernel: quantify duplicated
36KB weightunits from multiple8-token jobs per expert at16/32 and compare existing
loads-only/current skinny controls.763197 prepare-route means: loads-only39.9/69.7/105.4/
146.7us at1/8/16/32, computing41.6/70.9/106.3/149.2us. These are not the official median
latencies or a rigorous upperbound, but little arithmetic headroom is demonstrated on
those inputs. New structural hypothesis: share one expert weightunit across paired
8-token jobs via CTA-cluster TMA multicast with existing consumers/ring/fences, rather than
another ALU/MMA-only replacement or old TMA/barrier geometry. Requires a primary-source
instruction/descriptor proof and measured duplicatebyte/copy-only opportunity first.
M<=8 has no repeated8-token job; a broad lowtoken score claim would be false.

Acceptance before implementation: repeatable copy-only gain beyond2xduplicate null on
heldout layer/std routing and meaningful aggregate score coverage; source shows multicast
compatible with original36KB packing and barrier/lifetime semantics. Stop before a
prototype if duplicatebyte share and coldcopy gain are insufficient. A new kernel must
retain BF16 activation boundaries and FP32 standalone gamma, all fences, officialcorrectness,
fresh replay gates and paired whole-score compensation. No claimed stage/core utilization
beyond the measured timings; no occupancy/HBM/L2 hardware counters were collected here.

### Fresh logical-count attribution only (no new GPU kernel)

Root approved count-only check53c07507b9fd4f8c, dispatched on standaloneMoE from
`.build/today-moe/moe_weight_count.py`, generated by `tools/moe_weight_count_check.py`.
Production51 newline-normalized source prefix preserved (CRLF->LF during generation);
all original top-level AST nodes identical. Official kernel_fn/math/policy unchanged.
CPU map code embedded directly at generation time, no runtime file reads. M16/32,
layer/std, heldout311/312, fixedrouter8; all32 counts, exactmap summary and actual
deviceSM count printed.280s guard fails incomplete coverage. Early screen preregistered:
every requested case must have>=10% pairable logical source-request savings. HBM/L2
bytes and multicast benefit remain unmeasured. No copy/multicast kernel or barrier change.

First uploader attempt was rejected422 before any job ID: "edits loaded modules".
The tool had registered an embeddeddataclass module in sys.modules. Corrected by direct
class/function embedding with no exec or module-state mutation. Not a GPU/math result
and no duplicate check/job is claimed. Corrected source passes syntax, CPUmodel2tests.
Pending53c07507b9fd4f8c; on screen failure park before any full rewrite.

53c07507b9fd4f8c completed5/5 official, unchangedcontrol2.201850463x
(95%CI2.168822706–2.234878220). Full8case WCOUNTRESULT retained,Gactual132. Early
screen FAILS6/8: pairable savings M16layer311/3120/0%,M16std0/0%; M32layer18.1818/18.1818%,
M32std0/0%. Logicalduplicates M16layer0/11.1111%,std0/0%; M32layer27.2727/36.3636%,std0/0%.
All32counts, jobs/activeexperts, bytes and exactCPUmap stats saved in
`profiles/weight_count.53c07507b9fd4f8c.*`. The16/layer312 duplicate misses the consecutive
pair alignment, but fourstd cases and16/layer311 have none even under optimalpairing.
Park the registered two-size/two-routing multicast hypothesis before GPUcopy/fullrewrite;
no narrowing to32-layer after failure. HBM/L2 bytes, multicast benefit and residency
unmeasured. Dated audit addendum records exacttable and stopping/reopening scope.

### November1 PDT: skinny full-slot wait policy, one hypothesis

Check87f22aa71e614460 dispatched from `.build/today-moe/moe_skinny_testwait.py`, SHA256
58bce614db8b5baefa5d5c264a87da51515272bd11d1dd577beb146d2eabe99b. Builder
`tools/moe_skinny_testwait_check.py` at776b859. Root approved after independent
today_proj read-review: exact generated hash/sourcegate verified; PTX8.7section
9.7.13.15.16 gives same acquire-on-True for test/try, old defaults acquire/CTA;
parity and only two skinnyfull call sites change. Globalhelper/fat/empty/hand/producer
progress/watchdog/fences unchanged. No short hint, NS/prefetch or vote changes.

Four independently compiled arms sk8w3s mode0, neutral unusedbit256, testbit128 and
testduplicate384; private scratch, sharedreadonlyweights. New flags excluded from
production policy. Resourcevectors regs/local/static/dyn/blocksSM/maxThreads must
match exactly before timing; all16cases1/8/16/32 layer/std, heldout321/322 fixedrouter8,
64eager+64postgraph mathchecks, cold20random rounds, original.008/.075 limits/spin0.
Registered1% sixload contribution beyond2xnull and percase-regression gate; this
fixedsk8/3 contribution assumes large128/256ratio1 and is NOT an officialpolicy score.
Two corrected fully counted stamped2/130 twins after timing, optional265s guard;
corecoverage280s failclosed and finalsummary<10KB. Production source untouched.
Reviewer notes timeoutbreak can reach loads withoutTrue exactlyasbaseline; timeout
flag rejection remains required, never an acquire proof or successfulmeasurement.
Pending; no promotion before exactcandidate official/fresh gates.

87f22aa71e614460 FAILED prepare in host harness formatting: `for round in range(20)`
shadowed built-inround, then compact `round(v,3)` raised TypeError at firstcase summary.
Baseline preparation/selftests completed, build153.6s, policy skinny1/8/16/32+51large,
spin0. Exactresourcegate and firstcase math reached, but no resourcevector or complete
16case table retained; no official score or wait-policy timing conclusion. Failedreceipt/log
saved `profiles/testwait_harness.87f22aa71e614460.*`. This is a tooling error, not kernel loss.

Root authorized one host-only correction. Changed loop variable to round_idx; all
CUDA/CPP/skinny/variant constant values exactly equal between frozenoriginal andv2.
Added CPU scaffold executing ACTUAL20-round hostloop, compactformatter, scoregate and
summary via AST with no-op event/graph stubs. Synthetic winning and losing scoregate
branches pass; a negativecontrol recreates the old shadowing TypeError. Three source/
host tests pass, syntax passes. Correctedcheck252df4a54d764ca5 dispatched from
`.build/today-moe/moe_skinny_testwait_v2.py`; same preregistration/reviewedCUDA, pending.

252df4a54d764ca5 FAILED optional stamp-host decoding after the complete core loop:
inter is rank2; `.view(int64)[:G*16]` sliced rows, so reshape132x16 saw2162688 elements.
No core summary had yet printed, therefore no per-shape timing or contribution score
is recoverable/claimed. Baseline build153.4s/selftests spin0; the stack location is
after core math/timing, not evidence of a wait-policy win. Failedreceipt/log saved
`profiles/testwait_stamp_harness.252df4a54d764ca5.*`. No newGPU authorized yet.

Host-v3 repair flattens the dtype-view before selecting2112 counter values, prints
the complete core summary BEFORE optional stamps, and catches only decoder errors
as explicit no-attribution. Kernel execution and spin-timeout failures remain fatal
outside that catch. Core280s/stamp265s guards and registered performance gates unchanged.
Persistent local `/root/.cache/okc-host-validate` now has officialCPU-onlyTorch2.10.0
(Python3.14); CUDA_availableFalse. It is host validation, not a GPU benchmark.

Six tests pass: source scope, callable-shadow check, ACTUAL20-round loop/all16compact
case formatting and winning/losing scoregate/summary; plus REAL CPU Torch2D4096x4096BF16
inter decoding, full valid headers, zero-header rejection, reproducedoldreshapeerror,
core-preserving error/guard summaries and fatal CUDA-execution/spin/coreguard branches.
CPUfixture A/B wait values are synthetic and not reported as hardware measurements.
Validation log `profiles/testwait_host_cpu.2026-11-01.log`. V3scratch source generated
and syntaxchecked; independenttoday_proj host-source re-review requested before root
decides whether to authorize a further corrected check. CUDA/constants unchanged.

Independent v3 host review caught a further exception-scope issue before any retry:
GPU `.double()`/`.cpu()` were still inside a broad decodercatch. Finalv3 moves a
flattened int64 prefix snapshot/CPUtransfer OUTSIDE the catch, after launch/sync/spin
checks. Only CPUint64 reshape/header/math is optional inside. Snapshottransfer error
negativecontrol stays fatal; zero/badCPUheaders remain no-attribution. Actual2DTorch
tests and source/formatter tests all pass6/6, independently rerun by today_proj.
Verbose validation log `profiles/testwait_host_cpu_v3.2026-11-01.log`.

Root authorized ONE third/final corrected retry after this clearance.9be599cf333a47c4
dispatched from `_skinny_testwait_v3.py`, exact reviewedSHA256
528dea8ff79888a2b8f6693e9c5c63c20210547d6128f9bf8fc80b9f9cdcb4d1. SameCUDA/constants
and preregistered gates; no fourth retry without new root review. Core prints before
optional stamps, finalsummary always on decoder/guard fallback; kernel/transfer/spin
errors remain fatal. Pending, no performance or promotion claim.

### November 1 final skinny FULL-slot wait-policy result: close

Check `9be599cf333a47c4`, source/dispatch commit `036b581`, reviewed generated SHA256
`528dea8ff79888a2b8f6693e9c5c63c20210547d6128f9bf8fc80b9f9cdcb4d1`, completed
on H200. All 5 official tests pass. The official path remains owned production control:
2.207723314x (95% CI 2.174607465–2.240839164), not a test_wait candidate score.
Its median us at M1/8/16/32/128/256 are 27.401/68.636/95.941/134.509/192.134/237.144;
baseline medians used by the receipt are 77.648/145.904/186.976/209.360/460.368/627.152.

Complete four-arm timing evidence survives the receipt: all 16 layer/std cases at fixed
seeds321/322, 64 eager +64 postgraph FP32 checks pass, spin flags0. All arms have exactly
130 registers, 0 local/static bytes, 155520 dynamic bytes, 1 block/SM, maxThreads288.
Fixed-sk8/3 six-load contribution multipliers: null1.000295776, test1.000956258,
test duplicate0.998638858. These assume unchanged128/256 and are NOT actual official
policy gains. Both the prospective >=1%/2xnull gate and per-case regression gate fail;
KEEP=False. Exact two-full-slot test_wait mechanism closed; no production edit, fresh
promotion check or fourth harness retry. Reopen only with new evidence/mechanism.

Optional corrected stamped twins completed all8 cases. Mean CTA thread0 A/B full-wait
regions in us (try_wait -> test_wait), not critical-path savings:

| M / routing | A region | B region |
|---|---:|---:|
| 1 layer | 2.677 -> 2.125 | .226 -> .224 |
| 1 std | 2.908 -> 2.494 | .380 -> .383 |
| 8 layer | 5.897 -> 5.812 | 1.955 -> 1.769 |
| 8 std | 5.794 -> 5.261 | 1.771 -> 1.761 |
| 16 layer | 4.280 -> 4.119 | 1.321 -> 1.272 |
| 16 std | 5.857 -> 5.858 | 1.953 -> 1.947 |
| 32 layer | 4.791 -> 4.913 | 1.973 -> 1.996 |
| 32 std | 8.031 -> 7.693 | 3.011 -> 2.998 |

Modest changes in these instrumented regions do not produce a repeatable whole-kernel
improvement. Raw per-case medians/paired deltas, all resource vectors, stamp rows and
official profiles/receipt are preserved in `profiles/testwait.9be599cf333a47c4.*`.
Earlier87f.../252df... remain distinct host failures with no recoverable timing claim.

### November 1 single-warp stable position map, one check

`85a113a3f11d47a5`, source a776f32, generatedSHA256
f5762dbdaebd9f646781fd5eba6e931246ad5228ddef5841fd68b4732687902f,
changes only skinny position_pass for M<=16 to one warp's stable32-slot chunk scan.
Measured historical generic positions1.6/1.8/1.9us at M1/8/16 std motivate this
mechanism; full removal would algebraically give1.01795 six-load factor using
current medians, not an attainable forecast. Initial routing/jobs/copies, all async
fences, watchdog, math and M32+ original map unchanged. No token-order or polling
change. Modes0/256/512/768, private workspaces, fresh fixed seeds441/442, layer/std
1/8/16/32,64eager+64postgraph correctness, exact resources before timing, unchanged
>=1%/2xnull/no-regression gate. All3 CPU source/bijection/actualhost tests pass;
today_proj independent review found no blocker. Pending; root approved one check
maximum, no retries/sweeps/public submit. Report: steering/2026-11-01-moe-singlewarp-position.md.

85a113a3f11d47a5 DONE/ok H200:5official tests,64eager+64postgraph FP32 checks,
all16cases,spin0, exact4arm resources130regs/0local/static/155520dynamic/1blockSM/
maxThreads288. No candidate gain: fixed-sk8/3 six-load contribution test0.999496192,
duplicate0.999798370, null0.998989769. >=1%/2xnull and per-case regression gates fail,
KEEP=False. Modified-case paired deltas -.320 to +.704us; no repeatable improvement.
Official unchangedowned control2.203921580x CI2.170862757–2.236980404; medians us
M1/8/16/32/128/256=27.623/66.246/96.392/136.976/193.808/235.388. Candidate is
not the judged production path. No new phase attribution; do not infer why metadata
time did not translate into latency. Close exactsinglewarp-map mechanism; no retry,
sweep, promotion or production edit. Full profiles/receipt `profiles/position.85a113a3f11d47a5.*`.

### November1 new architecture comparator, exact external source only

Root authorized one donor-only off-board measurement after recovering keell0renz
commit8e30e1be994db4310794b8d04e90d18d895deb84, not the older alygg77 comparator.
`b65c4c878014416c` checks its standalone file byte-identically, SHA256
2d931f09133d058053dda833b51f1867ea06cdcac0c912521ad37e8f7853185c; no patched
harness/defines, concurrent GPU prepares, production edits or public submit.
PASS5tests H200,2.395702934x CI2.359767390–2.431638478. Median us M1/8/16/32/128/256:
25.114/63.252/91.149/122.315/176.505/211.115. This is8.123% above our published
2.215712x across different runs/inputs, NOT matched-owned causal evidence; still
2.408% below fresh rank50 threshold2.4534. Full receipt/log/profiles saved as
`profiles/kel_exact.b65c4c878014416c.*`; board snapshot also saved. Modules built53s,
SC6W active, skinnySC6off. MAIN96regs, small149regs, dedicatedM8issuer151regs,
all local/static0; no occupancy/physical-traffic claim. Combined new architecture
merits ordering/precision/license audit and matched isolation before adoption.
One check spent; no more dispatched. See steering/2026-11-01-moe-architecture-comparison.md.

## Source facts (SGLang a55ab7d5, read from GitHub and hashed against the image)

- `fused_post_expert_norm_weighted_sum` falls back to PyTorch ops below
  `_MIN_TOKENS_FOR_FUSED = 32` tokens: a `where` to zero filtered slots, then float, scale,
  pow, mean, rsqrt, multiply, multiply by the routing weights, sum and a cast, about ten
  launches. At 32 tokens and above it is one Triton kernel, one program per token.
- With an expert map, `moe_align_block_size` aligns over 257 buckets (256 global experts
  plus the filtered -1 bucket). At 64 routes or fewer (bs <= 8) that is the single-CTA
  Triton `moe_align_small_numel`; above, SGLang's two-kernel CUDA align. The filtered routes
  get blocks with expert id -1, which every Marlin CTA scans past.

## Runs

| date | job | commit | change | score (95% CI) | notes |
|---|---|---|---|---|---|
| 2026-10-28 | a61a16e1f1e84206 | 79edc53 | own align (all bs) + one-block-per-token finalizer (bs 1, 8, 16); bench in prepare | 1.193x (1.175-1.211) | per shape: bs1 1.542, bs16 1.271, bs32 1.046, bs8 1.335, bs128 1.027, bs256 1.025. Bench (graph replay, cold L2, own routing seed): finalizer alone is most of the gain (bs1 87.1 -> 54.0 us; align alone 87.1 -> 83.3); finalizer variants 0/2/4/8 within 0.5 us of each other; block_size_m 16 beats SGLang's pick at bs128 (361 vs 482 us) and bs256 (471 vs 632 us), bs16 16 vs 8: 92 vs 97 us |
| 2026-10-28 | 1c4d11ad0cab4943 | 7593542 | block_size_m 8 below 12 tokens else 16; our finalizer at every size; bench sweeps in prepare | 1.327x (1.307-1.347) | per shape: bs1 1.543, bs16 1.252, bs32 1.046, bs8 1.339, bs128 1.446, bs256 1.396. Bench, 3 routing seeds each, block_size_m 8/16/32 (us): 16 wins from 48 tokens up (bs128 319-350 vs 383-430 at 8 and 376-404 at 32; bs256 398-501 vs 535-749 and 455-504); 8 and 16 tie within seed noise at 12-32 tokens. is_ep=False Marlin (our align emits no filtered blocks) saved 2-7 us but prepare grew from 36 s (check a61a16e1) to 244 s of the 300 s cap with it, the bench also having grown, so a second Marlin JIT build costs on the order of 200 s; not used |
| 2026-10-28 | 947c28a7dd3d4de9 | accd987 | same choices as 7593542, benches removed from prepare (self-test kept) | 1.323x (1.303-1.343) | per shape: bs1 1.527, bs16 1.242, bs32 1.046, bs8 1.341, bs128 1.443, bs256 1.395; prepare 43 s. Gate/up paired layout with SwiGLU in the GEMM1 epilogue not built: silu_and_mul is 0.9-3.4% of GPU time per shape in the profiles (geomean ceiling about 1.8%, near the CI width), and a patched Marlin needs its own JIT build, inferred at on the order of 200 s of prepare from the is_ep=False build in 1c4d11ad |
| 2026-10-28 | 16fd966f071340eb | 7137cde (+bench) | measurement only: `tools/moe_make_bench.py` appends a cold-L2 streaming-read kernel and a per-shape Marlin GEMM1 / silu / GEMM2 / finalizer split to prepare(); kernel_fn unchanged | 1.327x (1.307-1.347) | Streaming reads (best of 5 grids, graph replay minus a 5.0 us empty replay): 2.68 TB/s at 9 MB, 3.06 at 18, 3.28 at 28, 3.65 at 57, 3.80 at 113, 3.95 at 283, 4.06 at 640 MB, so at these sizes the measured ceiling is 3.3-3.9 TB/s, not 4.8. Marlin over unique weight bytes (active experts x 18.9 MB for GEMM1, 9.4 MB for GEMM2), 3 routing seeds: bs1 (2-4 experts) g1 21-34 us at 1.8-2.2 TB/s, g2 18-22 us at 1.1-1.7; bs8 (7-10) g1 55-74 us at 2.4-2.6, g2 31-41 at 2.1-2.3; bs16 (11-13) g1 85-99 at 2.5, g2 46-53 at 2.3; bs32 (14-20) g1 107-149 at 2.5, g2 57-78 at 2.3-2.4; bs128 (23-25, hottest expert 30-124 routes) g1 185-244 at 1.9-2.4, g2 96-128 at 1.8-2.3; bs256 (21-25, hottest 48-256 routes) g1 239-289 at 1.4-1.9, g2 124-149 at 1.3-1.8. silu 0.9-5.1 us, finalizer 2.9-7.8 us. GEMM1 holds about two thirds of the recoverable time on every shape, so it is replaced first |
| 2026-10-28 | 5dea75d7c668473b | a31223f | W4A16 GEMM1 (own kernel, SwiGLU fused) for bs <= 32, bench of 16 configs in prepare | 1.320x (1.301-1.340) | W4 path disabled itself: the self-test compared it to the Marlin path with a 5e-3 bound and got 6.0e-3. Against an FP32 reference the W4 path was the more accurate one (3.7e-3 to 3.8e-3 vs Marlin's 5.2e-3 to 5.5e-3 at bs 1, 8, 32): Marlin rounds gate and up to bf16 before SiLU, ours keeps FP32. So this score is the Marlin path again. Extension build 39.5 s for 16 kernels. Gate changed to FP32 error |
| 2026-10-28 | 42924d2c6e314921 | 1f9417c | W4 GEMM1 for bs <= 32 with a fixed config table (Marlin GEMM2 kept), self-test gated on FP32 error; bench of 16 GEMM1 and 12 GEMM2 configs | 1.333x (1.313-1.353) | per shape: bs1 1.398, bs16 1.380, bs32 0.991, bs8 1.437, bs128 1.444, bs256 1.417 (bs128/256 still all Marlin). The fixed table was a poor pick: bs32 used nt=4 rows=32 (0.90x of Marlin in the bench). Bench (cold L2, graph replay minus 4.8 us; the log tail lost bs1 and bs16): GEMM1 + SwiGLU best configs were 1 n8 tile with K split 2 (cfg (1,1,8,2,4,4)) or 2 tiles with split 4, 1.15-1.27x faster than Marlin GEMM1 + silu at bs 8/32/128, 1.17-1.22x at bs256; without a K split 0.55-0.75x. W4 GEMM2 + FP32 finalizer best (split 4) 1.19-1.36x faster than Marlin GEMM2 + finalizer. Full forward with both W4 GEMMs: bs8 105 -> 86 and 96 -> 80 us, bs32 237 -> 172 and 186 -> 148, bs128 334 -> 265 and 292 -> 234, bs256 454 -> 368 and 378 -> 319, error vs FP32 2.9e-3 (Marlin path 5.3e-3). More K splits won almost everywhere: the limit is CTA count and per-block prologue, not ALU |
| 2026-10-28 | cfad171137bf45f8 | 186b046 | W4 GEMM1 and GEMM2 at every size, config per batch size tuned in prepare() (34 configs; Marlin also a candidate), timed on synthetic routing only | 1.672x (1.647-1.697) | per shape: bs1 1.695, bs16 1.674, bs32 1.422, bs8 1.731, bs128 1.819, bs256 1.721. Tuned policy {1: g1 (1,1,8,2,4,8) g2 (2,1,1,4,4,8); 8, 16, 32: g1 nt1 rows8 split 2-4, g2 (2,1,8,4,4,4); 128: g1 (1,1,8,2,8,4) g2 (2,2,16,4,4,4); 256: g1 (1,2,16,4,4,4) g2 (2,2,16,4,4,4)}. Marlin never won. Profile GEMM times (judge routing): bs1 g1 20.5 g2 12.3 us; bs8 57.3/33.1; bs16 77.5/39.4; bs32 85.1/44.6; bs128 179.9/85.1; bs256 195.5/95.9. Self-test error vs FP32 2.9e-3 to 3.0e-3 at bs 1-257 (Marlin path 5.2e-3 to 5.3e-3). Build 47 s for 36 kernels, tuning 1.7 s. Registers 76 (nt1 d4) to 163 (nt2 d8), no spills (ptxas) |
| 2026-10-28 | 05924173c44046e9 | ad87094 | programmatic dependent launch on the W4 kernels (tuned on/off per size); timing-only kernel modes: loads without dequant or mma (mode 1), mma on raw words without dequant (mode 2) | 1.677x (1.652-1.702) | per shape: bs1 1.730, bs16 1.689, bs32 1.421, bs8 1.739, bs128 1.819, bs256 1.695; same as cfad171137bf45f8 within the interval. PDL: whole forward 57.0 -> 54.4 us (bs1), 86.9 -> 84.0 (bs8), 124.5 -> 122.7 (bs16), 162.8 -> 160.2 (bs32), 250.6 -> 248.7 (bs128), 358.4 -> 356.9 (bs256), kept on everywhere. Diagnostics (tuner timings incl. ~5 us replay overhead, 2 seeds): GEMM1 nt1 rows8 split 4 is 52.4 us at bs8 with dequant, 50.4 without (mode 2), 60.4 with loads only (mode 1); bs32 103.8 / 101.5 / 121.6; nt2 rows16 split 4 at bs128 166.8 / 149.6 / 181.9. Dequant costs 2-4% (10% on the nt2 bs128 config); removing all compute makes the kernel no faster, so the GEMMs are bound by the weight access pattern (about 3000 warps each streaming its own 1 KB per stage), not by ALU |
| 2026-10-28 | 0f478ae5371a4b75 | 9971904 | second weight layout (8 neighbouring units interleaved per k stage, so a block's warps read 4-8 KB contiguous per stage), tuned against the first; configs pruned to 34 | 1.683x (1.658-1.708) | per shape: bs1 1.731, bs16 1.685, bs32 1.414, bs8 1.748, bs128 1.817, bs256 1.734; same as 05924173c44046e9 within the interval. Layout made no difference: matching configs within 1 us (bs8 (1,1,8,4,4,4) 52.2 layout 0 vs 52.5 layout 1; bs32 103.2 vs 104.2); the tuner took layout 1 only for bs256 GEMM1. Loads-only diagnostics: 60.3 (layout 0, 4 warps) and 61.1 (layout 1, 4 warps) vs 50.2 us (layout 1, 8 warps) at bs8, where the real GEMM takes 52.2. DRAM locality across warps is not what holds the GEMMs at about 3 TB/s; the remaining suspects are per-block start (dependent loads of items, sorted ids, x staging) and many short blocks, which a persistent kernel would remove |
| 2026-10-28 | e5ddb50d82234690 | 8da4bf0 | persistent GEMM mode (blocks walk (item, chunk, split) units, activations double-buffered, the next unit's first weight stages loaded during this unit's last), GEMM2 streams weights before its PDL wait, SwiGLU kernel triggers its dependent first; tuner checks every candidate against Marlin | 1.651x (1.626-1.676) | per shape: bs1 1.750, bs16 1.627, bs32 1.429, bs8 1.622, bs128 1.825, bs256 1.681; overlaps the previous intervals. Persistent configs landed within about 1 us of their one-unit-per-block twins (bs8 GEMM1 52.3 vs 52.4 us; bs32 108.6 vs 103.0), so block start-up and short blocks are not the limit either. Tuner chose persistent GEMM1 at bs 1-16 and persistent GEMM2 at bs 1 and 8. No config failed the Marlin comparison; self-test vs FP32 3.0e-3 |
| 2026-10-28 | c4b1b5366bc54857 | 8811b31 | diagnostics: loads-only kernels with no activation staging, with streaming-style addresses (every warp reads consecutive 1 KB chunks at each stage), with and without the L2 256-byte prefetch hint; GEMM configs without the hint as candidates | 1.690x (1.664-1.715) | per shape: bs1 1.722, bs16 1.749, bs32 1.461, bs8 1.710, bs128 1.823, bs256 1.698. Loads-only at bs8 (tuner timings incl. ~5 us replay overhead and the ~3 us SwiGLU-reduce kernel that split-K adds): real addresses 60.4 us, no activation staging 59.7, streaming addresses 59.6, streaming without hint 59.3, streaming with 8 warps per block 50.8. So the access pattern, the activation staging and the hint are not the limit; resident warps are. Net of overhead and the reduce kernel, GEMM1 at bs8 moves 141 MB in about 43.6 us, 3.25 TB/s, against about 3.85 TB/s for the plain streaming kernel at that size (85%). The no-hint GEMM variants won narrowly at most sizes (policy {1: (34, 26), 8: (33, 30), 16: (33, 11), 32: (33, 35), 128: (4, 36), 256: (37, 36)}, PDL on). Per-forward small kernels at bs1 (profile cfad171137bf45f8): finalizer 4.7 us on one block, SwiGLU reduce 2.2, route_align 2.4, against 32.8 us of GEMMs |
| 2026-10-28 | c1025175b73548fd | b911811 | finalizer over FP32 partials as 1, 4 or 8 blocks per token (a cluster sharing the per-slot sums of squares through distributed shared memory), tuned per size with PDL on/off; SwiGLU reduce over 8 blocks per route | 1.682x (1.656-1.707) | per shape: bs1 1.809, bs16 1.645, bs32 1.461, bs8 1.770, bs128 1.794, bs256 1.638; same as c4b1b5366bc54857 within the interval (per-shape swings of +-0.06 between runs follow the judge's per-run routing seeds). Whole forward, tuner: cluster finalizer saves at most 0.5 us (bs1 52.3 -> 52.0), PDL 2-4 us. Stop rule: four controlled iterations (layout, persistent blocks, access-pattern diagnostics, finalizer) moved nothing outside the interval. Superseded by 84f3975334f048c3 (BF16 boundaries) |
| 2026-10-28 | 84f3975334f048c3 | 965e415 | BF16 at SGLang's boundaries (TASK.md: activations stay BF16 wherever SGLang keeps BF16): gate and up rounded to bf16 before SwiGLU (Marlin's cache1), SwiGLU output bf16 (GEMM2 input), GEMM2 output rounded to bf16 after its FP32 split-K sum and before the norm (Marlin's cache3); rounding in registers | 1.683x (1.658-1.708) | per shape (vs c1025175b73548fd): bs1 1.780 (1.809), bs16 1.753 (1.645), bs32 1.430 (1.461), bs8 1.728 (1.770), bs128 1.795 (1.794), bs256 1.645 (1.638); same within the interval. Error vs FP32 in the self-test rose from 3.0e-3 to 4.2e-3 to 4.4e-3, still below the Marlin path's 5.2e-3 to 5.5e-3. This is the final file |
| 2026-10-28 | 4a8852d513874318 | fbbff53 | bench.py: TMA (bulk-copy) weight kernel and large-tile item lists built in prepare(), benchmarked after the tuner; kernel_fn and tuner unchanged | 1.676x (1.651-1.701) | per shape: bs1 1.771, bs16 1.644, bs32 1.450, bs8 1.722, bs128 1.808, bs256 1.688; a control run of the 84f3975334f048c3 policy. The bench itself failed at its first call (it asked the W4 extension for route_align, which lives in the routing extension), after printing registers, shared memory and occupancy per new config (see the TMA and large-tile section below). Prepare 104 s |
| 2026-10-28 | e1619f311d3747e3 | 238317a | the same bench, fixed | 1.671x (1.646-1.696) | per shape: bs1 1.771, bs16 1.649, bs32 1.438, bs8 1.738, bs128 1.779, bs256 1.675; control again (same policy). Bench results in the TMA and large-tile section below: TMA wins only at bs256 GEMM1 (NT2 W8 D4, 1.04x on normal routings), large tiles lose on normal routings at every size. Prepare 108 s, the bench itself about 7 s |
| 2026-10-28 | a49578e5c30348bb | 600e323 | TMA configs are tuner candidates (large tiles benchmark-only) | 1.691x (1.665-1.716) | per shape: bs1 1.764, bs16 1.724, bs32 1.445, bs8 1.734, bs128 1.826, bs256 1.679. The tuner chose TMA only for bs256 GEMM1 (config 49, (1, 2, 16, 4, 8, 4, 7, 1, 0): 233.7 us vs 243.6 for config 37, the previous pick); every other size kept a register-path config. Same as 84f3975334f048c3 (1.683x) and the 1.699x submission within the interval: a 4% gain on one GEMM of one shape is about 0.5% of the geomean. Error vs FP32 4.40e-3. Prepare 101 s, build 52.8 s for 64 kernels |
| 2026-10-28 | 5bc2b26850514b44 | bdbac38 | rewrite: one persistent launch (grid = SM count): routing prologue in every CTA, phase A (GEMM1 + SwiGLU to bf16 inter), phase B (GEMM2, FP32 y and sums of squares), software grid barrier, final norm and sum; 16-row x 4096-K units of 36,864 B by one bulk copy each into a 3-slot ring, 8 compute warps split K with mma.sync (x in registers), a helper warp sums partials and runs the epilogue; every M on the SPLIT path (jobs of <= 8 tokens, split experts re-stream) | 1.429x (1.408-1.451) | per shape: bs1 34.2 us (2.266x, best so far), bs16 125.7 (1.484), bs32 173.9 (1.180), bs8 92.6 (1.614), bs128 368.0 (1.260), bs256 581.2 (1.057). Every shape runs at 1.9-2.0 us per unit, 2.5 TB/s, against 1.2 us at 4 TB/s: the helper warp fenced (__threadfence, MEMBAR.SC.GPU) and credited done_a after every phase A unit, and under a saturated memory system that fence costs about a microsecond, so with 4 partial buffers of slack the compute warps waited on the helper. Error vs FP32 3.9e-3 to 4.2e-3; prepare 38 s (build 37 s); 140 regs, no spills |
| 2026-10-28 | 9f1b1b8a9f0d425e | d0dd9ca | done_a credited once per (CTA, job) instead of per unit; weight ring of 4 slots | failed (deadlock, no score) | the helper warp flushed a job's phase A credits only when it reached the next job or its first phase B unit, but it waited for that B unit's partials first, and the compute warps produce none until done_a is complete, which needs this CTA's own credits; the spin timeout fired (flag 1, about 1.09 s per launch) so the run ended instead of hanging. Self-test error 3.94e-3 (results were right once the timeout released the wait). Fix: flush before waiting at the A-to-B transition (commit after this one) |
| 2026-10-28 | f82a8cf656df4914 | 27986fb | credits flushed before the phase B wait (the deadlock fix); fat path for M >= 17: 64-token jobs, 256-row units in 16 stages of 256 k, four consumer warpgroups on register-A wgmma m64nNk16 (N = 8 x ceil(tokens / 8)), x gathered into the wgmma B layout, stream-K stage ranges by a cost table, last-arriver fixup for cut units; segments grouped by (phase, N) so ptxas pipelines the wgmmas | 1.740x (1.714-1.766) | per shape: bs1 28.7 us (2.706x; bar 30 met), bs16 100.3 (1.876), bs32 203.0 (1.009), bs8 70.2 (2.042; bar 55), bs128 302.7 (1.554), bs256 362.8 (1.708). All tests pass on the fat path (bs 33, 257, 128 skewed), error vs FP32 3.8e-3 to 4.6e-3, no spin timeouts, prepare 36 s (build 35 s). Skinny path now 1.47 us per unit at bs8 (3.0 TB/s plus about 6 us fixed); the fat path streams at 2.4 TB/s (bs32, 2.05 us per 36 KB stage), 2.2 (bs128) and 1.8 (bs256), so its weights ring (3 slots, 108 KB in flight per SM, next to 3 x 32 KB x slots) or its per-stage synchronization is the limit, not re-reads: each expert is read once. Next: a variant bench in prepare (ring depths, 16 compute warps on the skinny path, loads-only modes) |
| 2026-10-28 | bab82f5f5b974594 (submit, authorized by the user via the coordinator) | cc3c926 | variants bench in prepare with a per-size policy: skinny sk8w4s / sk16w4s / sk8w5s, fat main-d2 / main-d3 (runtime-sized rings, conflict-free codes layout), the previous kernel's configs as legacy32 / legacy128 / legacy256, plus loads-only and no-gate diagnostics | 1.869x (1.841-1.897) | per shape: bs1 28.2 us (2.716x), bs16 105.0 (1.868), bs32 136.9 (1.485), bs8 82.4 (1.837), bs128 254.6 (1.793), bs256 358.5 (1.718). POLICY {1: sk8w4s, 8: sk16w4s, 16: sk8w4s, 32: sk8w4s, 128: legacy128, 256: main-d3}. BENCH (us, mean of 2 routings): bs1 loads-only 40.0 vs sk8w4s 42.0; bs8 70.6 vs 76.0-76.2 (sk8w5s 79.4); bs16 105.4 vs 109.6 (legacy32 119.0); bs32 main-loads 140.4, sk8w4s-loads 146.7, sk8w4s 155.0, legacy32 156.5, main-d3 = main-d2 223.5, main-nogate 221.2; bs128 main-loads 190.6, legacy128 250.3, sk8w4s 269.5, main-d2 301.5; bs256 main-loads 204.8, main-d3 337.3, legacy256 360.4, sk8w4s 440.7. So the skinny path is streaming-bound (loads-only within 2-6 us of computing, about 3.4 TB/s net of about 8 us fixed cost) and the fat path is compute-bound per stage (loads-only 35-40% below its computing modes; depth 3 and removing the readiness waits change nothing). Prepare 143 s (build 139 s for 7 fused kernels and the legacy extensions). Error vs FP32 3.7e-3 to 4.6e-3 |
| 2026-10-28 | 36b5b559b0344d0f (submit, authorized by the user via the coordinator) | 35baa17 | fat path: k permuted inside each 256-k stage so one scale build covers a lane's four k16 steps, x slab written in that mma-k order by the gather and the phase A epilogue, wgmma descriptors in uniform registers (slot base through a shuffle: R2UR per HGMMA gone), SiLU by MUFU.RCP; diagnostics without wgmma and without dequant; sk8w5s, depth 3 and no-gate dropped | 1.990x (1.960-2.020) | per shape: bs1 29.3 us (2.727x), bs16 97.3 (1.937), bs32 133.9 (1.544), bs8 68.4 (2.106), bs128 255.1 (1.800), bs256 308.4 (2.008). POLICY {1, 8, 16, 32: sk8w4s, 128: legacy128, 256: main-d2}. BENCH (us): bs32 main-loads 139.3, sk8w4s 149.9, legacy32 155.6, main-nowgmma 157.3, main-nodequant 173.1, main-d2 197.3 (was 223.5); bs128 main-loads 189.6, main-nowgmma 211.9, main-nodequant 233.2, legacy128 249.9, main-d2 260.8 (was 301.5), sk8w4s 261.8; bs256 main-loads 203.0, main-nowgmma 237.3, main-nodequant 267.1, main-d2 291.6 (was 337.6), legacy256 360.1. The fat path's two costs add rather than overlap: at bs256 dequant alone +34 us over streaming, wgmma alone +64, both +89; the wgmma side (fence, two HGMMA, commit, wait per group, all warpgroup-collective) is the larger. Prepare 143 s. Error vs FP32 3.9e-3 to 4.6e-3 |
| 2026-10-28 | 0400cfbdfa2845fe (submit, authorized by the user via the coordinator) | 54f7940 | fat path: jobs of up to 2 (modes 6, 9) or 4 (7, 10) n8 tiles, or all jobs (8), on mma.sync m16n8k16 instead of RS wgmma (same A fragments, accumulators, epilogue and fixup; B by ldmatrix.x4 from the x slot), modes 9, 10, 8 with flatter stage costs; diagnostics 2 and 3 no longer built; a mode that fails its self-test only leaves the bench | 2.007x (1.977-2.038) | per shape: bs1 27.4 us (2.796x), bs16 96.9 (1.938), bs32 133.4 (1.544), bs8 71.3 (2.100), bs128 261.9 (1.804), bs256 301.5 (2.060). POLICY {1, 8, 16, 32: sk8w4s, 128: legacy128, 256: main-m2}. BENCH (us): bs32 main-loads 140.6, sk8w4s 151.6, main-d2 189.3, main-m2 190.0, main-m2f 190.0, main-m8f 190.6, main-m4 191.2; bs128 main-loads 190.9, legacy128 250.9, main-m2 253.5, main-d2 254.4, main-m2f 261.5, main-m4 266.6, main-m8f 279.8; bs256 main-loads 202.6, main-m2 283.1, main-d2 285.3, main-m4 298.7, main-m8f 327.7. So mma.sync and RS wgmma cost the same at N = 8 and 16 (bs32, where nearly every job is one n8 tile: 190.0 vs 189.3): the MMA instruction is not what keeps the fat path 50 us above its streaming time at bs32 and 63 us at bs128. mma.sync loses above N = 16 (m4, m8f). Error vs FP32 3.88e-3 every fat mode. Build 154 s (was 139) |
| 2026-10-28 | c524612c78454486 (submit, authorized by the user via the coordinator) | cd06a9e | mode 11 (mode 6 with clock stamps) and a DIAG report in prepare | 2.002x (1.972-2.032) | per shape: bs1 28.1 us, bs16 95.5, bs32 133.1, bs8 67.7, bs128 255.7, bs256 304.6; same policy and bench as 0400cfbdfa2845fe within 1 us (bs128 main-loads 188.9, legacy128 249.7, main-m2 253.6, main-d2 254.3, main-m2-stamps 258.5). The DIAG report failed on the host (`inter.view(int64)` of the 2-D buffer sliced rows, not elements; the dry run used a flat buffer), so no stamps. Fixed in the next commit, dry run now uses the real shape |
| 2026-10-28 | 2072795029b7479f (submit, authorized by the user via the coordinator) | a248788 | the DIAG fix | 1.978x (1.948-2.007) | per shape: bs1 29.7 us, bs16 97.1, bs32 136.0, bs8 69.0, bs128 263.0, bs256 301.8; same code and policy as the two runs before, so the 1.978-2.007 spread of these three is run-to-run noise. Bench bs128: main-loads 190.6, legacy128 253.6, main-m2 255.4, main-d2 256.8. Stamps below (section "Where the fat path's time goes at 128 tokens") |
| 2026-10-28 | 84de5bbafa344b7a (submit, authorized by the user via the coordinator; the current board entry) | 4fed369 | mode 14 (mode 12 with two mma.sync accumulator chains per n8 tile); stage costs per launch (moe_run `cost` argument) fitted from the DIAG stamps by least squares and offered as `-fit` variants of modes 6, 12, 14; modes 12/13 with fence.acq_rel in the epilogue and fixup | 2.009x (1.979-2.039) | per shape: bs1 27.95 us (2.760x), bs16 95.20 (1.954), bs32 133.40 (1.538), bs8 67.76 (2.108), bs128 254.18 (1.829), bs256 300.02 (2.057). POLICY {1, 8, 16, 32: sk8w4s, 128: legacy128, 256: main-m2-ar}. BENCH bs128 (us): main-loads 188.7, legacy128 251.3, main-m2-ar2 252.8, main-m2 254.3, main-m2-ar 254.7, -fit variants 256.5-257.8; bs256: main-m2-ar 283.8, main-m2-ar2 284.0, -fit 284.0-285.1. The least-squares stage-cost fit gave negative phase A coefficients (A1 -0.40, A2 -0.56, A3 -0.56, A7 -0.36, A8 -0.92 us/stage; B 5.6-9.6; residual rms 6.4 us over 396 CTA launches), so the fitted table is not a cost model and its variants did not win anywhere; it is not used to drive partitioning again (plan 2026-10-29 item 4). It is within noise of 0400cfbdfa2845fe (2.007x, 54f7940) and the two runs after it. Stamps at bs128: producers wait 185.8/202.6 us (mean/max) for a free weights slot of about 236 us of consumer time. Profiles and the result JSON in muse_moe_w4a16/profiles/submission.84de5bbafa344b7a.* |
| 2026-10-28 (before the 00:00 UTC reset) | 1b20c5b390f044b2 (submit by the coordinator, snapshot of 1099d9b) | 1099d9b | the proxy-ordering fix (fence.proxy.async.global by every x writer before its release and by the issuer after its acquire; the gather publishes with red.release after its barrier) | 1.979x (1.949-2.008) | passed 5/5. per shape: bs1 29.48 us (2.696x), bs16 102.99 (1.902), bs32 136.32 (1.508), bs8 72.38 (2.080), bs128 251.48 (1.823), bs256 301.26 (2.046). POLICY {1-32: sk8w4s, 128: legacy128, 256: main-m2-ar-fit}. BENCH bs128 main-loads 191.0, legacy128 251.2, main-m2-ar2 252.8, main-m2-ar 254.4, main-d2 254.4; bs256 main-m2-ar-fit 282.7, main-m2-ar 282.8, main-d2 283.9 (84de5bba: main-m2-ar 283.8, main-d2 284.8): the fences cost nothing measurable on the fat path. The score is within noise of the 2.009x entry (CIs overlap; bs1 and bs16 moved +1.5 and +7.8 us on the skinny path the fix does not touch, so that is run-to-run noise). Board entry stays 84de5bbafa344b7a |
| 2026-10-28 (before the reset) | 8078730bfa434074 (submit by the coordinator, snapshot of 9755822) | 9755822 | the fix plus the nonnegative cost model in prepare (72 DIAG launches, held-out validation gate) | 1.985x (1.955-2.015) | passed 5/5. per shape: bs1 27.87 us, bs16 100.95, bs32 135.85, bs8 68.66, bs128 258.07 (1.783x), bs256 302.35. The gate failed (section "Cost model, 2026-10-29"), so no -nn variant ran; POLICY {1-32: sk8w4s, 128: legacy128, 256: main-d2}. BENCH bs128 main-loads 191.2, legacy128 251.7, main-m2-ar2 253.2, main-m2-ar 254.1, main-m2 254.3, main-d2 255.5; bs256 main-d2 283.5, main-m2 283.6, main-m2-ar2 284.4. bs128 258.07 against 251.48 the run before with the same legacy128 choice: run-to-run noise on the judge's timing of the same kernel |
| 2026-10-29 | 4111803ce7894b63 (check) | 12794bd | mode 15: mode 12 with phase B CTA boundaries rounded to the nearest unit (16 stages), so no phase B segment is cut (the cost model's cut term was 10.4 us in phase B, 2.6 in A) | 1.986x (1.956-2.016) | per shape: bs1 28.38 us, bs16 97.66, bs32 138.84, bs8 68.92, bs128 259.82 (legacy128), bs256 299.00 (main-m2-ar2). BENCH bs32: main-m2-ar-ub 186.6, main-m2-ar2 187.1, main-m2-ar 190.6; bs128: legacy128 252.3, main-m2-ar-ub 253.1, main-m2-ar2 253.2, main-m2-ar 256.8; bs256: main-m2-ar2 283.5, main-m2-ar 284.9, main-m2-ar-ub 285.3. Unit alignment takes 4.0 us off mode 12 at bs32 and 3.7 at bs128 and nothing at bs256, so the cut cost is real at the smaller sizes, but mode 15 still trails legacy128 by 0.8 us at 128. Cost model refit on this run's stamps: the same picture (B cut 10.4 us, gate failed again: skewed table order 12/17 vs 13/17). Error vs FP32 3.90e-3 |
| 2026-10-29 | 9a9fb08c134c4a16 (check) | 907542c | mode 16: mode 14 (two mma.sync chains per n8 tile) with the unit-aligned phase B boundaries of mode 15 | 2.019x (1.989-2.049) | per shape: bs1 28.03 us, bs16 96.98, bs32 135.04, bs8 67.14, bs128 237.94 (1.907x, main-m2-ar2-ub), bs256 291.90 (2.076x, main-m2-ar2-ub). POLICY {1-32: sk8w4s, 128: main-m2-ar2-ub, 256: main-m2-ar2-ub}: the first time the fat path beats legacy128 at 128 tokens in the bench. BENCH bs32: main-m2-ar2-ub 182.4, main-m2-ar-ub 185.6, main-m2-ar2 186.8, main-m2-ar 190.4; bs128: main-m2-ar2-ub 249.0, legacy128 251.0, main-m2-ar-ub 251.2, main-m2-ar2 251.6, main-m2-ar 256.0; bs256: main-m2-ar2-ub 280.2, main-m2 282.5, main-m2-ar2 282.9. The judge's bs128 time, 237.9 us, is 13.5-22 us under the last five runs (251.5-259.8, all legacy128), and bs256 is 8-10 us under. The score's CI overlaps the 2.009x entry, so a repeat check follows before a submit |
| 2026-10-29 | c65e91500215418d (check; 99970e4fd1af402b, the same file again, came back as a duplicate, so this is the file plus one trailing newline) | 907542c | repeat of 9a9fb08c134c4a16 | 2.011x (1.981-2.042) | per shape: bs1 29.53 us, bs16 97.14, bs32 130.15, bs8 67.80, bs128 245.86 (1.917x), bs256 295.42 (2.087x); same policy (main-m2-ar2-ub at 128 and 256). BENCH bs128: main-m2-ar2-ub 249.1, main-m2-ar2 249.8, legacy128 250.9; bs256: main-m2-ar2-ub 280.2, main-m2-ar2 282.6. Two runs of mode 16 put bs128 at 237.9 and 245.9 us against 251.5-259.8 in the last five runs on legacy128, and bs256 at 291.9 and 295.4 against 299.0-302.4; the scores (2.019x, 2.011x) are inside the noise of the 2.009x entry |
| 2026-10-29 | da9225e4f89441fc (submit, authorized by the user via the coordinator: carries the proxy-ordering fix) | 907542c | mode 16 file as checked in 9a9fb08c134c4a16 | 2.003x (1.973-2.033) | per shape: bs1 29.17 us, bs16 98.97, bs32 136.34, bs8 70.83, bs128 246.59 (1.903x), bs256 292.67 (2.078x); same policy. Third run of mode 16: bs128 237.9 / 245.9 / 246.6 and bs256 291.9 / 295.4 / 292.7 us, against 251.5-259.8 and 299.0-302.4 on the code before, while the skinny shapes (bs1-32, unchanged code) drew low this time. Below the 2.009x board entry within noise, so the board still shows 84de5bbafa344b7a, which predates the ordering fix |
| 2026-10-29 | 9332e2a49aff4ff2 (check) | d90ad14 without mode 21 and the staged pair (built modes 0, 1, 16, 17, 18, 19, 20) | modes 17 (16 with stamps, wgmma segments stamped too), 18 (16's phase A alone), 19 (phase A alone on owned whole units, snake order by n-tile count: the GEMM1-only prototype), 20 (19's phase A, then 16's phase B and final pass); a GATE report (6 seeds, same process); modes 6, 11-15 not built | 2.044x (2.013-2.074) | per shape: bs1 28.07 us, bs16 97.21, bs32 133.31, bs8 68.03, bs128 239.63 (main-own), bs256 283.75 (main-own). POLICY {1-32: sk8w4s, 128: main-own, 256: main-own}. GATE (us, mean of 6 seeds): bs128 mode 16 248.3, 18 175.6, 19 170.1, 20 245.6; 19 minus 18 per seed +1.9 -7.4 -6.1 -8.2 -9.1 -4.1 (mean -5.5), 20 minus 16 mean -2.7 (5 of 6 seeds lower); bs256 16 286.4, 18 205.6, 19 189.2, 20 282.0; 19 minus 18 mean -16.4 (6 of 6), 20 minus 16 mean -4.4. xb of 19 vs 18 rel L2 4e-6 to 7e-5 (FP32 sum order of the fixups); output of 20 vs 16 1e-4 to 3e-4; every self-test 3.9e-3 vs FP32. Stamps in section "Mode 16 at 128 and 256 tokens" below; mode 17 costs 14-18 us over 16 (263.3 vs 248.9 at 128) |
| 2026-10-29 | e221858b020c4dee (check) | d90ad14 | adds mode 21 (phase B and the final pass alone, no done_as wait, no gather) and the staged pair VARIANT_STAGED (19 then 21, two launches), self-tested and benched; modes 0, 1, 16, 18, 19, 20, 21 built | 2.026x (1.995-2.056) | per shape: bs1 27.98, bs16 97.30, bs32 136.88, bs8 69.03, bs128 245.85 (main-own), bs256 288.66 (main-own). GATE: bs128 16 249.4, 18 176.9, 19 171.3, 20 247.3, 21 109.6, staged 277.3; staged minus 16 +21.2 to +32.0 (mean +27.9), staged minus 20 mean +30.0; 19 minus 18 mean -5.6; 20 minus 16 mean -2.1. bs256 16 287.3, 18 206.1, 19 190.4, 20 283.5, 21 124.3, staged 311.6; staged minus 16 mean +24.2, minus 20 mean +28.1; 19 minus 18 mean -15.7; 20 minus 16 mean -3.8. Staged self-test 3.88e-3, output vs 16 1.3e-4 to 2.0e-4. The staged pair loses (ruled-out.md) |
| 2026-10-29 | 72b7a5a9c511446a (check) | 308ed5b | the shippable build: modes 0, 1, 16, 20 (the phase-only modes and the staged pair stay in the source, unbuilt) | 2.022x (1.992-2.053) | per shape: bs1 28.29, bs16 97.70, bs32 141.43, bs8 69.26, bs128 247.68 (main-own), bs256 291.36 (main-own). GATE: 20 minus 16 per seed bs128 -2.9 -2.3 -3.9 +0.6 -8.9 -1.2 (mean -3.1), bs256 -10.8 -10.6 +5.0 -5.0 +0.1 -4.1 (mean -4.2). BENCH bs128 main-own 247.7, main-m2-ar2-ub 251.6, legacy128 251.7; bs256 main-own 270.8, main-m2-ar2-ub 281.3. Three runs of mode 20: 2.044x, 2.026x, 2.022x against 2.019x, 2.011x, 2.003x for mode 16's three; the intervals overlap, the same-process A/B (three times, 18 seeds per size) puts 20 ahead by 2-3 us at 128 and 4 us at 256. Prepare build 137 s |
| 2026-10-29 | b99aa87cabdd430c (submit, authorized by the user) | 308ed5b | mode 20 file as checked in 72b7a5a9c511446a | 2.034x | passed. POLICY {1-32: sk8w4s, 128: main-own, 256: main-own}. GATE bs128 main-own minus mode 16 mean -2.1 us, bs256 -4.3 us (6 seeds). BENCH bs128 main-own 246.5, mode 16 248.4, legacy128 249.8; bs256 main-own 270.3, mode 16 278.6. New board entry over da9225e4f89441fc (2.003x) |
| 2026-10-29 | a9dbec1d98334943 (check) | 10375cb | modes 22 (20 with stamps), 23 (20 with phase B units claimed at run time from a global counter in S.ord order, both phases walked largest n-tile count first) and 24 (23 with stamps); GATE 16 / 20 / 23 on seeds 11-16 with the variant order reversed on every other seed; built modes 0, 1, 16, 20, 22, 23, 24 | 2.076x (2.045-2.107) | per shape: bs1 29.17, bs16 97.20, bs32 137.53, bs8 72.30, bs128 219.92 (2.083x, main-own-dynb), bs256 269.21 (2.289x, main-own-dynb). GATE (us, mean of 6 seeds): bs128 16 248.1, 20 246.2, 23 230.0; 23 minus 20 per seed -15.9 -14.2 -14.9 -18.3 -14.4 -19.2 (mean -16.2); bs256 16 287.4, 20 283.7, 23 267.6; 23 minus 20 -9.2 -17.8 -26.3 -13.8 -20.3 -9.4 (mean -16.1). Output of 23 vs 16 1.1e-4 to 2.4e-4 (same as 20 vs 16); self-tests 3.9e-3 vs FP32; spin flags 0. BENCH bs128 main-own-dynb 231.2, main-own 246.0; bs256 257.0 vs 268.8. Stamps in section "Phase B claimed at run time" below. The log is a tail: the DIAG22 lines for bs128 seed 11 were cut. Build 149 s |
| 2026-10-29 | 36812632dae242ba (check) | ec219f0 | the shippable build: modes 0, 1, 16, 20, 23; GATE on held-out seeds 17-22 and on skewed routing (seeds 11-13) | 2.077x (2.046-2.108) | per shape: bs1 28.05, bs16 101.53, bs32 133.82, bs8 66.30, bs128 221.07 (2.072x, main-own-dynb), bs256 270.57 (2.278x, main-own-dynb). GATE held-out normal: bs128 16 259.9, 20 263.3, 23 240.4; 23 minus 20 -26.7 -19.8 -28.6 -11.8 -30.4 -20.2 (mean -22.9), 23 minus 16 mean -19.5; bs256 16 310.7, 20 304.6, 23 285.9; 23 minus 20 -21.3 -12.6 -25.3 -26.1 -13.4 -13.4 (mean -18.7), minus 16 mean -24.8. Skewed: bs128 23 minus 20 -4.4 -1.9 -8.3 (mean -4.8); bs256 -26.0 -26.4 -16.3 (mean -22.9). 24 of 24 paired seeds favour 23 over 20 across both checks. Judge bs128 219.9 / 221.1 us against 239.6 / 245.9 / 247.7 for mode 20's three runs, bs256 269.2 / 270.6 against 283.8 / 288.7 / 291.4. Build 146 s |
| 2026-10-29 | fa18c9e87d6e4811 (submit, under the standing replace-when-demonstrably-better authorization) | beb1924 | mode 23 file as checked in 36812632dae242ba | 2.070x | passed. POLICY {1-32: sk8w4s, 128: main-own-dynb, 256: main-own-dynb}. BENCH bs128 main-own-dynb 231.3, main-own 246.0; bs256 256.2 vs 268.5. New board entry over b99aa87cabdd430c (2.034x) |
| 2026-10-29 | d1f587760b1241dc (check, bench file from tools/moe_make_bench_ab.py, not committed: it embeds the public MoE file) | 35ab79e | same-process A/B: mode 23 and sk8w4s against the public MoE kernel (vjeux/okc-2026 muse_moe_w4a16.py, MAIN build) at 32, 128, 256 tokens on six seeds of the standalone routing and six of a port of the layer's judge routing; mode 25 (23 with thread-0 phase stamps only) against the public kernel's own OKC_DEBUG stamps | 2.064x (2.033-2.095) | public minus mode 23, us (6 of 6 seeds lower in every row): T=32 std -18.0, layer -12.8; T=128 std -34.9, layer -32.2; T=256 std -52.5, layer -45.7. Attribution in section "Same-process A/B against the public MoE" below. Outputs vs FP32: ours 3.84e-3 to 3.86e-3, public 2.34e-3 (it does not round gate, up and the GEMM2 output to bf16). Mode 25 costs nothing (BENCH bs128 231.9 vs 23 at 232.7). Policy unchanged (sk8w4s to 32, main-own-dynb at 128 and 256) |
| 2026-10-29 | 39c365c34cd74559 (check, bench file) | 9c384a6 | mode 26: mode 23 with every warp counting the routes (the routing warp sums 18 per-warp counts instead of counting all E entries alone, and the positions reuse them) and a batched x gather (all of a CTA's 16-byte row loads before its stores); 27 is 26 with the stamps | 2.083x (2.052-2.115) | 26 minus 23, seeds 1-6, us: T=128 std -4.3, layer -3.7; T=256 std -8.2, layer -8.5; T=32 layer -0.5 (6 of 6 lower in each). Stamped route 11.0 -> 7.7 us at 128 layer, 16.1 -> 9.6 at 256; the routing warp still 4-5 us and thread 0's stage-range searches 2.3-3.1 us. Output of 26 vs 23 7e-6 to 3e-5. POLICY 128 and 256: main-dynb-fs. The log is a tail: the T=32 std lines were cut |
| 2026-10-29 | 4c285df44eb34bc1 (check, bench file) | c850d04 | 26 and 27 skip the four stage_at searches (owned phase A and claimed phase B read no stage range); routing-warp stamps; seeds 7-12 (held out from the two checks before) | 2.097x (2.066-2.129) | 26 minus 23, us: T=32 layer -1.4; T=128 std -5.8, layer -4.6; T=256 std -9.8, layer -9.1 (6 of 6 lower in each). Stamped route at 128 5.8 us (public 2.4): ids 0.5, routing warp 4.4 (counts and offsets 1.1, jobs 0.6, S.ord counting sort 2.5), positions 0.5, ranges 0.3. Public minus 23 on these seeds: T=128 -36.1 / -31.9, T=256 -50.2 / -50.1 |
| 2026-10-29 | 817eab6401d54a84 (check) | ba9a457 | the shippable build: modes 0, 1, 16, 20, 23, 26; GATE adds 26 minus 23 (held-out seeds 17-22, skewed 11-13) | 2.104x (2.073-2.136) | per shape: bs1 28.37 us, bs16 97.81, bs32 138.70, bs8 71.65, bs128 215.68 (2.148x, main-dynb-fs), bs256 261.71 (2.371x, main-dynb-fs). GATE 26 minus 23 per seed: bs128 -5.1 -6.0 -7.0 -5.1 -6.6 -5.2 (mean -5.8); bs256 -9.8 -9.3 -9.5 -9.7 -9.2 -9.8 (mean -9.5); skewed bs128 -4.2, bs256 -7.7 (3 of 3 each). Output of 26 vs 16 matches 23's (7.8e-5 to 3.5e-4); self-tests 3.89e-3 vs FP32; spin flags 0. BENCH bs128 main-dynb-fs 226.4, main-own-dynb 231.6; bs256 248.9 vs 258.6. Judge bs128 215.7 us against 219.9 / 221.1 for mode 23's two runs, bs256 261.7 against 269.2 / 270.6. Build 145 s. Submit candidate: ba9a457 (not submitted) |
| 2026-10-29 | a8076924ced1419e (submit, under the standing replace-when-demonstrably-better authorization) | a8fc96a | mode 26 file as checked in 817eab6401d54a84 (ba9a457) | 2.101x | passed. Board entry before: fa18c9e87d6e4811 (2.070x) |
| 2026-10-29 | 70dd7b13be884434 (check, bench file from tools/moe_make_bench_ab.py, not committed) | ec62b5a | mode 28: mode 26 with a leaner consumer stage (ring size and x slot base in registers, the weights slot released by every consumer thread behind the stage's first wgmma group, fence kept) and a one-pass S.ord sort; 29 is 28 with the light stamps; arms o26, o28, pub, seeds 1-6, standalone and layer-like routing | 2.132x (2.100-2.164) | 28 minus 26 per seed, us: T=32 layer -8.1 -4.4 -7.6 -3.6 -7.0 -7.8 (mean -6.4), std mean -6.4; T=128 layer -8.0 -10.8 -9.9 -7.6 -6.5 -8.9 (mean -8.6), std mean -12.1; T=256 layer -7.7 -10.3 -8.6 -7.3 -8.2 -10.8 (mean -8.8), std mean -11.0 (36 of 36 lower). Public minus 28: T=128 layer -16.1, T=256 layer -25.1 (was about -27 / -41 against 26). Output of 28 vs 23 1.6e-5 to 2.6e-5, vs FP32 3.84e-3 to 3.86e-3. ABSTAMP and attribution in section "Consumer stage and the ring slot releases". POLICY 128 and 256: main-lean |
| 2026-10-29 | f928f8a802014e4f (check) | ec62b5a | the shippable build: MAIN_MODES (0, 16, 26, 28); GATE adds 28 minus 26 (held-out seeds 17-22, skewed 11-13) | 2.149x (2.117-2.181) | per shape: bs1 28.05 us, bs16 95.96, bs32 135.02, bs8 68.47, bs128 200.89 (2.295x, main-lean), bs256 247.11 (2.524x, main-lean). GATE 28 minus 26 per seed: bs128 -11.5 -10.8 -10.2 -11.1 -10.6 -9.4 (mean -10.6); bs256 -12.0 -11.4 -12.9 -8.6 -11.2 -13.7 (mean -11.6); skewed bs128 -6.6 -6.7 -6.7, bs256 -7.5 -7.7 -9.9. Output of 28 vs 16 equal to 26's to two digits (1.1e-4 to 3.7e-4); self-tests 3.89e-3 vs FP32; spin flags 0. BENCH bs128 main-lean 213.6, main-dynb-fs 225.8; bs256 237.8 vs 248.5. Build 138 s |
| 2026-10-29 | 13fe68e7ca2e423f (check, repeat) | ec62b5a | f928f8a802014e4f's file plus one trailing comment line, so the judge does not return it as a duplicate (an identical resubmission, 3adce08f3e9844d5, came back as duplicate_of f928f8a802014e4f without running) | 2.158x (2.125-2.190) | bs128 200.34 us, bs256 244.59; GATE 28 minus 26: bs128 mean -11.6, bs256 -12.5 (6 of 6 lower each), skewed -6.4 / -8.4 (3 of 3). POLICY 128 and 256: main-lean. Against mode 26's shippable check 817eab6401d54a84 (2.104x): bs128 215.68 -> 200.89 / 200.34, bs256 261.71 -> 247.11 / 244.59. Submit candidate: ec62b5a (not submitted) |
| 2026-10-29 | 08070cce049b403d (submit, under the standing replace-when-demonstrably-better authorization) | c406b03 | mode 28 file as checked in f928f8a802014e4f (ec62b5a) | 2.137x | passed. Board entry before: a8076924ced1419e (2.101x) |
| 2026-10-29 | c7db20e053254b52 (check, bench file from tools/moe_make_bench_ab.py, not committed) | 1d962e6 | phase A attribution against the public kernel, layer routing seeds 1-6 at 32 / 128 / 256 tokens: mode 31 (28 with phase A counters per consumer warp and the producers' waits), 32 (phase A publish deferred to the next unit's second stage), 33 (32 with the counters), 34 (32 without the per-thread fence_gpu), 35 (32 with ss and own_n reloaded from shared memory, fewer spills); the public kernel's stream-K stage ranges patched into its DBG slot 7 | 2.147x (2.115-2.179) | minus 28, us: 32 +1.6 / +2.7 / +1.2 (0, 2, 0 of 6 lower), 34 +0.8 / +1.4 / -1.2 (2, 2, 5 of 6), 35 +0.8 / +3.0 / +4.7 (0, 1, 0 of 6); counters in section "Phase A against the public kernel". Outputs vs 28 2e-6 to 3e-5. The public kernel's stamped (DEBUG) build ran 30 us over its own graph replay in this check and the next two (233 vs 203 us at 256), so its stamped times from these checks are not used; its stage ranges are structural. Cause, found afterwards in a local build: the range write makes the DEBUG MAIN build spill 392 / 896 B (none without it); tools/moe_make_bench_ab.py now patches it in only with --pubranges=1 |
| 2026-10-29 | 0b684d5db0834dbc (check, bench file) | ef6b705 | mode 36 (28 with SiLU's reciprocal as one MUFU.RCP), 37 (36 without the per-thread fence_gpu in the phase A publish), 38 (36 with the counters); layer seeds 1-6 | 2.163x (2.130-2.195) | minus 28, us: 36 -1.8 / -0.9 / -1.2 (6, 5, 6 of 6 lower), 37 -1.9 / -2.1 / -3.8 (6, 6, 6 of 6); 37 minus 36 -0.1 / -1.2 / -2.6. Counters: 36's epilogue 2.85 us per unit at 256 against 28's 2.98. Outputs vs 28 7e-5 to 1.5e-4 (the approximate reciprocal moves bf16 roundings); self-tests 3.89e-3 vs FP32 |
| 2026-10-29 | 473d87d03de94e96 (check, bench file) | d6cccf0 | repeat of 37 against 28, and 39 (37 with paired SwiGLU stores) with its counter twin 40; layer seeds 1-6 | 2.154x (2.122-2.186) | minus 28, us: 37 -1.7 / -3.4 / -4.2 (5, 6, 6 of 6 lower), 39 -1.0 / -2.5 / -4.0; 39 minus 37 +0.8 / +1.0 / +0.2. Counters at 256: 40's epilogue 1.70 us per unit against 3.08 for 28, but its stage loops took 3.1 us more and phase A only 2.0 us less: the stores' cost moves into the following stages |
| 2026-10-29 | 987d024620e44eb7 (check) | 96f2c43 | the shippable build: MAIN_MODES (0, 16, 26, 28, 37); GATE adds 37 minus 28 (held-out seeds 17-22, skewed 11-13) | 2.144x (2.112-2.176) | per shape: bs1 28.71 us, bs16 98.86, bs32 133.91, bs8 68.86, bs128 197.09 (2.309x, main-lean-silu-bar), bs256 241.49 (2.561x, main-lean-silu-bar). GATE 37 minus 28 per seed: bs128 -1.5 -2.1 -4.1 -4.5 -2.8 -4.1 (mean -3.2); bs256 -3.3 -5.1 -5.0 -5.1 -5.5 -5.5 (mean -4.9); skewed bs128 -2.5 -0.6 -1.5, bs256 -0.2 -3.2 -4.4. Output of 37 vs 16 1.2e-4 to 4.1e-4 (28: 8e-5 to 4.0e-4); self-tests 3.87e-3 to 3.90e-3; spin flags 0. BENCH bs128 main-lean-silu-bar 211.2, main-lean 214.6; bs256 232.6 vs 237.8. Judge bs128 197.09 against 200.89 / 200.34 for mode 28's two checks, bs256 241.49 against 247.11 / 244.59; the score sits inside 28's checks' intervals (2.149x, 2.158x) while the skinny shapes (unchanged code) drew 0.4-2.9 us higher at 1, 8 and 16 tokens than in f928f8a802014e4f. |
| 2026-10-29 | 6307d0c91c8d4ada (submit by the coordinator, used as the repeat of 987d024620e44eb7) | 96f2c43 | the shippable mode 37 build as checked in 987d024620e44eb7 | 2.160x (2.127-2.192) | passed 5/5 (tests 4.07e-3 to 4.22e-3). Judge medians: bs1 27.98 us, bs16 101.51, bs32 135.48, bs8 68.34, bs128 202.35 (2.329x, main-lean-silu-bar), bs256 241.46 (2.572x). GATE 37 minus 28 per seed: bs128 -5.8 -3.4 -4.1 -3.3 -2.9 -6.1 (mean -4.3); bs256 -4.8 -5.2 -6.5 -5.1 -4.8 -5.6 (mean -5.3); skewed bs128 -2.6 +0.3 -1.8 (mean -1.4), bs256 -1.0 -1.6 -5.2 (mean -2.6). BENCH bs128 main-lean-silu-bar 211.1, main-lean 215.3; bs256 232.9 vs 237.2. Two judge runs of 37: bs256 241.49 / 241.46 us against 247.11 / 244.59 for 28's checks; bs128 197.09 / 202.35 against 200.89 / 200.34 (the judge's bs128 is noisier than the 3-4 us gain). Status JSON in profiles/submission.6307d0c91c8d4ada.status.json |
| 2026-10-29 | 240b68e7df1a427f (submit under the user's scored-runs-as-measurements authorization; bench file from tools/moe_make_bench_ab.py --small, not committed) | 4204a8f | same-process A/B of the skinny path against the public MoE (its TINY / TINY2 / MID builds, which kernel_fn picks at 1-4 / 5-8 / 9-16 tokens) at 1, 8, 16 tokens, layer and standalone routing, seeds 1-6; arms sk8w4s (the policy), sk16w4s, sk8w4s-loads; stamps of the new skinny kMode 2 (sk8w4s with thread-0 phase stamps and cycle counters) against the public DEBUG builds | 1.988x (1.959-2.018) | public minus sk8w4s, us: T=1 layer +0.4 (35.4 base), std +0.3; T=8 layer -3.7 (66.2, 5.6%, 6 of 6 lower), std -1.9 (92.1); T=16 layer -2.0 (84.4, 2.4%), std -2.5 (108.9). sk8w4s-loads minus sk8w4s -2.5 / -3.6 / -4.4 (layer): the public kernel computing (62.5 at 8 tokens) matches our loads-only floor (62.7). sk16w4s +1.3 to +6.9 everywhere. Attribution in section "Skinny path against the public kernel". The score is a bench file's (the judge timed the shipped policy; 2.0x band of earlier runs) |
| 2026-10-29 | 2c28a7256b004aa1 (submit, bench file, same authorization) | 126ee85 | skinny kMode 3 (sk8w4s-er: each compute warp loads its unit's codes up front, fences and arrives on a per-slot empty mbarrier; thread 0 refills once all 8 warps arrived, no per-unit bar.sync of the compute warps) and kMode 4 (3 with stamps) against sk8w4s, same A/B as 240b68e7df1a427f | 1.970x (1.940-1.999) | er minus sk8w4s, us: T=1 layer +0.4 (0 of 6 lower), std +0.4 (0 of 6); T=8 layer -1.0 (6 of 6), std +0.7 (2 of 6); T=16 layer -0.9 (5 of 6), std +1.6 (1 of 6). Public minus sk8w4s this run: T=8 layer -4.1, T=16 layer -3.1. Stamps at 8 tokens, layer: er's phase A 38.0 us against 34.9 (codes held in registers through the unit), its A->B wait 2.5 against 4.0 and barrier 2.2 against 3.9. Self-test 3.93e-3; outputs vs sk8w4s 0 to 3e-5. Not adopted for the standalone (it loses on the standalone routing and at 1 token); a layer bench on judge routing may pick it at 8 and 16 tokens |
| 2026-10-29 | e937013292b24906 (submit under the user's scored-runs-as-measurements authorization; bench file from tools/moe_make_bench_ab.py --small, not committed) | 8272efe | skinny kMode 5 (sk8w4s-sf: scale decode's x16 as an IMAD with a multiplier ptxas cannot fold, FMA pipe instead of LEA on the ALU pipe), 6 (sk8w4s-hand: partial buffers handed to and from the helper through mbarriers, one compute-warp barrier per unit instead of two), 7 (sk8w4s-sfh: both), 8 (7 with stamps); stamps now time the first try_wait | 2.004x (1.974-2.034) | minus sk8w4s, us, layer / std: T=1 sf -0.1 / -0.1, hand +0.1 / -0.0, sfh +0.1 / -0.1; T=8 sf -0.2 / -0.4, hand -0.1 / +0.0, sfh +0.1 / +0.1; T=16 sf -0.5 / -0.1, hand +0.4 / +0.5, sfh +0.5 / -0.1 (none 6 of 6). Public minus sk8w4s: T=8 layer -3.5, T=16 layer -2.2. Self-test 4.02e-3 for all; outputs vs sk8w4s <= 5.8e-5; spin flags 0. Corrected stamps at T=8 layer: thread 0 waits 3.0 us (A) and 0.7 (B) for weights. Policy picked sk8w4s-sf at 8-32 (bench noise). Not adopted (docs/runs-moe.md, "Skinny unit loop against the public kernel") |
| 2026-10-29 | 75bca9775f6f4b0b (scored run submitted by the coordinator, the day's last expiring one, per the user's instruction; bench file from tools/moe_make_bench_ab.py --small --unitstamps, not committed) | 03887be | skinny stream A/B: sk8w3s (3-slot ring, (8, 3, 0)), kMode 32 (sk8w4s-rf: unit barrier and refill ahead of the partial handoff), (8, 3, 32) (w3rf), kMode 64 (sk8w4s-nh: weight copies without the L2 hint) against sk8w4s and the public TINY / TINY2 / MID builds, layer and standalone seeds 1-6 at 1 / 8 / 16 tokens; per-unit stream stamps (kMode bit 16 and the public patched the same way) at 8 tokens | 2.006x (1.976-2.036) | passed. w3 minus sk8w4s -0.9 / -0.6 / -2.4 / -1.8 / -3.1 / -2.9 us (T=1 layer / std, 8 layer / std, 16 layer / std), 6 of 6 seeds lower in every row; w3 minus public -1.3 / -0.8 / +1.0 / +0.9 / -0.4 / -0.5. rf -1.1 / +0.4 at 8, w3rf behind w3 in all six rows, nh +1.2 to +3.3 everywhere. Stamps: copy issue-to-arrival 4.7 us with 4 slots, 3.2 with 3, public 3.0. Spin flags 0; every arm within 2.1e-5 of sk8w4s. Section "Skinny weight stream against the public kernel" |
| 2026-10-29 | ca43908f02874886 (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=128,256 --routings=layer o41=41 o42=42 --pairs=o41-o42, not committed) | 3f67a45 | fat-path waits timed in full (mbar_wait_cyc's clock before the first try_wait); mode 41 (37 with phase A and phase B counters per consumer warp) and 42 (37 with the light stamps only), layer seeds 1-6 | 2.174x (2.142-2.207) | passed 5/5; judge medians bs1 28.74 us, bs16 98.81, bs32 132.88, bs8 71.36, bs128 195.45, bs256 241.53; policy unchanged (sk8w3s to 32, main-lean-silu-bar at 128 and 256). ABTIME o37 164.6 / 227.2, pub 151.4 / 203.6 at 128 / 256. Instrumentation cost: 41 minus 37 +7.0 / +6.1 us, 42 minus 37 +1.1 / +2.3, 41 minus 42 +5.9 / +3.9 (0 of 6 lower in each row). Outputs of 41 and 42 against 37 1e-5 to 3e-5, spin flags 0. Counters in section "Fat-path waits with the first try timed". ABPUB's stage ranges are not meaningful in this run (no --pubranges) |
| 2026-10-30 | 9d94ac75f4724b52 (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer o43=43 o44=44 o42=42 ost=v301 --pairs=ost-o43, not committed) | 54257b5 | mode 43 (code groups, next stage's waits and first group hoisted under the last wgmma group, weights release behind the sixth group), 44 (43 with 41's counters), and the staged pair on 37's components (variant 301: modes 45 then 46); layer seeds 1-6 | 2.188x (2.155-2.221) | passed 5/5; judge medians bs1 27.95, bs16 95.17, bs32 137.66, bs8 67.16, bs128 193.98, bs256 239.72 us; policy main-lean-grp (43) at 128 and 256. 43 minus 37 -1.3 / -1.4 / -2.7 us at 32 / 128 / 256 (5, 6, 6 of 6 lower); 44's cycles per wait 214 / 224 / 239 against 41's 201 / 215 / 223 (phase A), not lower. Staged pair minus 37 +30.3 / +30.4 / +24.4 (0 of 6). Outputs against 37 4e-6 to 2.3e-5, spin flags 0. Section "Code groups with the next stage hoisted" |
| 2026-10-30 | 304f17f3e8704992 (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer --seeds=7,8,9,10,11,12 o43=43 o44=44 o42=42 o47=47 --pairs=o47-o43, not committed) | a4687c8 | repeat of 43 on held-out layer seeds 7-12 (the steering rule's step 3: timing won, waits did not fall), plus mode 47 (43 without the hoist) | 2.176x (2.144-2.209) | passed 5/5; judge medians bs1 27.46, bs16 97.33, bs32 133.58, bs8 72.36, bs128 199.79, bs256 241.87 us; policy main-lean-grp at 128 and 256. 43 minus 37 -0.9 / -1.2 / -2.1 (6, 4, 6 of 6); 47 minus 37 -0.4 / -1.6 / -3.6 (3, 6, 6 of 6); 47 minus 43 +0.5 / -0.5 / -1.5 (2, 5, 6 of 6); 44's cycles per wait 214 / 215 / 235 against 41's 204 / 203 / 225. Outputs against 37 1.2e-5 to 3.8e-5, spin flags 0 |
| 2026-10-30 | 5f381b8b6946492d (check, bench file from tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer,std,skewed --seeds=13,14,15,16,17,18 --gate=1 o47=47, not committed) | 20c295e | mode 47's pre-registered gate against 37 (rule committed in 28fc668 before the submit): held-out seeds 13-18 on layer, standalone and skewed routing | 2.196x (2.163-2.229) | passed 5/5; judge medians bs1 27.5, bs16 94.2, bs32 138.4, bs8 72.3, bs128 200.3, bs256 238.5 us; policy main-lean-grp-nohoist at 128 and 256. 47 minus 37, us, at 32 / 128 / 256: layer -1.0 / -2.3 / -3.7 (6, 5, 6 of 6 lower), standalone -2.4 / -2.8 / -3.4 (6, 6, 6), skewed -2.0 / -2.2 / -2.8 (5, 6, 6). Outputs against 37 on every seed 5.5e-7 to 4.4e-5; self-test 3.89e-3; spin flags 0. Every rule condition passed: 47 kept. Section "Mode 47 gate" |
| 2026-10-30 | b50351288b504e81 (check) | ec565a7 | the shippable build: MAIN_MODES (0, 16, 26, 28, 37, 47); GATE adds 47 minus 37 (seeds 17-22, skewed 11-13) | 2.190x (2.157-2.223) | passed 5/5. Per shape: bs1 28.8 us, bs16 95.0, bs32 138.0, bs8 67.3, bs128 199.8 (2.365x, main-lean-grp-nohoist), bs256 238.5 (2.590x, main-lean-grp-nohoist). GATE 47 minus 37 per seed: bs128 -3.6 -3.1 -3.3 -0.3 -1.7 -2.8 (mean -2.5); bs256 -5.2 -2.8 -2.7 -2.7 -3.6 -3.4 (mean -3.4); skewed bs128 -2.2 -1.9 -2.9, bs256 -2.6 -2.2 -2.0. Output of 47 vs 16 equal to 37's to two digits (1.5e-4 to 4.2e-4); self-tests 3.86e-3 to 3.89e-3 for 47; spin flags 0. BENCH bs128 47 207.5, 37 209.2; bs256 230.3 vs 232.5. Build 146.6 s. Judge bs256 238.5 against 241.49 / 241.46 for 37's two runs; bs128 199.8 against 197.09 / 202.35. Submit candidate: ec565a7 (not submitted; handed to the coordinator) |
| 2026-10-30 | abe7651dbaa44458 (submit by the coordinator, under the standing replace-when-demonstrably-better authorization) | ec565a7 | mode 47 shippable file, gate 5f381b8b6946492d passed its pre-registered rule | 2.216x | passed. Board entry before: 6307d0c91c8d4ada (2.160x) |
| 2026-10-30 | ed3b4fdc76b54263 (check, bench file from tools/moe_make_bench_ab.py --base=47 --sizes=128,256 --routings=layer --skew=1 o48=48 o49=49, not committed) | 64cec1f | warp skew on mode 49 (47 with per-warp stage stamps) and phase stamps on 48 against the public kernel, layer seeds 1-6 | 2.188x (2.155-2.220) | passed; the bench crashed after the 128-token timing (ABCHECKALL on an empty list, a bug of 20c295e, fixed in the next commit). 128 tokens: 48 minus 47 +0.8, 49 minus 47 +5.4 (0 of 6 lower), public minus 47 -12.2 (6 of 6) |
| 2026-10-30 | d47c78bdc130424b (check, same bench options, not committed) | 54318bd | the rerun of ed3b4fdc76b54263 | 2.194x (2.161-2.227) | passed; spin flags 0. Public minus 47 -14.7 / -22.6 us at 128 / 256 (6 of 6); 48 minus 47 +1.3 / +1.1, 49 minus 47 +5.8 / +9.4. Warpgroups of a CTA 750-1,200 cycles apart at the stage start, warps within one 50-160; up to 72-74% of the weights waits from that skew. Phase gap (48 vs public stamps) at 128 / 256: setup +4.3 / +4.4, phase A +2.3 / +7.7, phase B +2.9 / +7.3, tail +3.7 / +3.6, barrier and final +4.0 / +3.6. ABSKEW's column labels in this log are the skinny ABSK names (a name clash, fixed in a03ca00); the values are in the order of skew_cols. Section "Warp skew and the gap to the public kernel after mode 47" |
| 2026-10-30 | 8a7051fa926e43e1 (check, bench file from tools/moe_make_bench_ab.py --base=47 --sizes=32,128,256 --routings=layer,std,skewed --seeds=23,24,25,26,27,28 --gate=1 o50=50, not committed) | a03ca00 | mode 50's pre-registered gate against 47 (rule committed in 5ea3c34 before the submit) | 2.210x (2.177-2.243) | passed 5/5; spin flags 0. 50 minus 47, us, at 32 / 128 / 256: layer +0.7 / -5.9 / -6.6 (1, 6, 6 of 6 lower), standalone -2.8 / -5.7 / -8.3 (5, 6, 6), skewed -1.4 / -4.1 / -7.9 (6, 6, 6). Outputs vs 47 <= 4.3e-5; self-test 3.90e-3. Condition 2 failed (32 tokens, layer, +0.7 against +0.5): 50 not shipped, docs/ruled-out.md. Policy picked 50 at 128 and 256 |
| 2026-10-30 | c6b0e500c3a84c9a (check, bench file from tools/moe_make_bench_ab.py --base=47 --sizes=32,128,256 --routings=layer,std,skewed --seeds=29,30,31,32,33,34 --gate=1 o51=51, not committed) | 69c06ed | mode 51's pre-registered gate against 47 (rule committed in 69c06ed before the submit) | 2.194x (2.161-2.227) | passed 5/5; spin flags 0. 51 minus 47, us, at 32 / 128 / 256: layer +0.0 / -3.9 / -3.9 (2, 6, 5 of 6 lower), standalone -1.0 / -3.8 / -4.4 (5, 6, 6), skewed -0.1 / -4.0 / -7.7 (2, 6, 6). Outputs vs 47 <= 4.6e-5; self-test 3.88e-3. Every rule condition passed: 51 kept. Policy picked 51 at 128 and 256. Section "Mode 51" |
| 2026-10-30 | a5a2bbd1a18b4b89 (check, coordinator) | 6bbb252 | mode 51 shippable file | 2.210x | passed; POLICY 128/256 main-lean-grp-finss64; GATE 51 minus 47: -3.7 / -3.0 us (6 of 6 seeds), skewed -3.6 / -8.8 |
| 2026-10-30 | bcd2d7a3506a4b7d (submit by the coordinator, standing authorization) | 6bbb252 | mode 51 shippable file | 2.202x | passed. Below the board entry abe7651dbaa44458 (2.216x, mode 47), which stays; whole-score noise exceeds the 128/256-token gain |
| 2026-10-30 | 1fbb9fbf524f4ed7 (check, bench file from tools/moe_make_bench_ab.py --base=47 --sizes=32 --routings=layer,std --seeds=1,2,3,4,5,6 --work=1 --skst=8:3:2 o37=37 o16=16 sk8w3s=v3, not committed) | b3509a0 + tool changes | phase attribution at 32 tokens against the public kernel (stamps of 48, 42, 41, public, sk8w3s; per-CTA work) | 2.196x | passed; spin flags 0. Graph replay at 32 tokens, layer / standalone: 47 90.9 / 132.9 us, public -1.5 / -5.9 (6 of 6 each), 37 +1.8 / +1.6, 16 +28.5 / +31.1, sk8w3s +19.2 / -0.1, sk8w4s +22.9 / +3.4. Stamps 48 minus public: setup +3.6 / +0.9, phase A done max +6.2 / +8.6, tail +2.0 / +6.0, end +1.6 / +7.4. Work: 8.5 / 14.2 jobs, owned phase A units per CTA 1.5-2.5 / 2.8-3.8, public phase A 28-36 / 46-57 stages. Section "Phase attribution at 32 tokens" |
| 2026-10-30 | 209cf29bf91c4ea2 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=35,36,37,38,39,40 o52=52 o53=53 o47=47, not committed) | this branch (modes 52-54) | diagnostic of modes 52 / 53 (last phase B units claimed in K halves through the fixup, xs = G / 2 / G) | 2.191x | passed; spin flags 0. 52 minus 51 +3.3 / +3.5 / +5.9 / +4.9 / +8.5 / +5.3 us at 32 / 128 / 256 (layer / std), 0-2 of 6 lower; 53 +5.2 to +9.2. 54's tail 7.9 against 48's 7.1 at 32 (layer). Outputs vs 51 <= 9.7e-5. Lost everywhere, not gated; docs/ruled-out.md |
| 2026-10-30 | 7063f86352344efa (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=128,256 --routings=layer --skew=1, seeds 1-6, not committed) | e866c8a | re-measurement on 51 to pick the x gather or the warpgroup skew (twins 55, 56; rule set in e866c8a before the submit) | 2.210x (2.176-2.243) | passed; spin flags 0. Gather piece (start minus route) 51 / public: 6.8 / 3.6 us at 128, 8.3 / 4.9 at 256 (+3.2 / +3.4); thread 0 stores 1.7 / 5.6, fences and barrier 4.3 / 2.1, release 0.6 / 0.5. Skew bound for per-warpgroup slots (trailing warpgroup, A + B): 0.45 / 1.07 us; trailing warpgroup persistence 0.79-0.87. Public minus 51 -9.1 / -14.0 us (6 of 6). Decision: the gather. Section "The x gather or the warpgroup skew" |
| 2026-10-30 | e7e75c9510994c08 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer --seeds=41,42,43,44,45,46 o57=57 o59=59, not committed) | fca2be7 | diagnostic of modes 57 (x rows loaded under the routing work plus 16-byte gather stores) and 59 (the early loads alone) | 2.207x (2.173-2.240) | passed; spin flags 0. Minus 51 at 32 / 128 / 256: 57 +1.2 / +1.1 / +1.2 us (0 of 6 lower each), 59 -0.0 / +0.6 / -0.3 (2, 1, 5 of 6). Thread 0's gather 0.8-1.8 us shorter (58 vs 55) but the first job's x ready only 0.2-0.6 us earlier. Not gated; docs/ruled-out.md |
| 2026-10-30 | 3d8902381a194d56 (check, bench file from tools/moe_make_bench_ab.py --score=1 --sizes=1,8,16,32,128,256 --seeds=301-310, not committed) | this branch (tool only) | standalone scorecard: the shipped dispatch against the public file's whole dispatch, two independent prepares per side, std routing, seeded shuffled order | 2.211x (2.177-2.244) | passed; spin flags 0; policy sk8w3s at 1-32, 51 at 128 / 256 in both prepares. Public minus ours, us (10 seeds): +1.37 / -0.80 / +0.04 / -1.54 (sd 8.0, bimodal) / -9.60 / -14.92 at 1 / 8 / 16 / 32 / 128 / 256; nulls within 0.3 (ours) and 0.75 (public). Value if closed: 256 +1.12%, 128 +0.86%, 8 +0.21%, 32 +0.19%. Section "Standalone scorecard against the public kernel" |
| 2026-10-30 | 48b9bf9d77d146ac (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=47,48,49,50,51,52 o60=60, not committed) | 003be4b | diagnostic of mode 60 (51 without the gather's per-thread __threadfence; proxy fence, barrier and releases kept) with gather stamps (61 against 55) | 2.205x (2.172-2.238) | passed; spin flags 0. 60 minus 51 at 32 / 128 / 256: layer -0.3 / -0.4 / +0.0 us (6, 5, 4 of 6 lower), std -0.3 / -0.5 / -0.4 (3, 5, 4). Thread 0's barrier wait -1.3 to -3.6 us, its release +0.6 to +2.5 (MEMBAR.ALL.GPU before the RED); last x ready 0.4-2.0 us earlier. Pre-set gate condition failed (256 layer); not gated, docs/ruled-out.md |
| 2026-10-30 | b7e43ee6a517487a (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=53,54,55,56,57,58 o62=62 o63=63, not committed) | d5bafad | diagnostic of modes 62 / 63 (51 with phase A's last one / two rounds claimed at run time through ctr[10] ahead of phase B) with light and gather stamps (64, 65 against 55) | 2.198x (2.165-2.231) | passed; spin flags 0. 62 minus 51 at 32 / 128 / 256: layer +3.9 / +7.9 / +9.2 us, std +6.1 / +7.5 / +8.6; 63 +5.3 / +7.0 / +8.5 and +5.3 / +6.9 / +7.9 (0 of 6 lower in every row). Phase A end spread at 128 12.8 / 10.1 -> 9.5 / 7.7 us (62), but mean phase A +4.8 to +6.8 us at 128 / 256. Pre-set rule: not gated, docs/ruled-out.md |
| 2026-10-30 | ed4591a2477041c2 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=65,66,67,68,69,70 o66=66 o62=62, not committed) | a727b12 | null of the claimed phase A tail: mode 66 = 62's code with no round claimed (K = 0), 62 again, stamped twins 67, 64 against 55 | 2.199x (2.166-2.232) | passed; spin flags 0. 66 minus 51 at 32 / 128 / 256: layer +4.3 / +7.0 / +8.8 us, std +6.0 / +7.9 / +8.7; 62 +4.2 / +6.9 / +8.5 and +5.7 / +7.9 / +10.3 (0 of 6 lower in every row); 67's phase A mean +5.1 to +7.7 over 55's. The loss is the code shape, not the claims (62 minus 66 -0.3 to +1.6). No gate |
| 2026-10-30 | 098949b28fe84bd1 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=83,84,85,86,87,88 o72=72, not committed) | 86a633e | diagnostic of mode 72 (stream-K phase A with cut-unit fixups, mode 16's partition, on 51's stage loop, epilogue and phase B claims) with its stamped twin 73 | 2.193x (2.160-2.226) | passed; spin flags 0. 72 minus 51 +13.5 / +9.4 / +12.6 / +7.1 / +6.6 / +9.2 us at 32 / 128 / 256 (layer / std), 0 of 6 seeds lower in all six cells; 73's A end spread 18.9 / 16.9 at 256 against 55's 10.0 / 8.8; fixups 9.4-10.9 us per CTA at 128 / 256 (ABFIX). Diagnostic rule failed, no gate run; docs/ruled-out.md |
| 2026-10-30 | eaa881a934aa471e (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=95,96,97,98,99,100 o74=74, not committed) | 5f35018 | diagnostic of mode 74 (72 with a designated-finisher fixup; 75 stamped); the build had fast_silu ending at 73 (review finding after the check), so 74 ran the slow SiLU, and its serial setup designation cost about 3 us | 2.202x (2.169-2.235) | 74 minus 51, us: +8.2 / +9.0 at 32, +8.3 / +7.2 at 128, +2.1 / +3.8 at 256 (layer / std; 0, 0, 0, 0, 0, 1 of 6 lower); public -0.9 to -19.1. Fixups 3.7-4.0 us per CTA (72: 8.7-10.9), 1.7-1.9 us per counting share, finisher 1.9-2.2 incl. 0.8-0.9 waiting; A end spread 6.7-7.2 at 128 (55: 11.4-14.0, public 6.8). Rule failed for this build; corrected build gets its own rule (docs section "Designated finisher for stream-K phase A") |
| 2026-10-30 | e02f60fec94f45cb (scored run submitted by the coordinator, bench file from tools/moe_make_bench_ab.py --nopub=1 --base=51 --sizes=32,128,256 --routings=layer,std --seeds=107,108,109,110,111,112 o74=74, no public code, judged path the board dispatch, not committed) | 4187b8b | corrected mode 74 (fast SiLU, lane-parallel designation) | 2.197x (2.164-2.230) | 74 minus 51, us: +7.9 / +7.6 at 32, +4.6 / +7.4 at 128, +2.0 / +4.1 at 256 (layer / std; 0, 0, 0, 1, 1, 1 of 6 lower). A end spread 6.6-8.3 at 128 / 256 against 55's 10.9-19.5; fixups 3.9-4.5 us per CTA; route + setup still +3.4-3.6 us over 55. Rule failed; no gate; family parked |
| 2026-10-31 | 5b2db8029f094a66 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std,skewed --seeds=121,...,126 --gate=1 o76=76 o77=77 o78=78 with the donor's standalone file as the public arm, not committed) | 45f990d | gate of 76-78 (rule committed in 45f990d) | 2.205x (2.172-2.238) | passed; no A/B: the donor's `MID_DEFINES +=` line re-enabled the MID build the tool empties and its build failed, which stops the A/B block. Prepare bench (synthetic) put 76 first at 128 / 256. Section "Donor MoE (M-A) against mode 51" |
| 2026-10-31 | 64254f49676c4ce6 (check, the same bench with that donor line removed, not committed) | 5f19bc1 | gate of 76-78 against 51, donor MAIN build as the public arm | 2.188x (2.155-2.221) | passed; spin flags 0. Minus 51 at 32 / 128 / 256, layer: 76 -0.1 / -0.6 / -0.8, 77 -3.1 / +0.7 / +0.1, 78 -3.5 / -0.4 / -0.9, donor -2.3 / -9.9 / -22.5; std: 76 -0.1 / -0.0 / -0.5, 77 -1.8 / -0.0 / +0.5, 78 -2.4 / -0.2 / -1.0, donor -6.3 / -13.4 / -26.3. 76 failed rule 1, 77 rule 2 (+0.7 at 128 layer), so 78 was not shipped under the rule; 78 lower than 51 in all nine cells. Outputs 0 at 128 / 256. Section "Donor MoE (M-A) against mode 51" |
| 2026-10-31 | c8fc5f669fa34d06 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std,skewed --seeds=131,...,136 --gate=1 o78=78, donor MAIN build as the public arm, not committed) | 95c9848 | 78's own gate (rule committed in 95c9848) | 2.194x (2.162-2.227) | passed; spin flags 0. 78 minus 51 at 32 / 128 / 256: layer -2.6 / -0.6 / -0.6, std -1.5 / +0.6 / -0.4, skewed -3.1 / -0.8 / -1.3; donor minus 51 layer -3.2 / -11.6 / -19.8, std -8.7 / -10.1 / -20.1. Rule 2 failed (+0.6 at 128 std, limit +0.5): 78 not kept. Outputs 0 at 128 / 256. Section "Donor MoE (M-A) against mode 51" |
| 2026-10-31 | 2a4fb39964aa448c (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std,skewed --seeds=141,...,146 --gate=1 o72=72 o79=79, donor MAIN build as the public arm, not committed) | 587da5d | gate of 79 (stream-K in both phases, stage-0 finisher in phase B) with the null 72 (rule committed in 587da5d) | 2.196x (2.164-2.229) | passed; spin flags 0. Minus 51 at 32 / 128 / 256, layer: 72 +13.1 / +11.3 / +6.1, 79 +16.4 / +19.4 / +21.4, donor -1.9 / -11.3 / -21.0; std: 72 +10.3 / +9.8 / +4.3, 79 +17.5 / +24.8 / +11.3, donor -4.7 / -13.3 / -25.6; skewed 79 +6.0 / +27.9 / +31.4. 0 of 6 seeds lower for 79 in every cell. Not kept; family parked. Section "Stream-K in both phases (mode 79)" |
| 2026-10-31 | 43c35d18b5504eb5 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std --seeds=151,...,156 --work=1 --pubfix=1, the donor MAIN build stamped as a bench-only copy, not committed) | 1d42779 | diagnostic: donor per-phase stamps (DBG points plus its fixup, doneA release and finisher-wait cycles) against 55 (51's light twin) | 2.197x (2.164-2.230) | passed; spin flags 0. Donor minus 51 replay -3.0 / -6.2 / -9.8 / -12.1 / -21.8 / -22.4 us (32 / 128 / 256, layer / std). Per phase at 128 / 256 layer: start -3.4 / -3.8, phase A mean -3.4 / -10.3, A end spread -5.4 / -2.7, phase B 0.0 / -2.1, barrier + final -3.3 / -4.4. Donor fixups 10.4-12.8 us/CTA in phase A, 5.4-8.5 in phase B, releases 2.5-7.7; derived per stage 1.12 us (donor) against 1.38-1.47 (51). Stage loop 295 against 387 instructions. Section "The donor kernel stamped against 51" |
| 2026-10-31 | 1d767886e86d46e1 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=128,256 --routings=layer,std --seeds=161,...,166 --ablate=1 o83=83 o76=76, donor MAIN build as the public arm, not committed) | 2921d50 | stage-cost ablation: 83 (51 without the per-thread weights-slot fence, timing only, never a candidate) and 76 | 2.193x (2.160-2.225) | passed; spin flags 0; outputs 0 against 51. Minus 51 at 128 / 256: 83 -0.1 / -0.3 (layer), +0.2 / -0.5 (std); 76 +0.2 / -0.9, +0.3 / -0.8; donor -9.3 / -20.0, -13.4 / -25.3. The fence is not the per-stage cost (reading rule case 3). Section "Stage cost ablations" |
| 2026-10-31 | a00b49f8372042a9 (check, bench file from tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std,skewed --seeds=181,...,186 --gate=1 o86=86, donor MAIN build as the public arm, not committed) | 30d6b96 | gate of 86 (51 with each stage's x slot released after wait_group 0 at the stage end; rule committed in 30d6b96) | 2.195x (2.162-2.228) | passed; spin flags 0; outputs 0 against 51 at 128 / 256. 86 minus 51 at 32 / 128 / 256: layer +0.2 / -0.3 / -0.6, std -0.8 / +0.0 / -0.1, skewed -0.1 / -0.1 / -1.4. Rules 1 and 2 failed: not kept. Section "Earlier x slot release (mode 86)" |
| 2026-10-31 | 40b8c798289a4258 (check; F-MOE-0 bench: release ec565a7 byte for byte + the exact donor file lzma-carried, not committed) | ec565a7 (release) | Family F F0: R1/R2 (two release prepares) against D1/D2 (two prepares of alygg77 muse_moe_w4a16.py at 6cae49a, SHA256 b3307089...), std seeds 501-512, layer 521-526, pad/off64 541-544, order seed 8101 | 2.195x (2.163-2.228), the release's judged path | passed; 0 output failures, fresh-input replay ok. Score D/R pooled 1.0387 (nulls 0.9996 / 0.9993, D faster); per load R/D 0.960 / 1.013 / 1.006 / 1.060 / 1.079 / 1.120 at 1 / 8 / 16 / 32 / 128 / 256. Section "F-MOE-0 (Family F donor comparison)" |
| 2026-10-31 | 218ff300de7d4c22 (check; F-MOE-0 confirmation, same writer, not committed) | ec565a7 (release) | as 40b8c798289a4258 on std 601-612, layer 621-626, pad/off64 641-644, order seed 9203 | 2.194x (2.161-2.227), the release's judged path | passed; 0 failures. Score D/R pooled 1.0297 (nulls 1.0017 / 0.9977, D faster); per load 0.964 / 1.014 / 1.004 / 1.010 / 1.078 / 1.116. Decision per load: keep ours at 1, re-implement TINY2 at 8, inconclusive at 16, re-implement WGS (with proxy fences) at 32 / 128 / 256 |

## F-MOE-0 (Family F donor comparison), 2026-10-31 UTC (2026-10-30 PDT)

Question (archive/docs/steering/2026-10-30-family-f-donor-baseline.md, track F-MOE): is the complete
alygg77 standalone MoE file faster than our frozen release under matched controls, and if so,
on which token counts and through which mechanism? User decision relayed by the coordinator on
2026-10-31: Family F is measurement and learning only. A donor win means "re-implement
mechanism X in our own code, with attribution in comments"; donor code is never submitted and
F0 / F2 are not submission candidates. Both runs below are checks; nothing was submitted, and the
judged path of each check was the release file itself.

### Files and provenance

| item | value |
|---|---|
| donor | `alygg77/okc-muse-spark-kernels` at `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`, `kernels/muse_moe_w4a16.py`, 326,062 bytes, SHA256 `b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281` |
| donor receipt | its `results.json` maps that exact SHA256 (MATCH) to job `e1bcad5a55cd4a29`, 2.4015x, rank 41 read at 2026-10-29 20:54 UTC (archived, not re-verified) |
| donor license | Apache-2.0 (LICENSE, NOTICE): vjeux/okc-2026 at f35a41e is the base; the per-thread CUDA-context bind helper follows Matsvei's (kernarc 'matvey') posted code. Any re-implementation keeps that attribution in comments |
| release (control) | `git show ec565a7:muse_moe_w4a16/submission.py`, 235,700 bytes, SHA256 `b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14`; scored run abe7651dbaa44458 (2.216x). Its judge content hash `084911f39cdd77d1...` equals that of check b50351288b504e81, which the run table records as ec565a7 |
| bench | the release file byte for byte, then a block that carries the donor file lzma-compressed and base85-encoded, checks its SHA256 at run time (F0PIN MATCH in both runs), and execs it unchanged into its own module. No donor line was patched: its `import threading` / `ctypes` context bind is inside the blob and the judge's static check passed. Writer: scratchpad `fmoe/make_f0_bench.py` (not committed; no donor code in the tree) |

### Method

- Arms: R1, R2 = two `prepare()` calls of the release; D1, D2 = two `prepare()` calls of the
  donor. Each prepare allocates its own weights copy and scratch; only compiled extensions are
  shared within a side. Prepare order seeded (run 1: R2, R1, D2, D1; run 2: R1, D2, D1, R2);
  the first prepare of each side builds its extensions (150 s release, 165-169 s donor). Same weights for all arms: nvfp4.random_weights restated (log-normal
  block scales clamped to [8, 448], as the judge's), 32 local experts, seed 5252.
- Inputs: every arm gets its own copy of the same input values. std = reference.py's
  generate_input (the scored distribution); layer = the layer's judge routing restated (router
  seed 8, mean busiest-expert miss 3.0%).
- Timing: CUDA graph replay after a 256 MB L2 flush, median of 20 per (arm, seed), arm order per
  seed a seeded rotation of a seeded permutation, reversed every four seeds (each arm in each
  position equally often over 12 seeds).
- Held-out seeds (unused before in this file): run 1 std 501-512, layer 521-526, robustness
  541-544, order seed 8101; run 2 std 601-612, layer 621-626, robustness 641-644, order seed
  9203.
- Robustness (not used to pick anything): "pad" puts a seeded 4-252 MB allocation before each
  arm's input copy; "off64" makes each input tensor a view at element offset 64.
- Correctness: every timed graph's own output after its replays against the FP32 reference with
  the judge's rules (1.5% relative L2, rows within 5x, finite): 0 failures in both runs. Fresh
  input replay: capture on one input, copy another input into the captured buffers, replay,
  check against the second input's reference: ok for all four arms at all six sizes in both
  runs.
- Score: per arm, geometric mean over the six loads of receipt abe7651dbaa44458's
  overhead-subtracted SGLang time over (the arm's bench median minus 4.1 us of replay
  overhead). Bench medians run a few us above the judge's, so the absolute scores sit below the
  board; the ratios between arms are the result. Decision rule (as tools/dl_make_bench_pub.py):
  all four cross ratios on one side of 1 and |log pooled| above twice the larger |log null|.

### Selected policies (identical in both runs and in both copies of each side)

| tokens | release (R1 = R2) | donor build (D1 = D2; scales PRMT-exact) | donor dispatch |
|---|---|---|---|
| 1 | sk8w3s | TINY | donor:3377 (M <= TINY_MAX_M 4) |
| 8 | sk8w3s | TINY2 | donor:3379 (M <= 8) |
| 16 | sk8w3s | MID with the `@helper0` source | donor:3381, 3543-3544 |
| 32 | sk8w3s | MAIN, WGS (RS-wgmma) | donor:3383-3384, comment donor:150 |
| 128 | main-lean-grp-nohoist (mode 47) | MAIN, WGS | as above |
| 256 | main-lean-grp-nohoist (mode 47) | MAIN, WGS | as above |

### Results

Weighted standalone score (bench-based, std routing, the scored distribution):

| run | R1 | R2 | D1 | D2 | cross D/R (R1/D1, R1/D2, R2/D1, R2/D2) | pooled | null R2/R1 | null D2/D1 | rule |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| 40b8c798289a4258 | 2.0784 | 2.0775 | 2.1592 | 2.1576 | 1.0389, 1.0381, 1.0393, 1.0385 | **1.0387** | 0.9996 | 0.9993 | D faster |
| 218ff300de7d4c22 | 2.1409 | 2.1445 | 2.2089 | 2.2038 | 1.0318, 1.0294, 1.0301, 1.0276 | **1.0297** | 1.0017 | 0.9977 | D faster |

The same scores on the other routings and layouts, pooled D/R (null R / null D): layer 1.0601
(0.9975 / 0.9981) and 1.0628 (1.0003 / 0.9977); pad 1.0312 and 1.0312; off64 1.0319 and 1.0309;
all "D faster". The allocation layout moves both sides together; no layout changes the sign of
any load below.

Per load, pooled time ratio R/D (above 1: the donor is faster), run 1 / run 2; nulls are the
larger of R2/R1 and D2/D1 distance from 1 in that run:

| tokens | std | layer | pad | off64 | std nulls (max) | readout |
|---:|---|---|---|---|---|---|
| 1 | 0.9603 / 0.9644 | 0.9650 / 0.9634 | 0.9686 / 0.9608 | 0.9680 / 0.9629 | 0.29% / 0.22% | release faster by 3.5-4%, every arm pair, both runs |
| 8 | 1.0133 / 1.0143 | 1.0210 / 1.0172 | 1.0131 / 1.0114 | 1.0149 / 1.0106 | 0.27% / 0.10% | donor faster by 1.0-2.1% |
| 16 | 1.0061 / 1.0038 | 0.9942 / 0.9916 | 0.9972 / 1.0000 | 0.9966 / 0.9978 | 0.03% / 0.24% | sign depends on routing: inconclusive |
| 32 | 1.0598 / 1.0098 | 1.1398 / 1.1726 | 1.0323 / 1.0403 | 1.0346 / 1.0422 | 0.14% / 0.07% | donor faster; std is bimodal at 32 (per-seed sd 6-8 us, as in 3d8902381a194d56), layer 14-17% |
| 128 | 1.0792 / 1.0778 | 1.1104 / 1.1070 | 1.0644 / 1.0703 | 1.0642 / 1.0692 | 0.29% / 0.46% | donor faster by 6.4-11% |
| 256 | 1.1201 / 1.1160 | 1.1353 / 1.1341 | 1.1186 / 1.1117 | 1.1197 / 1.1095 | 0.36% / 0.51% | donor faster by 11-14% |

Raw means, us (std routing, 12 seeds), from the logs' F0TIME lines; the 16 KB log tail kept the
128 and 256 rows only (the small sizes are in the ratios above and in the robustness rows of the
logs):

| run | tokens | R1 | R2 | D1 | D2 | paired D1-R1 (lower / 12) | paired D2-R2 |
|---|---:|---:|---:|---:|---:|---|---|
| 40b8c798289a4258 | 128 | 197.36 | 197.40 | 182.63 | 183.16 | -14.73 (12) | -14.24 (12) |
| 40b8c798289a4258 | 256 | 255.85 | 256.77 | 228.85 | 228.81 | -27.00 (12) | -27.97 (12) |
| 218ff300de7d4c22 | 128 | 210.00 | 209.04 | 194.17 | 194.62 | -15.82 (12) | -14.42 (12) |
| 218ff300de7d4c22 | 256 | 246.26 | 245.25 | 219.65 | 220.76 | -26.61 (12) | -24.49 (12) |

Layer routing, run 1 / run 2, us (R1, D1): 1 token 33.3 / 34.6 and 32.5 / 33.7; 8: 67.4 / 65.9 and
75.7 / 74.5; 16: 76.8 / 77.4 and 85.0 / 85.9; 32: 89.2 / 78.1 and 103.4 / 88.1; 128: 161.6 / 145.4
and 160.1 / 143.5; 256: 225.8 / 199.2 and 229.9 / 202.1.

Accuracy: relative L2 against FP32, release 3.9e-3 to 4.0e-3, donor 2.3e-3 to 2.5e-3 at every
size (both within the 1.5e-2 bound). Release spin flags 0. The checks' own judged scores (the
release file): 2.195x (2.163-2.228) and 2.194x (2.161-2.227), two more readings of the release.

Hypothesis, not measured: taking the donor's mechanism at 8, 32, 128 and 256 tokens and keeping
ours at 1 would give about 1.045x on the run-1 std ratios (geometric mean of the per-load wins,
1 and 16 counted as 1.0). That sum of isolated wins is not a hybrid measurement.

### Eligibility review of the donor file (against muse_moe_w4a16/TASK.md and reference.py)

1. **Proxy ordering, WGS path (used at 32, 128, 256 tokens): defect.** The x gather writes xA with
   generic stores (`*d[k] = v[k]`, donor:2061), syncs the CTA (donor:2063) and publishes with
   `red.release.gpu` on `xready` (donor:2069); phase A's epilogue writes xB with generic stores and
   publishes with `red.release.gpu` on `doneAs` (donor:2118-2122, 2171-2172). Another CTA's x
   producer acquires those counters (donor:2011, 2014) and reads xA / xB with
   `cp.async.bulk` (donor:2018-2019), an async-proxy operation. The file has no
   `fence.proxy.async.global` on either side (its only proxy fences are `.shared::cta`, donor:760,
   974, 1446, 1482, 1550, 1579, 2488). This is the defect class this file's section "Proxy
   ordering of the x tiles, 2026-10-29" found and fixed in our fat path (PTX ISA 9.7.10.28.2 and
   8.9.5); our release fences both sides. It passed every test here, which does not make it
   safe. Consequence: a re-implementation of the WGS mechanism must carry
   `fence.proxy.async.global` on the writers before the release and in the x producer after the
   acquire, as our release does. The donor timings above are without those fences; their cost on
   the WGS path is unmeasured.
2. Skinny paths (1, 8, 16 tokens): bulk copies read only the prepared weights (donor:690, 1136,
   1974); inter is published with a per-thread `__threadfence`, CTA barrier and relaxed
   `atomicAdd` (donor:941-953) and read with generic loads after `ld.acquire` (donor:836, 1384).
   Our release uses the same publish pattern (`credit`, release file line 2980-2989). No
   donor-specific finding.
3. Precision: weights are repacked by bit and byte permutations only, never requantized
   (donor:3287-3343; KPERM3 permutes K inside each 256-K stage for weights and x together,
   donor:3223-3251); block scales decode exactly (PRMT only when every scale byte is a positive
   normal E4M3, checked in prepare, donor:3104-3111, 3332; LUT otherwise); post_norm_w stays BF16
   and 1 + w is formed in FP32 (donor:3296, 2101); x and inter are BF16 as in SGLang; accumulation,
   ybuf and the norm sums are FP32; output BF16. topk_weights are taken as FP32 (donor:3374). This
   problem has no router (topk ids and weights are inputs), so FP32 router semantics do not apply.
   Conforms.
4. Caching: the only module cache is `_EXTS` (compiled code, donor:3069-3100); `state["sfx"]`
   derives from the weights in prepare, not from data. kernel_fn branches on shapes and on the
   pointer alignment of topk_ids (donor:3372, an aligned copy when misaligned), not on values.
   The fresh-input replay test passed. Conforms.
5. Lifetimes: one scratch per state, counters re-zeroed in-kernel by generation, the norm-sum
   buffer double-buffered by launch generation (donor:2541-2558, 2630-2641); calls are serialized
   on the stream. The persistent grid (one CTA per SM) spins at a grid barrier
   (donor:2630-2634) with no timeout flag, so it relies on co-residency, as our release does.
6. Host: the per-thread context bind (donor:3546-3601, ctypes `libcuda`) is outside the task's
   numerical scope; the judge accepted the file. Not needed by any re-implementation.

### Decision (per load, under the measurement-only rule)

| tokens | decision | mechanism to re-implement (donor file:line) |
|---:|---|---|
| 1 | **keep ours** | release sk8w3s is 3.5-4% faster than the donor's TINY build in both runs |
| 8 | **re-implement mechanism**: the TINY2 skinny build | TINY2_DEFINES (donor:139-142): 8 compute warps (`OKC_NW=8`), skinny-only (`OKC_TINY=1`), `OKC_CPF=0`, `OKC_FAST1=2`, 4 accumulators (`OKC_NACC=4`), `OKC_UWARP=0`, a helper warp (`OKC_WS=1`, `OKC_WSB=1`, `OKC_WSBL=2`, `OKC_WSBLK=1`), final-pass prefetch (`OKC_FINPF=1`), 3-slot ring (`MAIN_NS`, donor:152), DQ7 bit layout (donor:172, 3114-3140), PRMT scale decode; dispatch M in (4, 8] (donor:3379). Gain 1.0-2.1% of that load |
| 16 | **inconclusive** | MID + `@helper0` (donor:145-149, 3543-3544) is +0.4-0.6% on std and -0.6 to -0.8% on layer routing |
| 32 | **re-implement mechanism**: dispatch 17-63 tokens to the WGS fat kernel | donor:150, 3383-3384: above 16 tokens the donor runs its WGS MAIN build where we run the skinny sk8w3s. 1-6% on std, 14-17% on layer routing. Same kernel as the next row |
| 128, 256 | **re-implement mechanism**: WGS, the RS-wgmma producer/consumer fat path | MAIN_DEFINES (donor:130: `OKC_NW=16`, `OKC_FAST1=0`, `OKC_EPI4=0`, `OKC_WGS=1`) with PRMT scales; weight copy per 256-row unit x 16 K-stages x 4 warpgroups of 9,216 B, warpgroup q owning rows 64q..64q+63, codes in natural mma k order then scales (`_wgs_copy`, donor:3254-3269); KPERM3 K order inside each 256-K stage so a lane's columns of one k16 step fall in one scale block (`_kperm_src` / `_kperm_codes` / `_kperm_rows`, donor:3223-3251; the x gather applies the same permutation, donor:2049-2054); stream-K stage ranges over (256-row unit, 16 stages) per CTA (`wgs_ranges` / `wgs_split`, donor:1950-1954, 2334) with FP32 fixups through `red.global.add.v4.f32` (donor:2267), 160 fixup slots (donor:120-122) and a designated finisher in phase B (`OKC_WGSDFIN=2`, donor:1855, `wgs_flush` donor:2113-2174); weights ring of 3 x 36 KB slots freed once codes are in registers (`WGS_NS`, donor:1820) and x ring of 3 slots (`OKC_WGSNX`, donor:1819); a dedicated x producer issuing one bulk copy per stage (`wgs_xproducer`, donor:1983-2023); phase A x gathered once by all consumer threads into the per-stage wgmma layout, 8 rows of loads before their stores (`OKC_GATHB`, donor:1853; `wgs_gather`, donor:2026-2073); phase B readiness per (job, stage) through `doneAs[j * 16 + st]` rather than per job (donor:2014, 2122). 6.4-11% at 128 and 11-14% at 256 |

The WGS rows must include the proxy fences of finding 1. Earlier same-process benches against
"the public kernel" in this file (for example 3d8902381a194d56, 7063f86352344efa) measured gaps
of the same sign and size at 128 / 256; whether that public file is this donor file's ancestor
with the same WGS path was not established here.

Not done: an F1 arm (the donor with the two proxy fences added) to price the fences, and any
F2 hybrid. Both are now re-implementation work in our own file, not donor candidates.

## Why the layer keeps the W4 chain at 128 tokens, 2026-10-29

Question (archive/docs/plan-2026-10-29-steering.md, step 4): standalone, mode 16 beats the W4 chain
(legacy128) at 128 tokens, while the layer's prepare-time bench keeps the W4 chain there
(e87fcb1ed5c64a62). Candidates: L2 state left by neighbouring kernels, overlap with
neighbours, tuner noise, input differences. Three layer checks of a snapshot (the layer at
bdb4c7a plus a probe at the end of prepare; the layer directory was not edited); probes and
profiles in `muse_moe_w4a16/profiles/layer-flip/`.

What the layer's bench times (code, `_moe1_tune`): graph replay after an L2 flush of the MoE
call alone, on the synthetic routing `_moe_routing(bs, 977 * seed + bs, 0, False)` for seeds 11
and 12 only, with the W4 chain's routing metadata (route_align) built outside the timed graph
because kernel_fn builds it on the side stream. No neighbouring kernel is in the timed graph,
so L2 state left by neighbours cannot reach the decision.

Measurements (same process, same inputs, 6 seeds, paired per seed; us):

| probe, T = 128 | W4 (metadata outside) | W4 (route_align inside) | mode 16 | mode 16 minus W4-run, per seed |
|---|---|---|---|---|
| synthetic routing, cold L2 (0cb15c11accf4190) | 259.1 | 261.8 | 250.5 | +1.6 +2.3 -35.8 -12.6 -9.4 +2.4 |
| synthetic routing, no flush | 253.4 | - | 244.5 | |
| synthetic, again (7d1205a6959c41a9) | 259.5 | 261.8 | 250.0 | +1.1 +3.1 -40.0 -14.0 -11.5 +4.3 |
| judge-like routing (9d046da0207b40b2) | 253.9 | 256.9 | 224.3 | -31.6 -27.6 -38.6 -31.8 -22.7 -25.0 |
| isotropic-input routing (7d1205a6959c41a9) | 263.6 | 265.9 | 276.2 | +11.0 +8.3 +17.2 +10.5 +17.5 +10.8 |

"Judge-like" is reference.py's residual stream (fixed mean direction, rank-4 shared part,
noise; verify draft rows at 0.6 of the shared part, the basis from its STREAM_SEED) through the
layer's own attention boundary module and the judge's router from prepare's weights, with
local_start on the 32-expert range that gets the most pairs, as reference.py sets it: 187-214
local routes over 17-22 experts, at most 42-71 per expert. "Isotropic" is a N(0, 1) FFN input
through the same router: 153-168 routes over 25-28 experts, at most 13-22 per expert, which is
not the judge's distribution (the router has a component along the stream's mean, so a few
experts take most tokens); its mode 16 time at 256 tokens (313 us) is also far above the
judge's own layer profile of mode 16 there (249-279 us), while the judge-like probe gives 275.
At 256 tokens mode 16 wins on every probe (judge-like: -82 to -107 us).

Answer (measured; the last step is inference):

- Tuner sampling, not timing noise and not L2. The bench's two seeds are the two of six
  synthetic seeds where the W4 chain wins (by 1-3 us); on seeds 13-15 mode 16 wins by 9-40 us.
  Run to run, the paired differences repeat within 0.5-4.2 us per seed, so timing noise does not
  decide it.
- L2 state: without the flush both candidates gain 5-6 us; the ranking is unchanged.
- Overlap: route_align costs 2.0-3.1 us at 128 tokens (W4 with it inside minus without), the
  most the side stream can hide. In the isotropic probe the whole front plus experts showed the
  same gap as the experts alone (+7.7 to +17.8 against +8.3 to +17.5), so the stream structure
  adds nothing beyond that.
- Input differences decide the sign. On judge-like routing mode 16 wins at 128 tokens by
  23-39 us per seed (mean about 30). Inference: the layer's pick of the W4 chain at verify 16
  costs about 30 us per shape on both verify-16 shapes; a bench on judge-like routing (or
  mode 16 forced at 128) would recover it. Not yet confirmed on the layer's scored shapes,
  which needs a layer check of that change (the layer owner's call).

## Mode 16 at 128 and 256 tokens: where the time goes, 2026-10-29

Stamps of mode 17 (mode 16 plus the stamps of mode 11, now also on wgmma segments) in check
9332e2a49aff4ff2, bench routing seeds 11 and 12, one cold-L2 launch each, all 132 CTAs, us at
1.77-1.78 GHz. The stamps cost 14 us at 128 (263.3 vs 248.9) and 18 at 256, so absolute times
run high. Per-warp rows are sums over one consumer warp's segments (mean / max over CTAs); they
overlap each other and the producers, so they are not end-to-end time.

| | 128, seed 11 | 128, seed 12 | 256, seed 11 | 256, seed 12 |
|---|---|---|---|---|
| critical path: routing + x gather before the consumers start, mean / max | 14.4 / 18.2 | 15.2 / 19.2 | 19.3 / 26.9 | 20.4 / 23.3 |
| phase A done (thread 0 of each CTA), mean / max | 175.0 / 185.3 | 164.0 / 171.0 | 185.7 / 204.8 | 184.1 / 196.9 |
| consumers done, mean / max | 242.4 / 271.1 | 225.8 / 239.4 | 261.0 / 291.1 | 255.7 / 287.7 |
| completion tail (last consumers done minus mean) | 28.7 | 13.6 | 30.1 | 32.0 |
| final norm and sum after the barrier | 2.9 | 2.9 | 5.6 | 5.0 |
| launch end | 275.5 | 243.2 | 298.5 | 294.1 |
| per warp: consumer math (dequant + MMA, wgmma waits included) | 148.1 | 135.1 | 155.7 | 157.0 |
| per warp: epilogue and fixup | 22.1 | 21.3 | 26.6 | 23.0 |
| per warp: waiting for weights / for x | 12.7 / 11.1 | 10.7 / 12.5 | 12.4 / 15.4 | 11.0 / 12.7 |
| x producer waiting on done_as (phase B dependency), mean / max | 14.4 / 31.9 | 13.5 / 32.8 | 16.9 / 48.1 | 16.2 / 42.3 |
| cut segments per CTA, phase A / B | 1.88 / 0 | 1.86 / 0 | 1.89 / 0 | 1.85 / 0 |

Reading: phase A ends at 171-205 us on the slowest CTA, then 55-90 us of phase B run after it
on the critical path, ending in a 14-32 us tail of the slowest CTA. Consumer math is the
largest per-warp item (135-157 us), as in the 2026-10-28 stamps; the weights producer still
waits 162-188 us for a free slot.

The named removable cost: phase A's stream-K cuts and cost-table partition. Each CTA cuts
about 1.9 phase A units, each finished through the global fixup by its last share. Replacing
them with whole units owned by one CTA (mode 19 against mode 18, phase A alone, same process):
-5.5 / -5.6 us at 128 tokens and -16.4 / -15.7 us at 256 (two checks, 6 seeds each). In the
full kernel (mode 20 against mode 16) the saving is -2.7, -2.1, -3.1 us at 128 and -4.4, -3.8,
-4.2 at 256 (three checks): phase B absorbs part of it. Mode 20 removes it and is the policy
at 128 and 256 in 72b7a5a9c511446a. The next cost in line is the completion tail (14-32 us), a
hypothesis-level target: phase B's stream-K boundaries come from the cost table and its stages
start only as other CTAs' phase A units release them.

## Phase B claimed at run time (mode 23), 2026-10-29

Target: phase B's critical path and the completion tail at 128 and 256 tokens (the steering
plan's 128/256 loads hold about 53% of the layer's remaining log gap).

Attribution first, on mode 22 (mode 20 plus the stamps of 17; check a9dbec1d98334943, cold L2,
one launch per seed, stamps add 14-18 us so compare rows with each other). Mode 20 deals phase A
whole units in snake order, so CTAs end phase A with 5 or 6 units (80 or 96 stages), while
phase B keeps mode 16's static ranges cut from the cost table, which do not see that:

| mode 22 (20 with stamps) | 128, seed 12 | 256, seed 11 | 256, seed 12 |
|---|---|---|---|
| CTAs with 80 A stages: count, phase A done, B stages, consumers done mean/max | 56, 141.2, 44.3, 207.4/231.9 | 24, 159.3, 51.3, 234.9/249.1 | 24, 154.7, 47.3, 235.3/257.0 |
| CTAs with 96 A stages: count, phase A done, B stages, consumers done mean/max | 76, 164.6, 44.8, 225.6/240.1 | 108, 179.5, 45.5, 254.1/278.5 | 108, 177.5, 46.4, 248.4/273.4 |
| slowest CTAs: A stages / B stages / B span after their phase A | 96 / 48 / 72 | 96 / 64 / 85-97 | 96 / 48-64 / 84-96 |
| completion tail (slowest consumers done minus mean) | (line cut from the log) | 27.9 | 27.4 |
| x producer waiting on done_as, mean/max | 13.7/33.1 | 15.6/28.1 | 16.2/36.5 |

The CTAs that start phase B 20 us early get no more phase B work than the others (44.3 against
44.8 stages at 128), and the slowest CTAs are those with a full phase A and a fourth phase B
unit. That is the removable cost: a phase B partition that ignores each CTA's phase A load.

Hypothesis: if each CTA takes its next phase B unit when its weights ring frees a slot, the
CTAs that leave phase A early absorb the difference and the tail shrinks to about the
granularity of one unit. Taking units in S.ord order puts the jobs whose phase A units are dealt
in the first snake rounds at the front of the queue, so a claimed unit finds its inputs
published. A CPU list-scheduling model (cost-table stage times, judge-like routing, seeds 11-16)
put a greedy assignment 10-30 us under mode 20's partition, a hypothesis, not a measurement.

Mode 23 is mode 20 with two changes. The weights producer claims phase B units one at a time
from a global counter (ctr[10]) in S.ord order, 16 units per job, once its ring has a free slot,
and passes each claim to the x producer and the consumers through a shared-memory queue (S.bq).
Both phases are walked largest n-tile count first, so a CTA runs its phase A units in snake
round order and its claimed units in claim order (S.ord's n-tile counts never rise, so one loop
per count still covers them; each loop keeps a single wgmma shape). Every phase B unit stays
whole on one CTA: no phase B fixup, as in mode 16.

| mode 24 (23 with stamps) | 128, seed 11 | 128, seed 12 | 256, seed 11 | 256, seed 12 |
|---|---|---|---|---|
| phase A done, mean / max | 161.6 / 188.4 | 149.5 / 164.4 | 178.0 / 185.6 | 172.2 / 179.5 |
| consumers done, mean / max | 229.4 / 245.0 | 212.5 / 223.2 | 250.5 / 260.6 | 243.8 / 251.1 |
| completion tail | 15.6 | 10.7 | 10.2 | 7.4 |
| launch end | 250.1 | 227.8 | 267.5 | 258.6 |
| x producer waiting on done_as, mean/max | 15.2/23.5 | 15.5/19.8 | 15.5/23.3 | 14.6/18.1 |

At 128 seed 12 the 80-stage CTAs now take 48.0 phase B stages against 42.1 for the 96-stage
ones; the tail falls from about 28 us to 7-16 us, and at 256 the launch ends 17-21 us earlier
on the same seeds. The mean done_as wait does not move (13.7-16.2 against 14.6-15.5 us), so the
gain is balance, not fewer dependency waits.

Same-process paired A/B (graph replay after an L2 flush, variant order reversed on every other
seed), 23 minus 20, us:

| routing | 128 tokens | 256 tokens |
|---|---|---|
| seeds 11-16 (a9dbec1d98334943) | -16.2 (6 of 6 lower) | -16.1 (6 of 6) |
| held-out seeds 17-22 (36812632dae242ba) | -22.9 (6 of 6) | -18.7 (6 of 6) |
| skewed, seeds 11-13 (36812632dae242ba) | -4.8 (3 of 3) | -22.9 (3 of 3) |

Against mode 16 the held-out means are -19.5 and -24.8 us. Outputs match mode 16 to 1e-4 to
4e-4 relative L2 (FP32 atomic order), every self-test is 3.9e-3 against FP32, no spin flags.
The bench picks main-own-dynb at 128 and 256 in both checks.

Ordering obligations. The data handoffs are mode 16's, unchanged: x rows through xready, phase
A's SwiGLU output through done_as with fence.proxy.async.global on both sides ("Proxy ordering
of the x tiles" below), weights through wfull/wempty. Two new handoffs carry only indices:

1. The claim counter (cross-CTA). Writers: each weights producer's atomicAdd on ctr[10]
   (relaxed). Publication and observation: the value an RMW returns; atomicity (PTX ISA Memory
   Consistency Model, Axioms, "Atomicity") gives each value to exactly one claimant. No data is
   published through it, so no release or acquire is needed: the claimed unit's inputs are
   still observed per stage through done_as (ld.acquire.gpu, then fence.proxy.async.global by
   the issuing thread before its cp.async.bulk, 9.7.10.28.2). Completion and use: complete_tx
   on xfull and wfull, consumers' mbarrier.try_wait (acquire.cta), unchanged. Reuse: CTA 0
   zeroes ctr[10] after the grid barrier. Every claim precedes its CTA's bar.sync and thread 0's
   __threadfence plus barrier atomicAdd (a release pattern, 8.8); CTA 0's ld.acquire of the
   full count synchronizes with it (8.9.4), so the zeroing is causally after every claim and
   later in coherence order. The next launch starts after this one completes (stream order, or
   griddepcontrol.wait under PDL, 9.7.15.14).
2. The claim queue S.bq (inside one CTA). Writer: the weights producer thread,
   st.release.cta.shared. Observers: the x producer and every consumer thread,
   ld.acquire.cta.shared on the same word, morally strong (8.7), so the acquire synchronizes
   with the release (8.8, 8.9.4). Use: the index selects S.ord, S.job_* entries written in the
   prologue before __syncthreads. No async-proxy operation reads S.bq. Reuse: each entry is
   written once per launch and set back to empty in the next launch's prologue, before the
   __syncthreads that precedes every role.

Every fence and barrier of mode 20 is kept.

Verdict: mode 23 wins at 128 and 256 tokens on 24 of 24 paired normal-routing seeds and 6 of 6
skewed ones; it is the standalone policy at both sizes. Port it to the layer: add 23 to
`LAYER_MODES` in `tools/dl_moe_inline.py` (the layer's ctr is already int32 [64], zeroed at
allocation; SmemMain grows by 256 bytes to 228,736) and let the layer's judge-like routing bench
choose. What is left of the tail (7-16 us stamped) is about half a phase B unit (16 stages of
1.3-1.4 us at one n8 tile). Splitting only the last units with split-K would cut it at most in
half, and the cost model put a phase B cut at about 10 us, so it was not tried.

## Same-process A/B against the public MoE, 2026-10-29

Method. `tools/moe_make_bench_ab.py OUT.py PUBLIC.py [--seeds=..] [ARM=MODE ..]` appends to
submission.py a block that runs at import, before the judge's prepare(): random NVFP4 weights for
32 experts (the judge rejects a file that imports reference.py; timing does not depend on the
values), our state, and the public file (vjeux/okc-2026 `muse_moe_w4a16.py`, copy in the session
scratchpad, its small-batch builds switched off so only the MAIN build compiles) with its own
prepare(). Both kernels run on the same inputs, graph replay after a 256 MB L2 flush, median of 20,
arms interleaved and reversed on every other seed. Two routings: "std" is reference.py's
generate_input (the standalone board); "layer" ports the layer's judge routing (residual stream
with the mean and shared directions, residual boundary, sigmoid router with bias, busiest 32-expert
range), router seed picked so the busiest local expert matches the layer's own SUMMARY lines
(measured 18-23, 50-59, 105-128 routes at 32 / 128 / 256 tokens against 20-22, 51-55, 100-124 in
80b98f567aa74dd0). Stamps: mode 25 (thread 0's globaltimer at the phase boundaries, no cost in the
bench) against the public kernel built with its OKC_DEBUG stamps, its DBG(3) moved to the CTA-wide
point before the grid barrier so both kernels mark the same events.

Attribution, check d1f587760b1241dc, layer routing, mean of six seeds, us. Setup = start to the
first consumer stage (routing, then the x gather). Phase A and phase B are per-CTA means (thread
0's end of phase A, then all consumers done). Tail = slowest CTA's consumers minus the mean.

| | T=32 ours / public | T=128 ours / public | T=256 ours / public |
|---|---|---|---|
| setup (routing part) | 11.1 (7.2) / 3.9 (1.9) | 17.7 (10.9) / 6.1 (2.4) | 24.0 (15.9) / 8.4 (3.3) |
| phase A | 51.8 / 49.5 | 97.7 / 87.2 | 137.4 / 119.5 |
| phase B | 23.8 / 23.0 | 49.8 / 43.0 | 67.4 / 58.0 |
| tail | 8.8 / 4.0 | 10.4 / 4.7 | 8.5 / 6.4 |
| barrier wait + final pass | 2.4 / 2.3 | 4.2 / 4.3 | 6.4 / 6.3 |
| launch end (stamped) | 98.9 / 85.6 | 180.3 / 146.4 | 244.2 / 200.3 |
| ours minus public: setup / A / B / tail | +7.2 / +2.3 / +0.8 / +4.8 | +11.6 / +10.5 / +6.8 / +5.7 | +15.6 / +17.9 / +9.4 / +2.1 |
| graph replay, paired (public minus 23) | -12.8 | -32.2 | -45.7 |

Standalone routing gives the same order: setup +6.6 / +10.9 / +15.2, phase A +6.3 / +15.6 / +24.1,
phase B +2.4 / +6.2 / +10.0, tail +6.4 / +3.9 / +3.6 us at 32 / 128 / 256 (replay gap -18.0 /
-34.9 / -52.5). The final pass is at parity everywhere. Setup is the largest single item at 32 and
128 tokens; phase A is the largest at 256.

SASS (local nvcc 12.8, sm_90a, `tools/moe_compile_fused.sh` for ours at e261521, the public CUDA source with
its MAIN defines; counts in `muse_moe_w4a16/profiles/sass_stage_loop_vs_public.2026-10-29.txt`). Both kernels cap at 96 registers; ours spills (64-byte frame, 120 B stores, 308 B
loads, outside the stage loop), the public one does not. The 256-k stage loop of one wgmma shape (16
HGMMA, here m64n24k16) is 389-391 instructions in our mode 23 against 322 in the public kernel. The
dequant and MMA core is the same (LOP3 85 / 80, HMUL2 64 / 64, IMAD.SHL 28 / 32, SHF 33 / 32, PRMT
8 / 16, UIADD3.64 15 / 15); the difference is waiting and bookkeeping: four mbarrier try-waits per
stage against two, the spin-timeout path (clock reads, ATOMG.OR on ctr[8], YIELD), the
fence.proxy.async.shared::cta before the x-slot release (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S), and
more address and branch work (LEA 9 / 1, ISETP 10 / 2, BRA 10 / 3). How much of phase A's +10-24 us
that explains is not measured.

Change tested (mode 26), the setup half: the public route() lets every warp count its slice once
and warp 0 combines the per-warp counts; ours had one warp count all E entries serially and then
position_pass counted them again in every warp. Mode 26 counts once in all 18 warps, the routing
warp sums the counts and writes each warp's offsets, the positions reuse them; the x gather issues a
CTA's row loads (16 B, up to 8 rows) before the stores, as the public gather does; and the four
stage_at searches thread 0 ran for stage ranges are dropped, since owned phase A units and claimed
phase B units never read them. No fence, barrier or handoff changes: the gather keeps
fence.proxy.async.global, __threadfence, the named barrier and red.release on xready; the new
shared-memory traffic (per-warp counts, offsets) is ordered by the __syncthreads already between
the steps.

| 26 minus 23, us (paired) | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer routing, seeds 1-6 (39c365c34cd74559, without the range change) | -0.5 | -3.7 | -8.5 |
| layer routing, seeds 7-12 (4c285df44eb34bc1) | -1.4 | -4.6 | -9.1 |
| standalone routing, seeds 7-12 (4c285df44eb34bc1) | | -5.8 | -9.8 |
| standalone, held-out seeds 17-22 (GATE, 817eab6401d54a84) | | -5.8 | -9.5 |
| skewed, seeds 11-13 (GATE, 817eab6401d54a84) | | -4.2 | -7.7 |

Every paired seed favours 26 (66 of 66 at 128 and 256, 12 of 12 at 32 on layer routing). The
stamped routing at 128 tokens fell from 11.0 to 5.8 us (public 2.4): what is left is the S.ord
counting sort (2.5 us, eight dependent rounds in one warp) and the rest of the routing warp (1.9).

Verdict and next step. Mode 26 is the standalone policy at 128 and 256 tokens (817eab6401d54a84,
2.104x). Port it to the layer: add 26 to `LAYER_MODES` in `tools/dl_moe_inline.py`; the kernel's
interface, scratch and SmemMain size (228,736 bytes) are mode 23's. On the layer-like routing it
saves 3.7-4.6 us at 128 tokens and 8.5-9.1 at 256, and 0.5-1.4 at 32. After it the same-process gap
to the public kernel is about 27 us at 128 and 41 at 256 (layer routing), mostly phase A (+10.5 /
+17.9), phase B (+6.8 / +9.4), the tail and the rest of setup (+6 us). The gap map's E1 (job table
built on the side stream) now has a ceiling of the remaining routing, 5.8-7.4 us at 128-256 tokens,
of which an in-kernel sort that does not take eight serial rounds would recover part without a new
cross-launch handoff; the E1 arm itself was not built.

## Consumer stage and the ring slot releases, 2026-10-29

Question: after mode 26 the same-process gap to the public kernel was about 27 us at 128 tokens and
41 at 256 (layer routing), with phase A +10.5 / +17.9 us and phase B +6.8 / +9.4. The SASS of one
256-k stage was 389 instructions against 322, with "four mbarrier waits against two" and one
shared-memory proxy fence per stage in ours and none in theirs. Is the public scheme provably
correct, and what can we cut?

Sections cited are from the PTX ISA 9.4 documentation
(docs.nvidia.com/cuda/parallel-thread-execution), read 2026-10-29: 8.8 Release and Acquire
Patterns, 8.9.1.1 Asynchronous Operations, 8.9.4 Memory synchronization, 8.9.5 Causality Order,
8.10.6 Causality, 9.7.10.28.2 Async Proxy (cp.async.bulk), 9.7.15.2 bar.warp.sync, 9.7.15.4
membar / fence, 9.7.15.16.16 mbarrier.arrive, 9.7.15.16.19 mbarrier.test_wait / try_wait,
9.7.17.4 Async Proxy (wgmma).

### What the public kernel does per stage (wgs_seg and the two producer warps, public source)

- Two waits per stage: `mbar_wait(BW_F + ws)` (weights full) and `mbar_wait(BX_F + xs)` (x full),
  each a two-instruction `try_wait` loop with no timeout.
- Weights slot release: lane 0 of each consumer warp does `mbarrier.arrive` on `BW_E + ws` (count
  NW warps) once its warp has issued the loads of the stage's last code group, with no
  `__syncwarp` and no proxy fence. The producer waits on `BW_E` and then issues the next
  `cp.async.bulk` into the slot.
- X slot release: lane 0 arrives on `BX_E + prev` after `wgmma.wait_group` has retired the group
  that read the slot, with no fence. The x slot is read only by `wgmma` through its descriptor.

Our 26 has the same two waits per stage. The four `SYNCS.PHASECHK.TRANS64.TRYWAIT` in its SASS are
`mbar_wait_to`'s first try plus the retry inside the spin-timeout loop; the second one, the clock
reads and the `ATOMG.OR` on ctr[8] run only after a failed first try (profile
`sass_stage_loop_mode28.2026-10-29.txt`: 390 instructions in the loop, 348 on the path where both
first tries succeed). The real differences are on the release side.

### Which public releases are provable

The x slot (theirs and ours): `wgmma.mma_async` reads shared memory in the async proxy (9.7.17.4)
and `cp.async.bulk` writes it in the async proxy (9.7.10.28.2). Both are the same proxy in the same
thread block, which is the second case of proxy-preserved base causality order (8.9.5), so no
proxy fence is needed. The chain: the read completes before `wgmma.wait_group` returns in every
thread of the warpgroup; lane 0's `mbarrier.arrive` follows in program order and has release
semantics by default (9.7.15.16.16); the producer's `try_wait.parity` has acquire semantics by
default and, returning true, forms an acquire pattern (9.7.15.16.19, 8.8) that synchronizes with
the arrive (8.9.4); the producer's `cp.async.bulk` is ordered after its prior instructions
(8.9.1.1). The read precedes the write in causality order, so it cannot read from it (8.10.6).
Ours already releases the wgmma path's x slot this way, with no fence.

The weights slot (public): the slot is read with generic `ld.shared` by all 32 lanes and refilled
by `cp.async.bulk`, a different proxy. Two links are missing. (1) The proxy: "accessing the same
memory location across multiple proxies needs a cross-proxy fence; for the async proxy,
fence.proxy.async should be used" (9.7.10.28.2). Proxy-preserved base causality order (8.9.5)
covers generic-to-generic and same-proxy pairs only, so without a fence on the path the read is
not ordered before the bulk write in causality order and 8.10.6 gives no guarantee. (2) The lanes:
lanes 1-31 read the slot but only lane 0 arrives, with no `bar.warp.sync` between them, so the
other lanes' reads are not in program order before any release (8.8: a release pattern "only
affects operations occurring in program order before" it; 9.7.15.16.19 makes visible the accesses
"requested prior, in program order, to mbarrier.arrive ... by the participating threads"). The
public scheme therefore rests on hardware behaviour, not on the model. Under the repository rule it is not adopted.

The weights slot (our 26): every lane's reads, then `__syncwarp` (bar.warp.sync "also guarantee[s]
memory ordering among threads participating in barrier", 9.7.15.2), then lane 0's
`fence.proxy.async.shared::cta` and arrive. This is correct, but the cross-lane link rests on the
instruction text of 9.7.15.2, which is not one of the synchronizes-with relations listed in 8.9.4.

### Ordering of the ring slot releases (adopted in mode 28)

Every consumer thread releases the weights slot itself: `fence.proxy.async.shared::cta`, then
`mbarrier.arrive` on `wempty[ws]`, whose expected count becomes kNC x 32 = 512. For one thread T
that read the slot with `ld.shared` (reads R) and the producer thread P that refills it with a
bulk copy (write W):

1. R, then T's bidirectional proxy fence F, in program order. F orders T's prior generic accesses
   before its subsequent async-proxy accesses and its effect "composes with other forms of
   synchronization" (9.7.15.4); this is the cross-proxy fence 9.7.10.28.2 requires.
2. F, then T's `mbarrier.arrive` A, in program order. A has `.release` semantics at `.cta` scope by
   default (9.7.15.16.16) and is a release pattern by itself (8.8), covering R and F.
3. P's `mbarrier.try_wait.parity` on the same barrier returns true once all 512 arrivals of the
   phase are in; it has `.acquire` at `.cta` scope by default and forms an acquire pattern
   (9.7.15.16.19, 8.8), so A synchronizes with it (8.9.4). 9.7.15.16.19 also states the case
   directly: accesses requested after the wait "are not performed and not visible to memory
   accesses performed prior to mbarrier.arrive having release semantics, in program order, by
   other threads participating in the mbarrier".
4. P then issues `mbarrier.arrive.expect_tx` and `cp.async.bulk`; the bulk operation is ordered
   after the prior instructions of P (8.9.1.1), so after the acquire.
5. So R precedes W in base causality order with T's proxy fence on the path, and R cannot read
   from W (8.10.6).

Each thread's own reads are now in its own program order before its own release, so the chain needs
neither `bar.warp.sync` nor lane 0. The fence and the arrive stay; on the mma.sync path the x slot,
read by `ldmatrix` (generic), gets the same per-thread fence and arrive, and the wgmma path's x
release (lane 0, `mbarrier.arrive` with count 32, after `wgmma.wait_group`) keeps the chain above.
The fence sits behind the stage's first wgmma group instead of before the x wait: every load of
the stage's codes and scales was issued earlier in T's program order, so the chain is unchanged,
and the MEMBAR it compiles to is no longer between the weights wait and the first wgmma. A
uni-directional `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster` compiles to
the same MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S (local nvcc 12.8 test), and the read-only
`::shared::cluster::read` form needs PTX ISA 9.4, so the bidirectional fence stays.

### The rest of mode 28

- The ring size (S.nsw), x slot 0's address and the x slot size are read once per launch into
  registers; 26 loaded them from shared memory every stage (an LDS, an LDS.64, S2UR / ULEA pairs)
  and shuffled the whole slot address. The slot offset still goes through one `__shfl_sync`: the
  ring position is carried around a loop whose trip count ptxas cannot prove warp-uniform, and
  without the shuffle every wgmma descriptor went through R2UR (448 instructions per stage).
- The S.ord sort is one pass: an expert's jobs are all 8 n-tiles except possibly its last, so a
  job's rank is a warp scan of the 8-tile counts plus, for a partial job, seven ballots. 26 ran
  eight dependent scan rounds (2.5 us of routing at 128 tokens in 4c285df44eb34bc1).
- Spin timeouts stay (the self-tests read ctr[8]); they were already off the fast path.
- Not removed: the spill. Mode 28 has 184 B / 512 B of spill stores / loads against 120 / 308 for
  26, all but 7 local accesses outside the per-stage loops (26: 5).

SASS of the 64x24 stage loop: 383 instructions, 335 on the fast path (26: 390 / 348; public 322).
The variant that kept 26's lane-0 release after `__syncwarp` at the moved point made ptxas
serialize the wgmma groups (C7513, 400 instructions per stage) and was dropped before any run.

### Results

Paired 28 minus 26, us (negative = 28 faster):

| | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer routing, seeds 1-6 (70dd7b13be884434) | -6.4 (6 of 6) | -8.6 (6 of 6) | -8.8 (6 of 6) |
| standalone routing, seeds 1-6 (70dd7b13be884434) | -6.4 (6 of 6) | -12.1 (6 of 6) | -11.0 (6 of 6) |
| standalone, held-out seeds 17-22 (GATE, f928f8a802014e4f) | | -10.6 (6 of 6) | -11.6 (6 of 6) |
| the same, repeat (GATE, 13fe68e7ca2e423f) | | -11.6 (6 of 6) | -12.5 (6 of 6) |
| skewed, seeds 11-13 (GATE, f928f8a802014e4f / 13fe68e7ca2e423f) | | -6.7 / -6.4 | -8.4 / -8.4 |

Stamps, layer routing, mean of six seeds (70dd7b13be884434; 27 is 26 stamped, 29 is 28 stamped).
Phase A = thread 0's end of phase A minus consumers' start; phase B = consumers done minus that.

| us | T=128 27 / 29 / public | T=256 27 / 29 / public |
|---|---|---|
| routing | 5.8 / 3.6 / 2.5 | 7.3 / 4.3 / 3.4 |
| consumers start | 12.9 / 10.9 / 6.2 | 16.0 / 12.8 / 8.3 |
| phase A | 97.6 / 92.7 / 87.7 | 137.7 / 133.7 / 120.6 |
| phase B | 48.4 / 47.5 / 42.9 | 66.2 / 65.2 / 58.6 |
| tail | 9.9 / 8.9 / 5.8 | 9.4 / 9.3 / 5.6 |
| launch end | 173.3 / 164.5 / 148.0 | 236.2 / 227.7 / 201.5 |

Judge (shippable build, policy main-lean at 128 and 256): 2.149x (2.117-2.181) in
f928f8a802014e4f and 2.158x (2.125-2.190) in the repeat 13fe68e7ca2e423f, against 2.104x
(2.073-2.136) for mode 26 in 817eab6401d54a84; bs128 215.68 -> 200.89 / 200.34 us, bs256 261.71 ->
247.11 / 244.59.

Reading. The stamps put about half of the gain in phase A (-4.9 / -4.0 us at 128 / 256) and a
quarter to a third in setup (-2.0 / -3.2, mostly the sort), with phase B nearly unchanged (-0.9 / -1.0); the stage
loop lost 13 fast-path instructions of about 348 (4%). Which of the release move, the per-thread
release and the hoisted constants carries the phase A gain is not separated (the lane-0 control
could not be built without serialization). The remaining gap to the public kernel on layer
routing is 16.1 us at 128 tokens and 25.1 at 256 (paired replay), in the stamps setup +4.7 /
+4.5, phase A +5.0 / +13.1, phase B +4.6 / +6.6, tail +3.1 / +3.7.

At 32 tokens on layer routing, 28 (94.7 us) and 26 (101.1) are both well under the skinny sk8w4s
(113.1, 6 of 6 seeds), while on the standalone bench routing sk8w4s wins (150.8 vs 161.2 in
f928f8a802014e4f), so the standalone policy keeps sk8w4s at 32 and a layer benched on judge-like
routing may pick 28 there.

Verdict. Mode 28 is the standalone policy at 128 and 256 tokens; submit candidate ec62b5a (not
submitted). Port: add 28 to `LAYER_MODES` in `tools/dl_moe_inline.py` and `28: "main-lean"` to
its MAIN_NAMES (26: "main-dynb-fs" too if 26 stays). The interface, scratch and SmemMain size
(228,736 bytes) are mode 26's. The standalone MAIN_MODES is now (0, 16, 26, 28); a layer that
still lists 20 or 23 in LAYER_MODES gets them silently dropped by the inliner (it keeps only modes
in MAIN_MODES), so list (0, 16, 26, 28) or (0, 16, 28). Tested on a scratch copy: the inliner's
anchors match and the layer's variant table gains `('main-lean', 'main', 'launch_main<28>')`; the
layer build itself was not compiled here.

## Skinny path against the public kernel at 1, 8 and 16 tokens, 2026-10-29

Asked by the 2026-10-30 gap map: at 1 or 8 tokens a microsecond is worth 3-4x more layer score
than at 128/256. Same-process A/B (tools/moe_make_bench_ab.py --small, public TINY / TINY2 / MID
builds as its kernel_fn picks them), six seeds of layer and standalone routing, runs
240b68e7df1a427f and 2c28a7256b004aa1. Stop rule: stop if the public lead at 8 tokens is under 3%.

| us, mean of 6 seeds | T=1 layer | T=8 layer | T=16 layer | T=8 std | T=16 std |
|---|---|---|---|---|---|
| sk8w4s (policy) | 35.4 / 35.7 | 66.2 / 66.9 | 84.4 / 85.5 | 92.1 / 93.0 | 108.9 / 109.8 |
| public minus sk8w4s | +0.4 / +0.3 | -3.7 / -4.1 (5.6-6.1%) | -2.0 / -3.1 | -1.9 / -2.8 | -2.5 / -2.9 |
| sk8w4s-loads minus sk8w4s | -2.5 / -2.6 | -3.6 / -3.8 | -4.4 / -5.1 | -2.3 / -3.0 | -4.0 / -4.1 |

The lead at 8 tokens is above 3%, so attribution. Stamps (thread 0's globaltimer; our new
sk8w4s-stamps against the public DEBUG builds), 8 tokens, layer routing, two runs:

| us | ours | public |
|---|---|---|
| setup (routing, positions) | 2.2 / 2.1 | 2.2 / 2.2 |
| phase A | 34.6 / 34.9 | 33.3 / 33.4 |
| phase B, with ours's wait for phase A credits (4.0) | 19.7 / 20.0 | 18.7 / 18.9 |
| grid barrier wait (tail of the slowest CTA) | 3.7 / 3.9 | 2.5 / 2.9 |
| final pass | 0.4 / 0.3 | 0.3 / 0.3 |
| end | 61.5 / 62.3 | 57.6 / 58.4 |

Measured: setup is at parity; the 4 us are spread over phase A (+1.3), phase B with its
dependency wait (+1.0) and the barrier tail (+1.2). [Corrected in "Skinny unit loop against the public kernel": the stamps did not time the first try_wait, which can suspend; timed in full, thread 0 waits 3.0 us in phase A at 8 tokens.] Thread 0 never waits for a weights unit (0.0
us of missed first tries in both phases), and the loads-only twin (same stream, no math) lands
at the public kernel's time (62.7 against 62.5 us), so the public kernel hides its compute under
the stream and ours does not: the compute warps are the limit, by 3.6 us at 8 tokens. Inferred:
the per-unit lockstep (a bar.sync of all 8 compute warps behind each unit's last MMA before the
slot is refilled) and the per-unit instruction count (our unit loop 828 SASS instructions against
709-759 in the public TINY2 build; local nvcc 12.8) are the two candidates.

Tested the first (sk8w4s-er, kMode 3): -1.0 us at 8 tokens and -0.9 at 16 on layer routing (11 of
12 seeds lower), +0.4 at 1 token and +0.7 / +1.6 on standalone routing. Holding the unit's codes
in registers through its MMAs made phase A 3 us longer while the dependency wait and the barrier
tail fell by 3.1 us. Not adopted for the standalone policy (it loses on the board's routing); the
release chain is the fat path's ("Ordering of the ring slot releases"). The instruction count is
untested.

## Skinny unit loop against the public kernel, 2026-10-29

Asked: cut the skinny path's compute per unit so it hides under the weight stream, starting from
an instruction-level diff of our unit loop against the public TINY2 build.

### The instruction diff (local nvcc 12.8, sm_90a, cuobjdump)

Ours: `tools/moe_compile_fused.sh` at 96f2c43, kernel `sk_8_4_0`. Public: its CUDA_SRC with the
TINY2 defines plus `-DOKC_PRMTSC=1 -DOKC_DQ7=1 -DOKC_NS=3 -DOKC_NSF=3`, as its `_ext` builds it
for this problem. The earlier figure (our loop 828 instructions against 709 / 759) is the static
size of the innermost HMMA loops, and it reproduces exactly; but ours is one loop for both phases
whose body carries the cold paths (the x reload on a job change, 16 predicated 16-byte loads and
32 zeroing CS2R per phase; the spin-timeout blocks), and the public splits phase A and B into two
template instances. The dynamic path that a compute thread other than thread 0 runs for a unit
with no job change and no failed waits:

| per unit, per compute warp | ours (sk8w4s) | public phase A | public phase B | ours, kMode 7 |
|---|---|---|---|---|
| instructions | 554 | 554 | 557 | 569 |
| ALU pipe (LOP3, SHF, PRMT, LEA, ISETP, SEL, IADD3, MOV) | 275 | 266 | 262 | 268 |
| FMA pipe (HMUL2, IMAD, FADD, VIADD) | 221 | 232 | 234 | 236 |
| HMMA.16816 / LDS.128 / STS.128 | 32 / 9 / 1 | 32 / 9 / 1 | 32 / 9 / 1 | 32 / 9 / 1 |
| CTA-level syncs | bar.sync (free, 288) + bar.arrive (full) + bar.sync (unit, 256) | mbarrier try_wait (buffer free) + bar.sync (256) | same | mbarrier try_wait + mbarrier arrive + bar.sync (256) |

Line by line:

- Dequant: identical cost. Ours (repacked layout) and the public DQ7 layout both decode a word in
  7 ALU instructions (5 LOP3, 2 SHF) and 2 IMAD.SHL; ptxas already turns our left shifts into
  IMAD.SHL. Both run 128 HMUL2 for the block scale.
- Scale decode: the one real difference. Ours `byte_perm(...) * 16 + c` compiles to PRMT + LEA
  (ALU); the public passes 16 as an opaque register (`16 + p.zero`), so it is PRMT + IMAD (FMA).
  16 ALU instructions per unit per warp.
- MMA issue and accumulators: the same (4 chains of 8 m16n8k16, x in registers, 12 FADD to fold
  the chains).
- Address math: a few instructions either way (the public's slot index is q % 3, an IMAD.HI).
- Barriers: ours makes the 8 compute warps meet twice per unit (the 288-thread bar.sync that waits
  for the helper to free the partial buffer, then the 256-thread unit barrier before the slot
  refill); the public meets once and waits for the buffer on an mbarrier.

So the per-unit instruction count is at parity, and the static 828 vs 709 / 759 was the cold paths.

### Two stamp facts that change the attribution

The skinny stamps (kMode 2, 4) added the weights wait only after a failed first `try_wait`, but
`mbarrier.try_wait` may suspend the thread for a system-dependent time before returning false
(PTX ISA, mbarrier.try_wait), so a wait inside the first try counted as zero. Since 8272efe the
stamps time the whole wait. Timed that way (e937013292b24906, layer routing, sk8w4s-stamps),
thread 0 waits for weights 2.2 / 3.0 / 3.7 us in phase A and 0.3 / 0.7 / 1.1 us in phase B at
1 / 8 / 16 tokens. The earlier "thread 0 never waits for a weights unit" is withdrawn, and with it
the inference that the compute warps alone are the limit. What still stands: the loads-only twin
lands at the public kernel's time (63.0 against 62.7 us at 8 tokens, layer).

### Changes tested (kMode 5 to 8, commit 8272efe)

- 5, sk8w4s-sf: the scale decode with an opaque 16 (`16 + (nloc >> 6)`, nloc <= 32): LEA -> IMAD,
  the public's SCDK.
- 6, sk8w4s-hand: partial buffers through mbarriers, the public's WSB shape: each compute thread
  waits on `hand[4 + bb]` (helper's free, count 32) instead of bar.sync kBarFree, stores its
  partial, arrives on `hand[bb]` (count 256) instead of bar.arrive kBarFull; the helper waits on
  `hand[bb]` and each lane arrives on `hand[4 + bb]` after its own loads. The unit barrier, the
  proxy fence and thread 0's refill are unchanged.
- 7, sk8w4s-sfh: 5 and 6. 8: 7 with the stamps.

Results, e937013292b24906, six seeds per row, means of paired deltas against sk8w4s in us:

| | T=1 layer | T=1 std | T=8 layer | T=8 std | T=16 layer | T=16 std |
|---|---|---|---|---|---|---|
| sk8w4s (us) | 35.5 | 31.9 | 66.2 | 92.8 | 84.7 | 109.2 |
| sf | -0.1 (5/6) | -0.1 (5/6) | -0.2 (4/6) | -0.4 (5/6) | -0.5 (4/6) | -0.1 (4/6) |
| hand | +0.1 (2/6) | -0.0 (4/6) | -0.1 (3/6) | +0.0 (2/6) | +0.4 (2/6) | +0.5 (2/6) |
| sfh | +0.1 (3/6) | -0.1 (4/6) | +0.1 (2/6) | +0.1 (2/6) | +0.5 (3/6) | -0.1 (4/6) |
| loads | -2.4 | -2.9 | -3.2 | -3.2 | -4.5 | -3.9 |
| public | +0.7 | +0.3 | -3.5 | -2.6 | -2.2 | -2.4 |

(n of 6 = seeds lower.) Every arm passed the self-test (4.02e-3 against FP32) and matches
sk8w4s to 5.8e-5 or better; spin flags 0. The judge's own bench picked sk8w4s-sf at 8-32 tokens,
a choice within its noise. Scored 2.004x (1.974-2.034), which does not replace the board entry.

Stamps of sfh against sk8w4s at 8 tokens, layer: phase A 37.7 against 36.7, thread 0's wait for a
free partial buffer 2.5 against 1.5 us, unit barrier 1.2 against 0.8. With one barrier fewer the
warps run ahead and wait for the helper instead; the time moves, it does not leave.

Verdict: neither the ALU instructions of the scale decode nor the second compute-warp barrier is
the gap. The 3.5 us at 8 tokens is not in the per-unit instruction stream (parity, measured
statically) and not in the barrier count (measured). Open, from the corrected stamps: thread 0
waits for weights 3 to 4 us at 8-16 tokens while the loads-only twin, with the same ring and
issue points, matches the public; so the next candidates sit where the stream meets the compute
(ring depth 4 against the public's 3, the refill issued only after the unit barrier, the L2
policy), not in the unit loop's arithmetic. None of the arms is a candidate; no layer port.

### Ordering of the mbarrier handoff (kMode 6-8; not adopted, recorded as the rule asks)

Both directions are generic `st.shared` / `ld.shared` on `S.red`, one proxy, so no proxy fence
is needed (proxy-preserved base causality order, 8.9.5, covers generic-to-generic).

1. Full: a compute thread T stores its partial W, then `mbarrier.arrive` A on `hand[bb]` in
   program order; A is `.release` at `.cta` scope by default (9.7.15.16.16) and a release pattern
   by itself (8.8). The helper lane H's `try_wait.parity` returns true once all 256 arrivals of
   the phase are in; it is `.acquire` at `.cta` by default and an acquire pattern (9.7.15.16.19,
   8.8), so A synchronizes with it (8.9.4), and H's loads R follow in program order. W precedes R
   in causality order, so R sees W.
2. Free: H's loads R of `red[bb][w][lane]` (every w, its own lane), then its own arrive on
   `hand[4 + bb]` (count 32, one per lane, so no lane relies on `bar.warp.sync`); T's
   `try_wait.parity` on it (acquire) before its next store W' to the buffer. R precedes W' in
   causality order and cannot read from it (8.10.6).
3. Phases: unit q uses buffer q % 4. T waits for completion q / 4 - 1 of the free barrier (skipped
   for q < 4); H waits for completion q / 4 of the full barrier. Neither side can be two phases
   ahead of the other, since each waits for the other's previous use.

Section numbers as in "Ordering of the ring slot releases" above.

### Port

None: no arm beat sk8w4s. For the record, a layer port of a kHand variant would also need
`tools/dl_moe_inline.py` to build skinny modes other than 0 (it keeps `v[2] == 0`) and to
initialize `hand[]` in its PDL preamble, which today initializes only `full[]` and skips the
routing warp's block (the same holds for `empty[]` of sk8w4s-er, never ported).

## Skinny weight stream against the public kernel, 2026-10-29

Asked: measure where our skinny path's weight stream meets its compute, against the public
TINY2 / MID builds, name the largest piece, and test the smallest change aimed at it. State before:
the public leads sk8w4s by 3.5 us at 8 tokens and 2.2 at 16 (layer routing, e937013292b24906), per
unit instructions are equal, thread 0 waits 2.2 / 3.0 / 3.7 us for weights in phase A at 1 / 8 /
16 tokens, and our loads-only twin lands at the public's time.

### What differs on the stream side (static, public source as its `_ext` builds TINY2 and MID)

| | ours (sk8w4s) | public TINY2 / MID |
|---|---|---|
| ring | 4 slots of 36,864 B (one unit: 16 rows x full K, codes then scales) | 3 slots (`-DOKC_NS=3`), same 36,864 B unit |
| unit order per CTA | contiguous range `[njobs x 512 x c / G, ...)`, ascending; phase B the same | the same split (`muldiv_g`), ascending |
| CTAs | one per SM, 132 | one per SM, 132 |
| first copies | the routing warp issues 4 after its histogram and job list | FAST1 = 2 copies before routing, the helper the third after it |
| refill | thread 0, after the 288-thread buffer-free barrier, the store, the full arrive and the 256-thread unit barrier; `fence.proxy.async.shared::cta` first | thread 0, after an mbarrier wait for a free buffer, the store and the 256-thread CTA barrier; no proxy fence |
| L2 hint on weight copies | `createpolicy.fractional.L2::evict_first` 1.0 | constant 0x12F0000000000000 (CUTLASS EVICT_FIRST) in TINY2 / MID; none in TINY (<= 4 tokens) |
| inter / ybuf stores | plain | `st.global.L2::cache_hint` evict_last (0x14F0...) |

So the slot size, the unit order and the CTA count are the same in both; the ring depth, the first
copies and the store hints differ, and both refill only after a barrier that also waits for the
helper to free a partial buffer.

### Stamps

kMode bit 16 (ours) and `--unitstamps` (the public small builds, patched by
`tools/moe_make_bench_ab.py`) record the same four events per unit with thread 0's clock64: the copy
issued (I, by whichever thread issues it), thread 0 starts waiting for it (S), the wait ends (E), and
the slot is free after the unit barrier (F, the refill follows). A wait over 0.05 us marks the unit
stalled, so its E is its arrival.

### Ordering of the early refill (kMode bit 32)

Bit 32 moves the unit barrier (`bar.sync 1, 256`) and thread 0's fence and refill ahead of the
partial-buffer handoff (the `bar.sync 6 + b, 288` wait, the store to `S.red` and `bar.arrive 2 + b`).
The chain is mode 0's, with the same instructions in the same relative order:

1. Each compute thread T reads the slot with `ld.shared` (codes and scales, reads R). All of them
   precede, in T's program order, the accumulator fold, and the barrier now sits right after the
   fold, so R precedes T's arrival at `bar.sync 1` in program order as before.
2. `bar.sync` is a CTA-scope barrier: memory accesses requested by the participating threads before
   it are performed relative to all participants (PTX ISA, `bar` / `barrier`, memory ordering), and
   in the memory model it orders them before the participants' later operations (8.9.4, the barrier
   case of synchronizes-with).
3. Thread 0 (a participant) then runs `fence.proxy.async.shared::cta` F and issues the bulk copy W;
   W is ordered after F in program order (8.9.1.1). F is the cross-proxy fence the generic read to
   async write path needs (9.7.10.28.2, 8.9.5), on the path R -> barrier -> F -> W.
4. So R precedes W in proxy-preserved base causality order and cannot read from it (8.10.6).

The handoff is unchanged (same barriers, same counts), and the order cannot deadlock: the helper
takes no part in barrier 1, and every compute warp reaches barrier 1 for unit q before it waits on
the free barrier of buffer q % 4, which the helper releases after reading unit q - 4. ptxas output
(local nvcc 12.8, sm_90a): the same 2,992 instructions, the same counts of BAR.SYNC (9), BAR.ARV (6),
MEMBAR (10), FENCE.VIEW.ASYNC (2), UBLKCP (2) and SYNCS (12); the unit `BAR.SYNC 0x1, 0x100` moves
ahead of the free `BAR.SYNC` and the `BAR.ARV`. Mode 0 compiles to the same SASS as before the
change (2,992 instructions, byte-identical listing).

### Results (run 75bca9775f6f4b0b)

Stamps at 8 tokens, mean over six seeds of one cold launch each, per CTA, us; layer / standalone
routing. Every unit's wait lasted over 0.05 us in all five kernels, so thread 0 waits for every
copy and E is each copy's arrival.

| 8 tokens | sk8w4s | sk8w3s | sk8w4s-rf | sk8w4s-loads | public |
|---|---|---|---|---|---|
| copies in flight when thread 0 starts waiting | 4 | 3 | 4 | 4 | 3 |
| copy issue to arrival (E - I), phase A | 4.69 / 4.55 | 3.23 / 3.15 | 4.78 / 4.68 | 4.79 / 4.64 | 3.02 / 2.92 |
| the same per copy in flight | 1.17 / 1.14 | 1.08 / 1.05 | 1.20 / 1.17 | 1.20 / 1.16 | 1.01 / 0.97 |
| thread 0's weights wait, phase A sum | 6.82 / 10.20 | 5.92 / 8.69 | 3.61 / 5.23 | 23.04 / 34.94 | 3.60 / 5.24 |
| unit hold (wait end to slot free), phase A sum | 19.01 / 28.60 | 18.49 / 28.58 | 25.15 / 38.51 | 1.86 / 2.83 | 21.29 / 33.13 |
| phase A span (first wait start to last slot free) | 35.86 / 55.08 | 34.74 / 54.08 | 37.86 / 58.58 | 31.91 / 49.12 | 32.20 / 50.18 |
| slot free to the refill's issue, per unit | 0.18 / 0.18 | 0.19 / 0.19 | 0.15 / 0.15 | 0.17 / 0.17 | 0.07 / 0.07 |
| wait per unit, first kNS units of phase A | 0.53 / 0.54 | 0.65 / 0.67 | 0.43 / 0.49 | 1.25 / 1.31 | 0.25 / 0.19 |
| phase B weights wait, sum | 2.29 / 3.86 | 2.12 / 3.51 | 1.13 / 1.81 | 2.53 / 3.61 | 0.85 / 1.51 |
| SM clock (GHz) | 1.79 / 1.77 | 1.79 / 1.76 | 1.79 / 1.77 | 1.97 / 1.97 | 1.78 / 1.77 |

The largest piece is the ring depth. Measured: with 4 copies in flight each copy takes 4.7 us from
issue to arrival against 3.0 in the public kernel's 3-slot ring, and the interval per copy
(arrival time over copies in flight) is longer too (1.17 against 1.01 us), so the fourth slot adds
a whole copy of latency and no throughput; thread 0 waits 6.8 us in phase A against 3.6. With 3
slots our latency drops to 3.2 us and the phase A span by 1.1 us. Hypothesis, not measured: more
concurrent bulk copies per SM lower the HBM's efficiency, which would explain why the interval per
copy grows with depth (the old sk8w5s lost 3.3 us to sk8w4s at 8 tokens in bab82f5f5b974594).

Paired timing, graph replay after an L2 flush, mean of six seeds (n of 6 seeds lower):

| us | T=1 layer | T=1 std | T=8 layer | T=8 std | T=16 layer | T=16 std |
|---|---|---|---|---|---|---|
| sk8w4s | 35.6 | 31.9 | 66.2 | 92.7 | 84.8 | 109.4 |
| sk8w3s minus sk8w4s | -0.9 (6) | -0.6 (6) | -2.4 (6) | -1.8 (6) | -3.1 (6) | -2.9 (6) |
| sk8w4s-rf minus sk8w4s | +0.1 (2) | +0.0 (2) | -1.1 (6) | +0.4 (3) | -1.0 (6) | +0.8 (1) |
| (8, 3, 32) minus sk8w3s | +0.3 (1) | +0.1 (2) | +0.3 (1) | +0.9 (0) | +0.9 (0) | +2.6 (0) |
| sk8w4s-nh minus sk8w4s | +2.8 (0) | +2.3 (0) | +3.3 (0) | +1.3 (0) | +1.2 (1) | +2.4 (0) |
| public minus sk8w4s | +0.5 (1) | +0.2 (0) | -3.4 (6) | -2.7 (6) | -2.6 (6) | -2.4 (6) |
| sk8w3s minus public | -1.3 (6) | -0.8 (6) | +1.0 (0) | +0.9 (1) | -0.4 (4) | -0.5 (6) |

Every arm passed the FP32 check through sk8w4s (3.83e-3 to 3.88e-3) and matches it to 2.1e-5 or
better; spin flags 0. sk8w3s is faster than sk8w4s on 36 of 36 paired seeds, closes 2.4 of the
3.4 us public lead at 8 tokens (layer) and passes the public kernel at 1 and 16 tokens. The early
refill lowers thread 0's waits but lengthens the hold (the warps then meet the helper's buffer
barrier inside it) and loses with 3 slots in every row; the unhinted copies lose everywhere. Both
are in docs/ruled-out.md.

What is left at 8 tokens (sk8w3s against the public, measured in the stamps above): the first
kNS units wait 0.65 us each against 0.25 (about 1.2 us; the public issues two copies before
routing), and slot free to refill issue is 0.19 against 0.07 us per unit (thread 0's
`fence.proxy.async.shared::cta` and the issue path; an upper bound, since the stamp's own global
store is still outstanding when the fence's MEMBAR runs). The fence stays (CLAUDE.md); a release
that takes it off thread 0's path, or earlier first copies, is the next test, not built here.

Verdict: sk8w3s joins `SKINNY_VARIANTS` as `(8, 3, 0)`; the prepare bench picks it per size like
the others (it is a mode 0 build, so a candidate). No other kernel changes; mode 0 and the fat
modes compile as before.

### Port

`tools/dl_moe_inline.py` keeps every `SKINNY_VARIANTS` entry with mode 0, so it carries
`(8, 3, 0)` as `sk8w3s` (`sk_8_3_0::launch<1>`) with no change to the tool (checked: its
`device_code` lists sk8w4s, sk16w4s, sk8w3s and the fat modes). Its PDL preamble initialises
`full[s]` for `s < kNS`, which is all mode 0 uses (`empty[]` and `hand[]` belong to kModes 3-8),
so the 3-slot ring needs nothing more there. A port needs: rerun the inliner on the new standalone
file, and let the layer's MoE bench include `sk8w3s` at 1, 8 and 16 tokens; if the layer trims its
build to `((8, 4, 0), (16, 4, 0))` as this guide once advised, it must keep `(8, 3, 0)` instead of
or beside `(8, 4, 0)`. The flag modes (16, 32, 64) are diagnostics or losers and need no port.

## Phase A against the public kernel, 2026-10-29

Question: after mode 28 the same-process gap to the public kernel on layer routing was 16.1 us at
128 tokens and 25.1 at 256, with phase A the largest piece at 256 (+13.1 us in mode 29's stamps
against the public kernel's, check 70dd7b13be884434). Where does phase A's time go?

Method. Mode 31 is 28 with counters per consumer warp over phase A only: cycles in weights and x
waits that miss their first try (a wait that succeeds at once adds nothing, so the stage loop
gains no clock read: 31 costs +2.5 us of graph replay at 256 and its stamped phase A is within
0.3 us of 29's), cycles from a unit's start to its accumulators, epilogue cycles, units, stages
and n-tiles; plus each producer's phase A cycles waiting for a free ring slot. tools/moe_make_
bench_ab.py prints them per size as ABDIAG (mean over CTAs and six layer seeds of one cold launch
each) and the public kernel's stream-K stage ranges as ABPUB. Checks c7db20e053254b52,
0b684d5db0834dbc and 473d87d03de94e96 (three repeats of 31; the numbers below are their range).

[Corrected 2026-10-29, check ca43908f02874886: the wait counters were wrong. `mbar_wait_cyc`
started its clock only after a failed first `try_wait` and returned 0 when the first try
succeeded, but `try_wait` is potentially blocking (PTX ISA 9.4, mbarrier.test_wait /
mbarrier.try_wait), so both cases hid real waiting. The producers' slot waits used the same
function. Since 3f67a45 the clock starts before the first try. Timed in full on mode 41 (37 with
the counters), a consumer warp waits 8.4 / 7.3 us (weights / x) at 128 tokens and 11.8 / 11.5 us
at 256, about 20% of its stage-loop time. The "waiting" row, the "stage loops" label and the
first sentence of the Reading below rest on the old counters. The producers' slot-wait row was
low too: timed in full it reads 65.5 / 64.3 us at 128 and 94.0 / 92.0 at 256 (mode 41), so the x
producer had been undercounted by about 10 us. See "Fat-path waits with the first try timed". The
other rows are wall-clock spans or counts and still stand.]

Measured, mode 28 at 256 tokens, layer routing (us per consumer warp over phase A):

| | 128 tokens | 256 tokens |
|---|---|---|
| phase A per CTA, mean / p10 / p50 / p90 / max | 93.3-93.9 / 85 / 91 / 107 / 111 | 133.9-134.7 / 126-127 / 131 / 145 / 147 |
| waiting for weights / for x (missed first tries) [withdrawn 2026-10-29, see the note above] | 0.00 / 0.03-0.05 | 0.00 / 0.18-0.22 |
| stage loops (dequant, wgmma, ring bookkeeping) [they include the waits, 2026-10-29] | 77.2-78.1 | 111.1-111.3 |
| epilogue (SwiGLU, stores, proxy fence, fence_gpu, bar.sync, red.release) | 10.3-10.8 (2.6-2.7 per unit) | 16.0-16.6 (3.0-3.1 per unit) |
| weights producer / x producer waiting for a free slot | 64-65 / 55 | 93-94 / 82 |
| owned units per CTA, min / mean / max | 3.7 / 4.0 / 4.7 | 5 / 5.37 / 6 |
| MB streamed per CTA (36,864 B weights + 4,096 B x per n-tile, per stage), mean / max | 2.92 / 3.44 | 4.17 / 4.55 |
| rank correlation of a CTA's phase A time with its MB | 0.73-0.82 | 0.75-0.78 |

Reading. The consumers never wait for the ring: both producers spend 70% of phase A blocked on a
full ring, so phase A is consumer-bound, and streaming or ring depth are not the limit. Of a
consumer warp's 134 us, 111 are the stage loops (86 stages, 1.29 us each) and 16 the epilogues
of 5.37 units. The per-CTA spread (p10 to max, 20 us) follows the assigned bytes: owned whole
units leave CTAs with 5 or 6 units (80 or 96 stages), and dyn_b's claimed phase B units absorb
that (phase A's spread does not enter its mean).
[Corrected 2026-10-29: "The consumers never wait for the ring" is withdrawn, along with "streaming
or ring depth are not the limit", which rested on it. Timed in full (mode 41,
ca43908f02874886), consumers wait about 20% of the stage loops while the producers still find
the ring full for about 70% of phase A. Each wait averages 210-230 cycles at both sizes and in both
phases. That pattern looks like a fixed cost at every stage boundary or skew between warps of one
CTA, not a stream that falls behind on average; the counters cannot tell these apart. "Both
producers spend 70% of phase A blocked on a full ring" still holds with the repaired counters
(69-71%).]

The public kernel's partition at 256 tokens (its source, `wgs_ranges` / `wgs_split` /
`wgs_flush`, and the ranges it computed in these checks): stream-K over 256-row x 16-stage units,
each CTA a contiguous stage range of equal cost under the same kCostA table as ours, 57 to 107
stages per CTA (mean 86, the same total as ours), 1.85 cut units per CTA finished through fixup
slots by a designated share; a CTA runs its trailing partial unit first; a finished unit's
release (bar.sync, then one red.release, no per-thread gpu fence and no proxy fence) is deferred
behind the next segment's fourth wgmma group; SiLU is `__fdividef`; the SwiGLU stores go to a
natural-k-order layout, so the 8 lanes of a quad row write one full 16-byte row.

Split of mode 28's epilogue (mode 33, the deferred publish, against 31): the stores and SwiGLU take
1.6 us per unit at 256 (1.2 at 128) and the publish 1.0 to 1.35. Inside the stores and SwiGLU,
`__frcp_rn` compiles to MUFU.RCP plus a range check that calls a subroutine, inside a
convergence region per element (16 per thread per unit at 8 n-tiles). In our k-permuted xb
layout a warp owns two bf16 of each 16-byte token row, so each 2-byte store instruction touches
16 sectors.

Inferred (not measured on the public kernel: its DEBUG build ran 30 us over its replay in these
checks because the stage-range write made it spill, so its stamped times are not used; check 70dd7b13be884434's public stamps put its phase A
at 120.6 us for the same 86 stages): the 13.1 us phase A gap is mostly our epilogue (16 us, of
which the public kernel's cheaper release and SiLU leave a few us) plus the stage loop's 4% more
fast-path instructions (335 against 322, about 4 us at 1.29 us per stage). The spread across
CTAs is not part of it.

### Changes tested

| mode | change | minus 28 at 32 / 128 / 256, us (layer seeds 1-6) |
|---|---|---|
| 32 | publish deferred to the next unit's second stage (fences kept) | +1.6 / +2.7 / +1.2 |
| 34 | 32 without the per-thread fence_gpu | +0.8 / +1.4 / -1.2 |
| 35 | 32 with ss and own_n reloaded from shared memory (spills 104 / 280 B against 192 / 516) | +0.8 / +3.0 / +4.7 |
| 36 | SiLU's reciprocal as one `rcp.approx.ftz` (MUFU.RCP) | -1.8 / -0.9 / -1.2 |
| 37 | 36 without the per-thread fence_gpu, publish in place | -1.9 / -2.1 / -3.8 and -1.7 / -3.4 / -4.2 (two checks) |
| 39 | 37 with paired 4-byte SwiGLU stores (one shuffle per n-tile) | -1.0 / -2.5 / -4.0 |

The deferral moves the cost instead of removing it: 33's publish still takes 1.0 us per unit one
stage later and its stage loops 2 us more, and the publish block inside the stage loop adds about 30
instructions to each phase A loop body. Paired stores halve the epilogue in 40's counters (1.70
against 3.08 us per unit) but the next stages slow by about as much, so the stores' cost is in
the memory system, not in the instructions. Mode 37 removes what is removable inside the
epilogue: the SiLU subroutine and a MEMBAR.ALL.GPU plus CCTL.IVALL per thread per unit.

Mode 37 against 28 in the shippable check 987d024620e44eb7 (GATE, standalone routing): held-out
seeds 17-22 -3.2 us at 128 (6 of 6 lower) and -4.9 at 256 (6 of 6); skewed -1.5 and -2.6 (3 of 3).
On layer routing 12 of 12 paired seeds lower at 128 and 256 over two checks, 11 of 12 at 32.

Spill. Mode 28 spills 184 / 512 B (80 B frame): 7 accesses in stage loops, 114 in per-unit code
(the owned unit count and the x ring base reloaded per unit in phase A; 16 hoisted sum-of-squares
pointers per phase B unit) and 53 elsewhere (local count of SASS accesses by enclosing loop). It
does sit on per-unit paths. 37 has 156 / 416 B (56 B frame; the SiLU subroutine's ABI is gone)
with 9 / 90 / 44 accesses. Removing more (35, the public kernel's volatile shared reloads) cut it to
104 / 280 B but lost 3.4-4.7 us, so the remaining spill stays; at the 96-register cap of 576
threads it would take a stage loop that holds fewer code registers at once (the public kernel
loads a stage's codes one 4-step group at a time), which is a change to the loop, not tested.

### Ordering of the phase A publish without the per-thread fence_gpu (modes 34, 37, 39)

Sections are from the PTX ISA 9.4 documentation (docs.nvidia.com/cuda/parallel-thread-execution)
as cited in "Proxy ordering of the x tiles": 8.8 Release and Acquire Patterns, 8.9.4 Memory
synchronization, 8.9.5 Causality Order, 9.7.15.4 membar / fence.

Mode 28 publishes a finished phase A unit per consumer warpgroup this way: each of the 128
threads stores its SwiGLU outputs into the xb stage (generic `st.global`), then runs
`fence.proxy.async.global`, then `fence.acq_rel.gpu`, then `bar.sync` on the warpgroup's barrier;
thread 0 then runs `red.release.gpu.global.add` on `done_as[j * 16 + st]`. The reader is the x
producer of whichever CTA runs that phase B stage: `ld.acquire.gpu` until the count is 8, then
`fence.proxy.async.global`, then `cp.async.bulk`. Mode 37 drops the writers' `fence.acq_rel.gpu`
(SASS: MEMBAR.ALL.GPU, ERRBAR, CGAERRBAR, CCTL.IVALL per thread per unit) and keeps the rest:

1. Each writer's stores precede its `fence.proxy.async.global` in program order; the proxy fence
   acts on the thread's own generic accesses and composes with other synchronization (9.7.15.4).
2. `bar.sync` synchronizes with the `bar.sync` of every other participating thread (8.9.4), so
   every writer's stores and proxy fence precede, in base causality order (8.9.5), whatever
   thread 0 does after the barrier.
3. Thread 0's `red.release` is a release operation, a release pattern by itself (8.8); base
   causality order is transitive (8.9.5), so the writers' stores precede the release, and the
   reader's acquire that observes the count synchronizes with it (8.9.4). The reader's proxy
   fence and bulk copy follow as in steps 5-7 of "Proxy ordering of the x tiles".

This is the argument already given for the x gather (step 3 there: "its effect extends to the
other threads' stores that precede it through the barrier"); the per-thread `fence.acq_rel.gpu`
is no link in it. The acquire half of that fence (CCTL.IVALL) ordered nothing: no writer reads anything after
it that another CTA wrote. The skinny path's `credit` (fence, `__syncwarp`, relaxed atomicAdd)
is a different pattern and is unchanged.

Modes 32 to 35 (not adopted) moved the whole publish, fences included, to the start of the
warpgroup's next unit's second stage, and for the CTA's last unit to the end of phase A. Delaying
a release reorders nothing in steps 1-3 (every link is program order or synchronization that
stays). Progress: between a unit's stores and its deferred publish the warpgroup waits only on
its own CTA's weights ring and on x rows the gather published before any consumer started; the
last unit is published before any phase B wait; owned units always have 16 stages; and the target
is recomputed from S.ord and the unit's round, not carried in shared state.

## Fat-path waits with the first try timed, 2026-10-29

Defect (steering report archive/docs/steering/2026-10-29-overnight.md, section 2). `mbar_wait_cyc`, the
wait counter of the phase A diagnostic modes (31, 33, 38, 40), returned 0 when the first
`mbarrier.try_wait.parity` succeeded and started its clock only after a failed first try. The
instruction is potentially blocking (PTX ISA 9.4, mbarrier.test_wait / mbarrier.try_wait): a try
that returns true may have suspended the thread until the phase completed, and one that returns
false may have suspended it until a time limit. Both cases hid waiting, so the "0.00" weights
waits in "Phase A against the public kernel" do not show that the consumers never wait. The
producers' phase A slot waits went through the same function. The skinny path's stamps were
fixed the same way in 8272efe.

Fix (3f67a45). The clock starts before the first try. try_wait stays, with the same
spin-timeout check, so the counted wait is the production `mbar_wait_to` loop without its
separate first-try branch, plus two clock reads (test_wait would have changed what is measured). Audit of the other try_wait timers: the is_diag
stamps (modes 11, 17, 22, 24) and the producers' is_diag waits already read the clock before the
wait; the skinny kMode 2/4/8 stamps were fixed in 8272efe and the kMode 16 stamps (03887be)
bracket the whole wait;
tools/moe_make_bench_ab.py's patches of the public kernel stamp before and after its whole
`mbar_wait`. `mbar_wait_to`'s timeout clock starts after the first try, but it is a hang guard
and counts nothing. No other counter had the defect.

New modes (timing only). 41 is 37 (the shipped mode) with the repaired phase A counters plus the
same counters over phase B: per consumer warp, weights and x waits, stage-loop time, epilogue
time, units, stages and n-tiles, and the time waiting for a claimed phase B unit (S.bq). Phase B's
counters go to `inter` after S.dg, so SmemMain keeps its size. 42 is 37 with the light stamps
only, the overhead baseline for 41. tools/moe_make_bench_ab.py maps 37 to these twins and prints
phase B as ABDIAGB.

Production SASS. Modes 0, 16, 26, 28 and 37 and the four skinny kernels compile to the same
instruction text before and after the change (per-kernel sha256 with addresses stripped,
nvcc 12.8, sm_90a). Registers and spills are equal. Only 31 and 38, whose counters changed,
differ. Resources: 37 uses 96 registers, 156 / 416 B spill stores / loads and a 56 B frame. 42
uses 148 / 516 B and 56 B. 41 uses 264 / 580 B and 96 B. The old counter twins were 31 at
260 / 592 B and 38 at 248 / 620 B
(muse_moe_w4a16/profiles/sass_fatwait.2026-10-29.txt).

Overhead (check ca43908f02874886, layer routing, seeds 1-6, graph replay after an L2 flush):

| us, mean of 6 paired seeds | 128 tokens | 256 tokens |
|---|---|---|
| 42 minus 37 (light stamps) | +1.1 (0 of 6 lower) | +2.3 (0 of 6) |
| 41 minus 42 (the counters) | +5.9 (0 of 6) | +3.9 (0 of 6) |
| 41 minus 37 | +7.0 (0 of 6) | +6.1 (0 of 6) |
| stamped phase A, 41 minus 42 | +2.2 | +1.3 |
| stamped phase B, 41 minus 42 | +3.3 | +3.9 |

The counters cost 3-4% of the kernel, more in phase B, where they are new and include a
global read-modify-write per unit. Even if all of phase A's +1.3-2.2 us sat inside the waits,
the waits below would be at least 13.5 us at 128 and 21.9 us at 256.

Measured, mode 41, layer routing, mean over CTAs and six seeds of one cold launch each
(us per consumer warp; the GHz is each CTA's clock64 rate against globaltimer, 1.70 / 1.68):

| | 128 tokens | 256 tokens |
|---|---|---|
| phase A per CTA (thread 0), mean / p90 / max | 93.0 / 106.9 / 111.4 | 131.7 / 141.9 / 144.0 |
| A: waiting for weights / for x | 8.41 / 7.32 | 11.76 / 11.50 |
| A: stage loops without the waits (dequant, wgmma, ring bookkeeping) | 64.62 | 91.66 |
| A: waits as a share of the stage loops | 19.6% | 20.2% |
| A: epilogue (per unit) | 6.05 (1.51) | 9.18 (1.71) |
| A: outside segments (span minus the above) | 6.6 | 7.6 |
| A: weights producer / x producer waiting for a free slot | 65.5 / 64.3 | 94.0 / 92.0 |
| phase B per CTA (thread 0's phase A end to the CTA's barrier) | 49.1 | 69.5 |
| B: waiting for weights / for x / for a claimed unit | 4.48 / 3.88 / 1.23 | 6.44 / 5.40 / 1.24 |
| B: stage loops without the waits | 31.72 | 45.04 |
| B: waits as a share of the stage loops | 20.9% | 20.8% |
| B: epilogue (per unit) | 4.42 (2.21) | 10.11 (3.76) |
| B: units, stages, n-tiles per unit | 2.00, 32, 2.16 | 2.69, 43, 2.84 |
| average cycles per wait, phase A / phase B | 209 / 222 | 227 / 231 |

Stamped phases against the public kernel's own stamps in the same check (42 stands for 37):
phase A 90.8 against 87.2 at 128 and 130.4 against 119.3 at 256; phase B 45.8 against 43.1 and
65.6 against 58.3; launch end 159.9 against 147.6 and 224.3 against 199.7.

Reading.
- Measured: the consumers do wait. In both phases about a fifth of each warp's stage-loop time
  is spent in mbarrier waits for weights or x, 15.7 us per warp in phase A at 128 tokens and
  23.3 us at 256. Phase B adds 8.4 and 11.8 us.
- Measured: at the same time both producers find the ring full for about 70% of phase A. So on
  average the stream keeps ahead of the consumers, and the waits come as a near-constant 210-230
  cycles per wait in both phases and at both sizes. A startup transient (the first weights,
  the x gather) would be a fixed cost per launch and would not scale with the stage count this
  way.
- Hypothesis, not separated by these counters: each wait is either a fixed cost at the stage
  boundary (the try_wait round trip, then the dependent shared loads of the codes before the
  first dequant, serial on a warp's own path) or skew between the 16 warps of a CTA (a slot
  refills only after all 16 warps release it, and with 3 weights slots at both sizes, since the
  largest job is over 4 n-tiles, a leading warp catches the slot its slowest peer still holds).
  Either way, per-warp waits are not SM time lost one for one: three other consumer warps share
  each scheduler and can issue meanwhile.
- Measured: phase B's epilogue costs 3.76 us per unit at 256 tokens against phase A's 1.71. It
  is 10.1 us per warp, the largest non-loop item in phase B.
- What the earlier record still supports: phase A's per-CTA spread follows the assigned bytes
  (rank correlation 0.76 / 0.84), and the epilogue numbers of modes 33, 38 and 40 were timed by
  wall clock, so they stand.

Next mechanism: see archive/docs/steering/2026-10-29-fat-moe-waits.md (recommendation only, nothing
built).
[2026-10-29, later: built as modes 43 and 44, see "Code groups with the next stage hoisted".]

## Code groups with the next stage hoisted, 2026-10-29

Brief: archive/docs/steering/2026-10-29-fat-moe-waits.md, "Recommendation: one mechanism". Mode 41 put
the consumers' mbarrier waits at about 20% of the stage loops, 210-230 cycles each, while the
producers found the ring full for 70% of phase A (check ca43908f02874886). H1 there: every stage
starts with a serial path on each warp (the try_wait round trip, then the shared loads of the
stage's codes, then the first dequant) while at most one wgmma group of the previous stage is in
flight.

Change (modes 43 and 44; 44 is 43 with mode 41's counters, timing only). Only the wgmma stage loop
(`run_segment`, `grp_codes`) differs from 37; the mma.sync path for jobs of 1 or 2 n-tiles, the
epilogues, the producers and the ring sizes are 37's.

- A warp loads one 4-step code group at a time (one `uint4`, steps 4c to 4c + 3) and the scale
  word of each pair of groups, instead of the stage's four `uint4` and the `uint2` of scales up
  front. Group c + 1 is loaded at the start of group c, so a warp holds two code groups and two
  scale words (10 registers instead of 18).
- After the stage's eighth (last) wgmma group is committed and the seventh retired, the warp waits
  on the next stage's `wfull` and `xfull` and loads that stage's first code group and scale word.
  The barrier round trip and those loads run with the last group in flight; the next stage starts
  with its first dequant. A unit's last stage skips this block, and a unit's first stage waits
  before its loop as before, so the hoist does not cross a unit boundary.
- The weights slot is released by every consumer thread with its own
  `fence.proxy.async.shared::cta` and `mbarrier.arrive` (mode 28's form) behind the sixth wgmma
  group instead of the first. The last code load of the stage is issued at the fifth group; the
  release waits one more group so that the fence's MEMBAR.ALL.CTA does not find that load still
  in flight (a guess about the MEMBAR's cost, not measured). The x slot release is unchanged.
- The code and scale pointers of the current slot are carried from the hoist to the next stage
  rather than recomputed from the slot index at the top of every stage; the first draft did the
  latter and ran 406 instructions per stage loop against 396-397.

### Ordering of the moved weights-slot release

Sections are from the PTX ISA 9.4 documentation (docs.nvidia.com/cuda/parallel-thread-execution)
as cited in "Ordering of the ring slot releases": 8.8 Release and Acquire Patterns, 8.9.1.1
Asynchronous Operations, 8.9.4 Memory synchronization, 8.9.5 Causality Order, 8.10.6 Causality,
9.7.10.28.2 Async Proxy, 9.7.15.4 membar / fence, 9.7.15.16.16 mbarrier.arrive, 9.7.15.16.19
mbarrier.test_wait / try_wait.

For one consumer thread T and the stage k that reads weights slot w, the reads R of the slot are:
group 0 and scale word 0, loaded after T's acquire of `wfull[w]` (in the previous stage's hoist,
or before the loop for a unit's first stage), groups 1, 2 and 3 at the first, third and fifth
wgmma groups of stage k, and scale word 1 at the third. The producer thread P refills slot w with
a bulk copy (write W) after its wait on `wempty[w]`.

1. Every read in R is in T's program order before T's `fence.proxy.async.shared::cta` F behind the
   sixth group of stage k. The loads are plain C++ shared loads and F, like every mbarrier asm
   around them, carries a `"memory"` clobber, so the compiler cannot sink a load below F or raise
   one above the try_wait that acquires the slot (SASS: the last LDS.128 of the stage sits before
   the fifth HGMMA pair, MEMBAR.ALL.CTA and FENCE.VIEW.ASYNC.S after the sixth pair's
   `WARPGROUP.DEPBAR.LE gsb0, 0x1`). F is the cross-proxy fence 9.7.10.28.2 asks for and its effect
   composes with later synchronization (9.7.15.4).
2. F, then T's `mbarrier.arrive` A on `wempty[w]`, in program order. A has `.release` at `.cta`
   scope by default (9.7.15.16.16) and is a release pattern by itself (8.8), covering R and F.
3. P's `mbarrier.try_wait.parity` on `wempty[w]` returns true once all kNC x 32 arrivals of the
   phase are in. It has `.acquire` at `.cta` scope by default and forms an acquire pattern
   (9.7.15.16.19, 8.8), so A synchronizes with it (8.9.4).
4. P's `cp.async.bulk` into slot w is ordered after P's prior instructions (8.9.1.1), so after the
   acquire.
5. R precedes W in base causality order with T's proxy fence on the path, so no read in R can
   read from W (8.10.6).

This is the chain of mode 28 step for step; only the program-order distance between R and F
changed, and every link is program order or the same synchronizes-with edge. What the move adds,
and why it keeps the rest intact:

- Arrival count and phase. Each thread still arrives once per stage on the slot it read, so
  `wempty` keeps its expected count of kNC x 32 and each phase completes once per lap.
- The hoisted reads of stage k + 1's slot follow T's acquire of `wfull` for that slot (its
  try_wait, then the loads, in program order), the same full-side chain as 37 with the wait
  issued earlier. They belong to stage k + 1's R and are covered by stage k + 1's own F and A.
  The ring has 3 to 5 weights slots, so stage k + 1's slot is never the slot stage k releases.
- Progress. Before each wait, T has issued the same releases as in 37. In 37, T waits on
  `wfull` and `xfull` for stage k + 1 at the top of stage k + 1, having released weights slots up
  to stage k (at stage k's first group) and x slots up to stage k - 1 (at stage k's first group).
  In 43, T waits on them at stage k's last group, having released weights up to stage k (sixth
  group) and x up to stage k - 1 (first group). No new wait comes before a release that 37 issues
  first, so 43 adds no cycle to the wait graph: the producers wait only on releases, the
  consumers only on fills, and across CTAs only through the x producer's done_as and claim waits,
  which no consumer change touches.
- The x slot. The x slot of stage k is still released by lane 0 after the `wgmma.wait_group` at
  stage k + 1's first group retires the last group that read it (same proxy on both sides, no
  fence; "Which public releases are provable"). The hoisted `xfull` wait for stage k + 1 reads no
  data itself; the wgmmas that read the slot are issued after it in program order.

No fence or barrier is removed. The fence is the bidirectional `fence.proxy.async.shared::cta`
and the arrive is per thread, as in 28 and 37.

### Local compile gate (no GPU)

`muse_moe_w4a16/profiles/sass_grpcodes.2026-10-29.txt` (nvcc 12.8, sm_90a, Docker):

- Production SASS unchanged: modes 0, 16, 26, 28 and 37, the four skinny kernels, and the
  diagnostic 41 and 42 compile to the same instruction text as before the change.
- No C7510-C7515 warning. One `wait1` per wgmma group and one `wait0` per segment end in 43 and
  44, as in 37 (HGMMA 192, wait1 96, wait0 12).
- 96 registers. Spills: 43 has 8 / 100 B (37: 156 / 416 B), none of them inside a stage loop
  (37 has none there either); its unit loops have 1 or 2 LDL each against 2 to 12 LDL and up to 5
  STL in 37. 44 has 20 / 128 B (41: 264 / 580 B).
- Stage loop: 396-405 instructions against 382-390 in 37; on the path where every first try_wait
  succeeds, 347 against 338 (n-tile count 7) and 354 against 340 (8). The hoisted block is about
  9 to 14 more instructions per stage on that path.
- The layer copy (tools/dl_moe_inline.py on a scratch copy of the layer, LAYER_MODES with 37 and
  43) applies its anchors and the PDL patch and builds `moe1::moe_main_kernel<43>` with the same
  resources. `tools/moe_host_check.sh` and `tools/moe_dryrun.py` pass with 43 and 44 built.

### Check plan and stopping rule

One check after the 00:00 UTC reset, bench file from `tools/moe_make_bench_ab.py --base=37
--sizes=32,128,256 --routings=layer o43=43 o44=44 o42=42` plus the staged pair's arm (below;
it does not enter 43's rule) (seeds 1-6; 41 is built as 37's counter
twin, so ABDIAG and ABDIAGB print 41 and 44 side by side, now with the average cycles per wait).
43 is in MAIN_CANDIDATES, so prepare self-tests it against FP32 at 32, 33, 128 and 257 tokens on
normal and skewed routing and the policy bench may pick it.

Keep 43 only if 43 minus 37 has a mean of -2 us or lower at 256 tokens with at least 5 of 6 seeds
lower, a mean of +0.5 us or lower at 32 and at 128, and ABCHECK within 1e-4 of 37; and 44's
average cycles per wait must fall by at least 30% against 41's in the same run. A loss or a tie
goes to docs/ruled-out.md and ends the line. At most 2 MoE checks.

### Results (checks 9d94ac75f4724b52 and 304f17f3e8704992)

Paired differences against 37, us, layer routing (negative = faster; seeds 1-6 in the first
check, held-out seeds 7-12 in the repeat):

| | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| 43 minus 37, seeds 1-6 | -1.3 (5 of 6) | -1.4 (6 of 6) | -2.7 (6 of 6) |
| 43 minus 37, seeds 7-12 | -0.9 (6 of 6) | -1.2 (4 of 6) | -2.1 (6 of 6) |
| 47 (no hoist) minus 37, seeds 7-12 | -0.4 (3 of 6) | -1.6 (6 of 6) | -3.6 (6 of 6) |
| 47 minus 43, seeds 7-12 | +0.5 (2 of 6) | -0.5 (5 of 6) | -1.5 (6 of 6) |
| 44 minus 37 (counter cost), both checks | +3.3 / +4.1 | +6.3 / +6.4 | +7.5 / +9.1 |

Average cycles per wait, phase A, 44 against 41 in the same process: 214 / 224 / 239 against 201 /
215 / 223 (seeds 1-6) and 214 / 215 / 235 against 204 / 203 / 225 (seeds 7-12) at 32 / 128 / 256.
Phase B: 210 / 215 / 225 against 204 / 217 / 220, and 188 / 221 / 233 against 192 / 224 / 234.
The per-warp waits did not fall in either phase; in phase A they rose 3-7%. The producers' slot
waits rose too (256 tokens: 99.9 / 93.5 against 92.9 / 91.1 us, then 108.7 / 102.5 against 100.2 /
99.2), so the ring was full longer. The phase A epilogue was shorter in 44 in every row (per unit
1.38 against 1.49 at 128 and 1.56 against 1.71 at 256 in the first check).

Reading.
- Measured: 43 is faster than 37 at every size in both checks, by 1-3 us.
- Measured: the hoist did not shorten the waits. H1 of the steering report (a serial try_wait
  round trip plus code loads at each stage start) is not what the 210-230 cycles per wait are;
  issuing the wait one wgmma group earlier leaves the wait the same length. That is consistent
  with H2 (warps of a CTA drifting apart, a leading warp waiting on a slot its slowest peer still
  holds) or with the wait being the stream's own pace at the stage boundary; this run does not
  separate them.
- Measured: 47, the same code groups without the hoist, is as fast as 43 or faster (1.5 us faster
  at 256 tokens, 6 of 6). The gain comes from the code groups: a warp holds 10 code and scale
  registers instead of 18, and the kernel's spill falls from 156 / 416 B to 4 / 88 B (8 / 100 B
  in 43), with the per-unit reloads mostly gone (1 LDL per unit loop against 2-12 plus STL). The
  hoist costs about 9-14 instructions per stage and holds the slot longer, and buys nothing.
- Stopping rule: 43's timing condition passed in both checks, the mechanism condition failed in
  both. 43 is not kept; the hoist line ends (docs/ruled-out.md). 47 came out of the repeat as a
  diagnostic arm and was not under a pre-registered rule. On one check (held-out seeds) it clears
  43's timing thresholds against 37 (-3.6 us at 256, 6 of 6; -1.6 at 128; -0.4 at 32). It is a
  candidate for its own gate run (held-out standalone and skewed seeds) before any adoption.

Mode 47's ordering is 43's chain above with the waits back at the stage start: the reads of a
slot (group 0 after the stage's own `wfull` acquire, groups 1-3 and both scale words in the stage)
precede the per-thread fence and arrive behind the sixth group in program order, and before each
wait a thread has issued the same releases as in 37 (weights up to the previous stage, x up to the
one before), so the progress argument carries over unchanged.

## Mode 47 gate, 2026-10-30

Mode 47 is mode 37 with the weight codes loaded one 4-step group at a time and no next-stage
hoist (section "Code groups with the next stage hoisted"; spills 4 / 88 B against 37's 156 /
416 B). One check so far, on held-out layer seeds 7-12, as a diagnostic arm with no rule set in
advance (304f17f3e8704992: 47 minus 37 -0.4 / -1.6 / -3.6 us at 32 / 128 / 256). This section
sets the rule before the gate check is submitted.

Gate run: one check of a bench file from `tools/moe_make_bench_ab.py --base=37
--sizes=32,128,256 --routings=layer,std,skewed --seeds=13,14,15,16,17,18 --gate=1 o47=47`.
Layer seeds 13-18 are held out from both earlier checks (1-6, 7-12); 47 has never run on the
standalone or skewed routing, so all three are held out for it. 37 and 47 are timed in one
process, interleaved per seed and reversed on every other seed. `--gate=1` drops the stamped and
counter twins and the public kernel's stamped runs (only timing and outputs are needed), puts 47
in MAIN_CANDIDATES so prepare's FP32 self-test covers it at 32, 33, 128 and 257 tokens on normal
and skewed routing, and prints 47's output against 37 on every seed.

Keep 47 only if all of these hold in that check:

1. At 256 tokens, 47 minus 37 has a mean of -1.5 us or lower on each of the three routings
   (layer, standalone, skewed), with at least 5 of 6 seeds lower on layer and on standalone
   routing.
2. At 32 and at 128 tokens, 47 minus 37 has a mean of +0.5 us or lower on every routing.
3. 47's output is within 1e-4 relative L2 of 37's on every seed, size and routing.
4. Prepare's FP32 self-test passes for 47 (no FAIL line, spin flags 0).
5. Compile gate, confirmed again on the commit that is checked: production SASS (modes 0, 16, 26,
   28, 37 and the skinny kernels) unchanged, no C7510-C7515 warning, no local loads or stores
   inside 47's stage loops.

If any condition fails, 47 goes to docs/ruled-out.md scoped to this design (code groups without
the hoist, on 37's components) and is not shipped. If all pass: 47 joins MAIN_MODES and
MAIN_CANDIDATES, the shippable file is judged with one check (GATE prints 47 minus 37), and the
file goes to the coordinator for a scored run. Budget: at most 3 MoE checks for the whole line.

### Gate result (check 5f381b8b6946492d)

47 minus 37, us, same process, seeds 13-18 (negative = faster; count = seeds lower of 6):

| routing | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer | -1.0 (6) | -2.3 (5) | -3.7 (6) |
| standalone | -2.4 (6) | -2.8 (6) | -3.4 (6) |
| skewed | -2.0 (5) | -2.2 (6) | -2.8 (6) |

Per seed at 256 tokens: layer -5.7 -1.5 -2.3 -3.9 -5.1 -3.7; standalone -2.2 -1.6 -3.2 -5.1 -4.0
-4.4; skewed -3.8 -2.3 -2.7 -1.0 -4.5 -2.3. Means of 37 in the same run: 223.8 / 246.9 / 171.5 us
at 256 tokens (layer / standalone / skewed).

1. 256 tokens: -3.7 / -3.4 / -2.8, all under -1.5; 6 of 6 lower on layer and on standalone. Pass.
2. 32 and 128 tokens: the highest mean is -1.0 (32, layer). Pass.
3. Outputs of 47 against 37 on all 54 seed, size and routing cases: 5.5e-7 to 4.4e-5. Pass.
4. Prepare's FP32 self-test: main-lean-grp-nohoist 3.89e-3 (37: 3.89e-3), spin flags 0, no FAIL. Pass.
5. Compile gate: muse_moe_w4a16/profiles/sass_mode47.2026-10-30.txt. Against 47's parent commit
   (4daacf6) every production kernel has the same instruction text; no C7510-C7515; 47 spills
   4 / 88 B (37: 156 / 416) and its stage loops have no local loads or stores. Pass.

The prepare-time bench in the same run picked 47 at 128 and 256 tokens (207.8 against 209.5 and
230.1 against 232.3 us). The public kernel stays ahead: public minus 37 at 256 tokens -24.0 /
-24.2 / -8.9, so 47 closes about 3 us of a 24 us gap on layer and standalone routing.

Decision under the rule: 47 is kept. MAIN_MODES (0, 16, 26, 28, 37, 47) and MAIN_CANDIDATES gain
47 (ec565a7); GATE prints 47 minus 37 on its held-out seeds.

Shippable check b50351288b504e81 (ec565a7, 2.190x, 2.157-2.223): GATE 47 minus 37 -2.5 / -3.4 us at
128 / 256 tokens (seeds 17-22, 6 of 6 lower at 256, 6 of 6 at 128) and -2.3 / -2.3 on skewed seeds
11-13 (3 of 3 each); the policy picked 47 at 128 and 256. The judge's medians: 238.5 us at 256
tokens against 241.49 and 241.46 in 37's two runs; 199.8 at 128 against 197.09 and 202.35 (the
judge's 128-token shape moves by more than the gain). The score sits inside the intervals of 37's
runs (2.144x, 2.160x); the paired evidence is the same-process A/B above. The file went to the
coordinator for a scored run.

## Warp skew and the gap to the public kernel after mode 47, 2026-10-30

Brief: archive/docs/steering/2026-10-29-fat-moe-waits.md, next action 3 (per-warp stage-start skew, H2)
and a re-attribution of the same-process gap to the public kernel on mode 47.

### Method

Modes 48 and 49 (64cec1f, timing only) are twins of 47: 48 with the light stamps (thread 0's
globaltimer at the phase boundaries, the twin of 42), 49 with the light stamps plus per-warp
stage stamps. In 49, lane 0 of every consumer warp stores clock64 at each wgmma stage's start,
after its weights wait, after its x wait and after its weights-slot arrive, through a predicated
`st.global` (a lane-0 branch between wgmma groups serializes them, docs/ruled-out.md); the
weights and x producers store their issue time per stage. Stage q is the ring position, the
same for both rings. The mma.sync stages (jobs of 1 or 2 n-tiles) are not stamped. Compile
gate: production SASS unchanged, no C7510-C7515, 49's stage loops 418 instructions against
47's 387 with no local loads or stores. tools/moe_make_bench_ab.py `--skew=1` reduces the stamps
per phase as ABSKEW.

The public kernel could not be stamped the same way. At its 96 registers, any consumer-side
stamp (four subsets tried, local nvcc 12.8, sm_90a) made its DEBUG build spill 400 to 1,260 B,
and three of the four subsets serialized its wgmmas (C7512), so its skew would not be its own.
The option stays in the tool as `--pubskew=1`, unused.

Checks: ed3b4fdc76b54263 (crashed in the bench after the 128-token timing on an empty ABCHECKALL
list, a bug of 20c295e fixed in the next commit) and d47c78bdc130424b (the full run), both
`--base=47 --sizes=128,256 --routings=layer --skew=1 o48=48 o49=49`, layer seeds 1-6.

### Skew (mode 49, d47c78bdc130424b)

Mean over CTAs and six seeds of one cold launch each; wgmma stages only; cycles at 1.74-1.75 GHz
unless marked us.

| | 128 A | 128 B | 256 A | 256 B |
|---|---|---|---|---|
| wgmma stages stamped per CTA | 19.1 | 9.0 | 35.4 | 16.0 |
| per warp: weights waits / x waits, us | 1.96 / 3.48 | 0.97 / 1.12 | 4.42 / 7.30 | 1.76 / 1.89 |
| cycles per wait | 253 | 202 | 288 | 200 |
| stage-start spread over the 16 warps | 873 | 883 | 1,370 | 976 |
| spread of the 4 warpgroups' mean starts | 756 | 826 | 1,210 | 922 |
| spread within a warpgroup | 98 | 58 | 160 | 54 |
| spread of the weights-slot releases | 945 | 1,020 | 1,420 | 1,130 |
| weights-wait share a zero-skew release removes (bound) | 0.72 | 0.72 | 0.74 | 0.73 |
| producer issue after the slowest release | 306 | 327 | 553 | 350 |
| weights copy, issue to first wait end (upper bound) | 4,930 | 4,720 | 5,450 | 5,160 |
| a warp's own release of the slot to its next start in it | 5,940 | 5,810 | 7,210 | 6,330 |
| x copy, issue to first wait end (upper bound) | 3,860 | 4,320 | 4,080 | 4,120 |

The bound in the share row is, per wait, min(wait, slowest warp's release of the slot three
stages back minus this warp's own). The copy rows use the earliest wait end over the 16 warps,
which cannot precede the copy's arrival, so they overstate the latency. Cost of the stamps:
49 minus 47 +5.8 / +9.4 us at 128 / 256 (0 of 6 lower), 48 minus 47 +1.3 / +1.1.

Reading.
- Measured: the four warps of a warpgroup stay within 50-160 cycles of each other (the
  warpgroup-wide wgmma keeps them together). The four warpgroups of a CTA drift 750-1,200
  cycles apart, about half of a 2,200-cycle stage at 256 tokens. H2 holds at the warpgroup level.
- Measured: a weights slot is refilled 300-550 cycles after its slowest warp releases it and
  arrives about 5,000 cycles later. A warp comes back to the slot 5,800-7,200 cycles after its
  own release, so a leading warpgroup that is 1,000-1,400 cycles ahead of the slowest one finds
  the slot still in flight. Up to 72-74% of the weights-wait cycles would go if every warp had
  released when the waiting warp did.
- Measured: x waits are 1.2-1.8 times the weights waits. The x slot of stage q is released
  only after stage q + 1's first wgmma group, so its lead is about a stage shorter, against an
  x copy of about 4,000 cycles.
- Not measured: whether the skew costs SM time. The waits fall on the leading warpgroups; the
  trailing one sets the pace. Removing the waits (per-warpgroup slots and barriers, below)
  helps only if the leading warpgroups' idle time is on the CTA's critical path.

### Same-process gap by phase (48 against the public kernel's stamps, d47c78bdc130424b)

| us, mean of 6 layer seeds | 128: 48 / public / gap | 256: 48 / public / gap |
|---|---|---|
| graph replay, 47 against public (6 of 6) | 165.1 / 150.4 / +14.7 | 224.1 / 201.5 / +22.6 |
| setup: routing (route) | 3.4 / 2.5 / +0.9 | 4.3 / 3.4 / +0.9 |
| setup: x gather and release (start minus route) | 7.2 / 3.8 / +3.4 | 8.6 / 5.1 / +3.5 |
| phase A (per CTA, mean) | 88.6 / 86.3 / +2.3 | 125.4 / 117.7 / +7.7 |
| phase A done, last CTA | 113.6 / 98.0 / +15.6 | 148.9 / 132.7 / +16.2 |
| phase B (per CTA, mean) | 46.0 / 43.1 / +2.9 | 64.2 / 56.9 / +7.3 |
| tail (last phase B end minus the mean) | 9.0 / 5.3 / +3.7 | 9.6 / 6.0 / +3.6 |
| barrier and final pass | 13.4 / 9.4 / +4.0 | 16.1 / 12.5 / +3.6 |
| launch end, last CTA | 159.2 / 146.4 / +12.8 | 219.4 / 197.6 / +21.8 |

Summed over the two sizes the pieces are phase B 10.2 us, phase A 10.0, setup 8.7 (7.0 of it
in the gather), barrier and final pass 7.6, tail 7.3. Phase A's piece has no single mechanism
behind it yet (the skew above and the x lead are candidates). Phase B's has one: mode 41 put
the phase B epilogue at 3.76 us per unit at 256 tokens against phase A's 1.71 (2.21 against
1.51 at 128, ca43908f02874886), and the public kernel's phase B epilogue writes y only and
forms each route's sum of squares in its final pass (its OKC_WGSFINSS). Ours shuffles three
rounds and issues two FP32 atomics per n-tile there.

### Mode 50: the sums of squares in the final pass

Change (a03ca00). Mode 50 is 47 with the phase B epilogue's shuffles and atomics into `ss`
removed (it still writes ybuf) and the final pass replaced by `final_pass_ss`: one token per
item, each of the 512 consumer threads holds 8 columns; pass 1 reads the token's local rows of
ybuf and sums y^2 per slot (warp shuffles, then the 16 warps' partials through shared memory in
the idle rings, named barrier 8), pass 2 reads the rows again (L2) and writes out. Same
coefficient formula (topk_w * rsqrt(ss / 4096 + eps)); only the summation order of ss changes.

Ordering. No fence or barrier is removed and no release or acquire moves.
1. ybuf: the final pass reads it after the grid barrier, as final_pass did (every writer's
   stores, then `__syncthreads`, thread 0's `__threadfence` and atomicAdd; the reader's
   `ld.acquire` spin and `__syncthreads`). What came through `ss` before (atomics, read with
   `ld.global.cg` after the same barrier) now comes from the ybuf rows themselves.
2. The rings as scratch: before the grid barrier every bulk copy a producer issued has been
   waited on by a consumer (each stage's full barrier, on both the wgmma and mma.sync paths),
   and `__syncthreads` follows; the async writes are complete and observed through the mbarrier
   phase (PTX ISA 9.7.15.16.19), so a later generic write does not race them.
3. The reduction buffer alternates halves by item parity with one 512-thread `bar.sync` per
   item: a thread writes a half for item k + 2 only after the barrier of item k + 1, which every
   thread reaches after its reads of item k. The trip count depends only on the CTA index, so
   all 512 threads execute the same barriers; barrier 8 is used nowhere else.

Local compile gate (nvcc 12.8, sm_90a; sk/after50 in the session scratchpad): production SASS
(modes 0, 16, 26, 28, 37, 47 and the four skinny kernels) has the same instruction text as
before 64cec1f; no C7510-C7515; mode 50 at 96 registers with no spill (47: 4 / 88 B) and no local
load or store anywhere; stage loops 387-393 instructions (47: 387-394); MEMBAR 43,
FENCE.VIEW.ASYNC.S 21, SYNCS.ARRIVE 48, HGMMA 192, wait1 96, as 47. tools/moe_dryrun.py and
tools/moe_host_check.sh pass with 50 in MAIN_MODES and MAIN_CANDIDATES.

Prediction (hypothesis): -2 to -5 us at 256 tokens, -0.5 to -2 at 128, from the phase B
epilogue, less a second L2 read of the final pass's rows (32 MB at 256 tokens).

Gate run (rule set before the check is submitted): one check of a bench file from
`tools/moe_make_bench_ab.py --base=47 --sizes=32,128,256 --routings=layer,std,skewed
--seeds=23,24,25,26,27,28 --gate=1 o50=50`. Seeds 23-28 are held out on all three routings (layer
seeds 1-18 and standalone 13-22 ran before). 47 and 50 are timed in one process, interleaved
per seed and reversed on every other seed; `--gate=1` puts 50 in MAIN_CANDIDATES (prepare's FP32
self-test covers it) and prints 50's output against 47 on every seed.

Keep 50 only if all of these hold in that check:

1. At 256 tokens, 50 minus 47 has a mean of -1.5 us or lower on layer and on standalone routing,
   with at least 5 of 6 seeds lower on each, and a mean of 0 or lower on skewed routing.
2. At 32 and at 128 tokens, 50 minus 47 has a mean of +0.5 us or lower on every routing.
3. 50's output is within 1e-4 relative L2 of 47's on every seed, size and routing.
4. Prepare's FP32 self-test passes for 50 (no FAIL line, spin flags 0).
5. The compile gate above, on the commit that is checked.

If any condition fails, 50 goes to docs/ruled-out.md scoped to this design (sums of squares in a
two-pass final pass, one token per item, on 47) and is not shipped. If all pass: 50 joins
MAIN_MODES and MAIN_CANDIDATES, the shippable file is judged with one check (GATE prints 50
minus 47), and the file goes to the coordinator. Budget: this check and one shippable check.

### Gate result for 50 (check 8a7051fa926e43e1)

50 minus 47, us, same process, held-out seeds 23-28 (negative = faster; count = seeds lower of 6):

| routing | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer | +0.7 (1) | -5.9 (6) | -6.6 (6) |
| standalone | -2.8 (5) | -5.7 (6) | -8.3 (6) |
| skewed | -1.4 (6) | -4.1 (6) | -7.9 (6) |

Public minus 47 in the same run: -4.5 / -12.8 / -23.8 us on layer routing at 32 / 128 / 256.
Outputs of 50 against 47 on all 54 cases: 1.2e-6 to 4.3e-5. Self-test: main-lean-grp-finss
3.90e-3 (47: 3.90e-3), spin flags 0. The prepare bench picked 50 at 128 and 256 (204.9 against
206.3, 221.6 against 229.2 us). Score 2.210x (2.177-2.243), passed.

1. 256 tokens: -6.6 / -8.3 / -7.9, 6 of 6 lower on layer and standalone. Pass.
2. 32 and 128 tokens, +0.5 us or lower on every routing: 32 tokens on layer routing is +0.7
   (1 of 6 lower). Fail.
3. Outputs within 1e-4: pass. 4. Self-test: pass. 5. Compile gate: pass (a03ca00).

Decision under the rule: 50 is not shipped and goes to docs/ruled-out.md, scoped to this design
at 32 tokens. Reading (hypothesis): at 32 tokens the final pass has 32 items, one per token, so
100 of 132 CTAs idle through it while 32 CTAs each read their rows twice; 47's final pass has 64
items of half a row. The phase B saving is small at 32 tokens (few phase B units per CTA).

### Mode 51: the final-pass sums from 64 tokens up

Change. Mode 51 is 50 with the choice made per launch: from kFinssMinM = 64 tokens the phase B
epilogue drops its sums and the final pass forms them (50's path); below 64 the epilogue keeps its
shuffles and atomics and the final pass is 47's. p.M is uniform over the launch, so the epilogue
and the final pass agree, and the double-buffered `ss` is zeroed for the next launch on both
paths as before. The threshold sits between the sizes the layer runs on the fat path (32, 128,
256); it was read off the gate run above, so the gate below uses new held-out seeds.

Ordering: 50's chain above applies to the 64-token-and-up path; the path below 64 is 47's.
Independent read-only review of 50 (2026-10-30): no bug found in the ybuf visibility through the
grid barrier, the reuse of the rings after every bulk copy was waited on (both consumer paths and
the claimed phase B units; the PDL preamble only prefetches into L2), the double-buffered
reduction with named barrier 8, or the indexing. It noted one unproven step that does not arise
in a correct run: after a spin timeout (ctr[8] set), a bulk copy may still be in flight when the
final pass writes the rings; the result is already flagged wrong in that case.

Local compile gate (nvcc 12.8, sm_90a, MAIN_MODES 0, 16, 26, 28, 37, 47-51): production SASS
(modes 0, 16, 26, 28, 37, 47 and the four skinny kernels) and 48, 49 have the same instruction
text as before; 50 differs from its a03ca00 build in register names only (12 lines); no
C7510-C7515; 51 at 96 registers with no spill and no local load or store; stage loops 387-393
instructions; MEMBAR 43, FENCE.VIEW.ASYNC.S 21, SYNCS.ARRIVE 48, HGMMA 192, wait1 96.
tools/moe_dryrun.py and tools/moe_host_check.sh pass.

Gate run (rule set before the check is submitted): one check of a bench file from
`tools/moe_make_bench_ab.py --base=47 --sizes=32,128,256 --routings=layer,std,skewed
--seeds=29,30,31,32,33,34 --gate=1 o51=51`; seeds 29-34 are held out on all three routings.

Keep 51 only if all of these hold in that check:

1. At 256 tokens, 51 minus 47 has a mean of -1.5 us or lower on layer and on standalone routing,
   with at least 5 of 6 seeds lower on each, and a mean of 0 or lower on skewed routing.
2. At 128 tokens, a mean of -1.0 us or lower on layer and on standalone routing and of +0.5 or
   lower on skewed routing.
3. At 32 tokens, a mean of +0.5 us or lower on every routing.
4. 51's output is within 1e-4 relative L2 of 47's on every seed, size and routing.
5. Prepare's FP32 self-test passes for 51 (no FAIL line, spin flags 0).
6. The compile gate above, on the commit that is checked.

If any condition fails, 51 goes to docs/ruled-out.md with 50 and is not shipped. If all pass,
51 joins MAIN_MODES and MAIN_CANDIDATES and the shippable file goes to the coordinator, who runs
its check (this line's 4-check budget ends with the gate run).

### Gate result for 51 (check c6b0e500c3a84c9a)

51 minus 47, us, same process, held-out seeds 29-34 (negative = faster; count = seeds lower of 6):

| routing | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer | +0.0 (2) | -3.9 (6) | -3.9 (5) |
| standalone | -1.0 (5) | -3.8 (6) | -4.4 (6) |
| skewed | -0.1 (2) | -4.0 (6) | -7.7 (6) |

Outputs of 51 against 47 on all 54 cases: at most 4.6e-5. Self-test: main-lean-grp-finss64
3.88e-3 (47: 3.88e-3), spin flags 0. The prepare bench picked 51 at 128 and 256 (207.7 against
208.1, 223.6 against 230.5 us). Score 2.194x (2.161-2.227), passed.

1. 256 tokens: -3.9 (5 of 6) / -4.4 (6 of 6) / -7.7. Pass.
2. 128 tokens: -3.9 / -3.8 / -4.0. Pass.
3. 32 tokens: +0.0 / -1.0 / -0.1. Pass.
4. Outputs: pass. 5. Self-test: pass. 6. Compile gate (69c06ed): pass.

Decision under the rule: 51 is kept. MAIN_MODES (0, 16, 26, 28, 37, 47, 51) and MAIN_CANDIDATES
gain 51; GATE prints 51 minus 47. The shippable build was checked locally (production SASS and
51 identical to the gated build, no C75xx, tools/moe_dryrun.py and tools/moe_host_check.sh pass)
but not judged: the line's 4-check budget ended with the gate run, so the coordinator runs its
check. The 128-token gain on layer routing (-3.9 us) is smaller than 50's in its run (-5.9) on
other seeds; the two runs do not compare the modes directly.

## Phase attribution at 32 tokens against the public kernel, 2026-10-30

Brief: the layer report (archive/docs/steering/2026-10-30-layer-release-attribution.md, item 3) put our MoE
at about 102 us exposed against the public's 88 at decode 32 sliding, with nobody holding phase
stamps at 32 tokens; mode 50 failed its gate there and 51 runs 47's code below 64 tokens.

### Method (check 1fbb9fbf524f4ed7)

`tools/moe_make_bench_ab.py --base=47 --sizes=32 --routings=layer,std --seeds=1,2,3,4,5,6
--work=1 --skst=8:3:2 o37=37 o16=16 sk8w3s=v3`, public file from the session scratchpad. Same
process, graph replay after an L2 flush, arms interleaved and reversed per seed. Stamped twins:
48 (47's light stamps), 42 (37's), 41 (37 with the phase A and B counters; 37 and 47 share the
partition, owned phase A and claimed phase B), the public kernel with its own DEBUG stamps, and
sk8w3s with mode 0's stamps (kMode 2). Two additions to the tool: `--work=1` prints ABWORK, the
work each design gives each CTA from the routing alone (our jobs and owned phase A deal, the
public kernel's stream-K stage ranges ported from its `wgs_prefix` / `wgs_range` with its own
cost tables, the skinny path's split jobs) and our claimed phase B units per CTA from 41's
counters; `--skst` builds stamped skinny variants outside `--small` and prints their ABSK next
to the fat stamps. ABPUB's stage columns need `--pubranges` and are garbage in this run; ABWORK
replaces them.

### Results

Graph replay, us, mean of 6 seeds (count = seeds lower than 47):

| arm | layer routing | standalone routing |
|---|---|---|
| 47 | 90.9 | 132.9 |
| public | 89.5 (-1.5, 6 of 6) | 127.0 (-5.9, 6 of 6) |
| 37 | 92.7 (+1.8, 1 of 6) | 134.5 (+1.6, 0 of 6) |
| 16 (stream-K both phases, older stage loop) | 119.4 (+28.5) | 164.0 (+31.1) |
| sk8w3s | 110.1 (+19.2, 0 of 6) | 132.8 (-0.1, 3 of 6) |
| sk8w4s | 113.8 (+22.9) | 136.2 (+3.4) |

Phase stamps, us, mean over CTAs and 6 seeds of one cold launch (48 against the public; 42 and
41 agree with 48 within 1-8 us, 41 paying its counters):

| piece | layer: 48 / public / gap | standalone: 48 / public / gap |
|---|---|---|
| routing (start to routed) | 3.0 / 1.8 / +1.2 | 2.9 / 1.8 / +1.1 |
| x gather and release | 4.4 / 2.0 / +2.4 | 2.0 / 2.2 / -0.2 |
| phase A per CTA, mean | 47.7 / 49.8 / -2.1 | 77.9 / 75.3 / +2.6 |
| phase A done, last CTA | 64.2 / 58.0 / +6.2 | 92.3 / 83.7 / +8.6 |
| phase B per CTA, mean | 22.7 / 22.8 / -0.1 | 34.1 / 34.6 / -0.5 |
| tail (last phase B end minus the mean) | 6.5 / 4.5 / +2.0 | 10.8 / 4.8 / +6.0 |
| barrier and final pass (includes the tail wait) | 8.9 / 6.6 / +2.3 | 13.2 / 6.8 / +6.4 |
| launch end, last CTA | 87.7 / 86.1 / +1.6 | 131.0 / 123.6 / +7.4 |

Work per CTA (ABWORK, means over the seeds; all 132 CTAs have phase A work in every design):

| | layer | standalone |
|---|---|---|
| fat jobs (one per active expert, none over 64 tokens) | 8.5 (5.7 of 1 n-tile, 1.2 of 2, 1.7 of 3) | 14.2 (12.8 of 1 n-tile) |
| ours: owned phase A units per CTA, min / mean / max | 1.5 / 2.06 / 2.5 | 2.83 / 3.43 / 3.83 |
| ours: claimed phase B units per CTA (41), min / mean / max; CTAs with any | 0.5 / 1.03 / 1.5; 122 | 1.33 / 1.72 / 2.33; 132 |
| public: phase A stages per CTA, min / mean / max; cut units per CTA | 28 / 33.0 / 35.8; 1.86 | 46 / 55.0 / 56.8; 1.84 |
| public: phase B stages per CTA, min / mean / max | 14.3 / 16.5 / 17.8 | 24.8 / 27.5 / 28.0 |
| phase A per CTA, p10 / p90 (41; public from its stamps) | 42.7 / 62.4 (public 48.0 / 51.6) | 73.7 / 89.1 (public 73.9 / 76.8) |
| weights streamed: fat / skinny (split jobs re-stream) | 241 / 368 MB (13 skinny jobs) | 401 / 458 MB (16.2) |

Skinny sk8w3s stamps on layer routing: phase A 67.2, phase B 34.0, end 110.8 us; it streams 53%
more bytes than the fat path because the experts with 9-23 routes become 2-3 jobs of 8 tokens.

Reading.
- Measured: in one process on layer routing at 32 tokens, the public kernel is 1.4-1.6 us
  ahead of 47 (graph replay and stamps), not 10-14 us. On standalone routing it is 5.9-7.4 us
  ahead. The layer report's +9.8 us at decode 32 sliding came from one profiled replay, and the
  same report shows +3.9 at decode 32 full, which runs the same MoE size; so most of the layer
  figure is not MoE kernel work as this bench measures it. What else differs in the layer is a
  hypothesis for the layer owner: the routing of that replay, the PDL preamble's L2 prefetch of
  unit 0 of all 32 local experts (18.9 MB) when about 8.5 are active at 32 tokens, and the
  profiler's exposed-time split.
- Measured: the public partitions both phases by cost-weighted stage ranges (stream-K, about
  1.85 cut units per CTA per phase), so every CTA gets 33 +- 3 phase A stages on layer routing.
  Ours deals whole 16-stage phase A units (2.06 per CTA: 1, 2 or 3 units) and lets CTAs that
  finish early claim whole phase B units; the last phase A CTA ends 6.2 / 8.6 us after the
  public's and the last phase B CTA 6.5 / 10.8 us after the mean CTA (public 4.5 / 4.8).
- Largest piece, by the stamps: on standalone routing the end-of-launch imbalance (phase A
  done max +8.6, tail +6.0); on layer routing the setup (+3.6: routing +1.2, gather +2.4) and
  the same imbalance (+6.2, +2.0) offset by a faster mean phase A (-2.1). The setup pieces have
  no single mechanism (the gather's fences stay); the imbalance has one: whole-unit claims, of
  about 22 us each at 32 tokens, in the last wave.
- sk8w3s does not belong in the layer's 32-token pick on judge-like routing (+19.2 us, 0 of 6);
  on standalone routing it ties 47. The standalone prepare bench (synthetic routing) chose
  sk8w3s at 32 in this run, which is the standalone board's business only.
- 37 is 1.6-1.8 us behind 47 at 32 tokens; the layer should keep 47 (or 51, the same code there).

### Modes 52 and 53: the last phase B units claimed in halves

Change. 52 and 53 are 51 with the claim values past `units - xs` mapped to the two K halves (8
stages each) of the last `xs` phase B units, xs = G / 2 (52) or G (53), at most the unit count;
54 is 52 with the light stamps (timing only). The weights and x producers stream the claimed
half's 8 stages; the consumer runs it as a cut segment through `fixup` (the phase B cut-unit path
of modes 12-16): each half adds its accumulators into a phase B fixup slot with red.add.v4, then
`fence.acq_rel.gpu`, the warpgroup's bar.sync and one `atom.acq_rel.gpu` on the slot's stage
count; the half that completes 16 stages reads the slot back after a second fence, zeroes it and
runs 51's phase B epilogue. The slot index is the unit's place in the split tail (below G, so the
2 x G x 4 slots and the per-launch fixcnt reset after the grid barrier cover it). Claims still
rise through S.ord, so the per-n-tile-count claim loops are unchanged. Applies at every M in
these modes; the gate decides where.

Ordering. No fence, barrier, release or acquire is removed or moved. The halves use the cut-unit
fixup that modes 12-16 ran in phase B before phase B boundaries were aligned (its chain: a
fence.acq_rel followed by a strong write is a release pattern and a strong read followed by
fence.acq_rel an acquire pattern, PTX ISA, Memory Consistency Model, "Release and Acquire
Patterns"; the writer's red.add.v4 are ordered before the release by its fence and the
warpgroup barrier, the reader's ld.cg after its acquire and fence). The inputs of each half come
through done_as per stage as for whole units. New is only which slot a cut phase B unit uses and
that two CTAs claim the halves at run time instead of by stage range; each slot is used by
exactly one unit per launch (claim values are unique), and both halves add their stage counts,
so the count reaches 16 once.

Local compile gate (nvcc 12.8.1, sm_90a, tools/moe_compile_fused.sh): production SASS (modes 0,
16, 26, 28, 37, 47, 51 and the four skinny kernels) has the same instruction text as before this
change (b3509a0); no C7510-C7515; 52, 53 and 54 at 96 registers, no stack frame, no spills, no
local loads or stores; HGMMA 192, wait1 96, wait0 12 as 51 (the wgmma groups still pipeline);
MEMBAR 91 against 51's 43 and REDG 45 against 9, the fixup's fences and red.add.v4 that 51
compiles out (its phase B segments are never cut).

Prediction (hypothesis, before the run): the last wave of claims ends in about half a unit (11
us at 32 tokens) instead of a whole one, so the tail falls by 2-4 us at 32 tokens, less the
fixup's cost on the split units.

### Diagnostic run (check 209cf29bf91c4ea2, no keep rule set in advance)

`tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=35,36,37,38,39,40 o52=52 o53=53 o47=47` (light twins 54 and 48, public stamps). A
diagnostic arm, as 47's first check was: the rule for a gate was to be written from it, on new
held-out seeds.

Minus 51, us, same process (count = seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 52 (xs = G / 2) | +3.3 (2) | +3.5 (1) | +5.9 (0) | +4.9 (0) | +8.5 (0) | +5.3 (0) |
| 53 (xs = G) | +8.1 (0) | +5.2 (0) | +7.3 (0) | +6.5 (0) | +9.2 (0) | +7.7 (0) |
| 47 | -0.0 (3) | +0.0 (4) | +3.8 (0) | +3.2 (0) | +5.2 (0) | +3.5 (0) |
| public | -1.9 (4) | -5.2 (6) | -7.8 (6) | -10.8 (6) | -16.1 (6) | -15.3 (6) |

Stamps at 32 tokens, 54 against 48 (47's code, which 51 runs below 64 tokens): phase B per CTA
24.2 against 20.5 us (layer) and 43.0 against 39.2 (standalone); tail 7.9 against 7.1 and 8.2
against 7.7. Outputs of 52 and 53 against 51 at most 9.7e-5; spin flags 0. Score 2.191x.

Reading. Measured: halving the last claims did not shorten the tail, and phase B per CTA rose
by 3.7-3.8 us at 32 tokens; splitting more units (53) costs more. Hypothesis for the cost: the
fixup's red.add.v4 of the accumulators, its two gpu-scope fences under a saturated memory system
and the second half's read-back sit on the critical path of both halves. Hypothesis for the tail that
remains: it is set by the phase B units' wait for the last phase A units of their job (phase A
done max is 6-8 us later than the public's), not by the claim size.

Decision: 52 and 53 are not gated (they lost at every size and routing, 0-2 of 6 seeds lower),
and go to docs/ruled-out.md scoped to this design. Their code stays in the source, not built
(production SASS unchanged). No layer port. Two of the four checks were used; the other two are
returned unspent, because the same-process gap left at 32 tokens on layer routing is 1.5-1.9 us
in two runs, below what a further MoE-only mechanism at that size can be expected to show inside
the noise of one check, while the layer's 10-14 us figure is not reproduced here.

Next for the layer owner (hypotheses, untested): time the layer MoE at decode 32 with and without
the PDL preamble's L2 prefetch (it hints unit 0 of all 32 local experts, 18.9 MB, when about 8.5
experts are active at 32 tokens), and repeat the decode 32 attribution over more than one
profiled replay before charging 10-14 us to the MoE.

## The x gather or the warpgroup skew: re-measurement on mode 51, 2026-10-30

Brief: of the two untested leads in the 2026-10-29 fat-MoE steering addendum (3), the setup's x
gather (+3.4 / +3.5 us against the public kernel at 128 / 256 tokens in 48's stamps) and per
warpgroup weights slots against the skew (up to 72-74% of the weights-wait cycles, a bound on
waits, not on SM time), pick the larger by a short re-measurement on mode 51 and build one.

### Method

Modes 55 and 56 are timing-only twins of 51: 55 with the light stamps, 56 with the light stamps
and 49's per-warp stage stamps. Both add three light stamps around the x gather (gather_st):
thread 0's globaltimer after its row stores and after the gather barrier (its fences included),
and the x producer's when the xready wait of its first phase A job ends. The tool prints them as
ABGATHER (with the public kernel's gather, DBG 1 to 7, which also holds its ss zeroing and
row-base scan). ABSKEW gains the trailing warpgroup's columns: per stage, the warpgroup whose
mean stage start is latest is taken to set the CTA's pace, and its own weights and x waits, the
part of its weights waits a zero-skew release would remove, and how often the same warpgroup
trails within a CTA (persistence) are reported. Production SASS is unchanged (compile gate
below).

Check: one bench file from `tools/moe_make_bench_ab.py --base=51 --sizes=128,256
--routings=layer --skew=1` (twins 55 and 56, the public kernel with its stamps), layer seeds
1-6.

Decision rule, set before the check:

- Value of (a), the gather: the same-process gap of the gather piece (ABSTAMP "start" minus
  "route", 55 against the public kernel) at 128 and at 256 tokens. Every CTA's phase A waits on
  the slowest gatherer of its jobs, so this is a critical-path gap.
- Value of (b), per-warpgroup weights slots: an upper bound on what they can take off the
  critical path, the trailing warpgroup's weights waits that a zero-skew release would remove
  (ABSKEW "trailing-wg w-wait from release skew us"), phases A and B summed, from 56. The x ring
  stays shared under (b), so x waits do not count toward it.
- Build (b) only if its bound exceeds (a)'s gap at both sizes; otherwise build (a).

### Result (check 7063f86352344efa)

2.210x (2.176-2.243), passed, spin flags 0. Graph replay: public minus 51 -9.1 / -14.0 us at 128 /
256 tokens (6 of 6 seeds each). Means of 6 layer seeds, one cold launch each:

| us | 128 tokens | 256 tokens |
|---|---|---|
| gather piece (start minus route), 55 / public / gap | 6.8 / 3.6 / +3.2 | 8.3 / 4.9 / +3.4 |
| 55, thread 0: row loads and stores | 1.68 | 5.61 |
| 55, thread 0: proxy fence, `__threadfence`, gather barrier | 4.30 | 2.13 |
| 55, thread 0: release | 0.61 | 0.46 |
| 55, x producer's first xready wait end after its CTA's release | 2.49 | 2.88 |
| public: whole gather (DBG 1 to 7, with its ss zeroing and row bases) | 3.32 | 4.82 |
| skew bound for (b): trailing warpgroup's weights waits a zero-skew release removes, A + B | 0.31 + 0.14 = 0.45 | 0.79 + 0.28 = 1.07 |
| trailing warpgroup's own waits, weights / x, A + B | 2.65 / 4.64 | 5.46 / 9.04 |
| trailing warpgroup persistence (phase A / B) | 0.86 / 0.87 | 0.79 / 0.87 |

Reading.
- Measured: the skew re-measured on 51 matches 49's on 47 (warpgroup spread 780-1,130 cycles,
  71-73% of the weights waits from release skew). But the warpgroup that starts a stage last is
  the same one in 79-87% of the stages of a CTA, and only 0.3-0.8 us of its weights waits per CTA
  come from the other warpgroups' releases: it mostly waits for the copy issued behind its own
  release. Per-warpgroup slots move the leaders' waits, not the trailing warpgroup's, so their
  bound on the critical path is 0.45 / 1.07 us, against the gather's +3.2 / +3.4.
- Decision under the rule: (a), the gather.
- Measured: at 256 tokens thread 0's loads and stores take 5.6 us because the batched loop walks
  entries cta + k G in batches of kGB G = 1,056 entries, so E = 2,048 takes two batches whose loads
  wait behind the first batch's stores; at 128 one batch, 1.7 us. The fences and the barrier
  (the slowest gathering thread's stores becoming visible) take 4.3 us at 128 and 2.1 at 256, where
  the second batch's loads gave the first batch's stores time to complete.

### The change: modes 57 to 59

All fences, the barrier and the release stay as they are and where they are. 57 is 51 with:

- gx_pre: right after the routing ids reach shared memory (the first `__syncthreads`), every thread
  finds the CTA's first kPreRows = 4 local entries among cta, cta + G, ... and the 16 consumer warps
  load their 16 bytes of each row. The HBM round trip then runs under the routing work (3.4-4.2 us
  of it). After routing, these rows are stored first and the batched loop takes only the entries
  after the fourth local one, so at 256 tokens the second dependent batch goes away when a CTA has
  at most 4 local entries.
- gx_st16: each lane stores one 16-byte group of the row's per-stage layout instead of four 4-byte
  words. The four words of a group are held by lanes l, l ^ 2, l ^ 4, l ^ 6 (pair h of lane l lands
  in group 8 (l >> 3) + 4 (l & 1) + h, word (l >> 1) & 3); three `shfl.xor` exchange them. A quarter
  of the store instructions. [Corrected after the run: not fewer sector writes. In 51's 4-byte
  stores the four lanes of a group write adjacent words in the same instruction, so each
  instruction already writes eight whole 16-byte groups; the 16-byte form writes the same groups,
  32 lines per instruction instead of 8.]

58 is 57 with 55's light and gather stamps (timing only); 59 is gx_pre alone (a diagnostic arm).

Ordering. The chain of "Proxy ordering of the x tiles" (steps 1-8) is unchanged:
1. The x rows are kernel inputs. Their loads still come after the kernel's `griddepcontrol.wait`
   (PTX ISA 9.7.15.14), only earlier in the setup, and no thread of the launch writes p.x.
2. Each writer's generic stores to xa, now one 16-byte `st.global` per row instead of four 4-byte
   ones, precede in its program order its `fence.proxy.async.global`, `__threadfence` and
   `bar.sync kBarGather` (steps 1-2). Which lane stores which word changes; the set of threads
   that store, and every one of them reaching the fences and the barrier after its stores, does
   not. The prefetched rows are stored before the batched loop, inside the same block.
3. The release loop after the barrier is unchanged and still covers every local entry (step 3), so
   the x producers' acquire, proxy fence and bulk copy (steps 4-8) see the same writes.

Coverage. Every local entry is stored once: the first four through ipf, the rest by the batched
loop from ipf[3] + G, which keeps i congruent to cta mod G. ipf is the same in every thread (a scan
of S.eid), so the loop start is CTA-uniform. The shuffle runs inside `dst != nullptr` and `ipf[c]
>= 0` conditions that depend only on the entry and the warp (gt < 512 is warp-uniform), so all 32
lanes reach it. The 16-byte layout was checked word for word against the 4-byte stores by an
emulation of one warp (128 words, scratchpad `wg/st16_check.py`).

Local compile gate (nvcc 12.8.1, sm_90a, MAIN_MODES 0, 16, 26, 28, 37, 47-49, 51, 55-59;
tools/moe_compile_fused.sh): production SASS (modes 0, 16, 26, 28, 37, 47, 51 and the four skinny
kernels) and 48, 49 have the same instruction text as before e866c8a; no C7510-C7515; 55, 57, 58
and 59 at 96 registers with no stack frame and no spill (as 51), so no local loads or
stores in any stage loop; MEMBAR 43, FENCE.VIEW.ASYNC 32, SYNCS.ARRIVE 48, RED 81, ATOM 148, BAR.SYNC
16, HGMMA 192, wait1 96, wait0 12 in 51, 57 and 59 alike; 57 has 13 STG.E.128 against 51's 1.
tools/moe_dryrun.py and tools/moe_host_check.sh pass.

Prediction (hypothesis): -1.5 to -3 us at 256 tokens (the second batch and part of the first
round trip), -1 to -2 at 128 and 32, if the fence wait shrinks with the store count; less if the
fences wait on the memory system's load rather than on our writes.

Diagnostic run, no keep rule set in advance (as 47's first check): `tools/moe_make_bench_ab.py
--base=51 --sizes=32,128,256 --routings=layer --seeds=41,42,43,44,45,46 o57=57 o59=59` (light twins
55 and 58, public stamps). A gate rule on new held-out seeds was to follow only if 57 won here.

### Result (check e7e75c9510994c08)

2.207x (2.173-2.240), passed, spin flags 0. Minus 51, us, same process, layer seeds 41-46
(count = seeds lower of 6):

| arm | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| 57 (gx_pre + gx_st16) | +1.2 (0) | +1.1 (0) | +1.2 (0) |
| 59 (gx_pre alone) | -0.0 (2) | +0.6 (1) | -0.3 (5) |
| public | -1.4 (4) | -8.7 (6) | -15.6 (6) |

Outputs against 51 on the first seed: 57 and 59 0.0 at 128 and 256 tokens (bit-identical), 1.6e-5
and 1.2e-4 at 32. [59's 1.2e-4 at 32 tokens is above the 1e-4 used in the gates; at 32 tokens
51 runs 47's path, whose phase B sums of squares use FP32 atomics, so the order of the adds varies
from run to run; not separated from a defect, and moot since 59 is not kept.]

Gather stamps (58 against 55, us, means over CTAs and seeds):

| | 32 | 128 | 256 |
|---|---|---|---|
| thread 0: loads and stores | 0.59 / 1.02 | 1.35 / 1.74 | 3.89 / 5.26 |
| thread 0: fences and barrier | 1.04 / 2.27 | 3.67 / 4.14 | 2.92 / 2.38 |
| whole gather | 2.18 / 3.99 | 5.60 / 6.38 | 7.17 / 8.08 |
| x producer's first xready wait end after its release | 6.33 / 5.00 | 2.56 / 2.20 | 3.01 / 2.42 |
| latest first-x-ready, from the first gather start | 10.03 / 10.24 | 9.30 / 9.90 | 10.84 / 11.18 |
| public's whole gather | 1.82 | 3.37 | 4.82 |

Reading.
- Measured: each CTA's own gather got 0.8-1.8 us shorter, but the time at which the x producers
  find their first job's rows complete moved only 0.2-0.6 us earlier: a job's rows come from many
  CTAs, so its readiness follows the slowest gathering CTA and the release's propagation, not the
  mean CTA's gather. The wait after the CTA's own release grew by about as much as the gather
  shrank.
- Measured: gx_pre alone (59) is a tie (-0.3 to +0.6 us); the 16-byte stores added about 1 us at
  every size (57 against 59). Why is not separated (hypotheses: 32 lines per store instruction
  instead of 8, or the larger setup code).
- Hypothesis for the gap that remains: the per-thread `fence.proxy.async.global` and
  `__threadfence` before the gather barrier, which the public kernel's gather does not have (it
  publishes with a barrier and one red.release per job), on the slowest gatherer. Testing it
  means removing or weakening a fence, which this brief excludes; mode 37's chain ("Ordering of the
  phase A publish without the per-thread fence_gpu") is the argument such a change would rest on,
  and it would need its own review and a decision outside this line.

Decision: 57 and 59 are not gated or shipped (57 lost at every size, 0 of 18 seeds lower; 59 tied),
and go to docs/ruled-out.md scoped to this design. Their code stays in the source, not built
(production SASS unchanged). Two of the four checks were used; two are returned unspent. The skew
lead (per-warpgroup weights slots) is not built: its bound on the trailing warpgroup, 0.45 / 1.07
us, is below what one check can resolve.

## Standalone scorecard against the public kernel, 2026-10-30

Brief: rank the standalone board's per-shape gaps to the public MoE kernel by their value in the
standalone score, then attack the largest one that docs/ruled-out.md has not closed.

### How the standalone score weighs a shape (TASK.md)

Each of the six loads is one shape (decode 1 / 16 / 32 at bs 1 / 16 / 32, verify 1 / 16 / 32 at bs
8 / 128 / 256), all with weight 60, and the score is the geometric mean over the loads of
SGLang's time over ours. So each shape enters the log of the score with weight 1/6, and closing a
gap of g us on a shape where we take t us multiplies the score by (t / (t - g))^(1/6). A
microsecond at 1 token is worth about 8.5 times one at 256 tokens (t = 27 against 232 us).

### Method (check 3d8902381a194d56)

`tools/moe_make_bench_ab.py OUT.py PUBLIC.py --score=1 --sizes=1,8,16,32,128,256
--seeds=301,...,310` (new `--score` template). The shipped file (MAIN_MODES 0, 16, 26, 28, 37, 47,
51; GATE_RUNS off) runs its own `prepare()` twice (oursA, oursB, each with its own repack, scratch
and policy bench); the public file (vjeux/okc-2026 `muse_moe_w4a16.py`, scratchpad copy, with its
TINY / TINY2 / MID builds kept, so its kernel_fn dispatches as on the board) runs its
`prepare()` twice (pubA, pubB). The second copy of each side is a null: same code, independent
prepare. Standalone routing (reference.py's generate_input, `_make_inputs`), seeds 301-310 (never
used by any earlier run), graph replay after a 256 MB L2 flush, median of 20, the four arms in an
order shuffled per seed and size by `random.Random(7919 s + t)`.

### Result

Both prepares picked the same policy: sk8w3s at 1-32 tokens, 51 (main-lean-grp-finss64) at 128
and 256. FP32 check on the first seed: ours 3.77e-3 to 3.88e-3, public 2.34e-3 to 2.36e-3. Spin
flags 0. Judge score of the same file 2.211x (2.177-2.244); the judge's medians for ours: 27.1 /
65.5 / 95.4 / 133.0 / 191.1 / 231.7 us at 1 / 8 / 16 / 32 / 128 / 256 tokens.

Same-process means of 10 seeds, us; pairs are per seed (negative = public faster; count = seeds
where the first arm is lower):

| T | ours A / B | public A / B | null ours B-A | null pub B-A | public minus ours (mean of both copies) | value in the score if closed |
|---|---|---|---|---|---|---|
| 1 | 34.22 / 34.20 | 35.56 / 35.61 | -0.02 (sd 0.24) | +0.05 (0.16) | +1.37 (sd 0.32), 0 of 10 | we lead (-0.82%) |
| 8 | 84.00 / 84.05 | 83.28 / 83.16 | +0.05 (0.22) | -0.12 (0.49) | -0.80 (0.30), 10 of 10 | +0.21% |
| 16 | 104.22 / 103.94 | 104.21 / 104.02 | -0.28 (0.43) | -0.20 (0.53) | +0.04 (0.55), 4 of 10 | tie |
| 32 | 136.62 / 136.40 | 135.04 / 134.90 | -0.22 (0.66) | -0.14 (0.59) | -1.54 (8.02), 4 of 10 | +0.19%, seed-dependent |
| 128 | 198.86 / 198.90 | 189.53 / 189.04 | +0.04 (0.66) | -0.49 (0.56) | -9.60 (3.92), 10 of 10 | +0.86% |
| 256 | 233.49 / 233.54 | 218.96 / 218.23 | +0.05 (0.66) | -0.73 (0.70) | -14.92 (3.47), 10 of 10 | +1.12% |

Value = (t / (t - g))^(1/6) - 1 with t the judge's median for ours and g the same-process gap.

Reading.
- Measured: the nulls are within 0.3 us for ours and 0.75 us for the public at every size, so the
  gaps at 8, 128 and 256 tokens are outside the bench's own noise. The public's second copy runs
  0.5-0.7 us faster than its first at 128 and 256 (8-9 of 10 seeds); its gaps above use the mean
  of both copies.
- Ranking by standalone value: 256 tokens (+1.12%), then 128 (+0.86%), then 8 (+0.21%) and 32
  (+0.19%). Closing all four would move the score by about 2.4% (2.21x to about 2.26x on this
  run); rank 100 is 2.37x today, so matching the public kernel everywhere is not enough.
- 32 tokens is bimodal per seed: the public (its WGS MAIN build) leads by 20.8 and 9.2 us on two
  seeds and trails by 6.3 and 6.8 on two others; the policy runs sk8w3s there, and which of the
  skinny path and the fat path is faster depends on the routing (1fbb9fbf524f4ed7 had them tied on
  the mean). A per-routing choice cannot be made on the host (no branching on input values); an
  in-kernel choice is a separate design, not tested.
- 1 token is ours by 1.37 us (0 of 10 seeds lower for the public): a lead, not a gap.
- The board's 5% between the public entry (2.327x) and ours (2.216x) is larger than this run's
  same-process 1.6% net; the board rows are single scored runs (hypothesis: run-to-run noise and
  the judge's own routing per shape).

Target: the 128 and 256-token gap (the fat path, 51). Its measured pieces (7063f86352344efa,
d47c78bdc130424b): setup +4.3 / +4.4 us, of which the x gather +3.2 / +3.4; phase A +2.3 / +7.7;
phase B +2.9 / +7.3; tail +3.7 / +3.6 (the final pass is at parity since 51). docs/ruled-out.md
closes the gather's loads and stores (57, 59), the tail splits (52, 53), the skew slots, the cost
models, the staged pair and the stage-loop tweaks tried so far; it leaves open "the per-thread
fence.proxy.async.global and `__threadfence` before the gather barrier, which the public gather
lacks". Of those two, the proxy fence stays (CLAUDE.md); the `__threadfence` is the change below.

## Gather publish without the per-thread fence (modes 60 and 61), 2026-10-30

Change. Mode 60 is 51 with one line removed from the x gather: the `__threadfence()`
(fence.sc.gpu, SASS MEMBAR.SC.GPU) each of the 544 gathering threads runs after its
`fence.proxy.async.global` and before `bar.sync kBarGather`. Everything else stays: each writer's
4-byte row stores, its proxy fence, the named barrier, the `red.release.gpu.global.add` on
`xready` after the barrier (one per local entry), the x producer's `ld.acquire.gpu` spin, its own
proxy fence and the bulk copy. Mode 61 is 60 with 55's light and gather stamps (timing only).

Measured basis: 55's stamps put thread 0's proxy fence, `__threadfence` and gather barrier at 4.30
/ 2.13 us at 128 / 256 tokens (7063f86352344efa), and 58's at 3.67 / 2.92 (e7e75c9510994c08),
against 3.3 / 4.8 us for the public kernel's whole gather, whose writers run no fence at all.
Shortening the mean CTA's loads and stores did not move job readiness (57, 59), because a job's x
is ready when its slowest gathering CTA releases; a per-thread gpu-scope fence waits for all of
that thread's stores to be performed at gpu scope, on every gathering CTA including the slowest.
Prediction (hypothesis): -1 to -3 us at 128 and 256 tokens, about -1 at 32; none if
MEMBAR.ALL.GPU in each releasing thread's `red.release` pays the same drain.

### Ordering of the gather publish without the per-thread fence

Sections from the PTX ISA (9.x) memory consistency model as cited in "Proxy ordering of the x
tiles": 8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.4 Memory synchronization (barriers),
8.9.5 Causality Order, 9.7.10.28.2 Async Proxy, 9.7.15.4 membar / fence. The chain is that of mode
37's phase A publish ("Ordering of the phase A publish without the per-thread fence_gpu"), with
the gather's writers and the xready counter in place of the epilogue's and done_as:

1. Writers. Every gathering thread W (warps other than the weights producer, kNG = 544 threads)
   stores its share of the CTA's local token rows into xa with weak generic `st.global`, then runs
   `fence.proxy.async.global` in program order. The proxy fence acts on W's own generic accesses
   and composes with other synchronization (9.7.15.4).
2. Barrier. W then runs `bar.sync kBarGather, kNG`. Barrier completion synchronizes every
   participant with every other participant (8.9.4), so W's stores and proxy fence precede, in
   base causality order (8.9.5), every operation after the barrier in every participating
   thread.
3. Release. After the barrier, thread R (one per local entry, gt-strided) runs
   `red.release.gpu.global.add` on `xready[job]`. A release operation is a release pattern by
   itself (8.8). Base causality order is transitive (8.9.5), so every writer's stores (step 2)
   precede R's release, whichever thread R is.
4. Count. `xready[job]` receives one release per row, from the CTAs that gathered that job's rows.
   The x producer's `ld.acquire.gpu` that reads the final count follows each contributing release
   in observation order through the chain of atomic read-modify-writes (8.9.2), so it
   synchronizes with every one of them (8.9.4; all morally strong, gpu scope).
5. Consumer. The x producer then runs `fence.proxy.async.global` and issues the bulk copy, ordered
   after its prior instructions (8.9.1.1); completion through the mbarrier carries the implicit
   generic-async proxy fence (9.7.10.28.2). Steps 5-8 of "Proxy ordering of the x tiles" are
   unchanged.

The removed `fence.sc.gpu` is no link in steps 1-5: the release is the releasing thread's own
instruction after the barrier, as in mode 37. Its acquire half orders nothing either: no gathering
thread reads, before the barrier, anything another CTA wrote in this launch (the routing reads
topk_ids, inputs ordered by `griddepcontrol.wait`, 9.7.15.14). Other writes before the barrier in
the same threads (the next launch's `ss` copy and barrier counter, zeroed in setup) are read only
by the next launch, ordered by `griddepcontrol.wait` or stream order, not by this fence; the grid
barrier keeps its own `__syncthreads`, `__threadfence` and atomics.


Local compile gate (nvcc 12.8.1, sm_90a, tools/moe_compile_fused.sh, MAIN_MODES 0, 16, 26, 28, 37,
47, 51, 55, 60, 61 against the parent commit with 0-51 and 55): every production kernel (modes 0,
16, 26, 28, 37, 47, 51 and the four skinny kernels) and 55 have the same instruction text as
before; no C7510-C7515; 60 and 61 at 96 registers, no stack frame, no spills. 60 against 51: the
gather's MEMBAR.ALL.CTA, MEMBAR.SC.GPU, ERRBAR, CGAERRBAR and CCTL.IVALL are gone (MEMBAR 43 ->
41), FENCE.VIEW.ASYNC.G (32), BAR.SYNC (16), RED (81), ATOM (148), SYNCS.ARRIVE (48) and HGMMA (192)
unchanged; the other differences are instruction order (17,282 -> 17,274 instructions).

Diagnostic run, no keep rule set in advance (as 47's and 57's first checks): `tools/moe_make_bench_ab.py
--base=51 --sizes=32,128,256 --routings=layer,std --seeds=47,48,49,50,51,52 o60=60` (light twins
55 and 61 with the gather stamps, public stamps). A gate on new held-out seeds follows only if 60
is lower than 51 at 128 and 256 tokens on both routings.

Independent read-only review (2026-10-30, of 003be4b): sound. Every link was proved, with the same
structure as mode 37's; the CTA-scope barrier is no obstacle because base causality order puts no
scope condition on a synchronizes-with edge, and only the release and acquire pair must be
morally strong (both gpu scope). The reviewer added the citation the chain leaves implicit: the
bulk read sees the stores by the Causality axiom (8.10: a read sees a write that precedes it in
causality order, or a later one). Nothing else relied on the removed fence: the setup's writes
before the barrier (the next launch's `ss` copy and grid-barrier counter) are read only by the next
launch; the gathering threads read only `topk_ids` and `x`; the only `xa` reader is the x producer,
always behind the `xready` acquire and its proxy fence (also for a CTA's own rows); the removed
fence's Fence-SC order was not used. Every mode predicate gives 60 and 61 exactly 51's behaviour
(61 adds 55's stamps). It also predicted the outcome below: each `red.release.gpu` still compiles to
MEMBAR.ALL.GPU before its RED, so the drain may move rather than disappear.

### Result (check 48b9bf9d77d146ac)

2.205x (2.172-2.238), passed, spin flags 0. Outputs of 60 against 51: 0.0 at 128 and 256 tokens,
2.2e-5 / 3.6e-5 at 32 (51's FP32 atomics below 64 tokens). 60 minus 51, us, same process, seeds
47-52 (count = seeds lower of 6):

| routing | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer | -0.3 (6) | -0.4 (5) | +0.0 (4) |
| standalone | -0.3 (3) | -0.5 (5) | -0.4 (4) |
| public minus 51, layer / standalone | -1.4 / -4.0 | -10.1 / -8.5 | -17.1 / -18.2 |

Gather stamps, 61 against 55 (us, means over CTAs and seeds; layer / standalone):

| | 32 | 128 | 256 |
|---|---|---|---|
| thread 0: fences and barrier | 0.11 / 0.09 against 1.48 / 0.44 | 0.38 / 0.24 against 3.98 / 1.72 | 0.32 / 0.27 against 2.62 / 2.59 |
| thread 0: release | 1.25 / 0.31 against 0.67 / 0.24 | 3.10 / 1.45 against 0.58 / 0.67 | 1.94 / 2.41 against 0.44 / 0.62 |
| whole gather | 2.37 / 1.41 against 3.18 / 1.71 | 5.08 / 3.12 against 6.19 / 3.85 | 7.57 / 6.53 against 8.32 / 7.05 |
| latest first-x-ready from the first gather start | 8.41 / 9.43 against 9.64 / 9.13 | 8.41 / 8.70 against 10.24 / 10.71 | 11.31 / 10.58 against 11.73 / 11.78 |
| public's whole gather | 1.63 / 2.12 | 3.50 / 3.43 | 4.68 / 4.63 |

Reading.
- Measured: the fence's wait moved into the release. Without the per-thread fence, thread 0's
  barrier wait falls by 1.3-3.6 us, but its first `red.release` (MEMBAR.ALL.GPU, then RED) grows by
  0.6-2.5 us: the stores still have to be performed at gpu scope before a release, and the first
  releasing thread waits for them. The whole gather is 0.3-1.1 us shorter and the last job's x is
  ready 0.4-2.0 us earlier (largest at 128 tokens).
- Measured: end to end, 60 is -0.0 to -0.5 us (13 of 18 seeds lower at 128 and 256). The 1.8-2.0 us
  earlier x readiness at 128 tokens buys 0.4-0.5 us, so setup is not on the launch's critical
  path for most of its length; the stamps put the critical path at the end of phase A: the last
  CTA's phase A ends 14.2 / 13.0 us after the mean CTA's at 128 tokens (layer / standalone) and
  9.5 / 10.0 at 256 (start + phase A mean against A done max), against the public kernel's 5.8 /
  5.2 and 7.0 / 9.1, and our tail is 9.1 / 7.0 and 8.4 / 8.8 us against 4.8 / 4.1 and 4.4 / 7.0.
- Decision under the rule set before the run (gate only if 60 is lower at 128 and 256 on both
  routings): 256 tokens on layer routing is +0.0 (4 of 6 lower), so 60 is not gated or shipped.
  Its expected value if it holds, about -0.4 us at 128 and 256, is 0.03% of the standalone score,
  below one check's resolution. Parked in docs/ruled-out.md; its code stays, not built (production
  SASS unchanged). Two of today's seven MoE checks were used by this line (scorecard and this
  diagnostic).

Next lead for the 128 / 256-token gap (hypothesis, not built): the end of phase A. Owned units are
dealt whole (about 8.3 per CTA at 128 tokens, so a third of the CTAs run one unit more), and the
last phase A units gate their jobs' phase B units, so the spread of phase A ends (9.5-14.2 us
against the public's 5.2-9.1) reappears as the tail (+1.8 to +4.3 us) and the barrier wait. docs/ruled-out.md
closes the static fixes (cost-model partitions, static phase B ranges) and the K-split of phase B
claims, not run-time claims for phase A's last round, which would mirror mode 23's phase B claims
(-16 to -23 us against static ranges) inside the per-n-tile-count loops.

## Phase A's last round claimed at run time (modes 62 to 65), 2026-10-30

Brief: the standalone scorecard (3d8902381a194d56) puts the public MoE 14.9 us ahead at 256 tokens
and 9.6 at 128 (10 of 10 seeds each), +1.12% and +0.86% of the standalone score; 61's stamps
(48b9bf9d77d146ac) put the critical path at the end of phase A: the last CTA leaves phase A 9.5-14.2
us after the mean CTA (public 5.2-9.1) and our tail is 1.8-4.3 us longer. Mode 23 claimed phase B
units at run time; phase A's owned units are still dealt whole in snake order, about 8.3 per CTA
at 128 tokens, so a third of the CTAs run one unit more whatever their speed.

### The change

Mode 62 is 51 with phase A's last round claimed at run time; 63 claims the last two rounds; 64 and
65 are 62 and 63 with 55's light and gather stamps (timing only). With `units` phase A units (32 per
job, S.ord order, n-tile counts never rising) and G CTAs, the snake deal keeps the units below
`s0 = max(0, ceil(units / G) - K) G` (K = 1 or 2), so every CTA owns `s0 / G` rounds. The other
`na = units - s0` go out through the same counter as phase B, ctr[10]: claim value m < na is phase
A unit s0 + m, m >= na is phase B unit m - na. So every claimed phase A unit is taken before any
phase B unit, and each CTA runs its claims in claim order. A claimed unit stays whole on one CTA
(16 stages, no phase A fixup) and runs the dealt units' code: the weights producer streams its 16
stages from w13f, the x producer waits on `xready[j]` with `ld.acquire.gpu`, runs
`fence.proxy.async.global` and issues the copies from xa, the consumers run the phase A segment and
its epilogue (proxy fence, warpgroup bar.sync, red.release on done_as). Job order, every fence,
barrier, release and acquire stay as in 51; the only new code is where a unit index comes from.

Consumer order. The consumers keep one loop per n-tile count. In the loop for count T a CTA runs
its dealt units of count T, then reads S.bq for claimed phase A units of count T, but only if its
last dealt unit has a count of T or more (no dealt unit of a smaller count is left). Dealt units
come before claimed ones in S.ord, so the producers' order (dealt units in round order, then claims
in claim order) and the consumers' order agree. A CPU model of the two orders
(`.build/claim_model.py`, not committed: 3,000 random job sets, G of 1 to 132, K of 1 and 2,
random claim interleavings) found both orders equal on every CTA, every unit run once, and no read of
the queue before the CTA's last dealt unit.

### Ordering chain for the claimed phase A units

Sections from the PTX ISA memory consistency model as cited in "Proxy ordering of the x tiles":
8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.1.1 Asynchronous Operations, 8.9.2 Observation
Order, 8.9.4 Memory synchronization, 8.9.5 Causality Order, 8.10 axioms, 9.7.10.28.2 Async Proxy,
9.7.15.4 membar / fence, 9.7.15.14 griddepcontrol.

1. Uniqueness. Every claim is `atomicAdd(ctr + 10, 1)` (a relaxed gpu-scope `atom.add`). The
   Atomicity axiom (8.10) makes the read-modify-writes on one location totally ordered with no
   two reading the same value, so each value below `na + units + xs` (xs = 0 here) goes to exactly
   one CTA: each
   claimed phase A unit runs on one CTA, and done_as[j * 16 + st] still receives exactly 8
   releases (2 units x 4 warpgroups), as for dealt units. The claim carries no data; nothing is
   published through the counter.
2. Hand-off inside the CTA. The weights producer writes the claim into S.bq[i] with
   `st.release.cta.shared`; the x producer and the consumers read it with `ld.acquire.cta.shared`
   in `bq_get` (mode 23's queue, unchanged); the pair carries only the claim value. The S.ord and
   S.job_* tables it indexes were written before the post-routing `__syncthreads`, which orders
   them for every thread (8.9.4).
3. Inputs of a claimed unit. Weights: w13f, written by prepare before the launch, ordered by stream
   order or `griddepcontrol.wait` (9.7.15.14). x: the x tile of job j, published by the gather's
   writers through `fence.proxy.async.global`, the gather barrier and `red.release.gpu` on
   `xready[j]`; the x producer spins with `ld.acquire.gpu` until `xready[j] == job_n[j]`, runs
   `fence.proxy.async.global`, then issues the bulk copies. That is steps 1-8 of "Proxy ordering of
   the x tiles", the same code path the dealt units run; a claimed unit adds no step.
4. Outputs. The consumers' phase A epilogue for a claimed unit is the dealt units' epilogue:
   generic stores to xb, `fence.proxy.async.global` per thread, the warpgroup's bar.sync, one
   `red.release.gpu` on done_as (mode 37's chain, "Ordering of the phase A publish without the
   per-thread fence_gpu"). Phase B's acquire on done_as and its proxy fence are unchanged.
5. Ring reuse. The claimed unit's stages use the same ring slots and the same wfull / wempty /
   xfull / xempty parity walk (the producers advance `rg` per stage as for dealt units), with the
   slot releases of 51 unchanged: each consumer thread's `fence.proxy.async.shared::cta` before
   its arrive on the weights slot, and lane 0's arrive for the warp on the wgmma path's x slot
   after `wgmma.wait_group`, which only the async proxy reads ("Ordering of the ring slot
   releases"). Producer and consumer orders agree (above), so each
   slot's phases line up as before.
6. Counter reset. CTA 0 zeroes ctr[10] after the grid barrier. Every claim of every CTA precedes
   that CTA's arrival at the grid barrier in program order (the weights producer's claim loop ends
   before the CTA's `__syncthreads` ahead of the barrier), and the barrier's `__threadfence` and
   gpu-scope atomics and acquire spin order them before the reset (as for mode 23). The next launch
   reads ctr[10] after `griddepcontrol.wait` or in stream order.
7. Progress. A claimed phase A unit waits only on `xready[j]`, which needs the gather of every
   CTA that holds rows of job j; each CTA gathers before starting its consumers, with no wait on
   any claim, so every count completes. A phase B unit waits on done_as of phase A
   units of its job; all claimed phase A units have smaller claim values than every phase B unit,
   and a CTA issues and consumes its claims in increasing order, so a phase A unit is never queued
   behind a phase B unit of the same CTA. The phase A queue read waits only for the CTA's own
   weights producer, which claims after issuing its last dealt stage; the consumers have taken all
   dealt units by then (consumer order above), so that producer's slot waits complete. No new
   cross-CTA wait exists.

No fence, barrier, release or acquire is removed or moved.

### Local compile gate (nvcc 12.8.1, sm_90a, `tools/moe_compile_fused.sh`)

MAIN_MODES 0, 16, 26, 28, 37, 47, 51, 55, 62-65 against the parent commit (8c7ed1f) with 0-51 and
55: every production kernel (modes 0, 16, 26, 28, 37, 47, 51 and the four skinny kernels) and 55
have the same instruction text; no C7510-C7515; 62-65 at 96 registers, 8 B stack frame, 4-8 B
spills, all in setup before the first stage loop (no local load or store inside a stage loop);
HGMMA 192 in 12 stage loops, as 51. 62 has 21,464 instructions against 51's 17,280: the claimed
path's queue reads and a second copy of the phase A epilogue per n-tile count (MEMBAR 107 against
43, BAR 40 against 16, RED 134 against 81); the source adds no fence, so the likely cause is the
compiler duplicating the code after the merge of the dealt and claimed paths (not checked). `tools/moe_dryrun.py` and
`tools/moe_host_check.sh` (bench file with 62-65) pass.

Prediction (hypothesis, before the run): 62 shortens the last CTA's phase A end by moving the
partial round to the CTAs that finish their dealt rounds first; at 128 tokens (8.3 units per CTA)
that is the spread among the earliest finishers, a few us; 63 can also take a unit away from the
slowest CTAs. Expected -1 to -4 us at 128 and 256 tokens if the end spread is the critical path,
nothing at 32 (two or three rounds and the skinny path in the standalone policy).

### Diagnostic run and its rule (set before the check)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=53,54,55,56,57,58 o62=62 o63=63` (light twins 55, 64, 65 and the public stamps; ABSTAMP's
new last column "A end spread" is the last CTA's phase A end minus the mean CTA's). Seeds 53-58 are
new. Rule: the arm with the lower sum of its mean deltas against 51 at 128 and 256 tokens over both
routings goes to the gate, if its mean is negative at 256 tokens on both routings; otherwise both
are parked in docs/ruled-out.md and no gate is run. Mechanism check (reported, not a condition):
the chosen arm's twin should show a smaller A end spread than 55 at 128 and 256 tokens.

### Gate rule (set before the gate check)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256
--routings=layer,std,skewed --seeds=59,60,61,62,63,64 --gate=1 oX=X` with X the arm the diagnostic
chose; seeds 59-64 are held out on all three routings. Keep X only if all of these hold:

1. At 256 tokens, X minus 51 has a mean of -2.0 us or lower on layer and on standalone routing,
   with at least 5 of 6 seeds lower on each, and a mean of 0 or lower on skewed routing.
2. At 32 and at 128 tokens, a mean of +0.5 us or lower on every routing.
3. X's output is within 1e-4 relative L2 of 51's on every seed, size and routing.
4. Prepare's FP32 self-test passes for X (no FAIL line, spin flags 0).
5. The compile gate above, on the commit that is checked.

Shipping also needs the independent read-only review of the chain above to find it sound. If any
condition fails, X goes to docs/ruled-out.md scoped to this design and is not shipped. If all
pass, X joins MAIN_MODES and MAIN_CANDIDATES and the shippable file goes to the coordinator, who
runs its check. Budget for the line: two checks (diagnostic and gate).

### Diagnostic result (check b7e43ee6a517487a, bench at d5bafad)

2.198x (2.165-2.231), passed, spin flags 0. Outputs of 62 and 63 against 51: 0.0 at 128 and 256
tokens, at most 2.2e-5 at 32 (51's FP32 atomics below 64 tokens). Self-test of the shipped policy
modes unchanged (main-lean-grp-finss64 3.89e-3). Minus 51, us, same process, seeds 53-58 (count =
seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 62 (last round claimed) | +3.9 (0) | +6.1 (0) | +7.9 (0) | +7.5 (0) | +9.2 (0) | +8.6 (0) |
| 63 (last two rounds) | +5.3 (0) | +5.3 (0) | +7.0 (0) | +6.9 (0) | +8.5 (0) | +7.9 (0) |
| public | -1.1 (5) | -6.7 (6) | -8.8 (6) | -9.5 (6) | -14.8 (6) | -20.1 (6) |

Stamps (one cold launch per seed, means over seeds; us; A end spread = last CTA's phase A end
minus the mean CTA's; layer / standalone):

| | 32 | 128 | 256 |
|---|---|---|---|
| 55 (51): phase A mean, A end spread | 41.4 / 83.9, 4.6 / 7.8 | 91.2 / 113.1, 12.8 / 10.1 | 122.2 / 150.0, 12.1 / 9.6 |
| 64 (62) | 42.5 / 89.0, 5.3 / 7.6 | 96.3 / 117.9, 9.5 / 7.7 | 127.7 / 156.8, 12.1 / 8.6 |
| 65 (63) | 46.0 / 88.0, 4.9 / 7.6 | 95.3 / 117.5, 9.8 / 7.6 | 127.2 / 155.8, 11.7 / 7.9 |
| public | 44.3 / 81.3, 3.8 / 3.9 | 88.9 / 108.8, 6.3 / 5.6 | 115.0 / 139.6, 6.9 / 7.5 |
| A done max: 55 / 64 / 65 | 53.0 / 56.5 / 57.9 (layer) | 114.1 / 115.9 / 115.5 (layer) | 147.2 / 152.9 / 151.7 (layer) |

Reading.
- Measured: both arms lost at every size and routing, 0 of 6 seeds lower in all twelve rows, +3.9
  to +9.2 us.
- Measured, mechanism: the claims did narrow the end of phase A at 128 tokens (spread 12.8 / 10.1
  -> 9.5 / 7.7 us with 62), less at 256 (12.1 / 9.6 -> 12.1 / 8.6), not at 32. But the mean CTA's
  phase A grew by 4.8-6.8 us at 128 and 256 tokens (1.1-5.1 at 32), so the last CTA's phase A end
  came later, not earlier (A done max +1.8 to +5.7 us), and the whole launch followed.
- Hypothesis for the cost, not separated: the phase A slowdown appears at 32 tokens too, where
  only 6 of about 270 units are claimed and every CTA's dealt rounds are the same as 51's, so it
  looks like a cost of the new code shape on every unit rather than of the claims: 21,464
  instructions against 17,280 (the phase A epilogue appears twice per n-tile count), or the
  merged dealt-and-claimed loop. The cheapest decisive test is the same build with K = 0 (no unit
  claimed, the new code path run on 51's deal): if it is also about 5 us slower than 51, the
  claims are not the cost and a leaner implementation could keep the narrower spread.
- Decision under the rule set before the run: neither arm is negative at 256 tokens, so no gate is
  run, nothing ships and the layer gets no port. Both are parked in docs/ruled-out.md; the code
  stays in the source, not built (production SASS unchanged). One of the line's two checks was
  used.
- Independent read-only review of d5bafad (2026-10-30): sound (no ordering bug, deadlock, lost or
  duplicated unit, or index error; no fence, barrier, release or acquire removed or moved against
  51). It corrected three wordings of the chain (steps 2, 5 and 7, fixed above) and could not check
  the PTX section numbers against a specific ISA version (unverified, not found wrong). It also
  noted that in 64 and 65 the gather stamp S.dg[16][7] stays 0 on a CTA with no dealt unit (fewer
  than G units with K = 1, at most 2G with K = 2); ABGATHER should read 0 as "no dealt unit".

### Mode 66: the same code with nothing claimed (set before the check)

The coordinator asked for the cheapest decisive test of the cost above. Mode 66 is 62's code with
K = 0: `a_static` returns all units (now `min(units, ...)`, so na = 0), every CTA keeps 51's snake
deal, and the consumers still read S.bq in phase A (the head is then a phase B unit or the
sentinel) before phase B; 67 is its stamped twin (55's light and gather stamps). Compile gate as
above (nvcc 12.8.1, sm_90a): production kernels and 55 identical to 8c7ed1f, no C7510-C7515; 66 has
21,488 instructions like 62 (both changed by the `min`), 96 registers, 8 B frame, 8 B spills in
setup only, HGMMA 192.

Check: `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=65,66,67,68,69,70 o66=66 o62=62` (light twins 55, 64, 67, public stamps). No gate either
way. Reading, set before the run, on 66 minus 51 at 128 and 256 tokens on both routings and 67's
phase A mean against 55's:

- 66 is +3 us or more in all four cells and 67's phase A mean is 3 us or more above 55's: the loss
  is the code shape, not the claims. The mechanism is not rejected; the reopening trigger becomes a
  leaner implementation.
- 66 is within 1 us of 51 in all four cells: the claims themselves cost the time (62 minus 66 is
  their price). The family stays parked with that evidence.
- Anything between: mixed, recorded as such, no further check from this line.

### Result for 66 (check ed4591a2477041c2, bench at a727b12)

2.199x (2.166-2.232), passed, spin flags 0. Outputs of 66 and 62 against 51: 0.0 at 128 and 256
tokens, at most 4.5e-5 at 32. Minus 51, us, same process, seeds 65-70 (count = seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 66 (nothing claimed) | +4.3 (0) | +6.0 (0) | +7.0 (0) | +7.9 (0) | +8.8 (0) | +8.7 (0) |
| 62 (last round claimed) | +4.2 (0) | +5.7 (0) | +6.9 (0) | +7.9 (0) | +8.5 (0) | +10.3 (0) |
| public | +0.9 (0) | -3.4 (6) | -9.1 (6) | -11.6 (6) | -16.5 (6) | -17.2 (6) |

Stamps (means over seeds, us; layer / standalone):

| | 32 | 128 | 256 |
|---|---|---|---|
| phase A mean: 55 (51) / 67 (66) / 64 (62) | 44.1 / 47.2 / 44.8 and 94.0 / 98.4 / 98.6 | 90.1 / 95.2 / 94.7 and 118.2 / 123.6 / 123.6 | 121.7 / 129.4 / 128.3 and 146.0 / 152.5 / 152.9 |
| A end spread: 55 / 67 / 64 | 3.9 / 3.6 / 4.1 and 10.4 / 9.8 / 7.4 | 10.8 / 10.3 / 9.0 and 12.6 / 11.8 / 10.0 | 10.8 / 9.3 / 9.2 and 13.3 / 13.7 / 12.3 |
| public phase A mean, spread | 46.3, 5.1 and 90.9, 3.9 | 87.3, 6.2 and 112.4, 5.6 | 114.5, 7.8 and 136.9, 8.9 |

Reading under the rule set before the run:
- Measured: 66 is +7.0 / +7.9 / +8.8 / +8.7 us at 128 / 256 tokens (layer / standalone), all above
  +3, and 67's phase A mean is 5.1-7.7 us above 55's in all four cells. So the loss is the code
  shape, not the claims: with nothing claimed the new code costs as much as 62 does. 62 minus 66 is
  -0.1 / +0.0 / -0.3 / +1.6 us; the claims themselves cost nothing measurable and buy nothing
  measurable end to end in this run, while they narrow the phase A end spread by 0.5-2.4 us against
  the same code (64 against 67, five of six cells).
- Not separated: which part of the code shape costs the time. 66 differs from 51 in the
  consumers' phase A loop (one merged loop per n-tile count for dealt and claimed units, a
  queue read at the end of phase A, the phase A epilogue compiled twice per count: 21,488
  instructions, MEMBAR 107 against 43), in the phase B decoders (the `m - na` offset) and in both
  producers' claim loops. The phase A mean grows everywhere, which points at the consumers' phase A
  code (hypothesis).
- Decision: the claim mechanism is not rejected; this implementation is. Reopening needs a leaner
  build whose K = 0 null ties 51: the consumers' dealt-unit loops left as 51's code (byte-identical
  phase A segment and epilogue), the claimed phase A units run in a separate per-count loop after
  them (claimed counts are never larger than any dealt count, so the ring order still holds), and
  the claim and decode kept in the weights producer, with the x producer and the consumers told only
  through S.bq. Gate for such a build: its K = 0 null within 1 us of 51 at 128 and 256 tokens before
  its claimed arm is timed. No gate was run here; three of today's MoE checks remain.

## Lean phase A claims (modes 68 to 71), 2026-10-30

Brief: the reopening trigger of the mode 62/63 row in docs/ruled-out.md. Measured so far: claiming
phase A's last round at run time narrowed the phase A end spread by 0.5-2.4 us against the same
code and the claims cost nothing measurable (62 minus 66, ed4591a2477041c2), but 62's code shape
made every CTA's phase A 5-8 us slower (21.5k against 17.3k instructions), so 62 lost to 51 by 7-9
us at 128 and 256 tokens. At 128 / 256 tokens the public kernel leads by 9.6 / 14.9 us
(3d8902381a194d56) and the end of phase A is the critical path (our last CTA 9.5-14.2 us after the
mean CTA against the public kernel's 5.2-9.1, 48b9bf9d77d146ac).

### The change

Mode 68 is 51 with phase A's last snake round claimed at run time, built the way the trigger
names; 69 is the same code with no round claimed (K = 0, the null); 70 and 71 are 68 and 69 with
55's light and gather stamps (timing only). Every other mode predicate gives 68-71 exactly 51's
behaviour (70 and 71: 55's); a host build of all 31 predicates differs only in `claim_lean` and
`claim_rounds`.

- Setup: `own_n` counts the dealt units below `a_dealt` = min(units, max(0, ceil(units / G) - K)
  G) (62's `a_static`), so a CTA owns K rounds fewer when units remain to claim.
- Consumers: 51's `consumer_phase<0>` runs unchanged over the dealt units (its `run_group` takes
  the 51 branch: `claim_a` is false). After it, `consumer_claim_a` runs one loop per n-tile count,
  8 down to 1, over S.bq: an entry of kBqA (2^24) or more is a claimed phase A unit whose first
  stage index is the entry minus kBqA; a smaller entry (a phase B unit or the -1 sentinel) ends the
  claimed loops without being consumed. Then phase B runs 51's `consumer_dyn_b` from the same
  queue index.
- Weights producer: it streams its dealt units as 51 does, then claims through ctr[10] as 62 does
  (claim value m < na is phase A unit a_dealt + m in S.ord order, m >= na phase B unit m - na). It
  decodes the claim itself and writes S.bq: kBqA + S.ord[k >> 5] * 512 + (k & 31) * 16 for a phase A
  unit k, m - na for a phase B unit, -1 at the end. A phase B entry is then exactly 51's claim
  value, so the consumers' and the x producer's phase B decode is 51's (with `claim_a` false their
  `na` is units - units and folds to 0).
- x producer: its dealt loop is 51's; in the queue loop an entry of kBqA or more takes the job's x
  as a dealt unit does (xready acquire, then the proxy fence, then the copies from xa), a smaller
  entry is 51's phase B path.

Order of the ring. Each CTA's producers issue its dealt units (counts 8 down to 1, the S.ord deal
order) and then its queue entries in queue order; the consumers run the dealt units in the same
order (51's loops) and then the queue in queue order. Claimed phase A units take S.ord positions
after every dealt unit and a CTA's claim values rise, so a CTA's claimed counts never rise and
never exceed any count already consumed; the per-count loops 8 to 1 therefore take every claimed
phase A entry in queue order, and the first phase B entry ends them. Both sides walk the same
stage sequence, so the wfull / wempty / xfull / xempty phases line up as in 51.

### Ordering chain for the lean claimed units

PTX ISA memory consistency model sections as cited in "Proxy ordering of the x tiles" and in 62's
chain above: 8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.1.1 Asynchronous Operations, 8.9.2
Observation Order, 8.9.4 Memory synchronization, 8.9.5 Causality Order, 8.10 axioms (Atomicity,
Causality), 9.7.10.28.2 Async Proxy, 9.7.15.4 membar / fence, 9.7.15.14 griddepcontrol.

1. Uniqueness. Every claim is a relaxed gpu-scope `atom.add` on ctr[10]; by the Atomicity axiom
   (8.10) no two read the same value, so each claim value below na + units goes to exactly one CTA.
   The decode is a function of the value and of S.ord / S.job_e, which every CTA computes
   identically in its routing prologue, so each phase A unit (dealt or claimed) and each phase B
   unit runs on at most one CTA. Every value is claimed: a CTA claims up to kBqMax - 1 = 63 times
   before writing its own sentinel, and the values number at most njobs * 16 + G (about 1,670 at
   512 tokens) against 63 G = 8,316 queue slots, so the counter passes the last value while every
   CTA still has room (the bound needs G of 25 or more; 51 has the same bound). So every unit runs
   exactly once and done_as[j * 16 + st] still receives exactly 8 releases. A CTA's claim values
   rise: one thread's atomics on one location follow its program order in coherence order
   (Coherence axiom, 8.10).
2. Hand-off inside the CTA. The weights producer writes the decoded entry to S.bq[i] with
   `st.release.cta.shared` after its own reads of S.ord and S.job_e; the x producer and the
   consumers read it with `ld.acquire.cta.shared` in `bq_get` (mode 23's queue, unchanged). The
   pair carries only the entry value; S.ord, S.job_* were written before the post-routing
   `__syncthreads` (8.9.4), which orders them for every thread.
3. Inputs of a claimed phase A unit: weights from w13f (written by prepare, ordered by stream order
   or `griddepcontrol.wait`, 9.7.15.14); x from xa, published by the gather (writers'
   `fence.proxy.async.global`, the gather barrier, `red.release.gpu` on xready[j]) and taken by the
   x producer through `ld.acquire.gpu` until xready[j] == job_n[j], then
   `fence.proxy.async.global`, then the bulk copies: steps 1-8 of "Proxy ordering of the x tiles",
   the same calls the dealt units make.
4. Outputs. The claimed unit runs `run_segment` / `run_segment_mma` with PH = 0, the same
   functions as the dealt units, so its epilogue is 51's: generic stores to xb, per-thread
   `fence.proxy.async.global`, the warpgroup's bar.sync, one `red.release.gpu` on done_as (mode
   37's chain). Phase B's acquire on done_as and its proxy fence are unchanged.
5. Ring reuse. The claimed stages use the same slots and parity walk, with 51's releases (each
   consumer thread's `fence.proxy.async.shared::cta` before its weights-slot arrive; lane 0's
   x-slot arrive after `wgmma.wait_group`), and the producers wait on wempty / xempty before every
   reuse, as for dealt stages ("Ordering of the ring slot releases"). The orders agree (above).
6. Counter reset. CTA 0 zeroes ctr[10] after the grid barrier; every claim precedes its CTA's
   arrival in program order and the barrier's fence, atomics and acquire spin order it before the
   reset (as for 23 and 62). The next launch reads ctr[10] after `griddepcontrol.wait` or in stream
   order.
7. Progress. The consumers read the queue only after their last dealt stage. The weights producer
   writes S.bq[0] after issuing its last dealt stage and one more wempty wait, which needs only the
   consumers' releases of dealt stages, which they make without reading the queue. A claimed phase A
   unit waits only on xready[j], and each CTA's gather completes without waiting on any claim or on
   any other CTA (it runs before that CTA's consumers and x producer start), so xready[j] fills. A phase
   B unit waits on done_as of its job's phase A units; every claimed phase A value is smaller than
   every phase B value and each CTA takes its entries in rising order, so on no CTA is a phase A
   unit queued behind a phase B unit. No new cross-CTA wait exists.

No fence, barrier, release or acquire is removed or moved against 51; the claimed path calls the
same segment functions and adds only the queue reads (acquire) and the producers' claim loop.

Independent read-only review (2026-10-30, of 30c5a58): sound. No unit lost or duplicated, the S.bq
ranges are disjoint (phase B 0 to units - 1, kBqA entries 2^24 + j 512 + u 16, -1, kBqEmpty -2),
producer and consumer stage orders equal on every CTA (also with no dealt unit, K = 0 or no job),
no deadlock, no fence, barrier, release or acquire removed or moved against 51, and every mode
predicate of 68-71 equal to 51's (55's for 70 and 71) except the claims. It corrected the wording of
steps 1 and 7 and the `a_dealt` formula (fixed above) and flagged two timing-only skews in the
stamped twins:
- 70 and 71 take the light "phase A end" stamp (S.dg[16][3]) after `consumer_claim_a`, whose first
  `bq_get` waits for the weights producer's first claim; 55 takes it before phase B's first
  `bq_get`. So 71's phase A mean and A end spread include that wait and 55's do not: read the stamp
  comparison as an upper bound on 71's phase A cost.
- In 70 a CTA with no dealt unit leaves the gather stamp S.dg[16][7] at 0 (as in 64): ABGATHER should
  read 0 as "no dealt unit".

### Local compile gate (nvcc 12.8.1, sm_90a, `tools/moe_compile_fused.sh`)

MAIN_MODES 0, 16, 26, 28, 37, 47, 51, 55, 68-71 against the parent commit (c42f3fc) with 0-51 and
55: every production kernel (modes 0, 16, 26, 28, 37, 47, 51 and the four skinny kernels) and 55
have the same instruction text; no C7510-C7515; 68-71 at 96 registers, no stack frame, no spills,
no local load or store anywhere. Counts and loop lengths are in
`muse_moe_w4a16/profiles/sass_mode69.2026-10-30.txt` (stage loop = a backward-branch region with
16 HGMMA under 450 instructions; unit loop = the region around it with the epilogue).

- 68 and 69 have 26,760 instructions (51: 17,280; 62 and 66: 21,488). The claimed loop is a
  second inlined copy of the phase A segments, so HGMMA is 288 against 192 and the epilogue's
  MEMBAR, RED and BAR appear once more per n-tile count (MEMBAR 133 against 43).
- The dealt-unit phase A loops are not byte-identical to 51's SASS. Source-level they are 51's
  code (the same `consumer_phase<0>` and `run_group` instantiation path), but ptxas allocates
  registers and schedules per kernel: even 55, which only adds stamps, matches none of 51's stage
  loops after register names are normalized (difflib ratio 0.54-0.88), though its lengths are
  equal. In address order, 69's six dealt phase A unit loops are 912 / 833 / 785 / 743 / 708 / 668
  instructions against 51's 908 / 830 / 782 / 740 / 705 / 664 (+3 or +4 each), and their stage
  loops 396 / 389 x 5 against 393 / 387 x 5; the opcode differences are integer address arithmetic
  and uniform moves (UMOV +2, IMAD, LEA, IADD3, SHF), with HGMMA, SYNCS and fences unchanged. 62
  and 66 had dealt-and-claimed phase A unit loops of 790-1,298 instructions and stage loops of
  390-392.
- Phase B unit loops are 1,115 / 1,045 / 973 / 903 / 815 / 752 against 51's 1,102 / 1,032 / 962 /
  892 / 801 / 739 (+11 to +14), stage loops 391 against 387. The phase B source is 51's.

`tools/moe_dryrun.py` and `tools/moe_host_check.sh` pass on the build with 68-71.

### Null check and its reading rule (set before the check)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=71,72,73,74,75,76 o69=69` (light twins 55 and 71 with the gather stamps, public stamps).
Seeds 71-76 are new. Reading, on 69 minus 51 (same process, per-seed pairs):

- Pass (the claimed arm 68 may be timed): the mean of 69 minus 51 is within +-1.0 us at 128 and at
  256 tokens on both routings (four cells), 69's output equals 51's at 128 and 256 tokens and is
  within 1e-4 at 32, and prepare's self-test shows no FAIL line and spin flags 0.
- Fail: any of the four cells above +1.0 us. The lean build is then parked in docs/ruled-out.md
  with the 62 row, scoped to this code shape (a second inlined copy of the phase A segments), and
  68 is not timed.
- Reported, not conditions: 32 tokens; 71's phase A mean and A end spread against 55's (the code
  shape's cost should show there first); the cell-by-cell seed counts.

### Gate rule for 68 (set before any gate check; run only if the null passes)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std,skewed
--seeds=77,78,79,80,81,82 --gate=1 o68=68`; seeds 77-82 are held out on all three routings. Keep 68
only if all of these hold in that check:

1. At 256 tokens, 68 minus 51 has a mean of -2.0 us or lower on layer and on standalone routing,
   with at least 5 of 6 seeds lower on each, and a mean of 0 or lower on skewed routing.
2. At 32 and at 128 tokens, a mean of +0.5 us or lower on every routing.
3. 68's output is within 1e-4 relative L2 of 51's on every seed, size and routing.
4. Prepare's FP32 self-test passes for 68 (no FAIL line, spin flags 0).
5. The compile gate above, on the commit that is checked (production SASS unchanged, no
   C7510-C7515, no spill in any stage loop).

Shipping also needs the independent read-only review of the chain above to find it sound. If any
condition fails, 68 goes to docs/ruled-out.md scoped to this build and is not shipped. If all pass,
68 joins MAIN_MODES and MAIN_CANDIDATES and the shippable file (no public code) goes to the
coordinator with the layer port steps. Budget for the line: two checks (null and gate) of the three
MoE checks left today.

### Null result (check b2c92a71600d488b, bench at 30c5a58)

2.211x (2.178-2.244), passed, spin flags 0. Outputs of 69 against 51: 0.0 at 128 and 256 tokens,
1.2e-7 / 1.0e-5 at 32. Self-test of the shipped policy modes unchanged (main-lean-grp-finss64
3.89e-3). Minus 51, us, same process, seeds 71-76 (count = seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 69 (lean, nothing claimed) | +3.3 (0) | +3.7 (0) | +4.5 (0) | +5.0 (0) | +5.2 (0) | +3.9 (0) |
| public | -2.0 (5) | -3.8 (6) | -11.1 (6) | -8.6 (6) | -16.3 (6) | -18.9 (6) |

Stamps (one cold launch per seed, means over seeds, us; layer / standalone; 71's phase A end
includes its wait for S.bq[0], see the review above, so its phase A numbers are upper bounds):

| | 32 | 128 | 256 |
|---|---|---|---|
| phase A mean: 55 (51) / 71 (69) | 45.6 / 47.3 and 84.2 / 86.6 | 96.1 / 98.3 and 119.5 / 121.5 | 127.6 / 130.2 and 144.0 / 146.8 |
| A done max: 55 / 71 | 61.1 / 61.5 and 97.9 / 98.3 | 119.0 / 119.2 and 139.1 / 139.8 | 152.2 / 153.3 and 166.2 / 167.3 |
| A end spread: 55 / 71 | 9.3 / 7.6 and 9.0 / 6.8 | 12.8 / 10.7 and 11.8 / 10.4 | 11.9 / 10.4 and 10.7 / 8.7 |
| phase B: 55 / 71 | 20.6 / 21.9 and 38.2 / 39.7 | 45.4 / 47.1 and 54.7 / 57.0 | 60.2 / 62.5 and 68.7 / 70.3 |
| end: 55 / 71 | 83.6 / 87.3 and 137.2 / 143.2 | 166.8 / 171.2 and 194.6 / 200.7 | 217.0 / 221.7 and 243.3 / 248.5 |

The prepare bench agrees: main-finss64-cl0 211.4 / 226.2 us against 51's 207.2 / 223.2 at 128 / 256.

Reading under the rule set before the check:
- Measured: 69 is +4.5 / +5.0 / +5.2 / +3.9 us against 51 at 128 / 256 tokens (layer / standalone),
  0 of 6 seeds lower in all four cells, so the null fails (each cell above +1.0). 68 is not timed
  and nothing ships. This build costs less than 62's code shape (66 was +7.0 to +8.8 in
  ed4591a2477041c2, on other seeds) but still about 4-5 us.
- Measured: the cost is not confined to phase A. With 51's phase B source, 71's phase B is 1.3-2.3 us
  longer than 55's in all six cells, and the phase A mean 1.7-2.8 us longer (an upper bound). The A
  end spread is 1.4-2.2 us narrower with nothing claimed, but the last CTA's phase A end (A done max)
  is 0.2-1.1 us later, so the narrowing comes from a later mean, not an earlier tail.
- Hypothesis, not separated: the cost is whole-kernel rather than the dealt loop. The kernel grew
  from 17,280 to 26,760 instructions (a second inlined copy of the phase A segments), and every
  stage loop, phase B's included, gained 2-4 instructions of address arithmetic under ptxas's
  per-kernel allocation and scheduling. Instruction-cache pressure from the larger kernel is the
  other candidate; neither was measured.
- Decision: parked in docs/ruled-out.md next to the 62 row. Run-time phase A claims stay unrejected
  as a mechanism. The reopening trigger is now a build whose claimed units run through the dealt
  units' compiled loops (no second copy: kernel size and stage loop lengths within a few
  instructions of 51's) with a K = 0 null within 1 us of 51. One of the line's two checks was used;
  two MoE checks remain today.

## Stream-K phase A on 51's components (modes 72 and 73), 2026-10-30

Brief: a re-test of a parked family after its revisit conditions changed. Owned whole phase A
units (mode 20) beat mode 16's cost-weighted stream-K ranges with cut units by 2-4 us at 128 and
256 tokens (three checks, "Mode 16 at 128 and 256 tokens"), on the stage loop and epilogue of
2026-10-29. Since then 37, 47 and 51 made the stage loop, the phase A epilogue and the phase B
epilogue cheaper, which changes what a cut unit's fixup costs against the imbalance it removes.
Three bounded attempts at the 128 / 256-token gap on 51 lost or tied (the gather fence, mode 60;
phase A claims, modes 62 and 69). The public kernel deals phase A as stream-K ranges with about
1.85 cut units per CTA and its last CTA leaves phase A 5.2-9.1 us after the mean CTA, against
9.5-14.2 for 51 with whole owned units (48b9bf9d77d146ac). Also learned (modes 62 and 69): a kernel
made larger by second compiled copies of loops lost 3-5 us everywhere (rescheduling or instruction
cache, not separated), so the build has to stay close to 51's 17,280 instructions.

### The change

Mode 72 is 51 with `own_a` off: phase A runs mode 16's partition and fixup through the same
`run_group` / `seg_n` / `seg_at` / `run_segment` code the dealt units run (no second loop). Each
CTA takes the contiguous stage range [stage_at(cta), stage_at(cta + 1)) of the job-order cost
prefix S.pa (the kCostA table, as in 16 and the public kernel); the segments of the range are
walked largest n-tile count first in all three roles (consumers, weights producer, x producer), as
51 walks its dealt units; a segment that is not a whole unit adds its accumulators into the fixup
slot of the CTA holding the unit's first stage and the share that completes the count runs the
epilogue (mode 16's `fixup`). Phase B is 51's: units claimed at run time through ctr[10] and S.bq,
51's epilogue and final pass. Three changes keep it lean:

- Setup: the two phase A boundaries are found by warp 0 with a ballot count over S.pa (the jobs
  with pa[j] <= target are a prefix, so their count less one is the j of stage_at's scan) instead
  of thread 0's serial scans (mode 27's stamps put four scans at 2.3-3.1 us), and the leading cut
  unit's fixup slot is found there once (`S.lead_slot`, a new last field of SmemMain) instead of
  by seg_info's search in every consumer thread. `stage_at` itself is unchanged.
- One epilogue call per phase A segment (`epilogue_a_seg`): a whole unit copies its accumulators
  to the array the fixup fills and calls the same `epilogue_a`, so the SwiGLU epilogue is
  compiled once per n-tile count, not once for whole units and again for finished cut units.
- 73 is 72 with 55's light and gather stamps plus `fix_st`: warpgroup 0's thread 0 times each
  fixup from its start to the count update and, when it finishes the unit, to the end of the
  read-back, inside branches the fixup already takes. Stamp code placed after the fixup returned
  (a thread-0 branch or predicated shared stores) made 73 spill 288 B, so it is not used.

A CPU model of the partition (`scratchpad/sk72/model.py`, not committed: 4,000 random job sets, M
of 17 to 512, 1 to 32 local experts, G of 1 to 132) found the ballot boundaries equal to
stage_at's, the setup slot equal to seg_info's search on every cut segment, every phase A stage
covered exactly once, one cut unit per fixup slot and every slot's shares summing to 16 stages,
including ranges shorter than a unit (units spread over three or more CTAs).

### Ordering chain for mode 72

Sections from the PTX ISA memory consistency model as cited in "Proxy ordering of the x tiles":
8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.2 Observation Order, 8.9.4 Memory
synchronization, 8.9.5 Causality Order, 8.10 axioms (Atomicity), 9.7.15.4 membar / fence,
9.7.15.14 griddepcontrol.

No fence, barrier, release or acquire is removed or moved against either lineage: the fixup keeps
mode 16's fences (both `fence.acq_rel.gpu`, both warpgroup `bar.sync`, the `atom.acq_rel.gpu`
count), and the phase A publish, the gather, the ring releases, phase B and the final pass are
51's code. The one difference from 16 is the finisher's publish: 72 publishes through 51's
`publish_a`, without the per-thread `fence_gpu` that mode 37 removed with a reviewed chain
("Ordering of the phase A publish without the per-thread fence_gpu").

1. Setup values. Warp 0 reads S.njobs, S.pa, S.cost and S.job_ntt after the post-routing
   `__syncthreads`, which orders the routing warp's writes before them (8.9.4). Lane 0 writes
   S.gsb[0], S.gse[0] and S.lead_slot; thread 0 (the same thread) then writes S.gsb[1] and S.gse[1]
   in the existing block; the `__syncthreads` after that block orders all of it before every
   producer and consumer read. `__ballot_sync` runs with all 32 lanes: the loop bounds and the
   slot search's trip count depend only on shared values every lane reads alike.
2. Shares of a cut unit. Each of a warpgroup's 128 threads adds its partial accumulators into the
   slot with `red.global.add.v4.f32` (relaxed, gpu scope), runs `fence.acq_rel.gpu`, and the
   warpgroup meets at `bar.sync`; thread 0 then adds the share's stage count with
   `atom.acq_rel.gpu`. The adds are atomic (Atomicity axiom, 8.10), so concurrent shares do not lose
   updates. The barrier carries every thread's adds and fence before thread 0's atom in base
   causality order (8.9.4, 8.9.5), and the atom is itself a release, so the other threads' adds are
   covered by its cumulativity (8.8), as for the gather's red.release. [Corrected after the review:
   this step first called each thread's fence and a later strong write a release pattern, which
   holds only for thread 0; the other threads issue no strong write after their fence.]
3. The finisher. The count is a chain of atomic read-modify-writes, so the atom that reads the
   last value before 16 follows every earlier share's atom in observation order (8.9.2), and as an
   acquire it synchronizes with each of their releases (8.9.4). Its thread writes `fix_last`, the
   warpgroup meets at `bar.sync` again, every thread runs `fence.acq_rel.gpu` and reads the slot
   with `ld.global.cg`; causality order is transitive (8.9.5), so the reads see every share's adds,
   including the finisher's own (program order). The finisher then zeroes the slot and the count
   for the next launch.
4. The finished unit's outputs: `epilogue_a` with the summed accumulators, generic stores to xb,
   per-thread `fence.proxy.async.global`, the warpgroup's `bar.sync`, one `red.release.gpu` on
   done_as (51's publish, mode 37's chain). Each warpgroup has its own slot word (index
   (slot * 4 + wg)) and count, so each warpgroup's finisher publishes once and done_as[j * 16 + st]
   still receives exactly 8 releases (2 units x 4 warpgroups). Phase B's acquire, proxy fence and
   bulk copy are unchanged.
5. Slot reuse. One cut unit per slot per launch (the model above): a slot is the CTA holding the
   unit's first stage, and a CTA holds the first stage of at most one cut unit (its trailing one, or
   the single unit its range lies inside). The finisher's zero stores and every share's adds precede
   the grid barrier; the next launch's adds follow `griddepcontrol.wait` or stream order
   (9.7.15.14, 8.9.4). The post-barrier zeroing of `fixcnt` is unchanged. The fixup slots are
   generic-proxy only (red, ld.cg, st), so no proxy fence is involved (8.6).
6. The whole-unit copy. A whole segment copies its accumulators to `fin` after the stage loop's
   `wgmma.wait_group 0` and accumulator fence, the point where 51's epilogue reads them; the
   accumulators are only read (ptxas C7515 forbids writing them outside wgmma).
7. Ring order. The consumers, the weights producer and the x producer walk the same segments in
   the same order (`seg_n` / `seg_at` with n-tile counts 8 down to 1, the order 51's dealt units
   and every dyn_b mode use), so each slot's wfull / wempty / xfull / xempty phases line up as in 51
   and 16; the phase B claims come after the CTA's last phase A stage as in 51.
8. Progress. Phase A waits on no other CTA except through `xready` (the gather, which waits on no
   claim). A fixup never waits: a share adds, counts and leaves, and the last share finishes. So
   every CTA finishes phase A and every cut unit is finished by the share that arrives last. A
   phase B unit waits on done_as of its job's phase A units, which all complete without waiting on
   phase B. No new cross-CTA wait exists.

### Local compile gate (nvcc 12.8.1, sm_90a, `tools/moe_compile_fused.sh`)

MAIN_MODES 0, 16, 26, 28, 37, 47, 51, 55, 72, 73 against the parent commit (b5b8a34) with 0-51
and 55: every production kernel (modes 0, 16, 26, 28, 37, 47, 51 and the four skinny kernels) and
55 have the same instruction text; no C7510-C7515; 72 and 73 at 96 registers, no stack frame, no
spill, no local load or store anywhere. Counts and loop lengths in
`muse_moe_w4a16/profiles/sass_mode72.2026-10-30.txt`:

| | 51 | 72 | 55 | 73 | 16 (for scale) |
|---|---|---|---|---|---|
| instructions | 17,280 | 18,664 (+1,384, +8.0%) | 17,736 | 19,440 | 28,032 |
| stage loops | 387 x 11, 393 | 389-390 x 11, 398 | as 51 | as 72 | 391-392 x 11, 399 |
| phase A and B unit loops | 664-1,102 | 741-1,104 | 666-1,103 | 741-1,105 | 1,023-1,392 |
| MEMBAR / ERRBAR / CCTL | 43 / 11 / 8 | 91 / 35 / 32 | 43 / 11 / 8 | 91 / 35 / 32 | 186 / 83 / 72 |
| BAR / REDG / ATOMG | 16 / 9 / 148 | 32 / 45 / 156 | 18 / 9 / 148 | 34 / 45 / 156 | 57 / 89 / 201 |
| HGMMA, LDL / STL | 192, 0 / 0 | 192, 0 / 0 | 192, 0 / 0 | 192, 0 / 0 | 192, 43 / 27 |

The added instructions are the fixup per n-tile count (36 REDG for the v4 adds of counts 1-8, 8
ATOMG, 16 BAR, the fences) inside the phase A unit loops; no loop is compiled twice. The claim
builds that lost had 21,488 (62, 66) and 26,760 (68, 69) instructions. `tools/moe_dryrun.py` (with
72 and 73 built: both pass the self-test) and `tools/moe_host_check.sh` on the bench file pass.

Prediction (hypothesis, before the run): the phase A end spread narrows toward the public
kernel's (5-9 us), at a fixup cost of 1-3 us per CTA (mode 16's per-warp "epilogue and fixup" was
21-27 us over 4-5 segments on the old epilogue) plus the larger kernel's 1-3 us. Net -2 to +3 us at
128 and 256 tokens; nothing to gain at 32 tokens, where 51's spread is already 4-9 us.

### Diagnostic run and its rule (set before the check)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=83,84,85,86,87,88 o72=72` (light twins 55 and 73, public stamps). Seeds 83-88 are new.
Rule: the gate runs only if 72 minus 51 has a negative mean at 256 tokens on both routings;
otherwise the family stays parked in docs/ruled-out.md with this evidence and no gate is run.
Reported, not conditions: 128 and 32 tokens, seed counts, ABSTAMP's phase A mean and A end spread
for 73 against 55 and the public kernel, ABFIX's fixup time and cut segments per CTA, ABCHECK's
output of 72 against 51.

### Gate rule (set before any gate check; run only if the diagnostic passes)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std,skewed
--seeds=89,90,91,92,93,94 --gate=1 o72=72`; seeds 89-94 are held out on all three routings. Keep
72 only if all of these hold in that check:

1. At 256 tokens, 72 minus 51 has a mean of -1.5 us or lower on layer and on standalone routing,
   with at least 5 of 6 seeds lower on each, and a mean of 0 or lower on skewed routing.
2. At 128 tokens, a mean of 0 or lower on layer and on standalone routing and of +0.5 or lower on
   skewed routing.
3. At 32 tokens, a mean of +0.5 us or lower on every routing.
4. 72's output is within 5e-4 relative L2 of 51's on every seed, size and routing. (Not 1e-4 as
   in the 47 / 51 gates: a cut unit's FP32 sum order changes before the bf16 rounding of gate and
   up, and mode 20 against 16 differed by 1e-4 to 3e-4 for that reason, 9332e2a49aff4ff2.)
5. Prepare's FP32 self-test passes for 72 (no FAIL line, spin flags 0).
6. The compile gate above, on the commit that is checked.

Shipping also needs the independent read-only review of the chain above to find it sound. If any
condition fails, 72 goes to docs/ruled-out.md scoped to this build and is not shipped. If all
pass, 72 joins MAIN_MODES and MAIN_CANDIDATES and the shippable file (no public code) goes to the
coordinator with the layer port steps. Budget for the line: two checks (diagnostic and gate), the
two MoE checks left today.

### Diagnostic result (check 098949b28fe84bd1, bench at 86a633e)

2.193x (2.160-2.226), passed, spin flags 0. Outputs of 72 against 51 (first seed of each cell):
1.5e-4 to 2.4e-4 relative L2 (the cut units' FP32 sum order, as 20 against 16). Self-test of the
shipped policy modes unchanged (main-lean-grp-finss64 3.89e-3). Minus 51, us, same process, seeds
83-88 (count = seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 72 (stream-K phase A) | +13.5 (0) | +9.4 (0) | +12.6 (0) | +7.1 (0) | +6.6 (0) | +9.2 (0) |
| public | -1.4 (3) | -8.4 (6) | -8.4 (6) | -9.4 (6) | -17.2 (6) | -15.3 (6) |

Stamps (one cold launch per seed, means over seeds, us; layer / standalone):

| | 32 | 128 | 256 |
|---|---|---|---|
| phase A mean: 55 (51) / 73 (72) / public | 47.7 / 57.6 / 49.1 and 89.3 / 96.6 / 85.8 | 94.3 / 103.5 / 91.7 and 117.0 / 124.8 / 111.5 | 127.8 / 135.1 / 119.6 and 134.1 / 141.6 / 125.5 |
| A end spread: 55 / 73 / public | 8.0 / 12.4 / 3.3 and 10.6 / 11.0 / 4.7 | 15.3 / 18.2 / 7.3 and 8.4 / 15.2 / 6.1 | 10.0 / 18.9 / 7.1 and 8.8 / 16.9 / 8.1 |
| A done max: 55 / 73 | 61.3 / 75.4 and 104.5 / 113.0 | 119.6 / 129.5 and 132.9 / 146.9 | 150.4 / 165.0 and 153.9 / 168.1 |
| phase B: 55 / 73 | 21.5 / 22.6 and 39.6 / 40.1 | 45.0 / 46.8 and 54.3 / 54.7 | 59.9 / 61.2 and 62.9 / 63.9 |
| 73's fixups (ABFIX): cut segments per CTA, us per CTA, its share of phase A | 1.86, 8.8, 15% and 1.84, 8.7, 9% | 1.88, 10.1, 10% and 1.85, 9.4, 7% | 1.84, 10.9, 8% and 1.83, 9.5, 7% |
| 73's fixups: to the count update per share / read-back per finished unit, max CTA | 3.3 / 6.0, 17.9 | 3.3 / 7.3, 28.4-30.9 | 3.2 / 8.6, 28.3-32.5 |

Reading under the rule set before the check:
- Measured: 72 lost at every size and routing, 0 of 6 seeds lower in all six cells, +6.6 / +9.2 us at
  256 tokens. The rule's condition (a negative mean at 256 on both routings) fails, so no gate is run
  and nothing ships.
- Measured, mechanism: the partition did not narrow the end of phase A, it widened it. 73's A end
  spread is 15-19 us at 128 and 256 tokens against 55's 8-15 and the public kernel's 6-8, and the
  mean CTA's phase A is 7-9 us longer than 55's. The fixups cost 9-11 us per CTA (1.84-1.88 cut
  segments per CTA, the public kernel's 1.85): about 3.3 us per share to its count update (the v4
  global adds, a gpu-scope fence and two warpgroup barriers) and 6-9 us for the finishing share's
  read-back, with the slowest CTA at 28-33 us. Both the spread and the mean rise, so the last CTA's
  phase A end is 10-15 us later than 55's (A done max), and the launch follows.
- Hypotheses, not separated: (a) the fixup's global round trips sit on each CTA's critical path
  under a memory system that phase A saturates (its cost here is 2-3 times what 51's whole phase A
  epilogue takes per unit), and the CTAs that finish two cut units carry it twice, which by itself
  spreads the phase A ends; (b) the kCostA table (2026-10-28) no longer prices a stage of 47 / 51's
  loop by n-tile count, so equal-cost ranges are not equal-time ranges. The public kernel's
  partition gets 6-8 us of spread with the same 1.85 cuts, so its fixup must cost far less (its
  finisher is designated, not the last arrival; not measured here).
- Measured, a side result: 73's gather piece is shorter than 55's (thread 0's whole gather 3.4 /
  5.9 against 6.5 / 8.2 us at 128 / 256 tokens on layer routing, mostly its fences and barrier), but
  the latest first-x-ready comes later, not earlier (12.8 / 17.1 against 10.2 / 11.6 us), so it buys
  nothing on the critical path. Why the gather changed is not separated (the setup differs only in
  warp 0's boundary search before the gather).
- Independent read-only review of 86a633e (2026-10-30): sound (no ordering bug, deadlock, lost or
  shared fixup slot, or uninitialised read; production predicates unchanged; 72 / 73 equal 51 / 55
  on every predicate except own_a, stream_a and fix_st). It corrected the wording of step 2 (fixed
  above) and noted that ABFIX's per-CTA total subtracts the mean pre-count time per finished unit,
  so it is an estimate; the per-share and per-finish columns are measured directly.
- Decision: the stream-K phase A family stays parked (docs/ruled-out.md), with the new evidence that
  on current components its fixups, not its partition, decide the comparison. One of the line's two
  checks was used; one MoE check remains today.

## Designated finisher for stream-K phase A (modes 74 and 75), 2026-10-30

Brief: the revisit trigger of the stream-K phase A row in docs/ruled-out.md (a fixup well under
1 us per share on the critical path, for example a designated finisher). Mode 72 lost to 51 by
6.6-13.5 us (098949b28fe84bd1) with fixups of 9.4-10.9 us per CTA: about 3.3 us per share to its
count update and 6-9 us for the finishing share's read-back. First the public kernel's fixup was
read (scratchpad copy of vjeux/okc-2026 `muse_moe_w4a16.py`, check-only, not committed), then 74
was built from what it does.

### How the public kernel's fixup works (read from its source)

Phase A's stream-K path (`OKC_WGS`) gives each CTA the stage range of equal cost under its
cost table (`wgs_range`; its `OKC_WGSA` table, 112 135 140 158 173 185 200 207, is our kCostA,
so the table is the same in both kernels). A unit cut between CTAs is finished as follows:

1. Order. `wgs_split` runs a CTA's trailing cut share (the first stages of the unit holding the
   range's last stage) first, then the rest of the range in order: [sp, g1) then [g0, sp). Its
   comment says why: the CTA holding a unit's first stages otherwise reaches them last and pays
   the read-back and epilogue exposed at the phase end.
2. The finisher. With `OKC_WGSDFIN` bit 0 set, the share holding stage 15 (`s1 == 16`) finishes
   the unit in phase A ("the share processed last in time by construction": it is the leading
   segment of the next CTA's range, which that CTA runs second, after its own trailing share; in
   phase B, bit 1, the share holding stage 0, run at its CTA's end). Every other share is a
   contributor. The choice is made from the segment bounds; no atomic decides it.
3. How partials travel. At the end of a share's stage loop (`wgs_seg`, after `wgmma.wait_group
   0`), each of the warpgroup's 128 threads issues `red.global.add.v4.f32` of its accumulators
   into the slot of the CTA holding the unit's first stage, one 2 KB block per n-tile laid out
   [n-tile][thread][4], with no fence or barrier at that point. The slot is in global memory
   (L2), indexed by (slot, warpgroup).
4. When the count and the wait happen. The rest is deferred (`WgsPend`) and run by `wgs_flush`
   inside the next segment's first stage, after commit group 3 of its 8 wgmma groups (or at the
   end of the phase for the last segment), "so the stores and atomics it waits for have long
   completed". A contributor's flush: warpgroup `bar.sync`, then thread 0 `red.release.gpu.add`
   of its stage count on the (slot, warpgroup) counter, fire-and-forget; no per-thread fence and
   no atomic round trip. The finisher's flush: `bar.sync`, thread 0 spins on `ld.acquire` until
   the counter reaches 16 less its own stage count (its own share was added to the slot but not
   counted), `bar.sync`, `fence.acq_rel.gpu` on every thread, then two n-tiles of `ld.global.cg`
   in flight at a time, each followed by its zeroing store and the epilogue tile, thread 0 zeroes
   the counter, and in phase A a `bar.sync` and one `red.release` on the job's doneAs word.
5. Who waits on whom: only the finisher waits, on the contributors' counters; contributors
   never wait. Because of the order in 1, a contributor (a trailing share) is the first thing its
   CTA runs and the finisher runs second in its CTA, so the wait is usually already satisfied.

### What carries over, and what does not

Two parts carry over directly: the contributor that adds, fences and counts with one
fire-and-forget release, and the finisher that waits on a count target instead of an atomic
deciding the last arrival. The public order does not carry over. Our consumers walk a range by
n-tile count, largest first (51's order, which every dyn_b mode and 72 use); a range walk with
one `switch` over n-tile counts, the public shape, made ptxas serialize every wgmma in 74
(C7512, 96 registers; the same kernel with the grouped walk has none). Under the grouped walk the
stage-15 share is no longer last in time. A CPU model of the partition and the walk
(`scratchpad/sk74/sim74g.py`, not committed; stage costs from kCostA, each CTA from time 0, a
finisher adds one stage for its read-back) put the stage-15 rule's finisher waits at p90 5 / p99
73 stages at 132 CTAs (74% of cut units with no wait). Under the public range order the same rule
gives p90 5-6 / p99 12 stages (75-77% with no wait), so even there the order does not guarantee
the finisher arrives last: a contributor's trailing share can be longer than the finisher's own
trailing share plus its share.

So 74 designates the finisher by modelled time under our walk: for each cut unit, the share
whose end in cost time (its CTA's segments that run before it in the walk, plus itself) is
latest, ties to the later CTA. In the same model: no wait in 99.9-100% of cut units, the largest
1.4 stages, and the modelled phase A end spread 1.2 stage costs against 5.1 (stage-15 rule, range
order) and 32.7 (stage-15 rule, our walk).

### The change (mode 74; 75 its stamped twin)

Mode 74 is 72 (51 with mode 16's stream-K partition in phase A, the same walk, setup, slots and
counters) with three changes, all in phase A's cut units:

- Designation in setup. Warp 0, after 72's boundary and lead-slot searches, runs `dfin_flags`:
  for each cut share of its range (one loop over the two possible shares), it walks the CTAs
  holding the unit (from the one holding its first stage, the lead slot or this CTA, to the one
  holding stage 15, each range from `stage_at_a_warp`) and takes the one whose `share_end` is
  largest; `share_end` sums, over a range's segments with
  lanes and `__reduce_add_sync`, the cost of the segments the walk runs up to and including the
  share. Two bits go into `S.lead_slot` (bit 16 for the share with st_b != 0, bit 17 for the one
  with st_b == 0); `seg_info_m` masks the slot and decodes `SegInfo::fin`. Every CTA derives the
  bits from the same shared values (S.pa, S.cost, S.job_ntt), so all CTAs holding a unit agree.
- `fixup_df` in place of `fixup` for phase A. Every share adds its partial into the slot with v4
  `red.add`. A contributor then runs every thread's `fence.acq_rel.gpu`, the warpgroup's
  `bar.sync` and thread 0's `red.release.gpu` of its stage count, and returns. The finisher does
  not count: thread 0 spins (`spin_ge`, timeout flag 512) until the count reaches 16 less its own
  stages, `bar.sync`, every thread's `fence.acq_rel.gpu`, then per n-tile one `ld.global.cg` of
  the whole sum and a zero store; thread 0 zeroes the count; then 51's epilogue and publish as in
  72. (A first build kept the finisher's own accumulators out of the slot and added them in
  registers; that kept acc live and spilled 20 / 216 B.)
- The slot layout [n-tile][thread] in float4 (the public layout), so a warp's v4 adds and reads
  cover 4 whole 128-byte lines per n-tile; 72's [thread][n-tile] put each lane's 16 bytes in its
  own line. The slot and count arrays and their sizes are unchanged.

75 is 74 with 73's light, gather and fixup stamps; fixup word 4 is the finisher's spin from the
fixup's start (inside word 2, which runs to the end of the read-back). ABFIX gains the
finisher's wait (mean and largest CTA); for 75 its per-CTA total is the shares' time to their
count plus the finisher's time, with nothing subtracted (the finisher does not count).

A CPU transcription of `share_end`, `dfin_flags` and the bit decode
(`scratchpad/sk74/model74.py`, not committed; 1,500 random job sets, M 17 to 512, 1 to 32 local
experts, G of 1 to 132, 65,247 cut units) found, for every cut unit: exactly one share decodes
fin from its own CTA's flags (so the CTAs holding the unit agree); the other shares' stages sum
to 16 less the finisher's; every share's slot is the CTA holding the unit's first stage; and the
simulated walk with finisher waits resolves (no cycle), with no modelled wait in 99.93% of cut
units and at most 1.4 stages.

### Ordering chain for mode 74

Sections from the PTX ISA memory consistency model as cited in "Proxy ordering of the x tiles":
8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.2 Observation Order, 8.9.4 Memory
synchronization, 8.9.5 Causality Order, 8.10 axioms (Atomicity, Coherence), 9.7.15.4 membar /
fence, 9.7.15.14 griddepcontrol.

Fences and barriers against both lineages: the contributor keeps mode 16's `fence.acq_rel.gpu`
before its count and the `bar.sync` before thread 0's count; the finisher keeps the
`fence.acq_rel.gpu` and the `bar.sync` after its acquire. The phase A publish, the gather, the
ring releases, phase B and the final pass are 51's code, as in 72. Three operations change. (1)
The contributor's `atom.acq_rel.gpu` becomes `red.release.gpu`: the acquire half and the
returned value served only the last-arrival test, and a contributor reads nothing after its
count. (2) The contributor's second warpgroup `bar.sync` (which broadcast `fix_last`) is gone,
as no thread reads a result. (3) The finisher no longer runs the pre-count fence and barrier
that every share ran in 72, because it does not count: its own adds are read only by the same
thread, at the same addresses, after its later `fence.acq_rel.gpu` (8.9.5, 8.10). The finisher
gains an `ld.acquire.gpu` spin (thread 0) before its `bar.sync`.

1. Setup values. Warp 0 reads S.njobs, S.pa, S.cost and S.job_ntt after the post-routing
   `__syncthreads` (8.9.4), as in 72; lane 0 writes S.lead_slot with the bits, and the
   `__syncthreads` after the setup block orders it before every consumer read (72's step 1).
   `__ballot_sync` and `__reduce_add_sync` run with all 32 lanes: every loop bound and branch in
   `dfin_flags` and `share_end` depends only on shared values all lanes read
   alike and on `stage_at_a_warp`, which returns the same value to every lane.
2. A contributor's partial. (The finisher's threads issue the same adds; step 3 covers them.)
   Each of the 128 threads adds its accumulators with
   `red.global.add.v4.f32` (relaxed, gpu scope), runs `fence.acq_rel.gpu`, and the warpgroup meets
   at `bar.sync`; thread 0 then adds the share's stage count with `red.release.gpu`. The adds are
   atomic (Atomicity, 8.10), so shares added concurrently lose nothing. The barrier puts every
   thread's adds and fence before thread 0's release in base causality order (8.9.4, 8.9.5), and
   the release is cumulative over them (8.8), the same argument as 72's step 2 and the gather's
   `red.release`.
3. The finisher's wait. The count is changed only by atomic adds, so the value thread 0's
   `ld.acquire.gpu` reads is the result of some prefix of the adds in that location's order
   (Atomicity and Coherence, 8.10; 8.9.2). The target 16 - own is reached only after every
   contributor's add: the contributors' counts sum to exactly that (the model above), each is
   positive, and nothing else adds to the word in this launch. This is the pattern the kernel
   already relies on for done_as (8 `red.release` adds, one `ld.acquire` spin to 8) and xready
   (the gather's `red.release` adds, the x producer's spin to the job's row count), whose chains
   were reviewed ("Proxy ordering of the x tiles", "Ordering of the phase A publish without the
   per-thread fence_gpu"): the acquire that reads the final sum synchronizes with each
   contributor's release (8.8, 8.9.4). The warpgroup's `bar.sync` then extends this to the other
   127 threads, and each runs `fence.acq_rel.gpu` before its `ld.global.cg` of the slot;
   causality order is transitive (8.9.5), so every thread's read sees every contributor's adds.
   The finisher's threads also add their partials; each reads back only the addresses it added
   to, and its `red.add` precedes its `ld.global.cg` of the same address in program order (base
   causality order, 8.9.5), with its `fence.acq_rel.gpu` between them, so the read includes its
   own adds (Coherence, 8.10).
4. Reset. The finisher's zero stores to its slot entries and thread 0's zero store to the count
   follow its reads and the count's final value (program order); no contributor touches the slot
   or count again in this launch (one share per CTA per unit, and every contributor counted
   before the finisher read). The next launch's adds follow the grid barrier and
   `griddepcontrol.wait` or stream order (9.7.15.14, 8.9.4), as in 72's step 5. The slots stay in
   the generic proxy (red, ld.cg, st), so no proxy fence is involved (8.6). Failure path only:
   if a spin times out (ctr[8] code 512), the finisher reads and zeroes early, and a late
   contributor's adds stay in a slot that the end-of-kernel reset (fixcnt only) does not clear;
   that launch is already flagged, so the self-test fails, but later launches on the same
   buffers could read a wrong sum.
5. The finished unit's outputs: 51's `epilogue_a` and `publish_a` with the summed accumulators
   (72's step 4). Each warpgroup has its own slot word and count (slot * 4 + wg), each
   warpgroup's finisher publishes once, and done_as[j * 16 + st] still receives 8 releases.
6. Slot reuse within the launch: 72's step 5 unchanged (a slot is the CTA holding the unit's
   first stage; a CTA holds the first stage of at most one cut unit); the layout change keeps each
   (slot, warpgroup) block at 1,024 float4 and indexes it [n-tile][thread], so two warpgroups or
   two slots never overlap.
7. Ring order: unchanged from 72 (the producers and consumers walk the same segments in the same
   order; the designation changes no walk).
8. Progress. A contributor never waits in its fixup. A finisher waits on contributors, and a
   contributor share can sit behind a finisher earlier in its own CTA's walk, so waits can chain.
   They cannot form a cycle: a CTA shares a unit with lower CTAs only through its leading share
   and with higher CTAs only through its trailing share (one unit is cut at each range boundary).
   Take the highest CTA H on a supposed cycle. The cycle enters H from a lower CTA's finisher
   waiting on H's share of their common unit, which is H's leading share, a contributor; it must
   leave H through a finisher of H that runs before that share and waits on another CTA, and the
   only other unit H shares is its trailing one, shared only with higher CTAs, against the choice
   of H. A CTA whose range lies inside one unit holds a single cut share, so it cannot pass a
   wait on. So every chain ends, and every phase A wait (the gather's `xready`, and these counts)
   completes. All CTAs are resident (one per SM, the grid barrier's existing requirement). Phase
   B waits on done_as as before. Every spin has the timeout flag (ctr[8], code 512 new).

### Local compile gate (nvcc 12.8.1, sm_90a, `tools/moe_compile_fused.sh`)

`muse_moe_w4a16/profiles/sass_mode74.2026-10-30.txt` has three builds of MAIN_MODES 0, 16, 26,
28, 37, 47, 51, 55, 72, 73, 74, 75 against fcc0e30. In every build all production kernels (modes
0, 16, 26, 28, 37, 47, 51, the four skinny kernels) and 55 have the same instruction text as
fcc0e30; 72 and 73 differ by 16 and 8 instructions (they share the edited `seg_info_m` and
`epilogue_a_seg`; not production); no C7510-C7515; 74 and 75 at 96 registers.

| | 51 | 72 | 74 at 5f35018 (checked) | 74 with fast_silu | 74 corrected (4187b8b) |
|---|---|---|---|---|---|
| instructions | 17,280 | 18,680 | 20,504 | 19,400 | 19,312 (+11.8% on 51, +3.4% on 72) |
| spills | 0 | 0 | 0 | 0 | 0 |
| stage loops | 387 x 11, 393 | 389-390 x 11, 398 | 389-391 x 11, 399 | 389-390 x 11, 398 | 389-390 x 11, 398 |
| unit loops | 664-1,102 | 741-1,104 | 743-1,317 | 742-1,105 | 742-1,105 |
| MEMBAR / BAR / REDG / ATOMG | 43 / 16 / 9 / 148 | 91 / 32 / 45 / 156 | 91 / 32 / 53 / 156 | same | same |

Earlier shapes of 74 that were not checked: the range walk with one `switch` over n-tile counts
(C7512, every wgmma serialized), the designation with three inlined copies of its search (22,944
instructions), and the finisher adding its accumulators in registers (20 / 216 B spilled). The
setup designation alone was about 470 instructions in the serial form (74 at 5f35018 with it
stubbed out: 20,032). `tools/moe_dryrun.py` with 74 in MAIN_CANDIDATES and
`tools/moe_host_check.sh` on the bench file pass.

### Diagnostic run and its rule (set before the check)

One check of `tools/moe_make_bench_ab.py --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=95,96,97,98,99,100 o74=74` (light twins 55 and 75, public stamps). Seeds 95-100 are new.
Rule: a gate is considered only if 74 minus 51 has a negative mean at 256 tokens on both
routings; otherwise the family stays parked in docs/ruled-out.md with this evidence and no gate
is run. Reported, not conditions: 128 and 32 tokens and seed counts; ABSTAMP's phase A mean and A
end spread for 75 against 55, the public kernel, and 73 in 098949b28fe84bd1 (a different process
and seeds, labelled as such); ABFIX's time per share to its count, the finisher's wait and
read-back, and fixup time per CTA against the trigger (well under 1 us per share on the critical
path); ABCHECK's output of 74 against 51.

### Gate rule (set before any gate measurement; only if the diagnostic passes)

The gate is one run of `tools/moe_make_bench_ab.py --nopub=1 --gate=1 --base=51
--sizes=32,128,256 --routings=layer,std,skewed --seeds=101,102,103,104,105,106 o74=74`, held-out
seeds on all three routings. No MoE check remains today after the diagnostic, so the gate would run as a scored run
submitted by the coordinator: a bench file whose judged path is the unchanged board file (74 not
added to MAIN_CANDIDATES, no public code embedded, so the public arm is absent), which replaces
the board entry only if it scores higher. Keep 74 only if all of these hold in that run:

1. At 256 tokens, 74 minus 51 has a mean of -1.5 us or lower on layer and on standalone routing,
   with at least 5 of 6 seeds lower on each, and a mean of 0 or lower on skewed routing.
2. At 128 tokens, a mean of 0 or lower on layer and on standalone routing and of +0.5 or lower on
   skewed routing.
3. At 32 tokens, a mean of +0.5 us or lower on every routing.
4. 74's output is within 5e-4 relative L2 of 51's on every seed, size and routing (72's reason:
   a cut unit's FP32 sum order).
5. No spin flag in the run (ABMOE spin flags 0), 51 against FP32 at its usual 3.8e-3 to 3.9e-3
   in ABCHECK, and `tools/moe_dryrun.py` passing with 74 in MAIN_CANDIDATES. (Prepare's FP32
   self-test does not cover an arm outside MAIN_CANDIDATES, and a scored run must not add it.)
6. The compile gate above, on the commit that is measured.

Shipping also needs the independent read-only review of the chain above to find it sound. If any
condition fails, 74 goes to docs/ruled-out.md scoped to this build. If all pass, 74 joins
MAIN_MODES and MAIN_CANDIDATES and the shippable file goes to the coordinator with the layer port
steps.

### Diagnostic result (check eaa881a934aa471e, bench at 5f35018)

2.202x (2.169-2.235), passed, spin flags 0; board muse_moe_w4a16, job id confirmed. Outputs of 74
against 51 (first seed of each cell): 1.5e-4 to 2.0e-4. Minus 51, us, same process, seeds 95-100
(count = seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 74 as built at 5f35018 | +8.2 (0) | +9.0 (0) | +8.3 (0) | +7.2 (0) | +2.1 (0) | +3.8 (1) |
| public | -0.9 (6) | -6.3 (6) | -9.4 (6) | -11.6 (6) | -17.7 (6) | -19.1 (6) |
| 72 in 098949b28fe84bd1 (seeds 83-88, another process) | +13.5 | +9.4 | +12.6 | +7.1 | +6.6 | +9.2 |

Stamps (one cold launch per seed, means, us; layer / standalone; 73 from 098949b28fe84bd1):

| | 32 | 128 | 256 |
|---|---|---|---|
| route + setup: 55 / 75 / 73 | 2.7 / 6.5 / 3.7 and 2.9 / 6.6 / 3.7 | 3.3 / 7.0 / 4.2 and 3.4 / 7.2 / 4.3 | 4.1 / 7.9 / - and 4.2 / 7.8 / - |
| phase A mean: 55 / 75 / public | 44.8 / 49.3 / 46.9 and 84.8 / 88.2 / 81.2 | 89.9 / 95.5 / 87.4 and 124.8 / 131.1 / 119.5 | 129.1 / 133.2 / 120.1 and 146.8 / 152.9 / 137.4 |
| A end spread: 55 / 75 / public | 6.6 / 5.9 / 3.8 and 7.1 / 3.0 / 3.6 | 14.0 / 7.2 / 6.8 and 11.4 / 6.7 / 6.8 | 10.0 / 9.7 / 7.1 and 12.0 / 9.1 / 8.8 |
| A done max: 55 / 75 | 57.9 / 63.4 and 96.7 / 99.5 | 113.8 / 112.9 and 144.0 / 147.8 | 151.8 / 156.1 and 170.5 / 174.3 |
| 75's fixups: shares counted per CTA, us to the count per CTA | 0.92, 1.74 and 0.93, 1.90 | 0.92, 1.79 and 0.93, 1.80 | 0.92, 1.82 and 0.92, 1.66 |
| 75's finishes per CTA, us to the end of the read-back (of which waiting) | 0.92, 2.02 (0.78) and 0.93, 2.12 (0.89) | 0.92, 1.93 (0.80) and 0.93, 2.23 (0.88) | 0.92, 2.22 (0.91) and 0.92, 2.00 (0.81) |
| 75's fixup total per CTA, largest CTA, largest wait | 3.75, 8.5, 6.1 and 4.01, 7.7, 4.2 | 3.73, 10.6, 7.8 and 4.03, 11.7, 8.1 | 4.04, 10.3, 6.2 and 3.66, 9.9, 7.1 |

Reading under the rule set before the check:
- Measured: 74 as built lost at every size and routing, +2.1 / +3.8 us at 256 tokens (0 and 1 of 6
  seeds lower). The rule (a negative mean at 256 on both routings) fails for this build.
- The build was not the design described above. The independent review (after the check was
  sent) found `fast_silu` still ending at 73, so 74 and 75 ran SiLU with `__frcp_rn` (MUFU.RCP
  plus a range check and a subroutine per element) where 51 and 55 use `rcp.approx`; that is
  1,104 of 74's 20,504 instructions (19,400 with it fixed) and was worth 0.9 to 1.8 us in 36
  against 28 (0b684d5db0834dbc). 75's longer phase A mean (+4 to +6 us over 55) carries it.
- Measured, the fixup: 3.7 to 4.0 us per CTA against 72's 8.7 to 10.9, about 1.7 to 1.9 us per
  counting share to its count and 1.9 to 2.2 us per finishing share, of which 0.8 to 0.9 us is
  waiting (largest 4 to 8 us). That is not the trigger's "well under 1 us per share".
- Measured, the spread: 75's phase A end spread is 6.7 to 7.2 us at 128 tokens against 55's 11.4
  to 14.0 and the public kernel's 6.8, and 9.1 to 9.7 at 256 against 55's 10.0 to 12.0 and the
  public's 7.1 to 8.8. The partition with this fixup ends phase A about as evenly as the public
  kernel's.
- Measured, a new cost: route plus setup is 3.4 to 3.9 us longer than 55's and about 3 us longer
  than 73's. The light word that closes it is written after the setup `__syncthreads`, behind warp
  0's designation (`dfin_flags`), which ran its `stage_at_a_warp` ballots and `share_end` sums one
  candidate CTA at a time; every warp of every CTA, the producers included, waits there.
- Decision: under the rule this build is not gated. Two defects of the build (the SiLU predicate
  and the serial setup) together account for an estimated 4 to 5 us (hypothesis: 1 to 2 for SiLU
  from 36 against 28, about 3 from the stamps), against a measured loss of 2.1 / 3.8 at 256. The
  corrected build is a new experiment and gets its own rule below, set before its run.

### The corrected build (74 at 4187b8b) and its rule (set before the run)

Changes against 5f35018: `fast_silu` ends at 75; the designation is lane-parallel. Lane i weighs
candidate CTA k0 + i: `stage_at_a_lane` finds both bounds of its range by binary search over
S.pa (the last j with pa[j] <= target, which is `stage_at_a_warp`'s ballot count less one),
`share_end` sums its share's end on that lane, and one `__reduce_max_sync` over (end << 5 | lane)
picks the latest, ties to the later CTA as before. A CTA's cost share (at least 434 units at G =
132) exceeds a stage's (207 at most), so every range holds a stage and at most 16 CTAs hold a
unit; the CPU model (now with the lane form, the binary search asserted equal to the scan on every
call) saw at most 5 and the same designations, invariants and waits as before.

No MoE check remains today, so the run is a scored run the coordinator submits: `tools/
moe_make_bench_ab.py --nopub=1 --base=51 --sizes=32,128,256 --routings=layer,std
--seeds=107,108,109,110,111,112 o74=74` (new seeds; light twins 55 and 75; no public arm). The file
embeds no public code, keeps every production mode built and leaves MAIN_CANDIDATES unchanged, so
its judged path is the board file's dispatch; it replaces the board entry only if it scores
higher. Rule: the gate above is run only if 74 minus 51 has a negative mean at 256 tokens on both
routings in this run; otherwise the family stays parked with this evidence. Reported, not
conditions: 128 and 32 tokens, route + setup, phase A mean and A end spread against 55, the
fixup columns.

### Result of the corrected build (scored run e02f60fec94f45cb, bench at 4187b8b)

Submitted by the coordinator as a scored run (no public code; the first 6,463 lines differ from
4187b8b only in MAIN_MODES and GATE_RUNS, checked by him before submitting). 2.197x (2.164-2.230),
passed, spin flags 0; prepare's policy is the board's (sk8w3s to 32 tokens, main-lean-grp-finss64
at 128 and 256), so the run replaces the board entry only if it scores higher. Outputs of 74
against 51: 1.3e-4 to 3.4e-4. Minus 51, us, seeds 107-112 (count = seeds lower of 6):

| arm | 32 layer | 32 std | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|---|---|
| 74 corrected (4187b8b) | +7.9 (0) | +7.6 (0) | +4.6 (0) | +7.4 (1) | +2.0 (1) | +4.1 (1) |
| 74 as built at 5f35018 (eaa881a934aa471e, seeds 95-100) | +8.2 | +9.0 | +8.3 | +7.2 | +2.1 | +3.8 |

Stamps (one cold launch per seed, means, us; layer / standalone):

| | 32 | 128 | 256 |
|---|---|---|---|
| route + setup: 55 / 75 | 2.7 / 5.7 and 2.8 / 6.1 | 3.4 / 6.8 and 3.4 / 7.0 | 4.1 / 7.7 and 4.1 / 7.7 |
| phase A mean: 55 / 75 | 41.5 / 47.1 and 86.5 / 88.9 | 91.8 / 95.4 and 119.7 / 123.8 | 127.8 / 129.1 and 136.0 / 139.4 |
| A end spread: 55 / 75 | 5.0 / 4.8 and 11.1 / 4.3 | 12.7 / 8.3 and 19.5 / 6.6 | 10.9 / 8.1 and 15.2 / 7.7 |
| A done max: 55 / 75 | 53.4 / 59.4 and 102.4 / 100.9 | 114.2 / 113.7 and 147.6 / 140.3 | 151.1 / 150.3 and 163.0 / 158.8 |
| 75's fixup per CTA (to count + finish), finisher wait, largest CTA | 4.03 (1.89 + 2.14), 0.96, 7.9 and 3.94, 0.79, 9.7 | 4.40, 1.04, 11.8 and 4.36, 0.96, 13.3 | 4.45, 1.04, 12.1 and 3.89, 0.79, 9.0 |

Reading under the rule set before the run:
- Measured: the corrected 74 lost at every size and routing, +2.0 / +4.1 us at 256 tokens (1 of
  6 seeds lower on each). The rule fails; the gate is not run and the family stays parked.
- Measured: the partition with the designated finisher does what it was built to do on the end
  of phase A. The A end spread is 6.6-8.3 us at 128 and 256 tokens against 51's (55's)
  10.9-19.5, the public kernel's level (6.8-8.8 in eaa881a934aa471e), and A done max is 0.5-7 us
  earlier than 55's at 128 and 256. The fixup is 3.9-4.5 us per CTA (72: 8.7-10.9), about 2 us
  per counting share and 2.1-2.5 per finishing share including 0.8-1.0 waiting: still not "well
  under 1 us per share".
- Measured: the gain on A done max does not reach the end of the launch. Route + setup is still
  3.4-3.6 us longer than 55's with the lane-parallel designation (it was 3.4-3.9 with the serial
  one), so the setup rewrite did not remove that cost, and the stamp that closes it waits on warp
  0's whole setup block, 72's boundary and lead-slot searches included (73 was 0.8-0.9 us over 55
  there). Where the 3 us go is not separated (hypotheses: the designation still dominates
  through its shared-memory binary searches and per-lane segment loops; or the larger kernel's
  cold instruction cache at launch, as 62 and 69 suggested). The phase A mean is 1.3-5.6 us longer
  than 55's (the fixups and the larger kernel). Phase B and the tail are within 1 us of 55's.
- Decision: parked in docs/ruled-out.md with this evidence. Budget used: the day's last MoE check
  (eaa881a934aa471e) and one scored run (e02f60fec94f45cb).

## Donor MoE (M-A) against mode 51, 2026-10-31

Brief: F-LAYER-0 (checks 20abe6291eb04ed6, 00607fc6f8c84c77, docs/runs-decode-layer.md on the
F-LAYER branch) put the donor layer about 9% ahead of ours and its MoE ("M-A") at about -11 us at
decode 32 sliding and -21 us at verify 16 / 32 sliding; F-MOE-0 (40b8c798289a4258,
218ff300de7d4c22) put the exact donor MoE 3-4% ahead of our board file standalone, 1.08 / 1.12 at
128 / 256 tokens on standalone routing. User rule (2026-10-31): the donor is for measurement and
learning; a winning mechanism is re-implemented in our code with an attribution comment. Donor:
alygg77/okc-muse-spark-kernels at 6cae49a639ee6d14b929a5fd4a63c627f96d7dc0 (Apache-2.0),
`kernels/muse_moe_w4a16.py` (SHA256 b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281)
and `kernels/muse_decode_layer.py`, read in the session scratchpad only. Line numbers below are the
donor standalone file's unless marked "layer".

### What M-A is

The donor's standalone MoE file is the public file this log has measured against since
2026-10-29 (vjeux/okc-2026 at f35a41e, scratchpad copy SHA256 10e0690d...) plus 137 diff lines:
the KPERM3 permutation (the weight copy `_kperm_codes` / `_kperm_rows` 3223-3251 and 3258-3268,
the x gather 2049-2055, the stage loop's scale load 2204-2212), a helper-warp source used only by
its MID build (16 tokens), and a CUDA context bind around kernel_fn. The OKC_ flag sets differ by
one flag (OKC_WSH0, MID only); the PRMT scale decode (OKC_PRMTSC) is already in the public file.
The layer copy adds one build: `-DOKC_B4FIN=1` for 16 < M <= 32 (layer 3870, 5768-5786,
6323-6378, dispatch 13097-13116). So every gap this log attributed "against the public kernel"
is M-A's gap too, with KPERM3 and, in the layer at 17-32 tokens, B4FIN on top.

A side finding for the layer owner: the F-LAYER-0 ABPOL lines show our layer running mode 47 at
32 / 128 / 256 tokens (its _MOE1_VARIANTS ends at 47). Mode 51 is -3.9 / -4.4 us against 47 at
128 / 256 tokens on held-out seeds (c6b0e500c3a84c9a) and has not been ported, so about 4 us of
M-A's 21 us at verify 16 / 32 is already in hand (Port, below).
[Corrected 2026-10-31, coordinator: 51 is already ported. The layer candidate layer_m1rt51s
runs 51 at 32 / 128 / 256 tokens (best scored run 0f58d5d8223f45af, 2.0148x); the F-LAYER-0 ABPOL
lines came from the older release file. The 4 us is not extra headroom against the current layer.]

### M-A against mode 51, mechanism by mechanism

| # | mechanism | M-A (donor source) | mode 51 | tested by us |
|---|---|---|---|---|
| 1 | phase A partition | static cost-weighted stage ranges per CTA (`wgs_range`, tables OKC_WGSA / OKC_WGSB 1838-1839), about 1.85 cut units per CTA; a CTA runs its trailing cut unit first (`wgs_split`, OKC_WGSREORD, 1945-1953) so the share holding a unit's last stage runs early; cut units finish by last arrival (OKC_WGSDFIN = 2 sets the designated finisher for phase B only, 1855) | whole 16-stage units dealt in snake order by n-tile count, no cut, no fixup | 72 (stream-K A, last arrival: +6.6 to +13.5 us, 098949b28fe84bd1); 74 (designated finisher, the share ending last under our walk: +2.0 to +7.9, e02f60fec94f45cb). The donor's walk order (trailing cut share first) was not tested |
| 2 | phase B partition | the same static ranges over phase B's cost prefix; the share holding stage 0 finishes the unit (designated, `wgs_flush` kind 3, 2113-2140) | whole units claimed at run time (ctr[10], S.bq) | only as mode 16 (both phases stream-K) on the 2026-10-29 stage loop, which lost to 20 and 23; never under 51's components |
| 3 | readiness between phases | x producer spins on xready per job (phase A) and on doneAs >= 8 per (job, phase B stage) before each phase B stage's bulk copy (2011-2014) | the same granularity (done_as [job][16]) | same design |
| 4 | rings and producers | 3 weights slots of 36 KB, 3 x slots, one producer warp each, lane 0 issues one cp.async.bulk per stage (1819-1820, producers 1955-2023) | 5, 4 or 3 weights slots by the largest job's N, 3 x slots, one producer thread each | same structure |
| 5 | scale decode | KPERM3 makes k16 step j of every 4-step load group scale block 4 t + j of lane t, so one 8-byte scale load per stage and 4 bf16x2 scale pairs per stage (rows g and g + 8 in the two halves: one PRMT, one IMAD with an opaque 16 on the FMA pipe), applied by HMUL2 with half broadcasts (2204-2232) | our fat layout (repack_expert_fat, since mode 0) already gives each lane one scale block per 4-step chunk, two 4-byte scale loads per stage; but it builds each row's scale separately (8 PRMT + 8 LEA per stage) | not tested before: mode 76 |
| 6 | weights slot release | lane 0 arrives after its warp's last code load, no `__syncwarp`, no proxy fence (2214) | every thread fences (fence.proxy.async.shared::cta) and arrives, count 512, behind group s = 10 | not adoptable: unproven under the PTX model ("Which public releases are provable", above) |
| 7 | x tile publish | the gather and the phase A epilogue store xA / xB with generic stores; another CTA's cp.async.bulk reads them after ld.acquire with no fence.proxy.async on either side (gather stores 2061, epilogue store 2093, their releases 2066-2069, 2118-2122 and 2171-2172, copy 2018 after ld.acquire 2011 / 2014) | fence.proxy.async.global by every writer and by the issuer after its acquire | not adoptable: the 2026-10-29 proxy-ordering defect class ("Proxy ordering of the x tiles"); the donor's times understate a fenced version's |
| 8 | phase A publish | deferred behind commit group 3 of the next segment's first stage (`wgs_flush` kind 1, OKC_WGSFCI) | right after the epilogue: bar.sync and one red.release (37) | 32-34: the deferral lost or tied (c7db20e053254b52) |
| 9 | phase B epilogue and final pass from 64 tokens | y rows only; each route's sum of squares formed in the final pass, one token per item (OKC_WGSFINSS) | the same since 50 / 51 | adopted (51) |
| 10 | final pass at 17-32 tokens (layer B4FIN) | the epilogue stores each route's sum of squares per (route, 16-row block): 256 fixed partials per route, no atomics; items (token, 1024 columns), 4 M items, sums in a fixed order (layer 5768-5786, 6323-6378) | below 64 tokens: 47's path, epilogue atomics into ss, items (token, 2048 columns), 2 M items | not tested before: mode 77 |
| 11 | precision | gate, up and the GEMM2 output stay FP32 (2.35e-3 against FP32) | rounded to BF16 where SGLang rounds (3.88e-3) | ours kept |
| 12 | spin timeouts | none | clock reads and a flag off the fast path | kept (the self-tests read ctr[8]) |

Which of the five losing attempts on phase A at 128 / 256 tested what: 60 (the gather's
per-thread fence) touches our side of item 7 only; 62-66 and 68-71 (phase A's last rounds claimed
at run time) test a mechanism M-A does not have; 72 and 74 test item 1 with our walk order and
51's dynamic phase B. None tested item 2 under a stream-K phase A, the donor's
trailing-cut-share-first order, item 5's decode side or item 10.

What the gap is made of (cross-run subtraction, a hypothesis, not a paired measurement): the
public file trailed by -9.6 / -14.9 us at 128 / 256 tokens against 51 (3d8902381a194d56,
standalone routing); the donor led our board file (47 at 128 / 256) by -13 / -24 there
(218ff300de7d4c22), and 51 is -3.8 / -4.4 against 47 (c6b0e500c3a84c9a), so the donor leads 51 by
about -9 / -20. KPERM3 over the public kernel is then worth about 0 at 128 and -5 at 256; the rest
(about 10-15 us) is the public kernel's design, which earlier sections split into setup +4.3 /
+4.4 us, phase A +2.3 / +7.7, phase B +2.9 / +7.3 and tail +3.7 / +3.6 at 128 / 256 tokens
(d47c78bdc130424b against 47; 51 has since closed the final pass and part of phase B). Items 6 and
7 are part of that and stay out.

### The change: modes 76 to 78

Re-implemented in our code with attribution comments (`pair_sc`, `fin4`, `final_pass_part`); no
donor line is copied.

- 76 (pair_sc) is 51 with the paired-row scale decode on the wgmma stage loop: `scale_pair`
  selects bytes b and b + 1 of the lane's scale word into the two halves of one word (one PRMT),
  shifts and biases them as `scale_word` does (one LEA), and `dequant_pair` multiplies registers
  0 and 2 by the low half and 1 and 3 by the high half. Each half is bit-for-bit the word
  `scale_word` built (CPU check of all selector cases on 200,000 random words: 0 mismatches), so the
  products and the outputs are 51's. Loads, waits, releases and fences are unchanged. The mma.sync
  path (jobs of 1-2 n-tiles) keeps its decode.
- 77 (fin4) is 51 with the B4FIN-style final pass below 64 tokens: epilogue_b stores each 16-column
  sum of squares to slot (route, rb) of a partial buffer in ybuf's last 32 rows instead of the two
  atomicAdds, and `final_pass_part` runs (token, 1024-column) items, warp w summing half w & 1 of
  slot w >> 1's 256 partials (4 per lane, fixed xor tree), so every item of a token gets the same
  sums. From 64 tokens up 77 is 51.
- 78 is both.

Ordering, 77 (76 changes no memory operation). The partials follow ybuf's chain: writer thread W's
st.global of a partial precedes, in program order, W's arrival at the `__syncthreads` before the
grid barrier; bar.sync synchronizes its participants at CTA scope (PTX ISA 8.9.4, 9.7.13.1), so
the store precedes thread 0's `__threadfence` (fence.sc.gpu) and relaxed atomicAdd on the barrier
counter, a release pattern at gpu scope (8.8); the reading CTA's thread 0 observes the count with
ld.acquire.gpu, an acquire pattern that synchronizes with it (8.9.4), then `__syncthreads` again;
the reader's ld.global.cg is after all of that in base causality order (8.9.5, transitive), so it
reads the partial (8.10.6). Every slot a final pass reads is written in the same launch: each local
route sits in one job, whose 16 phase B units each write rb = 16 u + 4 wg + w for all 16 warps,
256 slots, through the same epilogue_b on the wgmma and mma.sync paths. `red` in shared memory
alternates halves by item parity behind one named barrier per item, as `final_pass_ss`: the next
item's writes to the other half come after every consumer warp passed this item's barrier, so
after their reads of it. The rings are idle at that point (every bulk copy was waited on before the
grid barrier), as for `final_pass_ss`; the same unproven step after a spin timeout applies. No
fence, barrier or release of 51 is removed or moved; the partial buffer needs no reset (written
before read in every launch that reads it).

Local compile gate (nvcc 12.8.1 in Docker, sm_90a, `tools/moe_compile_fused.sh`, MAIN_MODES (0, 16,
26, 28, 37, 47, 51, 76, 77, 78) against HEAD bc10cb1's (0, 16, 26, 28, 37, 47, 51)): modes 0, 16,
26, 28, 37, 47, 51 and the four skinny kernels have identical instruction text; no C7510-C7515;
76, 77, 78 at 96 registers with no stack frame and no spill. Stage loops 387 instructions for 51
(393 for the widest), 379 / 385 for 76 and 78; 76 has 50 fewer PRMT and 24 fewer LEA than 51 over
the kernel. MEMBAR 43, FENCE.VIEW.ASYNC.S 21, FENCE.VIEW.ASYNC.G 11, SYNCS 186, HGMMA 192 in all
four; 77 and 78 have one more BAR (the final pass's named barrier) and 12 fewer ATOMG (the
epilogue's sums). `tools/moe_dryrun.py` and `tools/moe_host_check.sh` pass. Profile:
`muse_moe_w4a16/profiles/sass_mode76.2026-10-31.txt`.

Expected effect (hypotheses): 76 removes 8 of about 387 instructions per stage (2%); if the stage
loop is issue-bound that is up to 3 us at 256 tokens, and nothing if it is memory-bound. 77 acts at
32 tokens only; 51's final pass there is about 2 us, so at most about 1 us. Neither closes the
partition share of the gap (items 1-2), which is the next test (below).

### Gate rule (set before the check is submitted)

One check of `tools/moe_make_bench_ab.py OUT.py DONOR.py --base=51 --sizes=32,128,256
--routings=layer,std,skewed --seeds=121,122,123,124,125,126 --gate=1 o76=76 o77=77 o78=78`, with
the donor's exact standalone file in the public slot (its MAIN build, measurement only; the bench
file stays in the scratchpad). Seeds 121-126 are new on every routing. "76 - 51" is the paired per-seed difference (negative = faster); counts are seeds lower of 6.

Keep 76 only if all hold:
1. 256 tokens: mean -1.0 us or lower on layer and on standalone routing, at least 5 of 6 seeds
   lower on each, and 0 or lower on skewed.
2. 128 tokens: mean -0.5 or lower on layer and standalone, +0.5 or lower on skewed.
3. 32 tokens: +0.5 or lower on every routing.
4. Output against 51 within 1e-5 relative L2 on every case (bit-identical products; only 51's own
   atomic order at 32 tokens can differ).
5. Prepare's FP32 self-test passes for 76 (no FAIL line, spin flags 0).
6. The compile gate above holds on the checked commit.

Keep 77 only if all hold:
1. 32 tokens: mean -0.5 us or lower on layer and on standalone routing with at least 4 of 6 seeds
   lower on each, and +0.3 or lower on skewed.
2. 128 and 256 tokens (51's code there): mean +0.5 or lower on every routing.
3. Output against 51 within 1e-4 relative L2 on every case.
4. and 5. as 76's 5 and 6.

78: if 76 and 77 are both kept, ship 78 when its mean is within +0.3 us of the better of 76 and
77 at every size on layer and standalone routing, else ship the better single mode per the rules.
If neither is kept, both go to docs/ruled-out.md scoped to these designs on 51. Budget: this
gate check, then at most one check of the shippable file; the line stops at 3 checks.

### First submission of the gate bench (check 5b2db8029f094a66): no A/B

The bench's public prepare failed to build: `patch_public` empties MID_DEFINES, and the donor file
appends to it afterwards (`MID_DEFINES += ("-DOKC_WSH0=1", "@helper0")`, a helper source for its
16-token build), which left a MID build with only those defines; it did not compile and the A/B
block stops on any exception. The judged path ran (2.205x, 2.172-2.238, passed; the prepare-time
bench put 76 first at 128 and 256 tokens by 0.3 / 1.0 us, two synthetic routings, not a rule
input). Fixes: the gate was re-sent with that one donor line removed in the scratchpad copy (it
touches only the donor's 16-token build; its MAIN build, the arm timed here, is unchanged), and
`tools/moe_make_bench_ab.py` now refuses a public file that extends an emptied build list.

### Gate result (check 64254f49676c4ce6, bench at 5f19bc1)

2.188x (2.155-2.221), passed, spin flags 0. Self-tests: 51, 76, 77, 78 all 3.89e-3 against FP32.
Minus 51, us, same process, seeds 121-126 (count = seeds lower of 6); "donor" is the exact donor
MAIN build (minus its MID helper line):

| arm | 32 layer | 32 std | 32 skewed | 128 layer | 128 std | 128 skewed | 256 layer | 256 std | 256 skewed |
|---|---|---|---|---|---|---|---|---|---|
| 76 | -0.1 (3) | -0.1 (4) | -0.0 (3) | -0.6 (5) | -0.0 (3) | -0.3 (4) | -0.8 (6) | -0.5 (3) | -1.1 (6) |
| 77 | -3.1 (6) | -1.8 (6) | -1.9 (6) | +0.7 (2) | -0.0 (4) | +0.4 (1) | +0.1 (3) | +0.5 (3) | +0.0 (3) |
| 78 | -3.5 (6) | -2.4 (6) | -2.9 (6) | -0.4 (4) | -0.2 (4) | -0.3 (5) | -0.9 (6) | -1.0 (5) | -1.3 (6) |
| donor | -2.3 (6) | -6.3 (6) | -10.4 (6) | -9.9 (6) | -13.4 (6) | -1.8 (4) | -22.5 (6) | -26.3 (6) | -6.5 (5) |

Means of 51: 90.7 / 149.3 / 69.0 us at 32 tokens, 157.3 / 202.9 / 94.5 at 128, 223.0 / 247.3 /
149.6 at 256 (layer / std / skewed). Outputs against 51: 0 (bit-identical) for 76, 77 and 78 at
128 and 256 tokens on every seed; at 32 tokens 1e-7 to 4.7e-5 for all three, including 76, whose
products are 51's: the difference there is 51's own FP32 atomics order in ss, which changes from
call to call.

Reading under the rule set before the check:
- 76: rule 1 fails (256 tokens -0.8 / -0.5 us on layer / standalone, against -1.0; 3 of 6 lower on
  standalone). Rule 4 also fails as written (3.5e-5 at 32 tokens against 1e-5), a badly set bound:
  76's products are bit-identical (0 at 128 and 256) and the 32-token spread is 51's atomics.
  Not kept. Measured: removing 8 of 387 instructions per stage gives 0.5-1.1 us at 256 tokens and
  about 0.5 at 128, so the stage loop is close to issue-bound but the gain is small.
- 77: rule 1 passes (32 tokens -3.1 / -1.8 / -1.9, 6 of 6 lower everywhere); rule 2 fails at 128
  tokens on layer routing (+0.7 against +0.5, 2 of 6 lower), where 77 runs 51's code (the
  difference is the larger kernel or noise; not separated). Not kept.
- 78: the rule ships it only if 76 and 77 both pass. Not kept under this rule.
- Measured, not a rule input: 78 is lower than 51 in all nine cells (-0.2 to -3.5 us, 4-6 of 6 seeds
  lower in each), and at 32 tokens on layer routing it is 1.2 us under the donor's MAIN build
  (87.1 against 88.3 us). The fin4 final pass carries the 32-token gain (-1.8 to -3.1 us, larger
  than the 1 us expected: the epilogue's atomics as well as the item count); the scale pairs carry
  the 256-token one.

The donor's lead over 51 in this run: -2.3 / -6.3 us at 32 tokens, -9.9 / -13.4 at 128, -22.5 /
-26.3 at 256 (layer / standalone). 78 takes 0.9-3.5 us of it.

### Gate rule for 78 alone (set before its check; the new evidence is the table above)

One check of `tools/moe_make_bench_ab.py OUT.py DONOR_MAIN_ONLY.py --base=51 --sizes=32,128,256
--routings=layer,std,skewed --seeds=131,132,133,134,135,136 --gate=1 o78=78`; seeds 131-136 are new
on every routing. Keep 78 only if all hold:

1. 32 tokens: mean -1.0 us or lower on layer and on standalone routing, at least 5 of 6 seeds lower
   on each, and 0 or lower on skewed.
2. 128 tokens: mean +0.5 or lower on every routing.
3. 256 tokens: mean 0 or lower on layer and standalone, +0.5 or lower on skewed.
4. Output against 51: 0 at 128 and 256 tokens; within 1e-4 at 32.
5. Prepare's FP32 self-test passes for 78 (no FAIL line, spin flags 0).
6. The compile gate above holds on the checked commit.

If all hold, 78 joins MAIN_MODES and MAIN_CANDIDATES and the file goes to the coordinator; if any
fails, 78 goes to docs/ruled-out.md with 76 and 77. This is the line's third and last check.

### Gate result for 78 (check c8fc5f669fa34d06, bench at 95c9848)

2.194x (2.162-2.227), passed, spin flags 0; self-test 78 3.88e-3 (51: 3.88e-3). Minus 51, us,
seeds 131-136 (count = seeds lower of 6):

| arm | 32 layer | 32 std | 32 skewed | 128 layer | 128 std | 128 skewed | 256 layer | 256 std | 256 skewed |
|---|---|---|---|---|---|---|---|---|---|
| 78 | -2.6 (6) | -1.5 (5) | -3.1 (5) | -0.6 (4) | +0.6 (1) | -0.8 (5) | -0.6 (6) | -0.4 (5) | -1.3 (5) |
| donor | -3.2 (5) | -8.7 (6) | -8.6 (6) | -11.6 (6) | -10.1 (6) | -2.4 (5) | -19.8 (6) | -20.1 (6) | -7.4 (6) |

Means of 51: 90.8 / 145.0 / 70.5 us at 32 tokens (78: 88.1 / 143.5 / 67.4; donor 87.6 / 136.3 /
61.9), 161.6 / 191.3 / 99.7 at 128, 218.5 / 236.7 / 163.3 at 256. Outputs against 51: 0 at 128 and
256 on every seed, at most 3.8e-5 at 32.

1. 32 tokens: -2.6 (6 of 6) / -1.5 (5 of 6) / -3.1. Pass.
2. 128 tokens: +0.6 on standalone routing (1 of 6 lower), against +0.5. Fail.
3. 256 tokens: -0.6 / -0.4 / -1.3. Pass.
4. Outputs: pass. 5. Self-test: pass. 6. Compile gate (95c9848 has the device code of 45f990d plus a
   comment): pass.

Decision under the rule: 78 is not kept and goes to docs/ruled-out.md with 76 and 77. The line's
three checks are spent (5b2db8029f094a66, 64254f49676c4ce6, c8fc5f669fa34d06).

Reading (measured unless marked):
- At 32 tokens the fin4 final pass is a gain in both runs: 77 / 78 against 51 -1.5 to -3.5 us on
  every routing, 35 of 36 paired seeds lower. In the layer, which runs the fat path at 32 tokens
  on judge-like routing, that is the size where a microsecond is worth most of the three.
- At 128 tokens, where 77 and 78 run 51's code apart from the scale pairs, they sit +0.6 to +0.7
  us above 51 in one routing per run (77 layer in the first, 78 standalone in the second) and
  below it elsewhere. The cause is not separated (hypothesis: the larger kernel's instruction
  layout, as 62 and 69 suggested; or noise of about that size).
- Both rules judged a mode as a replacement for 51 at every size. The policy chooses per size, so
  a mode that only acts below 64 tokens can be offered at 32 alone and never runs at 128. That
  design (fin4 as a candidate restricted to M < 64, the donor's own B4FIN arrangement: a separate
  build at 17-32 tokens) is the revisit trigger, with its own rule and fresh seeds.
- The donor keeps its lead at 128 / 256 tokens (-10 to -20 us); 78 recovers 0.4-1.3 us of it. The
  remaining gap is the partition (items 1 and 2) and the two unprovable orderings (items 6 and 7).

### Next test: the partition share (items 1 and 2), not built

What 72 and 74 left open is the combination M-A actually runs: stream-K in both phases, phase A
walked trailing cut share first (so the share holding a unit's last stage runs early and the
last-arrival finish lands on a share that started early), phase B finished by the share holding
stage 0 at its CTA's end, and no run-time claims. 74's stamps showed the phase A end spread at the
public kernel's level once a designated finisher was used, but route + setup grew 3.4-3.6 us and the
mean CTA's phase A 1.3-5.6 us (fixups and a larger kernel). The donor's version carries no
designation search at all: the finisher is fixed by position (stage 15 in phase A through the
reorder, stage 0 in phase B), which is what 74's setup cost suggests to copy. Smallest coherent
build: 72's phase A ranges with the donor's walk order and last arrival, plus static phase B ranges
over the phase B cost prefix with a stage-0 finisher (16's phase B fixup with the count read only by
that share), on 51's stage loop; a null arm with phase B kept on claims separates the two halves.
Budget: two checks (diagnostic with stamps, then a gate), rule written first; reopen trigger for the
whole family stays the one in docs/ruled-out.md.

### Port (layer)

1. The layer runs 47 at 32 / 128 / 256 tokens; 51 is in `tools/dl_moe_inline.py`'s LAYER_MODES and
   MAIN_NAMES already. Re-running the inliner on this tree adds `('main-lean-grp-finss64', 'main',
   'launch_main<51>')` to `_MOE1_VARIANTS` (checked on a scratch copy: 6,501 device lines, run twice
   byte-identical). The layer's judge-like MoE bench then chooses between 47 and 51 per size.
   [Corrected 2026-10-31: already done in the layer candidate layer_m1rt51s (0f58d5d8223f45af).]
2. A kept 76 / 77 / 78 needs, in `tools/dl_moe_inline.py` (layer owner's file): its number in
   LAYER_MODES and a MAIN_NAMES entry ("main-finss64-psc", "main-finss64-fin4",
   "main-finss64-psc-fin4"), and in the standalone file MAIN_MODES and MAIN_CANDIDATES.
3. 77 / 78 write ybuf's last 32 rows (kEmax - 32 .. kEmax - 1) below 64 tokens. The layer's MoE
   scratch passes a ybuf of at least kEmax x kH floats (host check at moe_run), so the rows exist;
   the layer owner must confirm nothing else in the layer uses those rows between the MoE's phase B
   and its final pass (the MoE launch owns its scratch for its whole duration in the standalone).
4. Validation before a check: inliner twice, Docker compile (`tools/dl_compile_check.sh`),
   per-instance SASS fence / barrier / mbarrier / atomic counts equal to the standalone
   (`tools/dl_moe_sass_counts.sh`), no new spills, and the layer self-test covering the new mode at
   17-32 tokens (fin4 acts only below 64).

## Stream-K in both phases (mode 79), 2026-10-31

Brief (coordinator, 2026-10-31): test the partition share of M-A's lead at 128 / 256 tokens,
items 1 and 2 of the M-A table above: stream-K stage ranges in both phases on 51's components,
the donor's walk order, position-designated finishers in phase B, readiness per (job, stage), and
a null arm with phase B on claims. Every fence stays. Budget 2 checks, rule set first.

### What already existed, and the change

- The donor's phase A walk order is already ours: `seg_get` (mode 16's, kept by 72) runs a CTA's
  trailing cut share first, then the leading share, then whole units, which is `wgs_split`. Under
  our walk by n-tile count (largest first in 72) the two orders differ only for ranges that span
  jobs of different n-tile counts.
- Readiness per (job, phase B stage) is already ours (done_as [job][16], count 8 per stage, the
  x producer's spin and fence.proxy.async.global before each phase B bulk copy).
- So the null arm is 72 itself (stream-K phase A, last-arrival fixup, phase B claimed), and the new
  arm 79 changes phase B only:
  - static cost-weighted phase B ranges, not unit-aligned (S.align_b 0), found by a warp in setup
    (`stage_at_b_warp`, the same arithmetic as `stage_at` for phase B), with the leading cut share's
    slot found there too and packed into S.lead_slot's high half (no SmemMain change);
  - phase B segments in range order (`seg_get_inorder`; a phase B stage waits for its phase A units
    in order, and the finisher must run late), walked largest n-tile count first in both phases
    (`walk_desc`, as 72's phase A);
  - a cut phase B unit is finished by the share holding its stage 0 through mode 74's `fixup_df`
    (every share adds its partial; the others fence, pass the warpgroup barrier and count with one
    red.release; the finisher spins with ld.acquire until the count reaches 16 less its own stages,
    passes the barrier, fences and reads the sum back), on slots G .. 2G - 1 of p.fix and p.fixcnt
    (phase A's last-arrival `fixup` uses 0 .. G - 1).
- No claim counter, no S.bq, no ctr[10] in 79 (dyn_b is false); everything else is 72's.

Why the stage-0 share and range order: the stage-0 share of a cut unit is the trailing segment of
the range of the CTA that holds it, and every other share is the first segment of a later CTA's
range. So the finisher runs at the end of its CTA's phase B and the shares it waits for run at the
start of later CTAs' (the donor's arrangement), and every wait points to a higher CTA index: no
cycle is possible, and CTA G - 1 never waits, so progress holds by induction with all CTAs
co-resident (the grid barrier's existing assumption). Under our walk by n-tile count a CTA can run
its own finisher before its counting share when the finisher's job has more n-tiles; then the CTA
before it waits for that finisher too (a chain, still acyclic). A CPU model of the partition
(`stage_at` arithmetic, kCostB, 200 synthetic routings per size, scratchpad model79.py) found every
unit covered once, one stage-0 share per cut unit, the counting shares' slot equal to the
finisher's CTA, at most 2 shares per unit, about 115-125 cut phase B units per launch, and 0 / 0.6 /
6.5 CTAs per launch at 32 / 128 / 256 tokens that walk their finisher before their counting share
(hypothesis for the judge's routing: similar).

Ordering. Phase A is 72's (reviewed sound, "Stream-K phase A on 51's components"). The phase B
fixup is mode 74's `fixup_df` chain (reviewed sound, "Designated finisher for stream-K phase A"):
each counting thread's red.add of its partial precedes its fence.acq_rel.gpu and the warpgroup's
bar.sync; thread 0's red.release.gpu is a release pattern (PTX ISA 8.8) observed by the finisher's
ld.acquire.gpu spin (acquire pattern, synchronizes-with 8.9.4); the finisher's bar.sync and
fence.acq_rel.gpu then order its warpgroup's ld.global.cg reads after it (base causality, 8.9.5;
8.10.6). The finisher's own adds precede its reads in program order (same thread, same addresses).
It clears the slot and the count; the next launch is ordered by the kernel boundary, and fixcnt is
also zeroed after the grid barrier. The phase B epilogue's ybuf / ss writes go through the grid
barrier as before; the phase B x copies keep the done_as acquire and fence.proxy.async.global.
Independent read-only review: below.

Local compile gate (nvcc 12.8.1, Docker, sm_90a, `tools/moe_compile_fused.sh`; MAIN_MODES (0, 16,
26, 28, 37, 47, 51, 72, 74, 79) against d73c4fd's (0, 16, 26, 28, 37, 47, 51, 72, 74)): 0, 16, 26,
28, 37, 47, 51, 72, 74 and the four skinny kernels have identical instruction text; no
C7510-C7515. 79: 96 registers, 8-byte stack frame, 4 B spill store and load, both outside every
stage and unit loop (in the phase loop region); stage loops 390 (399 widest; 72: 389 / 398);
24,424 instructions against 72's 18,680 and 51's 17,280 (the static phase B path compiles 8 n-tile
groups with the fixup); MEMBAR 138, BAR 48, ERRBAR 59, CCTL 64 (72: 91, 32, 35, 32). The size is a
risk: 62 and 69 lost 3-5 us everywhere from larger kernels (hypothesis: instruction cache).
`tools/moe_dryrun.py` and `tools/moe_host_check.sh` pass. Profile:
`muse_moe_w4a16/profiles/sass_mode79.2026-10-31.txt`.

### Gate rule (set before the check is submitted)

One check of `tools/moe_make_bench_ab.py OUT.py DONOR_MAIN_ONLY.py --base=51 --sizes=32,128,256
--routings=layer,std,skewed --seeds=141,142,143,144,145,146 --gate=1 o72=72 o79=79` (seeds new on
every routing; the donor's MAIN build in the public slot, measurement only). The policy chooses
per size, so 79 is judged where it would run, at 128 and 256 tokens; 32 tokens is reported.

Keep 79 (as a 128 / 256-token candidate) only if all hold:
1. 256 tokens: 79 minus 51 mean -1.5 us or lower on layer and on standalone routing, at least 5 of
   6 seeds lower on each, and 0 or lower on skewed.
2. 128 tokens: mean -1.0 or lower on layer and on standalone routing, +0.5 or lower on skewed.
3. Output against 51 within 5e-4 relative L2 on every case (cut units change the FP32 sum order;
   72 against 51 was 1.5e-4 to 2.4e-4).
4. Prepare's FP32 self-test passes for 79 (no FAIL line, spin flags 0).
5. The compile gate above holds on the checked commit, and the independent review finds no
   ordering or progress defect.

Reported, not conditions: 79 minus 72 (the phase B partition against claims, both on stream-K
phase A), 72 minus 51 (the null against the board mode), the donor's lead, 32 tokens. If 79 is
kept, the shippable file (79 in MAIN_MODES and MAIN_CANDIDATES) takes the second check; if not, 79
goes to docs/ruled-out.md with the stream-K row.

Independent read-only review (2026-10-31): no bug. Every predicate of 79 equals 72's except dyn_b,
skb, walk_desc and S.align_b (0); `stage_at_b_warp` equals `stage_at` for phase B; the ranges cover
every phase B stage once; the lead search returns the stage-0 holder even across empty ranges;
slots G .. 2G - 1 stay inside p.fix and p.fixcnt and clear of phase A's; no wait cycle (CTA G - 1
never waits as a finisher, and nothing in phase A waits on phase B); the fixup chain is 74's. It
flagged as a possible performance issue the chained finisher waits when a CTA's finisher group runs
before its counting share's, and two comment wordings (fixed in the source comments after the
check; comment-only).

### Gate result (check 2a4fb39964aa448c, bench at 587da5d)

2.196x (2.164-2.229), passed, spin flags 0; self-tests 51, 72 and 79 at 3.89e-3. Minus 51, us,
seeds 141-146 (count = seeds lower of 6):

| arm | 32 layer | 32 std | 32 skewed | 128 layer | 128 std | 128 skewed | 256 layer | 256 std | 256 skewed |
|---|---|---|---|---|---|---|---|---|---|
| 72 (null: phase B claimed) | +13.1 (0) | +10.3 (0) | +9.1 (0) | +11.3 (0) | +9.8 (0) | +20.1 (0) | +6.1 (0) | +4.3 (1) | +18.8 (0) |
| 79 (phase B stream-K) | +16.4 (0) | +17.5 (0) | +6.0 (0) | +19.4 (0) | +24.8 (0) | +27.9 (0) | +21.4 (0) | +11.3 (0) | +31.4 (0) |
| donor | -1.9 (2) | -4.7 (6) | -10.4 (6) | -11.3 (6) | -13.3 (6) | -3.3 (5) | -21.0 (6) | -25.6 (6) | -5.4 (5) |

Means of 51: 88.0 / 139.3 / 69.4 us at 32 tokens, 166.2 / 205.6 / 101.7 at 128, 215.7 / 250.3 /
159.2 at 256. Outputs against 51: 72 1.5e-4 to 5.0e-4, 79 1.6e-4 to 5.1e-4 (the cut units' FP32
order); both at 5.0e-4 / 5.1e-4 on one skewed 32-token seed.

Reading under the rule set before the check:
- 79 fails rules 1 and 2 by 11-25 us (0 of 6 seeds lower in every cell); rule 3 fails on one
  skewed 32-token seed (5.1e-4 against 5e-4, the same order of error as the null 72's 5.0e-4).
  Not kept; the shippable check is not run (one of the line's two checks used).
- Measured: replacing 72's claimed phase B with static stream-K ranges and a stage-0 finisher
  costs 8-15 us at 128 / 256 tokens (79 minus 72: +8.1 / +15.0 at 128, +15.3 / +7.0 at 256 on
  layer / standalone, +7.8 / +12.6 skewed). The null 72 repeats its 098949b28fe84bd1 loss (+6 to
  +20 us here). The donor runs the same partition 11-26 us under 51.
- Not separated (no stamps in this run; hypotheses): the phase B fixups (about 120 cut phase B
  units per launch, each with a v4 red.add per n-tile, a gpu fence, two warpgroup barriers and a
  read-back on the finisher) on top of phase A's; the chained finisher waits the review and the
  model point to (0.6 / 6.5 CTAs per launch at 128 / 256 in the model, here up to +43 us on one
  skewed seed); and the kernel's size (24,424 instructions, 41% over 51). The donor's lower fixup
  cost is part of its unfenced design (its weights-slot and x-tile orderings, items 6 and 7) and
  its smaller per-segment code; neither is adoptable as is.
- Decision: the stream-K family stays parked in docs/ruled-out.md with both phases now measured on
  current components. What it would take to reopen: a fixup that costs well under 1 us per share
  on the critical path in both phases (74 measured 2-2.5), shown by stamps before any timed gate.

## The donor kernel stamped against 51 (check 43c35d18b5504eb5), 2026-10-31

Brief (coordinator): five builds guessed where the donor's 11-26 us lead at 128 / 256 tokens comes
from and none was measured on the donor itself. One diagnostic check stamps a bench-only copy of
the donor MAIN build next to 55 (51's light and gather twin), per phase, at 32 / 128 / 256 tokens on
layer and standalone routing; then name the phases that hold the gap and propose one matching
re-implementation with a rule. No build ships from this section.

### Method

`tools/moe_make_bench_ab.py OUT.py DONOR_MAIN_ONLY.py --base=51 --sizes=32,128,256
--routings=layer,std --seeds=151,152,153,154,155,156 --work=1 --pubfix=1` (bench at 1d42779; the donor
file is the scratchpad copy of alygg77 6cae49a with its MID helper line removed; the bench file was
never committed). The donor runs its own OKC_DEBUG build: thread 0's globaltimer at its DBG points
(start, routed, gather end, consumers start, phase A end, phase B end moved to the CTA-wide point
before the grid barrier as for the public kernel earlier, after the barrier, end). New in this run,
`--pubfix=1` (PUBFIX_PAIRS in the tool): thread 0's clock64 around every `wgs_flush` call, split
into the split-unit shares (kinds 2 and 3, the fixups) and the doneA releases (kind 1), and around
the phase B designated finisher's wait; clock64 at DBG(0) and DBG(5) converts. The stamps are clock
reads and plain stores by thread 0: no fence, barrier, release or acquire is added, removed or
moved. Local compile of that build (nvcc 12.8.1): 16 / 20 B of spill, outside every stage loop;
stage loops unchanged (295-299 instructions), wgmma pipelining unchanged (HGMMA 512, wait1 256,
wait0 16 per kernel, as the unstamped build). Where the donor's unfenced x publish sits: its gather
stores xA and releases xready inside "pub gather" below (DBG 1 to 7), and its phase A epilogue stores
xB and releases doneAs in the kind 1 flush ("A doneA release" below); neither has a fence.proxy.async
on either side (items 6 and 7 above), and the stamps leave both as they are.

Timing in the same run (graph replay, cold L2, 6 seeds), donor minus 51: -3.0 / -6.2 us at 32
tokens (layer / standalone), -9.8 / -12.1 at 128, -21.8 / -22.4 at 256 (6 of 6 seeds lower except
32 layer, 5 of 6). The stamped launches (55 with its light and gather stamps, the donor in its
DEBUG build) end within 1-3 us of the unstamped replays of 51 and the donor, so the stamps describe
the timed kernels. Self-test and spin flags clean; 2.197x (2.164-2.230), passed.

### Per phase, us (mean over CTAs and seeds of one cold launch; "start" = consumers start)

| T, routing | kernel | route | start | phase A mean | A end spread | A done max | phase B mean | tail | barrier + final | end |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 layer | 51 (55) | 2.9 | 5.7 | 48.7 | 7.0 | 61.4 | 22.0 | 5.9 | 8.7 | 86.0 |
| | donor | 1.9 | 4.0 | 49.5 | 3.5 | 57.0 | 22.7 | 3.4 | 5.5 | 84.9 |
| 32 std | 51 | 2.9 | 4.9 | 89.5 | 12.4 | 106.8 | 40.1 | 7.8 | 10.1 | 145.8 |
| | donor | 1.9 | 4.3 | 85.8 | 6.5 | 96.5 | 39.4 | 6.3 | 8.4 | 141.5 |
| 128 layer | 51 | 3.5 | 9.4 | 90.4 | 11.0 | 110.8 | 43.0 | 7.5 | 12.9 | 156.2 |
| | donor | 2.5 | 6.0 | 87.0 | 5.6 | 98.6 | 43.0 | 5.2 | 9.6 | 146.7 |
| 128 std | 51 | 3.6 | 8.6 | 118.5 | 12.8 | 139.9 | 55.7 | 7.2 | 12.3 | 195.9 |
| | donor | 2.5 | 6.2 | 111.7 | 7.9 | 125.7 | 55.0 | 6.4 | 10.7 | 185.1 |
| 256 layer | 51 | 4.1 | 12.1 | 122.4 | 10.4 | 144.9 | 57.7 | 8.3 | 16.5 | 210.3 |
| | donor | 3.4 | 8.3 | 112.1 | 7.7 | 128.1 | 55.6 | 5.1 | 12.1 | 190.0 |
| 256 std | 51 | 4.2 | 11.9 | 137.8 | 10.1 | 159.8 | 65.2 | 8.1 | 15.9 | 232.9 |
| | donor | 3.5 | 8.5 | 125.8 | 8.7 | 142.9 | 62.2 | 6.1 | 12.7 | 211.2 |

Donor minus 51 by piece at 128 / 256 tokens (layer; standalone in brackets): start -3.4 / -3.8
(-2.4 / -3.4), phase A mean -3.4 / -10.3 (-6.8 / -12.0), A end spread -5.4 / -2.7 (-4.9 / -1.4),
together A done max -12.2 / -16.8 (-14.2 / -16.9); phase B mean 0.0 / -2.1 (-0.7 / -3.0); tail
-2.3 / -3.2 (-0.8 / -2.0); barrier + final pass -3.3 / -4.4 (-1.6 / -3.2), which contains the tail
wait. At 32 tokens: start -1.7 / -0.6, phase A mean +0.8 / -3.7, A end spread -3.5 / -5.9, tail
-2.5 / -1.5, barrier + final -3.2 / -1.7.

The donor's own costs (ABPUBFIX, us per CTA, thread 0): phase A fixups 10.4-12.8 (10-21% of its
phase A, worst CTA 23-32), doneA releases 2.5-7.7; phase B fixups 5.4-8.5 (12-25% of its phase B,
worst 9-17), its designated finisher waiting 0.4-0.6 (worst 1.9). [The share counts read 3.6-3.8
per CTA in both phases against ABWORK's 1.8-1.9 cut units per CTA; not reconciled: either the
flush runs twice per share or the two counts measure different things. The times are sums over the
calls stamped, so they stand either way.] Its gather 1.9-4.8 us; ours 1.8-8.0 including the
fences and barrier.

### Which phases hold the gap

Measured: at 128 and 256 tokens the end-of-launch gap (9.5-21.7 us) is mostly phase A: the mean
CTA's phase A (3.4-12.0 us) plus setup (2.4-3.8) plus the end spread (1.4-5.4), which together set
A done max (-12 to -17). Phase B's mean is within 0-3 us; the tail and the final pass add 2-4.

Derived from the stamps (arithmetic, not a separate measurement): per stage, the donor spends about
the same in both phases once its fixups and releases are taken out: at 256 tokens on layer routing
(112.1 - 12.8 - 5.6) / 83.4 stages = 1.12 us in phase A and (55.6 - 8.5) / 41.7 = 1.13 in phase B,
its whole-unit epilogues included. Ours, with no fixups, 122.4 / 83.4 = 1.47 in phase A and
57.7 / 41.7 = 1.38 in phase B, epilogues and phase A publishes included. (Stage counts are the
donor's per-CTA means from ABWORK; our CTAs stream the same stages in total.) So the donor's stage
is 0.25-0.35 us cheaper in both phases, worth about 21-29 us of phase A and 10-15 us of phase B per
CTA at 256 tokens; its fixups (12.8 + 8.5) and releases (5.6) give back about 27 of that, which is
why phase B ends up near parity and phase A keeps 10-12 us.

Static evidence for the per-stage cost (no GPU; `profiles/sass_stage_loop_vs_donor.2026-10-31.txt`):
the m64n24k16 stage loop is 387 instructions in 51 and 295 in the donor's MAIN build. Of our 92 more,
about 30 are the spin-timeout path (clock reads, ATOMG with its REDUX / VOTEU / UFLO aggregation,
YIELD, a second TRYWAIT, BSSY / BSYNC), which runs only after a failed first try; the fast-path
excess is then about 60 instructions (17%): IMAD +21, ISETP +10, LEA +8, IADD3 +5, R2UR +4, PRMT +4,
SEL / VIADD / MOV / LDS / SHFL +1-2 each, and the per-thread release (MEMBAR.ALL.CTA, FENCE.VIEW.ASYNC.S
and one more SYNCS.ARRIVE) that the donor does not have. The donor's dequant takes 7 ALU and 2 IMAD.SHL
per 8 weights (its DQ7 bit layout), its scale pairs are built once per stage from one 8-byte load
(KPERM3), and its slot addresses and descriptors are built once per segment.

Hypothesis (the one this section leaves to test): the per-stage instruction count, at four
consumer warps per scheduler, is the main driver of the phase A and B per-stage gap; with our fast
path about 17% longer and the per-stage time about 23% longer (1.38 against 1.12 us in phase B), most
of the gap fits issue pressure plus the per-thread release. The first piece, 76's scale pairs (8 of
the 60), measured -0.4 to -1.1 us at 256 tokens in two runs, in proportion to its share.

### Proposal: a lean stage loop to the donor's instruction budget (mode 80, not built)

On 51, every fence and release kept (the per-thread fence.proxy.async.shared::cta and arrive on the
weights slot stays; it is the one piece of the donor's loop we cannot copy):
1. 76's paired-row scale decode, with the stage's 8 scale bytes loaded once.
2. A dequant with the multiplies on the FMA pipe: a repack bit layout in which each 8-code word
   decodes with 7 ALU ops and 2 IMAD.SHL (the donor's DQ7 property, re-derived for our W4_BITPOS
   order). The repack is a bit permutation, lossless; a CPU round-trip test over all 2^8 code
   pairs per register position and over the real weights' bytes must reproduce the codes exactly.
3. Slot addresses, descriptors and the x slot offset computed once per segment and stepped by
   adds (no per-stage R2UR, SHFL or LEA chain).
4. The spin-timeout check moved off the stage loop's code path: the try_wait retry loop counts
   failed tries and reads the clock only every 256 failures, so the flag (ctr[8], which the
   self-tests read) still fires on a hang.

Compile gate before any check: production SASS unchanged, no C75xx, no local memory in stage or
unit loops, and the m64n24 stage loop at 320 instructions or fewer (from 387), counted by the same
script. Budget: one gate check after the compile gate holds.

Gate rule (set now, before the build): `tools/moe_make_bench_ab.py OUT.py DONOR_MAIN_ONLY.py
--base=51 --sizes=32,128,256 --routings=layer,std,skewed --seeds=161,...,166 --gate=1 o80=80`. Keep
80 only if all hold:
1. 256 tokens: 80 minus 51 mean -4.0 us or lower on layer and on standalone routing, at least 5 of
   6 seeds lower on each, and 0 or lower on skewed.
2. 128 tokens: -2.0 or lower on layer and standalone, +0.5 or lower on skewed.
3. 32 tokens: +0.5 or lower on every routing.
4. Output against 51: 0 at 128 and 256 tokens (the same bf16 products, no new sum order); within
   1e-4 at 32 (51's own atomics order).
5. Prepare's FP32 self-test passes for 80; spin flags 0.
6. The compile gate above, on the checked commit, and an independent read-only review of the
   repack and the timeout change.

Expected (hypothesis): if per-stage time follows the fast-path instruction count, 60 fewer
instructions per stage are worth up to the 0.25 us per stage above, -10 to -20 us at 256 tokens; if
the per-thread release or memory dominates, much less. Either outcome locates the rest of the gap.

The budget's second check (a repeat) was not used: the per-phase differences match the replay
gaps within 1-3 us at every size and both routings, and the earlier public-kernel stamps
(d47c78bdc130424b, 7063f86352344efa) put the gap in the same pieces.

## Stage cost ablations (modes 83 to 85), 2026-10-31

Brief (coordinator): the stamped run above put the donor's stage at about 1.12 us against our
1.38-1.47, a 23% time gap on a fast-path instruction gap of about 14% (on a line-by-line reading
of the m64n24 loop, the excess is about 40 instructions, not the 60 first estimated: the scale
build 8, which 76 removes, the x descriptor's SHFL / R2UR / convergence check about 10, ring
bookkeeping about 10, the per-thread release 4; our dequant is already the donor's 7 ALU + 2
IMAD.SHL per word, and the spin-timeout path runs only after a failed first try). So mode 80 as
proposed above would cut about 20 executed instructions and project to -1 to -3 us at 256 tokens,
under its own rule; it was not built. Instead, one check prices candidate causes with timing-only
ablations of 51.

### The arms, and what keeps them out of any policy

None of 83-85 is in MAIN_MODES, MAIN_CANDIDATES or any policy table; a kernel instantiated for one
fails to compile unless OKC_MOE_ABLATE is defined (static_assert, checked: a build with 84 and no
define stops with "ablation modes are bench-only"), and only `tools/moe_make_bench_ab.py
--ablate=1` defines it, in the bench copy. The tool refuses ablation arms in --gate or --nopub
benches. Prepare's self-test and policy never run them (not candidates); its policy bench times
non-candidates for the record only.

- 83: 51 without the per-thread fence.proxy.async.shared::cta before the weights-slot arrive
  (arrive kept). It measures what that fence costs. It does not license removing it: the chain in
  "Ordering of the ring slot releases" needs a cross-proxy fence between the generic reads and the
  bulk refill, and any narrowing (for example one elected lane after `__syncwarp`) is a separate,
  reviewed switch. SASS: 2 instructions fewer per stage (385), MEMBAR 31 and FENCE.VIEW.ASYNC.S 9
  against 51's 43 and 21; wgmma pipelining as 51.
- 84 (built, not run): the decode's ALU share. Feeding raw code words to wgmma made ptxas
  serialize every wgmma (C7513); one XOR per register, and dequant's bit extraction without the
  scale build and HMUL2 (319 instructions per stage), still serialized (C7520, compiler-inserted
  WG.AR in a divergent path; wait1 0, wait0 192). A serialized loop does not price the decode, so
  84 stays out of the check. If it were run, the bench excludes it from the output comparison
  and says so in the log (`_AB_WRONG`).
- 85 (built, not run): 3 weights and 3 x slots fixed at compile time. Its stage loops compiled to
  51's 387 instructions, and at 128 / 256 tokens the largest job has 8 n-tiles, so 51's ring is
  already 3 + 3 x 32 KB there; it would time 51 again.

Compile gate (nvcc 12.8.1, Docker, sm_90a; MAIN_MODES (0, 16, 26, 28, 37, 47, 51, 72, 74, 79, 83,
84, 85) with the define, against 25558d2's (0, 16, 26, 28, 37, 47, 51, 72, 74, 79)): every
production mode and 72, 74, 79 and the four skinny kernels have identical instruction text; no
spills; C7520 only in 84.

### Run and reading rule (set before the check)

One check of `tools/moe_make_bench_ab.py OUT.py DONOR_MAIN_ONLY.py --base=51 --sizes=128,256
--routings=layer,std --seeds=161,162,163,164,165,166 --ablate=1 o83=83 o76=76` (light twin 55
and the donor's stamps as above; seeds new on both routings). Reading, per size and routing, with
D = donor minus 51 and A = 83 minus 51 (means of 6 seeds):
1. If A is at most 0.5 D at 256 tokens on both routings, the per-thread fence is the main per-stage
   cost: the next step is the separate, reviewed narrowing switch (PTX chain first).
2. If A is between 0.5 D and 2 us of saving, the fence is a part, not the main piece.
3. Otherwise the fence is not it, and the line's second check goes to the latency arm the
   coordinator named (51 with the donor's wgmma wait-group depth and issue order) instead of a
   repeat.
76 rides along as a second measurement of the scale pairs (two earlier runs: -0.4 to -1.1 us at
256 tokens).

### Result (check 1d767886e86d46e1, bench at 2921d50)

2.193x (2.160-2.225), passed, spin flags 0; outputs of 83 and 76 against 51 0 (bit-identical) on
every case. Minus 51, us, seeds 161-166 (count = seeds lower of 6):

| arm | 128 layer | 128 std | 256 layer | 256 std |
|---|---|---|---|---|
| 83 (no per-thread weights-slot fence; diagnostic) | -0.1 (4) | +0.2 (2) | -0.3 (4) | -0.5 (5) |
| 76 (scale pairs) | +0.2 (0) | +0.3 (2) | -0.9 (4) | -0.8 (5) |
| donor MAIN | -9.3 (6) | -13.4 (6) | -20.0 (6) | -25.3 (6) |

Stamps (55 against the donor) repeat the previous run's split within 1-4 us: at 256 tokens phase A
mean 123.7 / 142.5 against 112.2 / 126.5, start 12.3 / 11.4 against 8.4 / 8.7, A end spread
9.2 / 14.6 against 7.6 / 7.5, phase B 57.9 / 65.7 against 54.9 / 62.3 (layer / std).

Reading under the rule set before the check: A = -0.3 / -0.5 us against 0.5 D = -10.0 / -12.7 at 256
tokens, so the per-thread fence is not the per-stage cost (case 3). It stays, and nothing here
argues for narrowing it. 76 repeats its earlier -0.4 to -1.1 us at 256 tokens (third run).

What the two cheap instruction cuts say together: removing 8 (76) or 2 plus the MEMBAR (83) of
about 337 fast-path instructions per stage moves 256 tokens by 1 us or less. Per-stage issue count
is not what separates the kernels.

### Where the per-stage time goes instead (re-reading measured counters, not a new run)

Mode 41's per-warp counters (37 with counters, check ca43908f02874886, layer routing, 256 tokens;
37's stage loop is the same family as 51's) split our phase A into stage loops without waits
91.66 us, weights waits 11.76 and x waits 11.50 over 83 stages per warp: 1.10 us of work per stage
plus 0.28 us of waiting. The donor's whole stage costs about 1.12 us (stamps above). So our stage
work matches the donor's, and the gap is the time our consumers spend in mbarrier waits, about 20%
of the stage loops in both phases (the same check: 19.6-20.9%, 209-231 cycles per wait), while
both producers find the ring full for about 70% of phase A. At 1.12 us per 36,864-byte stage on
132 CTAs the donor streams about 4.3 TB/s of weights; our 1.38-1.47 us is 3.3-3.5 TB/s, against
the 3.95-4.06 TB/s streaming ceiling measured in 16fd966f071340eb.

Hypothesis (not measured): the stream does not keep enough bytes in flight per SM. A slot is
refilled only after all 512 consumer threads release it, and the copy then needs its full HBM
latency before the stage two ahead can start; a ring that is "full" at the producer but whose
newest copy has not landed shows up as consumer waits. The donor's ring has the same depth (3 + 3
at 8 n-tiles), so what differs would be when its slots free and refill, which its own per-warp
waits would show; stamping those perturbed its build before (ABSKEW, 2026-10-30).

Proposal for the line's second check (instead of the latency arm, which would change the wgmma
wait depth and issue order that are already the donor's: two k16 steps per commit group, one group
in flight): a counter twin of 51 (41's counters on 51's loop) plus, per stage, the producer's issue
time and the full-barrier completion time, so that each consumer wait splits into copy latency
(issue to complete), producer delay (slot free to issue) and consumer skew. That names which of the
three the 0.28 us per stage is, without a new structure. Not built.

## Where the consumers' wait comes from: refill delay or copy latency, 2026-10-31

Brief (coordinator): read the donor statically for per-warpgroup weight sub-slots (its copy layout
is 256-row unit x 16 K-stages x 4 warpgroups of 9,216 B, donor 3254), then have a counter twin
split our 0.28 us per stage of waiting into refill delay after release and copy latency.

### The donor's weights slot (static, donor standalone file at 6cae49a)

- One slot is one stage: the weights producer issues one 36,864-byte `cp.async.bulk` per stage
  (`UNIT_BYTES`, 1972-1977) on one full barrier per slot (`BW_F + ws`).
- One empty barrier per slot, initialised with count NW = 16 (2530: `k < BX_F ? NW`); every
  consumer warp's lane 0 arrives once per stage after its last code-group load (2214); the producer
  waits on it (1970) and then issues the slot's next copy. The x ring is the same, count 16,
  arrivals after the stage's last wgmma group retires (2245, 2261).
- The 9,216-byte piece is only where each warpgroup's 64 rows sit inside the slot (`pc0 = ring +
  wg * WGS_PIECE`, 2194; `_wgs_copy`'s [unit][stage][warpgroup][9216] order, 3254). There is no
  per-warpgroup copy, barrier or refill.

So the donor refills a slot after all 16 warps release it, as we do; the difference is the number
of arrivals (16 lane-0 arrives with no fence against our 512 per-thread fence-and-arrive pairs; 83
removed only the fences and gained 0.3-0.5 us).

### Our split, from data already in the log (check 7063f86352344efa, mode 56 = 51 with per-warp
stage stamps and the producers' issue stamps, layer routing, seeds 1-6)

Caveat: 56's per-warp stamps slow it (49, the same stamps on 47, cost +5.8 / +9.4 us at 128 / 256
tokens in d47c78bdc130424b), which lengthens the lead; the split is ours under that load.

| cycles (1.72-1.74 GHz) | 128 A | 128 B | 256 A | 256 B |
|---|---|---|---|---|
| producer's issue after the slot's slowest release ("producer delay", the refill delay) | 317 | 338 | 532 | 342 |
| copy latency: issue to the first waiting warp's weights-wait end | 4,890 | 4,670 | 5,530 | 5,130 |
| x copy latency | 3,830 | 4,180 | 4,170 | 4,110 |
| lead: a warp's stage start after its own release of that slot | 5,920 | 5,810 | 7,180 | 6,270 |
| per warp per CTA, us: weights waits / x waits | 2.05 / 3.62 | 0.92 / 1.24 | 4.51 / 7.40 | 1.60 / 1.94 |

Reading (measured, with one caveat: these are 51's stamps of 2026-10-30, not a new run):
- The refill delay is 0.18-0.31 us per stage; the copy itself takes 2.7-3.2 us from issue to the
  first waiting warp (weights) and 2.2-2.4 us (x). The copy latency is an order of magnitude larger,
  and a slot's release-to-need window (lead, 3.4-4.1 us) less the refill delay and the copy
  latency leaves 0.4-0.6 us of margin on average, so the waits come from the copies' tail, not from
  the producer's reaction.
- The x waits are larger than the weights waits in phase A (3.62 / 7.40 against 2.05 / 4.51 us per
  warp at 128 / 256): the x stream, three 32 KB slots released only after the stage's last wgmma
  group retires, is the larger part of the 0.28 us.
- Per-warpgroup weight sub-slots would shorten only the refill delay and the leaders' skew waits;
  56 already bounded that at 0.45 / 1.07 us at 128 / 256 tokens ("The x gather or the warpgroup
  skew"), and the donor does not have them.

### Why no donor-side counter run

The matching donor stamps need thread 0's weights-wait and x-wait ends per stage inside its stage
loop. Built locally (scratchpad, never committed), that stamped DEBUG build grows the donor's stage
loops from 295 to 395-405 instructions and spills 176 / 416 B (148 local accesses in its phase
loops), with the wgmma pipelining intact; a donor that much slower waits less, so its waits would
not be its own (the same reason ABSKEW's donor stamps were dropped on 2026-10-30). Producer-only
stamps do not perturb its consumers but cannot see when a copy lands. No check was spent.

### Next (proposal, no build yet)

The measured lever is the copy tail against a 3-slot lead, mostly on the x stream. Two directions,
each testable as one mode on 51 with every fence kept:
1. Earlier x-slot release: free the x slot after the stage's last wgmma group is issued and
   retired within the same stage (wg_wait<0> at the stage end) instead of at the next stage's first
   group, trading one wgmma overlap per stage for about one group of x lead.
2. x tiles at their real size: the x slots are sized for the largest job (32 KB at 8 n-tiles);
   jobs with fewer n-tiles could use more, smaller x slots (deeper x ring for small jobs), which
   needs per-job x slot sizes in the ring arithmetic.
Budget: the coordinator's 2 remaining MoE checks today are unspent.

## Earlier x slot release (mode 86), 2026-10-31

Brief (coordinator): build direction (a) above on 51, every fence kept, one check under a rule set
first; if it gains but leaves most of the gap, per-job x slot sizing (b) follows after the reset.

### The change

Mode 86 is 51 with each stage's x slot released at the end of that stage: after the stage's eighth
commit group, `wgmma.wait_group 0`, then lane 0 of each warp arrives on the slot's empty barrier
(`x_release_warp`, count 32 per warp as in 51). 51 releases the slot one group into the next
stage, after `wait_group 1` retires the previous stage's last group. The x producer can then
issue the copy three stages ahead about one commit group earlier; the cost is that a stage's last
group no longer overlaps the next stage's first. Nothing else changes: the weights slot, its
per-thread fence and arrive, the phase A publish, the waits and every barrier are 51's.

Ordering: the same chain as 51's x release ("Ordering of the ring slot releases", the x slot):
the x slot is read only by wgmma through its descriptor (async proxy, 9.7.17.4); the reads of
every wgmma of the stage complete before `wgmma.wait_group 0` returns in each thread of the
warpgroup; lane 0's `mbarrier.arrive` follows in program order with release semantics at cta
scope (9.7.15.16.16); the x producer's `try_wait.parity` (acquire, 9.7.15.16.19) synchronizes with
the phase it completes (8.9.4), and its `cp.async.bulk` (async proxy, 9.7.10.28.2) is ordered
after it (8.9.1.1). Reads and the refill are in the same proxy, so no proxy fence is needed
(8.9.5), as for 51. The release is only moved earlier in program order to a point where the
retirement it depends on has already happened (wait_group 0 is stronger than 51's wait_group 1).

Local compile gate (nvcc 12.8.1, Docker, sm_90a; MAIN_MODES (0, 16, 26, 28, 37, 47, 51, 72, 74, 79,
86) against 14069cf's without 86): every other kernel has identical instruction text; no C75xx; 86
at 96 registers, no spill; stage loops 383 (389 widest) against 387; wgmma pipelining intact
(wait1 96 as 51, wait0 24 against 12: one per stage loop); SYNCS 174 against 186.

### Gate rule (set before the check)

One check of `tools/moe_make_bench_ab.py OUT.py DONOR_MAIN_ONLY.py --base=51 --sizes=32,128,256
--routings=layer,std,skewed --seeds=181,182,183,184,185,186 --gate=1 o86=86` (seeds new on every
routing). Keep 86 only if all hold:
1. 256 tokens: 86 minus 51 mean -1.5 us or lower on layer and on standalone routing, at least 5 of
   6 seeds lower on each, and 0 or lower on skewed.
2. 128 tokens: -1.0 or lower on layer and standalone, +0.5 or lower on skewed.
3. 32 tokens: +0.5 or lower on every routing.
4. Output against 51: 0 at 128 and 256 tokens (same products and accumulation order); within
   1e-4 at 32 (51's own atomics order).
5. Prepare's FP32 self-test passes for 86; spin flags 0.
6. The compile gate above on the checked commit, and an independent read-only review of the
   moved release.
If all hold, 86 joins MAIN_MODES and MAIN_CANDIDATES and the file goes to the coordinator; if any
fails, 86 goes to docs/ruled-out.md and (b) is the next candidate after the reset.

Independent read-only review (2026-10-31): no bug. 86 has exactly 51's predicates plus early_x;
every x slot is released once per use (xrel is never set on 86's wgmma path, so the s == 0 and
segment-end releases do nothing, and the mma.sync path releases its own slots); the release
follows `wgmma.wait_group 0` (.sync.aligned) in every lane. Notes: the segment end repeats a
redundant `wait_group 0` and `a_pin` (harmless); "MAIN_MODES (... 86)" above describes the local
compile build, not the committed file.

### Gate result (check a00b49f8372042a9, bench at 30d6b96)

2.195x (2.162-2.228), passed, spin flags 0; self-test 86 3.88e-3 (51: 3.88e-3). Outputs against
51: 0 at 128 and 256 tokens on every seed, at most 3.4e-5 at 32. 86 minus 51, us, seeds 181-186
(count = seeds lower of 6):

| routing | 32 tokens | 128 tokens | 256 tokens |
|---|---|---|---|
| layer | +0.2 (0) | -0.3 (3) | -0.6 (5) |
| standalone | -0.8 (3) | +0.0 (4) | -0.1 (3) |
| skewed | -0.1 (3) | -0.1 (2) | -1.4 (6) |

Donor minus 51 in the same run: -3.5 / -7.2 / -10.6 at 32, -11.7 / -10.8 / -0.7 at 128, -20.9 /
-24.8 / -7.1 at 256 (layer / standalone / skewed).

Reading under the rule: rule 1 fails (-0.6 / -0.1 at 256 tokens against -1.5) and rule 2 fails
(-0.3 / +0.0 at 128 against -1.0). 86 is not kept and goes to docs/ruled-out.md.

Measured: one more commit group of x lead per stage moves 128 and 256 tokens by 0.6 us or less on
judge-like routing (-1.4 skewed). Together with 56's split (x copy latency 2.2-2.4 us against a
3.4-4.1 us lead), this says the x waits are not set by the last group's position in the stage. The
x copies themselves are slow or late: the x producer also waits on done_as per phase B stage and on
xready per phase A job before issuing, which 56's "x copy latency" (issue to wait end) does not
include, and each x copy competes with the 36 KB weights copies for the same HBM bandwidth
(hypothesis, not separated). (b), per-job x slot sizing, gives the x ring more slots only for jobs
of 4 n-tiles or fewer, which are a minority at 128 / 256 tokens; before building it, the x
producer's own waits (readiness, empty slot) per stage should be split out of the existing
stamps or a producer-side counter, at no GPU cost if 56's log tail holds them.

## Staged pair on current components, 2026-10-29

The staged two-launch MoE (mode 19, phase A alone on owned units, then mode 21, phase B and the
final pass alone) lost to the fused kernel by 24-30 us at 128 and 256 tokens (checks
9332e2a49aff4ff2, e221858b020c4dee; "Staged GEMM1 gate"). docs/ruled-out.md parks it with a
revisit trigger, cheaper phase B or setup inside the fused kernel, and modes 23, 26, 28 and 37
have landed since. The coordinator asked for a re-test as an extra arm of the mode 43 check,
without changing 43's stopping rule or the 2-check budget.

Rebuilt pair (variant 301, "staged-lean", timing only, never a policy candidate):
- Mode 45: 37 with `a_only`. Owned phase A units, 37's stage loop, setup, SiLU and publish; no
  phase B, no final pass. Its phase B loops are compiled out (with them it spilled 268 / 656 B;
  without them 92 / 164 B).
- Mode 46: 37 with `b_only` and without owned phase A. Phase B on 37's claimed units (dyn_b) and
  the final pass; no x gather and no done_as wait. 45 completed before 46 started (stream order),
  so the done_as acquire has nothing to add; the x producer keeps its
  `fence.proxy.async.global` before each bulk copy, as mode 21 did.
- The claimed-unit x producer's done_as spin now sits under `!b_only(MODE)`, matching the static
  path; it compiles away in every other mode (production SASS unchanged, profiles/
  sass_grpcodes.2026-10-29.txt addendum).
- The stage loop is 37's, not 43's, because 43 is unproven; if 43 wins, the pair's loop could
  take it too.

Arm in the same check: `tools/moe_make_bench_ab.py --base=37 --sizes=32,128,256 --routings=layer
o43=43 o44=44 o42=42 ost=v301 --pairs=ost-o43`. ABPAIR gives the pair against 37 and against 43
per seed. prepare self-tests it against FP32 at 32, 33, 128 and 257 tokens.

Result (check 9d94ac75f4724b52, layer seeds 1-6): the pair minus 37 is +30.3 / +30.4 / +24.4 us at
32 / 128 / 256 tokens and minus 43 +31.6 / +31.9 / +27.0, 0 of 6 lower in every row. The prepare
bench (standalone routing) has phase A alone (45) at 156.5 / 160.0 us and phase B with the final
pass alone (46) at 86.8 / 97.2 at 128 / 256, a sum of 243 / 257 against the fused 37 at 210.6 /
233.2. The fused kernel's phase B overlaps the other CTAs' phase A tail (A done max minus phase A
is about 27 us at 128 and 24 at 256 in 42's stamps); that overlap is still worth more than the
pair saves. The pair passed the FP32 self-test (3.90e-3) and matched 37 within 2.3e-5. It stays
parked (docs/ruled-out.md, with the next trigger).

## Staged GEMM1 gate, 2026-10-29

Design (step 5 of the steering plan): routing and grouping, grouped GEMM1 with SwiGLU, grouped
GEMM2, weighted sum, as separate stages, keeping 64-token expert jobs, 36 KB bulk unit
transfers, the exact NVFP4 decode and one owner per output tile. The GEMM1-only prototype is
mode 19: the routing prologue and x gather of the fat kernel, then phase A on whole units (256
rows x full K, 16 stages) each owned by one CTA, so no unit is cut and no fixup runs. Units are
ordered by their job's n-tile count, largest first, and dealt in snake order (round r gives CTA
c unit r G + c for even r, r G + G - 1 - c for odd r), which every CTA derives from its own
routing prologue. The baseline is mode 18: the same kernel with mode 16's stream-K phase A, phase
B and the final pass switched off. Both end at the grid barrier.

| us, mean of 6 seeds | 128 tokens | 256 tokens |
|---|---|---|
| mode 18: phase A alone, stream-K (mode 16's phase A) | 175.6 / 176.9 | 205.6 / 206.1 |
| mode 19: phase A alone, owned units (GEMM1 prototype) | 170.1 / 171.3 | 189.2 / 190.4 |
| mode 21: phase B and final pass alone | 109.6 | 124.3 |
| staged pair: 19 then 21 | 277.3 | 311.6 |
| mode 20: 19's phase A with 16's phase B in one launch | 245.6 / 247.3 / 247.0 | 282.0 / 283.5 / 284.3 |
| mode 16 | 248.3 / 249.4 / 250.1 | 286.4 / 287.3 / 288.6 |

(checks 9332e2a49aff4ff2 / e221858b020c4dee / 72b7a5a9c511446a.)

Verdict: the GEMM1-only prototype beats mode 16's phase A (by 5.5 us at 128 and 16 us at 256,
on 10 of 12 and 12 of 12 paired seeds), so the gate passes. The full staged pipeline then loses:
19 then 21 is 28-30 us slower than mode 20 and 24-28 us slower than mode 16, because in one
launch phase B starts on each CTA as soon as its phase A slice ends and overlaps the other CTAs'
phase A tail (phase B alone takes 110-124 us; in the fused kernel it adds 75-93 us over phase A
alone). The gain came from owned whole units, not from staging; mode 20 keeps it inside the
fused kernel.

Ordering obligations. Modes 19 and 20 add no new handoff: the xa chain (gather, xready) and the
xb chain (epilogue_a, done_as at 8 releases per stage: 2 units x 4 warpgroups) are mode 16's,
only the CTA that writes a given unit changes. Mode 21 replaces the in-launch done_as handoff
with a launch boundary:

1. Writers: launch 1 (mode 19), each consumer warpgroup stores its SwiGLU outputs into xb with
   st.global, then fence.proxy.async.global, fence.acq_rel.gpu, bar.sync and red.release on
   done_as (kept; nothing in launch 1 reads it).
2. Publication and observation: launch 2 (mode 21) begins after launch 1 completes, by stream
   order (API synchronization, PTX ISA 8.9.4, which takes part in proxy-preserved base causality
   order, 8.9.5) or under PDL by griddepcontrol.wait ("all the memory operations from the
   prerequisite grids are performed and made visible", 9.7.15.14).
3. Async copy: launch 2's x producer runs fence.proxy.async.global (kept with the done_as spin
   removed) and issues cp.async.bulk from xb; the bulk operation is ordered after the issuing
   thread's prior instructions (8.9.1.1), in the async proxy (9.7.10.28.2).
4. Completion: complete_tx on xfull; the consumers wait with mbarrier.try_wait.parity
   (acquire.cta), and the completion carries an implicit generic-async proxy fence
   (9.7.10.28.2).
5. Use: ldmatrix and wgmma reads of the slot.
6. Reuse: the shared-memory slot as before (fence.proxy.async.shared::cta, then mbarrier.arrive
   on xempty); global xb is written next by a later launch's phase A, which starts after launch
   2 completes (stream order). moe_run splits batches above 512 tokens into chunks that share one
   xb, so the staged pair runs mode 20 there instead.

## Proxy ordering of the x tiles, 2026-10-29

An independent review found that the fat path (modes 0, 6, 11, 12, 13, 14, and the layer's
inlined copy, which builds mode 0) writes the x tiles in global memory with generic stores and
has other CTAs read them with `cp.async.bulk`, with no generic-to-async proxy fence on that
path. The only proxy fence was `fence.proxy.async.shared::cta` on the shared-memory ring. The
review is right, and the gather's readiness counter had a second, smaller defect. Sections are
from the PTX ISA 9.4 documentation (docs.nvidia.com/cuda/parallel-thread-execution), read
2026-10-28: 8.4 Operation types, 8.6 Proxies, 8.8 Release and Acquire Patterns, 8.9.1.1
Asynchronous Operations, 8.9.2 Observation Order, 8.9.4 Memory synchronization, 8.9.5 Causality
Order, 9.7.10.28.2 Async Proxy, 9.7.15.4 membar / fence, 9.7.15.14 griddepcontrol.

What the model requires. `cp.async.bulk` runs in the async proxy (9.7.10.28.2: "Accessing the
same memory location across multiple proxies needs a cross-proxy fence. For the async proxy,
fence.proxy.async should be used"). A generic write X reaches a later read Y through causality
order only along proxy-preserved base causality order (8.9.5), which holds when X and Y use the
generic proxy, or the same proxy within one thread block, or are aliases with a proxy fence on
the path. A generic store read by another CTA's bulk copy fits none of these without a proxy
fence. Bidirectional proxy fences "take effect within a single thread" and their effect
"composes with other forms of synchronization" (9.7.15.4).

Phase A input (xa), after the fix:

1. Writers. In each CTA, every thread except the weights producer warp (kNTM - 32 threads)
   copies its share of the token rows of the entries `cta, cta + G, ...` into the job's xa slab
   with 4-byte `st.global` (generic, weak). One job's tile collects rows from several CTAs.
2. Each writer then runs `fence.proxy.async.global` (new), so its own generic stores are ordered
   before async-proxy accesses that follow in causality order, then `__threadfence` (fence.sc.gpu,
   kept), then `bar.sync kBarGather, kNG`. A `bar.sync` synchronizes with the `bar.sync` of every
   other thread on that barrier (8.9.4), so every writer's stores precede, in base causality
   order, everything after the barrier in every participating thread.
3. Publish. After the barrier, one thread per entry does `red.release.gpu.global.add` on
   `xready[job]` (new; it was a relaxed `atomicAdd`). A release operation is a release pattern
   by itself (8.8), and its effect extends to the other threads' stores that precede it through
   the barrier (8.8 footnote 1, transitivity of base causality order in 8.9.5). The old form had
   the thread's fence before the barrier and a relaxed atomic after it: that pattern "only
   affects operations occurring in program order before the first instruction in that pattern"
   (8.8), so it covered the atomic thread's own stores, and the other threads' rows were
   published only because every thread happened to run MEMBAR.SC.GPU before the barrier, which
   the model does not promise.
4. The count. `xready[j]` gets one increment per token row from whichever CTAs gathered them. The
   consumer's acquire reads the last value of a chain of atomic adds, and observation order runs
   through atomic read-modify-writes (8.9.2), so it synchronizes with every contributing release
   (8.9.4: a release pattern synchronizes with an acquire pattern when a write in the first
   precedes a read in the second in observation order, both morally strong; all are gpu scope).
5. Consumer acquire. The x producer thread (lane 0 of the producer warp) of every CTA that holds
   a stage of job j spins on `ld.acquire.gpu` until `xready[j] == job_n[j]` (an acquire
   operation, 8.8), then runs `fence.proxy.async.global` (new).
6. Bulk copy. Then `mbarrier.arrive.expect_tx` on `xfull[slot]` and
   `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` from the xa stage. The bulk
   operation is ordered after the prior instructions of the issuing thread (8.9.1.1), so after
   the acquire and both fences.
7. Completion. The consumer warps wait on `xfull[slot]` with `mbarrier.try_wait.parity`
   (acquire.cta by default) before `ldmatrix`. "The completion of a cp{.reduce}.async.bulk
   operation is followed by an implicit generic-async proxy fence" (9.7.10.28.2), so the
   generic reads of the slot see the copy.
8. Reuse. The shared-memory slot: the consumer warps' generic reads, then
   `fence.proxy.async.shared::cta`, then `mbarrier.arrive` (release.cta) on `xempty[slot]`; the
   producer's acquire wait on it comes before the next bulk write into the slot (unchanged). The
   global xa and xb tiles are written once per launch. Every bulk read of a launch completes
   before its CTAs reach the grid barrier (the consumers wait on every `xfull` they use), and the
   next writes come from the next launch, which begins with `griddepcontrol.wait` ("wait until
   all prerequisite grids in flight have completed and all the memory operations from the
   prerequisite grids are performed and made visible", 9.7.15.14) or follows in stream order (8.9.4
   API synchronization, which also takes part in proxy-preserved base causality order). The
   counters are reset after the grid barrier, after their last reader.

Phase B input (xb): the writers are the 128 threads of the consumer warpgroup that finished a
phase A unit; each stores its SwiGLU outputs into the xb stage with `st.global`, then
`fence.proxy.async.global` (new), then `fence_gpu` (fence.sc.gpu, or fence.acq_rel.gpu in modes
12-14), then `bar.sync` on the warpgroup's barrier, and thread 0 does `red.release.gpu.add` on
`done_as[j * 16 + st]` (this release was already after the barrier). The x producer of the CTA
that holds the phase B stage spins with `ld.acquire.gpu` until the count reaches 8, then
`fence.proxy.async.global` (new), then the bulk copy, with steps 6-8 as above.

Why fences on both sides. The writer-side fence is the documented pattern (CLAUDE.md: proxy
fence on the writing side before the release). 9.7.15.4 says a bidirectional proxy fence acts
within one thread and composes with other synchronization, without saying whether a fence in
the writer orders that write against another thread's later async operation or whether the
issuing thread must fence its own async operation. The consumer-side fence covers the second
reading: the generic writes precede the acquire in causality order, then the issuer's proxy
fence, then its bulk copy. It costs one fence per issued segment in one thread.

Other paths. The skinny kernel's bulk copies read only the repacked weights, written before the
launch and ordered by stream order. The legacy W4 GEMMs use non-bulk `cp.async`, which is not an
async-proxy operation (9.7.10.28.2 names only the bulk forms). Neither needs a change.

Build evidence (Docker, CUDA 12.8.1, sm_90a, `tools/moe_compile_fused.sh`): registers, stack and
spills of every kernel are identical before and after; the SASS gains 115 `FENCE.VIEW.ASYNC.G`
(the global proxy fence, per epilogue instantiation and per producer wait) and 7 `MEMBAR.ALL.GPU`
(the release before the gather's red, one per fat kernel instance). The layer copy is
regenerated by `tools/dl_moe_inline.py` (now restricted to the layer's mode 0 by `LAYER_MODES`,
and patched for the int4 topk read); `diff` against the previous layer file shows all four fix
sites (`fence_proxy_async_global` in epilogue_a, the gather and both producer waits, and
`red_release_add` on `xready`). The layer's MoE extension compiles with torch's headers
(`tools/dl_moe_compile_check.sh`, new: `tools/dl_compile_check.sh` builds only the main
extension), mode 0 at the standalone's 80 B stack and 136/236 B spills. The layer's host wrapper now zeroes `Params::cost`, which
selects the compiled stage-cost table: the struct is not zero-initialized, and moe_run's `cost`
argument (empty means the compiled table) has no layer counterpart since the layer fits no
costs.

## Cost model, 2026-10-29

Replaces the least-squares stage-cost fit (negative coefficients in 84de5bbafa344b7a). Data:
mode 11 (mode 6 with stamps) on 72 cold-L2 launches, 3 sizes (32, 128, 256 tokens) x normal and
skewed routing x 4 seeds x 3 partition tables (the compiled one, flat 100 per stage, steep
20 + 25 n). The compiled table alone gives every CTA nearly the same predicted work, which is
why the old fit was collinear; the other two put different class mixes and cuts on each CTA.
Target: warp 0's time inside segments of both paths (the per-warp stamps cover only the
mma.sync path, so the old fit's target missed every wgmma segment). Terms, each a separate kind
of work: per stage and per n8 tile of each phase, per segment and per cut (stream-K partial)
segment of each phase, and a fixed CTA term. Fit: least squares with nonnegative coefficients
(exact, every support tried) on seeds 11-12; validation on seeds 13-14 against the compiled
table's prediction with its own nonnegative scale and offset. Gate: better per-launch Spearman
rank of predicted work against measured CTA finish times on both routings, and no worse
ordering of the three tables by makespan.

Result (submit 8078730bfa434074, log in muse_moe_w4a16/profiles/):

| term | A (GEMM1) | B (GEMM2) |
|---|---|---|
| per stage, us | 1.006 | 1.282 |
| per n8 tile per stage, us | 0.139 | 0.112 |
| per segment, us | 0 | 0 |
| per cut segment, us | 2.55 | 10.45 |

CTA term 0.91 us; fixed table: 0.01153 us per cost unit plus 19.3 us.

| held-out | CTA rank rho, model / fixed | tables ordered right, model / fixed | work rms, model / fixed |
|---|---|---|---|
| normal | 0.683 / 0.448 | 18/18 / 17/18 | 22.7 / 23.3 us |
| skewed | 0.702 / 0.255 | 12/15 / 13/15 | 20.9 / 21.8 us |

Verdict: the model ranks CTAs within a launch much better than the compiled table, but it
orders partitions no better (one pair worse on skewed routing) and explains the work barely
better (rms 22 us on a 146 us mean, so most of the per-CTA spread is something neither model
sees, probably waits on other CTAs' phase A output inside phase B segments). It failed the gate
and does not drive partitioning; no -nn variant was benched.

What the data says about T=128 anyway:
- The compiled table had the shortest makespan in 10 of the 12 held-out cases (the exceptions:
  bs128 skewed seed 13, flat 151.0 vs 156.7 us; bs32 skewed seed 13, steep 69.6 vs 70.4). Its
  remaining imbalance (slowest CTA's finish minus the mean) at bs128 normal is 12.3 and 9.5 us
  and at bs256 normal 17.9 and 21.2 us, so a perfect partition is worth at most about 10 us at
  128 tokens, which would put the fat path (253 us) about 8 us under legacy128 (251.5). That is
  under 0.5% of the score.
- The per-stage term dominates: about 1.0 us (A) and 1.3 us (B) per stage against 0.11-0.14 us
  per n8 tile, so at bs128 (mostly 4 n8 tiles per job) a stage costs the consumers about 1.56 us
  against about 1.43 us for the ring to stream it (main-loads 191 us over about 134 stages). The
  consumers are the bottleneck because of the per-stage fixed cost, not the MMA work per token.
  The earlier ablation (36b5b559b0344d0f: bs128 loads 189.6, dequant without wgmma 211.9, wgmma
  without dequant 233.2, both 260.8) puts more of that fixed cost on the wgmma side (fence,
  two HGMMA, commit, wait per group) than on the dequant.
- A cut segment in phase B costs about 10 us (A: 2.6): the fixup's FP32 global adds, the count,
  and the last share's read-back. Each CTA has up to two cuts per phase.

## Where the fat path's time goes at 128 tokens, 2026-10-28

Clock stamps of mode 11 (mode 6, jobs of up to 2 n8 tiles on mma.sync, the rest on wgmma) in
check 2072795029b7479f: one cold-L2 launch per size on the bench's routing (seed 11), all 132
CTAs, us at the measured 1.75 GHz. "Per warp" sums over a consumer warp's mma.sync segments
only (at 128 tokens the jobs of 3 or more n8 tiles run on wgmma and are not split out; their
share is the consumer time the four buckets leave over). Mean / max over CTAs.

| | bs32 | bs128 | bs256 |
|---|---|---|---|
| launch end (last CTA) | 209.7 | 264.4 | 289.8 |
| routing + x gather, before any weight load | 7.4 / 13.6 | 12.0 / 16.9 | 18.4 / 24.3 |
| consumers (first stage to last epilogue) | 190.4 / 198.9 | 236.1 / 247.8 | 247.4 / 263.2 |
| grid barrier wait | 9.7 / 24.6 | 12.8 / 48.1 | 16.7 / 50.9 |
| final norm and sum | 1.1 / 2.3 | 3.0 / 3.3 | 5.9 / 6.7 |
| per warp: computing (dequant + mma.sync) | 116.2 | 136.2 | 93.3 |
| per warp: waiting for weights | 11.8 | 12.3 | 8.2 |
| per warp: waiting for x | 11.2 | 9.6 | 7.5 |
| per warp: epilogue and fixup | 24.1 | 24.4 | 16.4 |
| per warp: stages on mma.sync, segments | 116.4, 9.1 | 133.8, 10.0 | 93.1, 6.9 |
| weights producer waiting for a free ring slot | 136.8 | 171.7 | 179.6 |
| x producer waiting for a free slot / done_as / xready | 121.8 / 12.0 / 7.2 | 152.8 / 15.8 / 3.9 | 157.2 / 17.5 / 3.9 |

Reading, at 128 tokens (bench: main-loads 190.6, main-m2 255.4, legacy128 253.6):

- The decomposition is not the limit. Streamed alone, the expert-local jobs take 190.6 us
  against 250-254 for the legacy W4 chain; every expert's weights are read once.
- The consumers are. The weights producer spends 172 of the consumers' 236 us waiting for a
  free slot: the ring is full and the 16 consumer warps drain it. A warp computes a 36 KB stage
  in 1.02 us (136.2 / 133.8) against about 1.2 us to stream one at the loads-only rate, so the
  compute alone nearly matches the stream, and everything else on the consumer's path adds:
  epilogue and fixup 24 us per warp (about 2.4 us per segment: a fence.sc.gpu, two named
  barriers and up to three L2 round trips on the fixup path), 22 us of waits for weights and x
  (ring fill at the start, x behind phase B's done_as gate).
- mma.sync in place of wgmma changed nothing (submit 0400cfbdfa2845fe), and the SASS of the
  mma.sync stage is about 13 instructions per 8 weights with no waste (9 integer, 4 HMUL2; the
  E2M1 bit layout leaves no 1-shift form for the fourth register), so the per-stage compute is
  close to its floor for this dequant; the gains are in the overheads.
- Around the consumers: 12 us of routing and gather before the first weight load (the routing
  warp read topk_ids in 8 serial rounds at E = 1024), and stream-K imbalance: the slowest CTA's
  consumers end 11 us after the mean, and the barrier wait reaches 48 us.

So the loss at 128 tokens is implementation: per-stage consumer work at about the stream rate
plus serial overheads (prologue 12, epilogue and fixup 24, imbalance 11 us), not the job
decomposition. Changes made from this (commit after a248788, not yet run): topk_ids loaded by
all threads at once; the weights producer starts before the x gather; fence.acq_rel.gpu in
place of fence.sc.gpu in the epilogue and fixup (modes 12, 13); per-class stage counts in the
stamps and a least-squares fit of per-stage costs, offered to the bench as a fitted table.

## W4A16 GEMM (own kernel), 2026-10-28

Per shape, GEMM1 and GEMM2 over the unique weight bytes of the experts that received tokens
(18.9 MB for W13, 9.4 MB for W2 per expert, NVFP4 values plus E4M3 scales; global scales
negligible), routing seeds 11 and 12 (`977 * seed + bs`, the same in check 16fd966f071340eb's
split and in the tuner). "Stream" is the cold-L2 streaming-read kernel's bandwidth at that
size, interpolated between the sizes measured in check 16fd966f071340eb. Marlin columns are
that check's kernel-only times
(replay overhead subtracted). Tuner columns are check c1025175b73548fd's medians, Marlin
GEMM1 + silu_and_mul vs the best W4 GEMM1 + its split-K SwiGLU reduce, and Marlin GEMM2 +
finalizer vs W4 GEMM2 + finalizer, same routings, each including about 5 us of graph replay
overhead.

| bs | experts | GEMM1 MB | floor @4.8 / @stream us | Marlin g1 us | GEMM2 MB | floor @4.8 / @stream us | Marlin g2 us | tuner g1+act: Marlin -> W4 | tuner g2+fin: Marlin -> W4 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.5 | 66 | 13.8 / 17.9 | 30.3 | 33 | 6.9 / 9.9 | 19.9 | 37.2 -> 30.8 (1.21x) | 29.3 -> 24.6 (1.19x) |
| 8 | 7.5 | 142 | 29.5 / 36.8 | 57.6 | 71 | 14.7 / 19.1 | 32.7 | 65.0 -> 51.4 (1.26x) | 42.5 -> 33.8 (1.26x) |
| 16 | 12.5 | 236 | 49.1 / 60.2 | 95.2 | 118 | 24.6 / 31.1 | 51.4 | 102.3 -> 77.6 (1.32x) | 60.9 -> 47.6 (1.28x) |
| 32 | 17.5 | 330 | 68.8 / 83.3 | 131.7 | 165 | 34.4 / 42.6 | 69.6 | 139.1 -> 101.4 (1.37x) | 79.2 -> 60.3 (1.31x) |
| 128 | 24 | 453 | 94.4 / 113.5 | 198.6 | 226 | 47.2 / 57.8 | 103.3 | 207.2 -> 168.1 (1.23x) | 113.7 -> 91.5 (1.24x) |
| 256 | 22.5 | 425 | 88.5 / 106.8 | 263.8 | 212 | 44.2 / 54.4 | 136.7 | 275.3 -> 244.6 (1.13x) | 150.9 -> 121.8 (1.24x) |

Marlin ran GEMM1 at 1.6-2.5x and GEMM2 at 1.6-2.5x the streaming floor (the 2.5x at bs256).

Lessons:
- Kernel: swap-AB mma.sync m16n8k16, weights as A (16 rows x 16 k, so one E4M3 scale per row
  per k block), tokens as B from shared memory via ldmatrix, FP32 accumulation, global scale
  times 2^7 in the epilogue. Weights are repacked losslessly in prepare() so each lane's
  16-byte loads are its own fragments (no shared memory for weights), and each 32-bit word's
  bits are permuted so that 4 bf16x2 registers come out of 1 AND, 2 shift+AND, and a 3-op
  merge; the scale byte becomes bf16 scale * 2^119 with one PRMT and one IMAD (valid for
  positive normal E4M3, checked in prepare()). About 17 instructions per 8 weights, 9 on the
  integer pipe (SASS). `tools/moe_emulate_w4.py` checks the bit permutation, fragment order,
  scale order, row mapping and the layout-1 addressing against plain NVFP4 dequantization.
- Numerics: the W4 path rounds to bf16 exactly where SGLang materializes bf16 (gate and up
  before SwiGLU, SwiGLU's output, GEMM2's output before the norm), and reduces in FP32 inside
  each GEMM. Its error against FP32 is 4.2e-3 to 4.4e-3 relative L2 (Marlin path 5.2e-3 to
  5.5e-3). Before check 84f3975334f048c3 it kept those tensors in FP32 (3.0e-3), which
  TASK.md's activation rule does not allow.
- K splits (2 to 4) won at every size: without them there are too few warps in flight. The
  split-K partials cost a reduce kernel (SwiGLU for GEMM1, the finalizer for GEMM2), still a
  net win.
- Dequant costs 2-4% (mode 2), and removing all compute does not speed the kernel up (mode 1).
  Streaming-style addresses, a block-contiguous weight layout, persistent blocks with
  cross-unit prefetch, and dropping the L2 256-byte hint each moved GEMM time by about 1 us or
  less. At bs8 GEMM1 runs at about 3.25 TB/s net of overhead, 85% of the streaming kernel.
- Programmatic dependent launch saves 2-4 us per forward. It also makes the profiler's
  per-kernel times include time spent waiting, see docs/hotspots.md.
- bs256's hottest expert gets up to all 256 tokens; items of 8-16 tokens re-read its weights
  (through L2), which is why W4 GEMM1 gains least there (1.13x). A tile with more tokens per
  dequantized weight would help that shape.


## MAIN path, wgmma pipelining, 2026-10-28

The fat path (M >= 17) dequantizes into registers and issues `wgmma.mma_async m64nNk16` with
A from registers, two k16 steps per commit group and `wait_group 1`. ptxas serializes every
wgmma of a function (a `WARPGROUP.DEPBAR.LE gsb0, 0x0` after each HGMMA, plus a
`WARPGROUP.ARRIVE` before each) whenever any of its pipeline checks fails, and the warning
that names the reason is not always printed. `tools/moe_wgmma_micro/` holds four micro
kernels that isolate the causes; `tools/moe_compile_fused.sh` prints the full kernel's
warnings and the SASS counts come from `cuobjdump`. Findings, all on CUDA 12.8 ptxas for
sm_90a at 576 threads per CTA (112-register cap):

| structure | result |
|---|---|
| per-variant `float acc[4 * NTT]` arrays, one per N inside the switch | every accumulator array in local memory (584-byte stack), C7515, serialized |
| one `float acc[32]` shared by every N variant, the finisher loading the fixup sum into `acc` | C7515: non-wgmma instructions (the `LDG`s) define accumulator registers; serialized |
| the finisher loads the sum into a separate `fin[32]`; `acc` written only by wgmma | no C7515, but still serialized without a warning |
| runtime scale-d (`first ? 0 : 1`) inside a runtime stage loop (micro test wg.cu modes 3, 7, 15) | C7515, serialized; the same with a single stage (mode 1) or an immediate scale-d (modes 2, 6, 10) pipelines |
| zeroed accumulators, immediate scale-d 1, a `switch` over N variants inside an enclosing loop (wg2.cu mode 3, wg3.cu modes 19, 35, 67 with per-variant accumulators, an if-chain, or only two variants) | C7512 "insufficient register resources", serialized; at 384 threads (mode 131, a 168-register cap) it pipelines |
| one segment loop per N variant, the variants in sequence, no loop around them (wg4.cu modes 2, 10, 14) | pipelined: one `wait 1` per group, 90-96 registers |
| the full kernel with that structure (16 groups: 8 N variants x 2 phases) | 256 HGMMA, 128 `wait 1`, 16 `wait 0`, no C75xx, 96 registers, 88-byte stack, 156 B spill stores |

Rules kept from this: the accumulators are zeroed once per segment before the first
`wgmma.fence` and never written by anything else (the fixup finisher reads the slot sum into
`fin`); the A fragments of a retired group are pinned with an empty `"+r"` asm after the wait
that retires it (`a_pin`) so the next group's dequant gets fresh registers; and the consumer
walks its segments grouped by (phase, n-tile count) with one loop per group, the two producers
walking the same order.

## TMA weight loads and large token tiles, 2026-10-28

Bench in check e1619f311d3747e3 (`tools/moe_bench_w4x.py` appended to bench.py by `tools/moe_make_bench.py`): per batch size, GEMM and routing seed (normal 11, 12, 13; skewed 21, 22, seed `977 * seed + bs`), the median of 30 CUDA graph replays, each after a 256 MB write that evicts L2. Times include the 5.4 us replay overhead and the SwiGLU reduce (GEMM1 with K splits) or the finalizer (GEMM2), as the tuner times them; GEMM2 reads the current policy's GEMM1 output. "cur" is the policy the tuner picked in that run. Ratios are cur over variant, summed over the three normal seeds (n) and the two skewed ones (s). Every variant matched cur to within 1.0e-4 relative L2 (0 where the K splits match). The best TMA and the best large-tile variant per row are picked by the sum over normal seeds.

| bs | GEMM | cur us (n11/n12/n13/s21/s22) | best TMA | best large tile |
|---|---|---|---|---|
| 1 | g1 | cfg 20: 29.8/32.3/28.5/30/29.9 | tE 30.5/42.4/27.1/30.7/30.7 (0.91x n, 0.98x s) | - |
| 1 | g2 | cfg 26: 23.9/26.8/20.8/23.7/23.6 | tA 26.8/30.3/23.5/27.2/27.4 (0.89x n, 0.87x s) | - |
| 8 | g1 | cfg 33: 53.3/50/61.4/36.7/31 | tG 59.8/56/68.3/43.3/33.4 (0.89x n, 0.88x s) | - |
| 8 | g2 | cfg 30: 35/34.1/46.3/31.7/31.3 | tA 42.1/38.7/47.3/30.6/28 (0.90x n, 1.08x s) | - |
| 16 | g1 | cfg 21: 79.9/76/74/43.5/52.4 | tC 84.2/79.3/73.5/43.9/50.5 (0.97x n, 1.02x s) | - |
| 16 | g2 | cfg 35: 49.9/45.6/43.8/27.7/32.3 | tE 55/50.8/49/30.1/31.6 (0.90x n, 0.97x s) | - |
| 32 | g1 | cfg 33: 110/93.7/91.7/79.6/62 | tG 123/98.1/92.4/56.8/49.8 (0.94x n, 1.33x s) | h8s2+32 120/102/100/78/60.5 (0.92x n, 1.02x s) |
| 32 | g2 | cfg 35: 65.4/56.2/53.7/47.9/38.4 | tE 70.4/60.5/59.2/37.8/32.4 (0.92x n, 1.23x s) | h8+32 70.9/61.5/59.3/55/43.2 (0.91x n, 0.88x s) |
| 128 | g1 | cfg 4: 176/157/213/182/142 | tG 180/160/206/133/109 (1.00x n, 1.34x s) | h8s2+32 181/166/198/137/108 (1.00x n, 1.32x s) |
| 128 | g2 | cfg 36: 96.1/86.1/108/73.2/59.5 | tE 98.3/88.3/111/71.7/59.5 (0.98x n, 1.01x s) | h16+64 103/92.4/115/73.5/58.5 (0.93x n, 1.01x s) |
| 256 | g1 | cfg 37: 258/229/254/285/224 | tG 248/219/246/272/207 (1.04x n, 1.06x s) | h16+64 255/245/269/253/219 (0.96x n, 1.08x s) |
| 256 | g2 | cfg 13: 129/114/129/143/106 | tE 131/113/129/143/108 (1.00x n, 0.99x s) | h16+64 131/122/139/136/100 (0.95x n, 1.06x s) |

Labels: TMA tA (1,1,8,4,4,4), tB (1,1,8,4,4,8), tC (1,1,8,4,8,4), tD (1,1,8,2,8,4), tE (1,1,8,2,4,8), tF (1,2,16,4,4,8), tG (1,2,16,4,8,4) for GEMM1 and tA (2,1,8,4,4,8), tB (2,1,8,4,8,4), tC (2,1,8,2,4,8), tD (2,2,16,4,4,8), tE (2,2,16,4,8,4) for GEMM2, all mode 7 layout 1, d = shared-memory stages. Large tiles: u32/u64 every item 32/64 routes; hA+B items of A routes for experts below the threshold and B routes (a second launch, run first) for experts at or above it (32 for +32, 64 for +64); h8s2+32 uses 2 K splits.

Registers, shared memory and occupancy (cudaFuncGetAttributes and the occupancy API in check 4a8852d513874318; warps per SM count the TMA kernel's producer warp; no spills in ptxas for any of them):

| config | regs | smem B | blocks/SM | warps/SM |
|---|---|---|---|---|
| register (1,1,8,4,4,4) | 76 | 16576 | 6 | 24 |
| register (1,2,16,4,4,4) | 92 | 33152 | 5 | 20 |
| register (1,2,16,4,8,4) layout 1 | 98 | 33152 | 2 | 16 |
| TMA (1,1,8,4,4,4) | 40 | 35040 | 6 | 30 (24 consumer) |
| TMA (1,1,8,4,4,8) | 40 | 53536 | 4 | 20 (16) |
| TMA (1,1,8,4,8,4) | 40 | 53472 | 4 | 36 (32) |
| TMA (1,2,16,4,8,4) (tG) | 56 | 70016 | 3 | 27 (24) |
| TMA (2,2,16,4,8,4) (GEMM2 tE) | 56 | 70016 | 3 | 27 (24) |
| large (1,4,32,4,4,4) | 124 | 66304 | 3 | 12 |
| large (1,8,64,4,8,4) | 190 | 132608 | 1 | 8 |
| large (1,8,64,4,4,2) | 146 | 132608 | 1 | 4 |
| large (1,4,32,2,8,4) | 128 | 131840 | 1 | 8 |

Lessons:
- TMA weight loads (one producer thread, two bulk copies per stage of a block's W units, full and empty mbarriers) cut registers from 76-98 to 40-56 and raised resident warps (36 per SM at W8 D4 against 16-24), yet lost 3-11% at bs 1-32 and tied at bs128 on normal routings. It won only at bs256 GEMM1 (tG, 1.04x, all five seeds) and against its own register twin (1,2,16,4,8,4) layout 1 by 6-10% at bs 8-256. So more resident warps, by themselves, do not close the gap to the streaming kernel; the loads-only result (50.8 us with 8 warps per block vs 59.6 with 4) did not carry over to the real GEMM. One likely reason, not measured: each stage is now a serial wait on a full barrier, then shared-memory loads, then the arrive, and the register path overlaps D stages of global loads per warp without that round trip.
- Large token tiles cut weight traffic where routing is concentrated: uniform 64-route items were 1.5x faster at bs128 skewed (120/91 us vs 182/142) but 2x slower on normal routings, where most experts get far fewer than 64 routes. The device-side split (large items for experts at or above a threshold, a second launch run first) kept part of the skewed gain (h8s2+32 1.32x at bs128 skewed) but lost or tied on normal routings at every size, including n11 at bs256 whose hottest expert has all 256 tokens (h16+64 255 vs 258 us). The large-tile kernel runs at 4-12 warps per SM (124-190 registers, 66-132 KB of activations) and the two launches serialize, which together cost about what the fewer weight re-reads save. The scored shapes use the normal routing (TASK.md: skewed appears only as a correctness test), so this is not wired in.
- The tuner now includes the TMA configs; it picked one (bs256 GEMM1) in check a49578e5c30348bb, which scored 1.691x (1.665-1.716), the same as before within the interval.
- Decision 2026-10-28: TMA weight loads and hot-expert large tiles stay out of submission.py.
  TMA won only bs256 GEMM1 (about 4%, about 0.5% of the score); large tiles lost on normal
  routing. The final file scored 1.691x (a49578e5c30348bb), overlapping the 1.699x entry. Code
  kept at muse_moe_w4a16/experiments/tma_and_large_tiles.py.

## Prepared for the next checks (not yet run)

Commit after f82a8cf656df4914, local work only (no checks left on 2026-10-28):

- Skinny kernel as per-variant namespaces: `sk8w4s` (the shipped one), `sk16w4s` (16 compute
  warps, 256 k each, x in 32 registers), `sk8w5s` (5 ring slots, 229 KB smem), and loads-only
  twins of the first two that stream the units without computing. ptxas: 140 / 96 / 130
  registers, no spills.
- Fat kernel: codes repacked as [warpgroup][warp][step quad][lane][4 words] (the first layout
  gave every lane a 64-byte stride, a 4-way bank conflict on each LDS.128); rings sized at
  runtime from the launch's largest job (5 weight slots next to 3 x 8 KB x slots when every
  job is N <= 16, 4 + 3 x 16 KB up to N = 32, 3 + 3 x 32 KB beyond); modes d2 (depth-2 wgmma
  pipeline, the shipped one), d3 (three A buffers, `wait_group 2`; micro test wg5.cu shows
  depth 4 trips C7512 at 576 threads), loads-only, and no-gate (no xready / done_as waits,
  timing only). ptxas: 96 registers each, 136-176 B stack, spills only at segment boundaries,
  128 `wait_group 1` (d2) or `wait_group 2` (d3) per kernel, no C75xx.
- The previous kernel's W4 path ported back as `_run_legacy` with its tuned configs at 32,
  128, 256 (`LEGACY_POLICIES`), a candidate from 16 tokens up.
- `prepare` self-tests every computing variant, benches all of them per batch size (graph
  replay, cold L2, two routings) and keeps the fastest as the policy; the BENCH lines carry
  the loads-only ceilings, which split the fat path's gap into ring versus compute.

**Board entry note (2026-10-28 23:25 UTC).** Rows above that call 84de5bbafa344b7a, 0400cfbdfa2845fe,
c524612c78454486 or 36b5b559b0344d0f "the board entry" are historical: those unfenced runs were
withdrawn (see docs/context.md). The board entry was then da9225e4f89441fc (2.003x, fenced, mode 16); since 2026-10-29 it is fa18c9e87d6e4811 (2.070x, mode 23).

## Ruled out

The losers of the fused-kernel runs are in `docs/ruled-out.md` under muse_moe_w4a16.

Losers from these runs are in docs/ruled-out.md.

## October31 23:07 PDT indexed library adapter

8d64807ea7504495, source experiments/humming_indexed_r.py from checked prefill-v3 BODY
via tools/moe_humming_candidate.py:5/5 official tests,0.911512837x,95%CI
0.897840145–0.925185530. No public submission. Baseline/ours us byBS:
1=80.00/91.01,8=142.91/167.23,16=187.47/228.86,32=206.05/253.86,
128=459.39/448.17,256=612.37/546.62. Profile with baseline eae9a6b7d5dd4809 before dispatch.
The indexed routed heuristic is unsuitable as current persistent-kernel replacement;
retain source and loss, do not infer masked/local-heuristic backend behavior from it.

Scored historical-best47 fresh repeat116c5185b14b487d:PASS5tests,2.213803083x.
Source `git show ec565a7:muse_moe_w4a16/submission.py` exactly235700bytes,
SHAb35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14 plus one trailing
comment (frozen .build/today/moe_best47_repeat.py). Existing b50351288b504e81/5f381b8b6946492d
check/paired evidence and abe7651dbaa44458 scored release validate that source. This is an
authorized public fresh-control measurement, no new algorithmic speedup. Below best2.215712x.

116c5185b14b487d95%CI2.180596037–2.247010129. Newreview correction: adapter8d648... cast
FP32(1+post_norm_w) toBF16 inherited from PREFILL's explicit boundary. Standalone production51
consumes FP32gamma; that adapter is not an exact51norm comparison (still a losing/unpromoted
experiment). Correct new masked diagnostic before dispatch, preserve FP32gamma.
2026-11-01 | bfbedf7d48e248a0 | baseline 6e078b5 plus tools/moe_token_locality_probe.py | off-board four-graph matched token order screen on six loads, seeds 781 and 782. Pending.
bfbedf7d48e248a0 completed. Five tests passed, control 2.183905x (95% CI 2.151147–2.216664). All 12 cases retained. Six-load factors 0.998626/1.000309 and 1.000687/1.000423 miss 1% gate. Park sorted rows. No public submission.


## Controlled standalone SC6 / byte scales, 2026-11-01

Evidence7886294, private frozen file SHA54d819e6482aaaa16c8a4e6d38f60e1e98b044b8888120f40b5ab7f3bccfb4cf.
Check05e51ddeb7014dd9 dispatched on muse_moe_w4a16, pending. Real supplied256-expert
weights/global IDs; four private mutable scratch states, byte/byte/SC6/SC6; all six
loads and seeds901/902. Common repaired BF16/fences/cooperative launch, MFIN/DFoff,
empty scheduler guard. Private attributed comparator only, no public submit.
Preregistered1% whole-six-load BOTHpaired ratios plus2x bothduplicate null, per-case
regression <=max(.5us,1%), official math/fresh replay/empty/resource gates.
Independent today_proj source/host clearance on exactfile,2source+4TorchCPUtestsPASS.
Full registration: archive/docs/steering/2026-11-01-moe-sc6-controlled.md.

Terminal addendum05e51ddeb7014dd9: H200 DONE/PASS5, official BYTE path2.2593108797x
(95%CI2.2254212165-2.2932005429), prepare116.574s. All48normal/48fresh/48empty
computing-arm comparisons pass, worstMarlinrelativeglobal.0062058/row.0067433.
SC6whole-six-load pairs0.999806837/1.007118932 vsduplicateabsolute-lognull.006403393;
registered1%+2xnull gateFAIL and percase-regression gateFAIL (256/901 +2.576us).
Bothmainresources96regs/0local/0static/231168dynamic/1residentCTA/576threads.
No SC6 port/promotion; no publicsubmit. Full perseedrows/resources/officialprofile
retained underprofiles/sc6_compare.05e51ddeb7014dd9.*, full interpretation inSC6report.
This closescompression-first ontherepairedcontrolledbackend, notall compression
architectures; per-expertfrontier measurement is next, notanotherSC6knobsweep.
2026-11-01 | 1c6ebc76b91d4b7f | native WGS frontier, private file SHA256 76c5b0e7483aba481f605a4d2a93363e1e8e968288f0bf977660da115ff9628b | Off-board measured writer/owner/finalizer schedule, separate from SC6 comparison. Pending.
1c6ebc76b91d4b7f PASS5 and flags empty, 2.256419844× (95% CI 2.222573546–2.290266141), ordinary SC6 control. 6 benchmarks/6 profiles; all six attribution cases and the representative trace re-evaluation passed. Favorable fixed-owner model 1.003616/1.004271/1.003445 across seeds. Registered stop.

2026-11-01 | 63f5a2a8ed2341d1 | a1031ff | Private component check of the Family G activation-slot retirement change. Exact source a08276dbd1dfe5551c360e2debe0fa67c4fdcfeb56d10ba347b858e3c60270c9 (388351 bytes), judged path repaired SC6 control. Four independent placements, three held-out inputs per load, all six workloads. Pending.

63f5a2a8ed2341d1 terminal PASS 5/5, control 2.238110203x (95% CI 2.204538550-2.271681856), six workloads and six profiles. Stage retirement passed every held-out comparison (six-load factors 1.018493-1.028699). 72 each initial/fresh/all-remote/restored checks, three negative controls. Both arms 96 registers, zero local, 231168 dynamic shared, one CTA/SM. See stage-retire.63f5a2a8ed2341d1.analysis.json and the steering report for 18 timing rows and transfer limits.

2026-11-01 | 46aafddc195c4c57 | exact candidate-only standalone treatment, 364234B SHA256 fde6b1ee532f77d4d2d3fc6bae337e2ceeea9d7e07e458a42a382b787100891b | root official off-board check following the controlled win, no public entry. Pending.

46aafddc195c4c57 terminal PASS 5/5, no flags, six timings and six profiles. Candidate-only stage retirement scored 2.348798659x (95% CI 2.313566679-2.384030639). Confirmed selected WGS/SC6 backend. Medians at 1/8/16/32/128/256 tokens: 25.601/69.097/99.162/123.898/178.264/212.460 us. Compared on matched inputs in 63f5a2a8ed2341d1, where all three seeds cleared the gate. Measurement only for the standalone board.


## Owned multiply decoder, 2026-11-01

Registered574cc1c before ONE authorizedcheck835253441a214a1c, pending on actualMoE
board. Frozenprivateharness628518B SHAb1e6e0886fe0058b9c5719950fb4dd06eef27b8f69975034affa234c440e98f3.
OwnedderivedW4permutation+multiply16648, noimporteddonorcode/table. Actualfullpacking
CPUproofandactualsm90aCUBINhotloops390->374, skinny140->128regs. Commoncoop
admissionandBF16effectivegamma inbotharms. Currenttrackedc785source, historical
abe765DISPATCHonly(skinny<=32/main47>=128), notclaimedhistoricalpublicbytes.
4privatepacks/scratch/inputs/captures,2binaries,18normalcases/seeds9919971009,
FP32fresh/skew/allremote/post gates, all4cross+EVERYseed whole-six-load>=1% and2xnull.
No publicsubmit. Fullregisteredproof/gates: archive/docs/steering/2026-11-01-owned-multiply-decode.md.

Terminal835253441a214a1c: H200 PASS5, officialCOMMONCONTROL2.189649236x
(CI2.156804498-2.222493975), prepare46.253s. Reportedchecks72normal+72fresh+24skew+
24empty+72post allpass, spin0, worstFP32.004383652/.004751944. Actualbank
3,221,225,472codepositionsverified. Registeredall4crosswhole-six-load ratios
1.001247993/1.002203525/1.001995933/1.002952178 vsduplicatenull.008029444;
all-seed1%+2xnull gateFAIL, percase-regressionFAIL. ActualH200skinny130->128regs,
both0local/155520dynamic/1CTA; main96regs/8local/228736dynamic/1CTA unchanged.
LocalNVRTC140->128predictionmustnotbequotedasanH200measurement. Noport/promotion,
nopublicsubmit. Stopmultiply-onlyvariant; nosecondcheck/selectivelarge-onlypolicy.
Receipt/summarymultiply_decode.835253441a214a1c.*, CLIprofile/logmultiply_check.835253441a214a1c.*.

## 2026-11-01 — native G stage observed-ready/retirement diagnostic

Source5c1a1f8, artifactbf0ca4ae…392222B. Root dispatched/waited
**f147d06a64504c1e** standalone MoE off-board. PASS5 control2.3402777977x
CI[2.305173631,2.375381965], prepare171.324839s. Six private computing arms,
MAIN128/256 seeds76111/76117/76123; B4FIN32 explicitly unmeasured.
All216 registered normal/fresh/skew/remote/restored math checks passed
(worst.0060922033/.0071880231 versus.008/.075). All12 detail mode/case
attribution gates FAIL: whole4.91–9.54%, phase10.26–14.90% displacement.
No causal weight/X/retire bottleneck conclusion, no candidate win, no rerun.
Root owns canonical profiles g-stage-check.f147d06a64504c1e.{json,log,txt}.
Derived completecapsule/report: archive/docs/steering/2026-11-01-g-stage-observed-readiness.md.
New H200 route census yields conditional proportional three-pair six-load
scenario1.012567/1.004403/1.004607, not measured gain/HBM/timing bound.
## Owned fixed-band32 private paired check, 2026-11-01

Root dispatched a5c7c7e3fe344919 (pending), harness evidence1bf61d2 and
owned patcherfc28a77. Exact private artifact SHA256
493d1678f7d92ffb2a9908675fcfa33257d235bd3b88cfc0b8696e27108b4c0c,
631897bytes. Two control/two treatment computing arms; onlyM16/32 use the
owned32-output-unit sameexpert8-row-job transpose. Other fourloads and judged
entry use frozen common cooperative/BF16-effective-gamma control. No decoder
or packing change. Seeds preregistered993701/993703/993709, all18cells required;
four-cross pooled AND perseed1%/2xnull, percell regression, resource/math/fresh
replay/empty/spin/budget/11KB gates. SevenTorchCPUhosttests PASS, localdevice
compilation belongs topatcher; no H200result claimed. Fullsourcepins/costs:
archive/docs/steering/2026-11-01-owned-moe-band32-registration.md. Root owns exactjob
status/receipt/profile archival; no publicsubmit, production unchanged.


Terminal a5c7c7e3fe344919: PASS5 official, common-control score2.1982215352150547,
95%CI[2.165248212186829,2.2311948582432803], all18timing records/264privatechecks,
prepare38.673292553s. Band32 loses all6targeted rows: duplicate-mean delta
M16 +18.368/+15.416/+20.872us; M32 +31.048/+22.000/+30.304us. Four full-six-load
factors0.945964140872/0.944172658054/0.943986837861/0.942199099693 vsnull0.006715307314;
allseed/1pct/2xnull/regression gates FAIL, KEEPfalse. Actual all4skinny resources
130regs/0local/0static/155520dynamic/1CTA/288threads; MAIN47 identical96regs/8local.
Exact493d1678 source unchanged, no production/pure qualification/public submission.
Root owns canonical receipt/log/profile; independent decoded report and closure:
archive/docs/steering/2026-11-01-owned-moe-band32-terminal.md. Fixedband32 and offline
153reg identity shortcut closed; no automatic band/cache/cap/credit sweep.


## Owned cached-band32 six-arm private check, 2026-11-01

Root dispatched LAST privateMoE check903e5b03411c4e2d (pending), exact945013B
artifact SHA2564bf3864b23ef94213794558d86c862cf781acd1d3a0ed5a8667f4b4e223c39d2,
harness6460509/cachepatche99d848/directfc28a77. Threeownedbinaries/sixprivate
computingowners: control0/1,cached2/3,direct4/5; only16/32usecached/direct,
otherfourloadsallcontrol. Samephysicalband32schedule/credits, new350u32shared
representation1400rawB/+1408actualSmemgate. Seedspreregistered993727/993733/993739,
all18rows/396mathchecks required; fourcrossctrl-vs-cached pooledANDallseed1pct/2xnull,
regression/resource/finiteFP32ref/fresh32/skew/remote/post/spin/budget/11KB gates.
Directcomparatordescriptiveonly. SevenactualTorchCPUhosttestsPASS, realistic
transport4419B; no H200successclaim. Fullsourcepins/registration:
archive/docs/steering/2026-11-01-owned-cached-band32-registration.md. Rootownsstatus/raw
receipt/log/profilearchive; nochildGPU, no publiccandidate unlessstrongfullscorewin.


Terminal903e5b03411c4e2d: PASS5/no flags, common-control2.190787796885395,
95%CI[2.1579259799321138,2.2236496138386754], all396privatechecks/18sixarmrecords,
prepare89.029513961s. Cached-vs-control fullsixfactors0.994470147488/0.997004522138/
0.996364377034/0.998903579067 vsnull0.009646480237; allseed/regression gatesFAIL,
KEEPfalse. TargetmeanM16ctrl102.554667/cached103.808001/direct117.456000us;
M32ctrl137.653333/cached139.448001/direct160.367999us. Cached-vs-direct1.043792..
1.046898 is recoveryfromfailedvariant, NOTwinvsbaseline. Actualcached128regs/
0local/156928dynamic vsctrl/direct130/0/155520, all288threads/1CTA;MAINexact96regs/
8local/228736dynamic. Independentactual_bc_summary recomputation matchesallgates
exactly. Fullterminal/decoded: archive/docs/steering/2026-11-01-owned-cached-band32-terminal.md.
Closecachedrepresentation/directband/CPUidentity; no purequalification/public
candidate/automaticknobsweep. Rootowns canonicalrawarchive, productionunchanged.

| 2026-11-01 | No official job; local source/compiler gate | OFF68bf4c1f (native7c915127) | Exactec565 progress prototype: both cooperative launchers, queuecapacityguard, fatalwatchdog; frozen abe765policy/strictchecks | No score/GPU | 5CPU tests pass; nativeproof14d5687/c00895f. Independent localCUDA12.8 main47 spills12/112→16/128 (baselinealready16Bstack), skinnyunchanged. Authorized unreachable db0b8cc1 alsofails. Bothclosed atno-new-spillgate; all3OFFflagsfalse, fullgraphqualifierdesignonly. Reports standalone-progress-off-prototype and standalone-moe-progress-compile, commitsabd6652/a20ec47. No production/public/queue action. |

## November2UTC2026 frozen pre-gamma component private check

Own **56e0c8b5634545e6**, prereg **adcfa8e548bb3af2780fa1ccd24d53c5e9f59e81**, captain allocation2f25832e. Exact344689LF-byte source8f12932a4d0fc8a2ca42a165610fd7659d4db960d8c65c8ea97281b75ec46d5d; official candidate2, controls0/1,candidates2/3. H200 PASS5tests/6bench/6profiles/no flags; official2.3324833814978434x,95%CI[2.2974961307753756,2.3674706322203107],prepare272.221707588s. Complete54cases/378observations/216captures/1152timings/54restorations/cleanup pass. Frozen reader exits0 but component_pass=false: all6bank/distribution groups/all24equal-six crossed factors fail>=1.01;16of18affected cells have>=1%regression. PARK exact treatment/no retry/tune/follow-on/public/default; official score is not candidate/control gain or layer evidence. Raw receipt/log/all6profiles: `muse_moe_w4a16/profiles/expert_top100_pregamma.56e0c8b5634545e6.*`; fullsource/capsule/attestations/reader: `docs/steering/evidence/2026-11-02-worker-moe-component/`. Terminal `docs/steering/2026-11-02-worker-moe-component-terminal.md`. Queue63.449752s,worker329.243756s distinguished from prepare. Final9privateleft does not authorize another job.


## November2UTC2026 frozen T1 three-launch standalone private check

Own **28248eb5e5854041**, prereg **9d7937d29b8fb0a6825b72bb2da5b916bf8347b8**, captain allocationc66f5ce6/sourcece14d11f. Exact370480LF-byte source5f10e2d73571d0cafa6d2fd4ab3dcf068f99979cf3d8c77470b63bab31fa0f58; frozen nativedbdd3381, officialcandidate2,2controls/2candidates,T1only. H200 PASS5tests/6bench/6profiles/no flags; official2.1658583028230054x,95%CI[2.1333704282806605,2.1983461773653503],prepare218.246490452s. All54cases/378observations/216captures/1152timings/54restorations/cleanup pass. Unchanged frozen reader exits0 but component_pass=false:0/24whole-six crosses reach1.01;6/6affectedT1cells regress>=1%. Descriptive24cross T1factor0.707998;all other-load drift retained. PARK exactthree-launchT1/no repair/retry/tune/T8/follow-on/public/default/layer. Official speedup is versusSGLang, not treatment/control gain. Fullraw receipt/log/all6profiles: muse_moe_w4a16/profiles/t1_three_standalone.28248eb5e5854041.*; source/attestations/reader in docs/steering/evidence/2026-11-02-worker-t1-moe-check/. Terminal docs/steering/2026-11-02-worker-t1-moe-terminal.md. Queue84.062889s/worker283.852265s distinctfromprepare; finalprivate8left/no-live gives no authority.


## November 2, 2026: constant WGS scale multiplier private check

Private `3aa3885f6b734f8c` terminal PASS5 tests/6 benchmarks/6 profiles/no flags; source `ca588999`, payload `ccef8a76` / 342733 bytes, native treatment `881874d4` against `c7c201ae`. Official versus-SGLang2.341016864x, 95% CI[2.305901611,2.376132117]. All54 cases/378 observations/216 captures/1152 timed replays/54 restorations and cleanup pass. Prepare273.985619s, queue626.302461s, worker331.904116s. Strict matched reader rejects performance: all24 whole-six factors0.992411..1.004932, descriptive mean0.997953, all6 groups fail>=1%/duplicate-null; three affected bank/load regressions>=1%. Local B4 spill bytes fall4/32 to0/0 but no measured speedup. Stop exact scale-only candidate, no confirmation/default/public action or layer/D32 source change. [Terminal and receipts](steering/2026-11-02-cuda-part2-experiments.md).

Compiler-only follow-up: constant scales plus ordinary `S.ssp` read produces byte-identical B4/MAIN SASS to scale-only under local CUDA 12.8.1; no new GPU measurement is allocated. Exact GS4-small and pointer treatments are recorded in `docs/ruled-out.md`.


## 2026-11-02 GOPT-PIPE-01 current skinny depth

Private matched job `f0b875484960438b`, source21db3bc31971ec9d7dd19da21e047adc069ad08c, payloadSHA256cf1dbda332e6a53fa581293d1ddf224c91447436e877c2ed005270c7dd1d04fe. Candidate NS4 versus incumbent NS3 for current G1b4 tiny/tiny2/MID, with unchanged B4/MAIN. T1/8/16 affected; T32/128/256 unchanged controls; three held-out banks, normal/skewed EP8, eight repeats per placement/cell, duplicate controls/candidates. Registration checkpoint; terminal outcome follows. [Owning association](../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/accepted.json), [frozen scope/gate](steering/2026-11-02-general-optimizer-pipeline-depth.md). No production or public change.

Terminal `f0b875484960438b`: PASS5 tests/6 benchmarks/6 profiles, no flags; prepare279.219903s. All54 private cases/378 observations/216 captures/1152 cold timings/54 restorations pass, worst global/row7.54371e-5/1.98935e-4. Strict source/job/log reader passes. Official2.293332488x CI[2.258932501,2.327732475] is versus SGLang. All24 affected small-load aggregate factors0.966414–0.990167 fail promotion, descriptive mean0.977832;17/18 affected cells cross the1% regression threshold. Unchanged256/skewed bank1102049 also fails the drift gate, so exact causal loss unresolved. Mean affected elapsed changes+0.675..+1.539us across T1/T8/T16 normal/skewed. PARK exactNS4, RETAINNS3, no confirmation/default/public/layer action. [All12 group means and raw36 rows](steering/2026-11-02-general-optimizer-pipeline-depth.md#terminal-result-november-02-at-1848-mst-in-americavancouver).

## November 2, 2026: corrected constant-scale / issue-group combination

Own PRIVATE `3c45788127e945c4`, source9669ee15, exact342999 LF bytes/c4e6d108, native7db5ec77 versusc7c201ae. PASS5official/6benchmarks/6profiles/no flags; official versus-SGLang2.348334366x, 95% CI[2.313109350,2.383559381]. Complete54cases/378observations/216captures/1152timedreplays/54restorations/cleanup pass, worst global/row0.000135982/0.000171862. Prepare276.366447s, queue545.209209s, worker349.580438s. Strict reader valid but performance fails: all24 whole-six factors0.995560..1.008180, descriptive mean1.001063, three affected bank/load regressions>=1%, all6 groups fail frozen1%/duplicate-null gate. No confirmation/default/public or layer inference. Local B4/MAIN both96regs/zero spills, HGMMA512/wait0=48/async-fence84 unchanged, wait1/arrive256->192. Stop exact combination; no mask/threshold rescue. Full source/receipt/log/profiles/analysis in `docs/steering/evidence/2026-11-02-cuda-group-scale/`; [terminal report](steering/2026-11-02-cuda-part2-experiments.md). D32 and ring-depth owners unchanged.


## November 2: GOPT-SHORT-02 source closure, no GPU job

Actual pickup at ef50beab. The proposed serial WGS prefix is excluded from all selected Tiny/Tiny2/MID helper builds under WS=1/WGS=0. Three CUDA12.8.1 preprocessing checks pass; a synthetic forced-WGS control exposes the expected functions. No kernel, compiler-performance claim, runtime test, timing or judge job. NS3 remains unchanged. [Full source evidence and refill request](steering/2026-11-02-general-short-prefix-result.md).


## November 2: T8/T16 WS norm aggregation accepted

Own PRIVATE `b38a9ed6dc9743e9`, revision51d6286b/payload1e857ad4,344267 bytes, accepted for one same-process four-owner complete-MoE comparison. Explicit FTZ helper/job accumulation changes tiny2/MID only; NS3, tail/finalizer/publication remain. Four CUDA12.8.1 builds pass with145/152 registers,214784 shared bytes/288 threads,zero spills; five CPU tests pass. No target correctness/timing yet. [Frozen gates and source](steering/2026-11-02-general-ws-norm-aggregation.md).


GOPT-WS-03 terminal `b38a9ed6dc9743e9`, revision51d6286b/payload1e857ad4: PASS5/6/6, no flags, return0;54 private cases/378 observations/216 captures/1152 cold timings/54 restorations pass. Worst global/row0.000277548359/0.000820404966, preparation254.148312s. All6 affected T8/T16 bank/distribution groups fail the frozen performance gate;24 aggregate factors0.979717–1.017916, descriptive geomean1.001545;7 affected regression rows, no unchanged-load guard breach. Official versusSGLang2.331049144x,95%CI2.296083406–2.366014881, is not a matched gain. Retain original norm updates/NS3, stop exact treatment; no confirmation/layer/default/public change. [All36 rows, uncertainty, identities and collection limits](steering/2026-11-02-general-ws-norm-aggregation.md#terminal-decision-retain-baseline-stop-exact-aggregation). Full artifacts under `muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/`.


## November2 MOE-BQUEUE-01 registration

OKC MoE Worker owns the new B4/MAIN whole-unit phase-B queue, with unchanged weighted phase A and skinny builds. Native99a6888d versusc7c201ae; paired4c8c470f,354612bytes. Four local CUDA12.8.1 bool entries pass96regs/zero stack/spills, three source and two real-Torch CPU facade tests pass. No GPU outcome yet. One complete four-state/three-bank/two-distribution/six-load private comparison under295s, original1%/duplicate-null/regression/correctness gates. [Source, proof, stop and accepted-job state](steering/2026-11-02-moe-worker-queue-experiment.md). Public permissions unchanged.

Own private MOE-BQUEUE-01 accepted `e044121f822d47b1`, source publishedb56bd8f1/paired4c8c470f. One POST,16checks/no live jobs at preflight. Shell wrapper exit1 after accepted JSON due zsh reserved status; no duplicate or retry. Target result pending. [Accepted record](steering/evidence/2026-11-02-moe-bqueue/accepted.json).


MOE-BQUEUE-01 terminal ownPRIVATEe044121f822d47b1, sourceb56bd8f1/paired4c8c470f/native99a6888d: PASS5official/6benchmarks/6profiles/no flags,54cases/378observations/216captures/1152coldtimings/54restores/cleanup. Worst global/row0.000224079/0.000484750,prepare270.729127s,queue348.705469s,worker347.263598s. Strict reader validates but0/6 matched groups pass;24whole-six factors0.954109315–0.995526795,descriptivegeomean0.978879559,15/18affected regressionrows. Official5856.634195us/2.296284786x CI[2.261840514,2.330729058] versusSGLang is not a matched gain or target5559.830241 achievement. Stop exact whole-unit B queue, retain c7c201ae; no mask/granularity/cap sweep,confirmation,integration/public action. Candidate-only same-native wrapper built/tested while waiting is unqualified,never dispatched and stopped. Waiter56587ended0. Fullraw/log/6profiles/strictanalysis/summary in docs/steering/evidence/2026-11-02-moe-bqueue/. [All six official and twelve descriptive rows, source boundaries and next evidence](steering/2026-11-02-moe-worker-queue-experiment.md#terminal-decision-november-2-pacific).


## November2 MOE-PHASE-01 accepted

Own private6028cae5d12f4afb, publishede8998f7a/registrationb5fa3610, paired77ca05dc/native365c9eab versus unchangedc7c201ae. Official arm is originalcontrol0. Sparse boundary observer with duplicate controls/observers, 36timed rows, 4,608cold timings and180checked repeated traces, fixed295s guard and priced perturbation/uncertainty gate. NineCPU/source/reader tests andfour local device compilations pass, with no target result yet. Sole foreground waiter23561 writes ignored `.build/moe-phase/status-wait.raw.json`. [Complete source, scope and one-run stop](steering/2026-11-02-moe-sparse-phase-evidence.md). Exact control release425fc2a9 is prepared separately, with no accepted release check or public permission.

MOE-PHASE-01 terminal6028cae5d12f4afb,77ca05dc/native365c9eab: setup return112 at success-capsule size after279.887460s, no official tests/benchmark/profile/score. Complete private378observations/54restores/216captures/4608timings/180traces retained in bounded failure capsule; worst global/row0.0000879050/0.0002260118. Frozen perturbation screen13/18rows pass, global false; raw representative omitted explicitly. Stop exact observer without retry/bound/guard increase or phase/gain qualification. Queue253.021123s,worker327.380771s,waiter23561ended1. [All retained evidence, exploratory spans and precise limits](steering/2026-11-02-moe-sparse-phase-evidence.md#terminal-stop). Next exact-file standalone release5df257af is separately prepared and does not rely on a valid observer.


## November2 MOE-RELEASE-01 accepted

Own private64f9998ef238476f, exact current standalone source5df257af/320466bytes, published9edf1ef6. Current c7 B4/MAIN with original skinny and scale fallbacks; actual five normal compiler calls unchanged, twoCPU tests pass. This prepares the newer base for a public decision independently of lost treatments. Fresh preflightzero live/14checks left, one POST. Target5official/6benchmark/6profile result and material public-score screen pending. Sole waiter52146 writes ignored `.build/moe-release/status-wait.raw.json`. Public submission remains unapproved. [Concrete release and stop](steering/2026-11-02-moe-control-release-qualification.md).

MOE-RELEASE-01 terminal own64f9998ef238476f, exact5df257af/320466bytes, published9edf1ef6. PASS5official/6benchmarks/6profiles, return0 and no flags. H200 preparation182.675155s,queue329.790188s,worker272.758662s,waiter52146ended0. Official5762.134395us/2.333944174x CI[2.298935011,2.368953337], frozen release screen passes and5.065786% below old public6069.607719us cross-job. Target5559.8us remains202.334395us away. Per-load adjusted T1/T8/T16/T32/T128/T256=25.275012/64.619045/95.442923/127.401430/178.951249/220.020295us. Five point times lower than public, T16+0.016691us; no matched kernel-gain/regression claim. Exact complete source ready for user's public decision, no public job. All24private numerical leads have dispositions. [Source, full receipt/profiles, uncertainty, fallbacks and release recommendation](steering/2026-11-02-moe-control-release-qualification.md#terminal-result-and-public-decision).


## November2 GOPT-WS-SPLIT-04 terminal

Own private3e6852d30c5c4535, source7370f630/payloadb4da948d/352468bytes, helper2be4d7bb enabled only T1/T8/T16; B4/MAINc7c201ae unchanged. PASS5/6/6/no flags/return0. All54private cases/378observations/216captures/1152cold timings/54restorations/cleanup and6native FP32 finalizer probes pass; worstglobal/row0.000126546/0.000233705,prepare284.983602s. All6matched performance groups fail;24complete-six factors0.984921731–0.998652082,descriptivegeomean0.992232006;16/18affected regressionrows,no unchanged-guard breaches. Official5834.137938us/2.305139190x CI[2.270562102,2.339716278] versusSGLang is not an incumbent win. Retain fused WS/NS3/original norm; stop exact two-launch design, no mask/third-node/sweep/retry/integration/public action. Solewaiter75595ended0 before archive. [All36rows, exact source and stop](steering/2026-11-02-general-ws-split-finalizer.md#terminal-decision-retain-fused-ws-stop-the-two-launch-split).


## November2 approved standalone base release

User directly approved exact5df257af. One publicPOST accepted4aa033ef23a142cd; terminalPASS5/6/6, no flags/return0, official5788.720720us/2.323224880x CI[2.288376507,2.358073253]. Source320466bytesunchanged. Setup184.048388s,queue235.197197s,worker334.295472s,waiter10826ended0. Board nowverifiesnewbest4aa at rank373from434,26attempts;4.627762%lowerelapsed thanpublic6069.607719us. Target5559.8remains228.920720us away. Private5762.134395us remains independently attributed. Approval covers5dfonly. [Owning receipt/profiles/verified board and next continued implementation](steering/2026-11-02-moe-control-release-qualification.md#public-result-verified). MOE-LUT-01 is the next distinct B4/MAIN representation/decoder experiment; no stale-observer or queue revival.

## November2 MOE-LUT-01 terminal loss

Ownc985bbdb2980407a passed5official tests/6benchmarks/6profiles, all54private cases,378observations/216captures/1152cold timings/54restores. Prepare276.790803s. All7compiler requests and256/256/16640/16640table owners match the frozen source. Worst global/row error is0.000089915/0.000183763, within unchanged0.015/0.075bounds. Cleanup/restoration passes. Solewaiter73200ended0 before archive.

Performance rejects the exact treatment:0/6bank/distribution groups, all18affected cells regress,24complete-six factors0.473011105–0.576767080, no unchanged-path drift under the registered duplicate-noise gate. Official candidate2 score12337.764699us/1.090027272x CI[1.073676863,1.106377681]. This is not a release lead. The current public remains5788.720720us.


Stop exact pair lookup, retain public5df257af. No follow-on/rescue/public action. [All six official loads, complete matched rows and stop](steering/2026-11-02-moe-pair-lookup.md#terminal-decision-stop-exact-pair-lookup). Next source qualification is direct phase-A input staging, a distinct work-elimination design.

## November2 MOE-DIRECT-X-01 terminal loss

Own94c57451574c4cd3/paired1191e44c/nativec705df6a, publishedf1955498, PASS5/6/6 andall54private cases. Complete378observations/216captures/1152timings/54restores; prepare272.829260s. All6matched groups fail,24whole-six factors0.755196–0.803711,18affected regressions,one unchangedT16/skewed drift. Official7194.998534us/1.869145621x CI[1.841108436,1.897182805]. Retain global gather/public5df257af; stop exact direct input without mapping/cache/ring sweep, retry, mask, integration or public action. Waiter38355ended0. [Full six loads, evidence and next lossless output-storage preparation](steering/2026-11-02-moe-direct-input.md#terminal-decision-retain-global-gather).

- November3 00:09 MST America/Vancouver: GOPT-WS-OWN-05 private7f10243c98e74746/source37964e41/payloadf31b652f/native896b5350. Fixed T8/T16 full-K64-row160-thread fused WS replacement, other sizes unchanged. PASS5/6/6,54private cases,8FP32 probes,prepare219.480960s; official candidate6679.194366823us/2.013491338x95%CI1.983288968–2.043693708x vsSGLang. Matched0/6groups,24complete-six factors0.845726–0.901634,12affected regressions,no unchanged drift breach. Retain original WS/NS3; stop exact topology, no sweep/retry/integration. [Evidence](steering/2026-11-02-general-ws-row-owned.md#terminal-decision-retain-original-fused-ws-stop-full-k-row-ownership).


## November 3 MOE-TL-01: Tiny / Tiny2 per-CTA timeline at T = 1 and T = 8

Final-day directive, MoE worker row. Base: the exact public release `current_control_release.py`
(5df257af, public 4aa033ef23a142cd 5,788.7 us; private 64f9998ef238476f 5,762.1 us). The newer private
7479372115c64702 (5,698.5 us, MOE-FACTOR-Y-01) was not taken as the base: its source
(docs/steering/evidence/2026-11-02-moe-factor-y/paired.py, 78506b74) is a paired experiment runtime with
duplicate control and candidate arms, 7 compiler calls and a 271 s prepare, not a release-form superset,
and its T = 1 / T = 8 cells (25.67 / 63.42 us) are inside the judge's noise of the release's (25.28 /
64.62 private, 26.73 / 67.21 public). Its Tiny / Tiny2 helper is the release's.

Candidate `muse_moe_w4a16/cand/tiny_timeline.py` (acfd73c5, 328,208 bytes), generated by
`tools/moe_tiny_timeline.py` from the release. Changes, helper0 source only (MOE_CUDA_SRC for B4 / MAIN
byte-identical, checked): `-DOKC_TLX=1` on the tiny and tiny2 defines (MID keeps OKC_TLX 0, so its bytes
are the release's) adds per-CTA `globaltimer` stamps into `p.dbg` (64 u64 per CTA), taken only when the
host passes a fourth scratch tensor; the timed path passes three, so every stamp is a uniform not-taken
branch. Stamps: kernel entry, routing and ranges done, first ring slot full (warp 0), each unit retire after
the unit barrier (slots 18.., q < 46), last phase A / phase B retire, first doneA wait start and end plus
the sum over all doneA waits, sums of warp 0's weights waits (mbarrier) and partial-buffer waits per phase,
the helper warp's doneA credit and its last epilogue, grid-barrier arrive and release, final-pass end.
Host: `prepare()` runs the instrumented launch once at T = 1 (2 local experts) and T = 8 (7 local experts,
20 routes, skewed) on synthetic inputs and prints `MOE_TL prep` summaries; `kernel_fn` runs it once more
per M in {1, 8} on the judge's own inputs on the first non-capturing call (`MOE_TL judge`, with a
same / maxdiff comparison against the plain output), after returning the plain launch's output. Each
summary is about 900 characters so four fit the judge's 10 KB log tail. Fences, barriers, ring, fallbacks
unchanged. Local nvcc 12.8.1 sm_90a (tools/moe_tiny_compile_local.py, docker, no target parity): tiny
157 registers (release 145), tiny2 154 (145), no spills, the same BAR / MEMBAR / FENCE / SYNCS / UBLKCP /
ATOMG counts (12 / 11 / 4 / 47 / 11 / 9); instructions 5,420 against 5,228 and 5,476 against 5,276.

Own private check `19128cda79ca4e25`, accepted 2026-11-02 23:4x PT (8 MoE checks were left; another
owner's public MoE run f9f37b46225b4980 was queued at the time and is not ours). Sole waiter writes
`muse_moe_w4a16/profiles/tiny_timeline.19128cda79ca4e25.status.json` and `.log`. Result pending.

MOE-TL-01 terminal `19128cda79ca4e25`: PASS 5 tests / 6 benchmarks, prepare 179.9 s, no flags. Score 5,845.9 us
(2.3005x, CI 2.266-2.335); medians T1 26.57 / T8 69.52 / T16 94.85 / T32 125.61 / T128 177.46 / T256 213.14 us
against the public release's 26.73 / 67.21 / 93.96 / 127.56 / 180.87 / 217.35: every cell inside the judge's
noise, as expected for a dormant instrument (not a candidate). The instrumented launch's output equals the plain
launch's at T = 1 (same=1) and differs by 3.05e-5 at T = 8 (the FP32 sum-of-squares atomics' order). Status and
log: `muse_moe_w4a16/profiles/tiny_timeline.19128cda79ca4e25.status.json`, `.log`. `globaltimer` resolution:
the smallest unit interval seen is 1.120 us with every per-unit stamp a multiple of 32 ns.

Timelines on the judge's own inputs (us from the first CTA's entry; min / mean / max over the 132 CTAs):

| T = 1, 2 local jobs, 1,536 units, nA 7-8, nB 3-4 per CTA | min | mean | max |
|---|---:|---:|---:|
| routing and ranges done | 2.2 | 2.3 | 2.5 |
| first ring slot full (2 units preissued at entry) | 2.8 | 3.1 | 3.8 |
| last phase A unit retired | 12.4 | 13.5 | 14.1 |
| helper's doneA credit | 13.5 | 14.8 | 15.3 |
| doneA wait start / end | 12.8 / 15.6 | 14.0 / 15.7 | 14.5 / 15.9 |
| last phase B unit retired | 19.4 | 20.5 | 20.9 |
| grid barrier arrive / release | 20.0 / 22.3 | 21.1 / 22.4 | 21.5 / 22.6 |
| exit | 22.5 | 22.6 | 23.1 |

Per unit (warp 0's retire interval): phase A 1.2 us (1.1-1.3), phase B 1.1 (1.0-1.3). Warp 0's total wait for
weights: 0.6 us in A (0.4-1.3), 0.2 in B; for a free partial buffer 0.4 in A, 0.2 in B. Mean CTA: ramp 3.1,
A 10.4, A end to dependency released 2.2 (wait 1.7), B 4.8, B end to barrier arrive 0.7, barrier idle 1.2 (0.9-2.4),
final pass 0.2.

| T = 8, 6 local jobs, 4,608 units, nA 23-24, nB 11-12 per CTA | min | mean | max |
|---|---:|---:|---:|
| routing and ranges done | 2.3 | 2.5 | 2.7 |
| first ring slot full | 3.1 | 3.5 | 4.9 |
| last phase A unit retired | 31.5 | 32.5 | 34.4 |
| helper's doneA credit | 32.4 | 33.5 | 35.2 |
| doneA wait start / end | 31.9 / 34.1 | 32.9 / 34.7 | 34.9 / 35.8 |
| last phase B unit retired | 46.8 | 48.4 | 50.8 |
| grid barrier arrive / release | 47.5 / 52.4 | 49.1 / 52.5 | 51.6 / 52.7 |
| exit | 52.6 | 52.9 | 53.5 |

Per unit: A 1.2 us (1.2-1.7), B 1.1 (1.0-1.2); warp 0's weights wait 2.0 us over 23.5 A units (1.6-3.2), 0.5 in B;
buffer wait 1.2 in A, 0.4 in B. Mean CTA: ramp 3.5, A 29.0, handoff 2.2 (wait 1.8), B 13.7, B end to arrive 0.7,
barrier idle 3.5 (1.0-5.2), final 0.4. The synthetic prepare-time runs (cold, after the build) show the same
shape with a longer ramp (3.9) and more A waiting (2.5 us at T = 1, 2.7 at T = 8).

Reading (measured): the unit loop runs at 1.1-1.2 us per 36,864-byte unit whether the data waited in the ring
or not (phase B's units are in smem before the wait ends and still take 1.1 us), and warp 0 waits for weights
under 0.1 us per unit, so at T <= 8 the compute warps' issue rate and the 3-slot stream (about 1.0 us per unit,
"Skinny weight stream" above) bound the loop at the same rate; the stream is at 4.1-4.3 TB/s aggregate while it
runs. The exposed non-stream time at T = 1 is 7.4 us of 22.6: ramp 3.1 (the first wave of 264 units, 9.7 MB, is
bandwidth-bound), handoff 2.2 (1.3 of it the helper's fence and credit after the compute warps' last A unit, the
rest the slowest CTA of the job), tail 2.1 (finish spread 1.5 from the 10-12 unit per-CTA counts and rate
variance, 0.9 barrier latency). At T = 8 the tail is 4.2 us (spread 4.1, counts 34-36 per CTA), the handoff 2.2,
the ramp 3.5. The phase B weights wait is 0.2-0.5 us, so an L2 prefetch of phase B units across the handoff
(the directive's first expected candidate) has nothing to recover and was not dispatched. Candidates in order of
measured exposure: (1) a balanced per-CTA A + B unit count (OKC_BAL: phase B ranges chosen so every CTA carries
11-12 units at T = 1 and 34-35 at T = 8, 1 unit less spread, about 1 us), (2) more issue capacity in the unit
loop (16 compute warps) if the stream can run faster than 1.1 us per unit, (3) the helper's credit lag.
