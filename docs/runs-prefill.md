# Prefill runs

All times and scores come from the named official job, not extrapolated decode results.
The October31 user goal explicitly includes this board and permits scored submissions.

| Date (PDT) | Job | Kind | Source/change | Score (95%CI) | Notes |
|---|---|---|---|---|---|
|2026-10-31|09ba15bd18fd4000|check|SGLang baseline copied with problem header, baseline_profile/baseline.py at e0ce491|1.000953x (0.985939–1.015967)|Passed; first SGLang hotspot map. Raw receipt and all downloaded profiles retained under baseline_profile/|
|2026-10-31|e6c12149c99a4a96|check|cand/marlin_m16.py; fixed expert M16|pending|NVFP4 weights/BF16 boundaries unchanged; all other baseline calls unchanged|
|2026-10-31|334149df33f64c54|check|cand/marlin_m32.py; fixed expert M32|pending|Same hypothesis, alternative tile granularity|
|2026-10-31|181880202c5742db|scored|Exact checked SGLang baseline plus scored-repeat comment|pending|Establish first board entry, not a claim of optimization|

## First hypothesis: MoE M-tile granularity

Baseline profile09ba15bd18fd4000 puts Marlin MoE at40–67% of profiled time, including
749.8us at turn1 sliding,2383.1us at turn4 sliding,11547us at first-turn sliding.
The baseline selects M64 at large shapes through the token-count heuristic. Fixed M16/M32
test whether reducing expert padding/altering occupancy is beneficial on actual routed work.
This is a hypothesis; no real-routing padding count or occupancy measurement is available yet.
Both candidates keep the same alignment, GEMM1, SwiGLU, GEMM2, per-expert norm and weighted
sum calls, and all activation/weight precision boundaries. No expert is skipped or cached.

Acceptance: all five official tests and scored-shape checks pass, then paired same-input
comparison against M64/current baseline on held-out routing inputs confirms weighted
three-load score gain beyond null variation. Cross-run scores alone cannot resolve small
differences. Stop fixed-tile promotion if no gain survives confirmation; record losses in
docs/ruled-out.md. A heterogeneous per-shape policy needs its own measured justification.

## Results, October31 PDT

Both candidates pass5tests but lose decisively: M16 e6c12149c99a4a96 scores0.715181x
(0.704453–0.725909); M32 334149df33f64c54 scores0.884745x (0.871474–0.898016).
No promotion or selective policy is justified. Keep M64/control. Candidate files remain
historical reproduction artifacts, not active submission candidates; original starter/default
was never replaced. All raw receipts and normalized profiles retained in profiles/ and
cand/profiles/. These losses close smaller fixed-M tiling, not expert-kernel redesigns.

First scored baseline181880202c5742db passed5tests at0.999115x (0.984128–1.014102).
This establishes an entry, not a speedup. Receipt profiles/scored.181880202c5742db.json.

Next investigation should target the underlying expert GEMM, which dominates the profile,
rather than repeat the tile sweep. Pinned SGLang wrapper inspected at
`a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/moe/moe_wna16_marlin.py`:
public Python signature has no thread-N/thread-K tuning knobs. It allocates FP32 c_tmp when
use_fp32_reduce and not use_atomic_add; do not toggle atomic output as a speed shortcut without
proving precision parity. No such toggle is built or authorized by this evidence.

## Capability check before a structural expert-backend experiment

Checke30ec1bf8e8142df dispatched from the unchanged validated baseline plus prepare-time
imports/version reports for Humming/FlashInfer/TileLang/CUTLASS and Humming grouped-layer API.
Source preserved in experiments/capabilities.py. No timing-path, arithmetic, precision or
policy changes. Purpose: determine installed judge capabilities before implementing a new
expert backend; do not assume a library from current online docs exists in the pinned image.

Primary [Humming source](https://github.com/vllm-project/humming/tree/948e6fd547e703065be51529ac1db5569256774e)
documents BF16 grouped NVFP4 with E4M3 group16 scales and per-expert FP32 global scales.
The required path is load_from_tensors on exact packed weights then transform inprepare,
not load_from_unquantized/requantization. Set use_f16_accum=False; retain BF16 GEMM1,
SiLU output, GEMM2, post-expert norm and weighted-sum boundaries. Expert layout is a device
int64 prefix array. Any implementation needs fixed-capacity route buffers and device-only
metadata generation; test helpers that use .item() to size outputs are not timed-call code.
No backend candidate has been built or measured yet. Capability check still running.

Capability result e30ec1bf8e8142df:5tests pass, unchanged baseline0.999768x
(0.984771–1.014764). Installed Humming0.1.12,FlashInfer0.7.0.post1,TileLang0.1.12,
CUTLASS4.8.0; HummingLayer/HummingLayerMethod import succeeds. This establishes availability,
not backend performance. Raw receipt/profile records retained in profiles/.

Compatibility correction: Hummingv0.1.12 at636ba85648c30ae2c2bfb335c9399593a67ecc1d
grouped_contiguous schedules its last expert from shape_m, ignoring actual prefixend.
Fixed capacityT*8 plus shorter activeprefix would therefore compute the unused tail as the
last expert: do not build that inefficient mapping. Owner is preparing INDEXED metadata with
only localexpert blocks, device num_tokens_padded, sentinel routeIDR=T*8 and appropriately
padded sorted-ID capacity. Loader/GEMM writer bounds and both GEMMtop_k mappings must be
verified before dispatch. No .item()/host count-based allocation is permitted in timedcalls.

Board correction: the public snapshot at22:27PDT returns no yelgabo row for prefill despite
the passed baseline scored0.999115x. The earlier 'establish an entry' language states intent;
it did not establish a ranked entry. Actual prefill speedup is still required.

First exact-Humming indexed candidate checkbe3cbc7f5a234c64 dispatched after root and
independent read-only source review. Frozen generator output SHA256
e53d374f6cf4239705cd3d20998ca3364ef0477dadd49031bafda0bff7324319 at MoE owner's
.build/today-moe/prefill_humming.py; helpertools/prefill_humming_indexed.py in thatworktree.
No .item()/host countallocation; metadata and outputs reset eachcall, everyshape's scratch
retained, failclosed on firstallocationduringcapture. G1topk8 andG2topk1 sentinel bounds,
deviceNP scheduler, BF16boundaries andexactweightload reviewed. This is only firstofficial
correctness/performance gate; freshinput queued zero-local/one-expert stress precedes scores.

## October31 23:07 PDT coordinator results

| Job | Source / purpose | Result |
|---|---|---|
|0de84000097946c9|Exact Humming0.1.12 indexed expert stages after installed-scale ABI correction|5/5 pass;1.176448889x,95%CI1.158802–1.194096|
|ece54526780a4f28|Pure-v3 plus appended2000freshM32/64matchedcontrol stress|5/5 official pass; fresh metadata/nonfinite/zero0; original envelope passes; never submit stress suffix|
|92408f1b65c643c4|Scored pure-v3,cand/humming_indexed_v3.py,SHAf3024d1b...|5/5 pass;1.146465578x,95%CI1.129268595–1.163662562; publicrank629/job verified|
|2bd4f06042864771|Four-arm zero/unfiltered vs nozero/filtered; routed heuristic unchanged|Pending; no promotion. PureSHA561e41804f402d3e3cf931a70c416347763a13e90d98384e403533d0468a017b|

Nozero hypothesis removes c1/c2/c3 whole-output clearing and restores baseline filteredSiLU.
Pinned SGLang activation.cuh skips nonlocal row before load; finalnorm has original localID
mask. Humming indexed localroute bijection writes every consumed row. Controls/candidates
own separate scratch/locks; poison all output buffersNaN after warmup before computing check.
Two heldoutseeds at753/3012/16384,12random cold rounds,4arms. Independent source review found
no ordering/missing-write blocker. Requires paired improvement beyond duplicate nulls and
fresh/official candidate validation before scoring. Source tools/prefill_humming_nozero.py;
exact generated pure artifact in experiments/humming_nozero.py. No Mheuristic/StreamK change.

Further Humming heuristic/fresh-control failures are owned in docs/runs-moe.md with exact
prefill board receipts; do not misattribute them to standalone MoE because of receipt folder.

## October31 23:15 PDT: nozero paired result and deterministic validation dispatch

Nozero four-arm2bd4f06042864771 passed5official tests and all24computing-arm checks
(4arms x2seeds x3sizes), including NaN-poisoned outputs. Actual judgedpath is nozero,
not control:1.262738192x,95%CI1.243797119–1.281679265. Never score this bench file.
Purecandidate remains SHA561e41804f402d3e3cf931a70c416347763a13e90d98384e403533d0468a017b.

|M|Z-R pairedus,seed271/272|Z2-R pairedus,seed271/272|R2-R nullus,seed271/272|
|---|---|---|---|
|753|-75.808/-80.512|-76.800/-78.592|+0.672/+0.064|
|3012|-314.400/-309.920|-313.792/-313.536|-1.728/-2.208|
|16384|-1705.791/-1752.192|-1699.680/-1749.184|-3.200/+4.000|

All6seed/shape comparisons improve far beyond duplicated nulls. Whole-output clearing
removed exactlyM*262144bytes per call:197394432/789577728/4294967296bytes at theseM.
FilteredSiLU additionally skips nonlocal rows; do not claim clearing alone caused all timing.
Original weight layout, Hummingcfg, BF16stage boundaries, FP32norm and ordering retained.

Deterministic routed/noStreamK74eb65dad3fa4e54 passed5official tests and fresh32/753/3012
2000queued+64eager each: all10reported stats exactly0, actualconfigsFalse. Score1.092871039x
CI1.076477973–1.109264104: it costs time, so not adopted as a speedup. Next root check
36d2347642e14e37 isolates nozero on this deterministic base, with same4arm/twoheldout bench
then unchanged3-size freshgate; actual judgedpath bothnozero+noStreamK, noMpolicy.
PureSHAa0fe61e771647b790e476b86159b7b43ea846a748a21ac09dd6151d36f74d855,
experiments/humming_nozero_nostream.py. Pending; no public submission until actual gates.

36d2347642e14e37 completed:5official tests plus24arm mathchecks pass; nozero gains repeat
on noStreamKbase. 32/753/3012 fresh all10stats0. Check1.197430766x,CI1.179469304–1.217110281.
Pure exactprefix/SHA verified before scored1530ec79d50a4f69 dispatch, excluding diagnostic
suffix. It completed1.216560352x,CI1.198311946–1.234808757, before/around scope deferral;
public board selects thisjob at23:26. No post-deferral submission occurred.

Combined e9bd4269a6584df8 also completes5official/fresh checks:1.273593087x,
CI1.254489190–1.292696983. Pure SHAabce3925a052d3cdc1c482e305904064945a7b4ccd2891c1b195fbe2d496b68d
is archived in experiments/humming_nozero_nostream_bounded.py, NOT submitted/promoted.
User now defers all prefill work; retain these completed receipts only. No further integration,
optimization, dispatch or public submission until user resumes this problem.
