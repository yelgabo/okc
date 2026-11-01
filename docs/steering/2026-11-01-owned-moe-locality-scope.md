# Owned standalone MoE locality: distinct mapping, missing score-covered opportunity

2026-11-01. Primary-source locality ledger: root `56319678130ffbb60062c5a49f4d40f4b3646b12`.
Reviewer base `49b46d3`. No external kernel adopted, kernel implemented, GPU job
requested or dispatched. This scope is the owned standalone incumbent, not G.

## Determination

A bounded same-expert job-by-output-unit transpose is **not the same experiment**
as the old two-CTA multicast screen or token sorting, and the inspected histories
do not establish that this exact mapping was benchmarked. Nevertheless, current
retained routing evidence does not justify implementing it or spending a check.
The two smallest loads cannot duplicate an8-row job; all four retained standalone
16/32 count cases have no duplicated expert payload. The positive32-layer counts
belong to a different routing workload. Do not promote them into standalone gains.
This is a scoped insufficient-evidence gate, not a universal cache-order closure.

## Exact owned source and dispatch

Public receipt `abe7651dbaa44458`, board `muse_moe_w4a16`, H200, PASS5, no flags:
speedup **2.2157115620135492**, CI **[2.182475888583346,2.2489472354437523]**.
The owning source is `git show ec565a7:muse_moe_w4a16/submission.py`,235700 B,
SHA256 `b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14`.
Receipt content hash is `084911f39cdd77d1d9ea6130d42b01cfe3a928b3fff3f8ac375ada4b444984ea`.
Its actual profiles select `sk_8_3_0::moe_fused_kernel<1>` for1/8/16/32,
`moe_main_kernel<47>` for128/256. Recent private common-cooperative/BF16-gamma
controls are different sources; they do not replace this public source identity.

Owned `_SKINNY_BLOCK` begins at Python line2645 in that commit; literal SHA256
`e26165cb897c685feec1d2bfd1bcdeed26cf765643791d8d64b997249e91b98d`.
Within that literal, `issue_unit` starts line93, `compute_loop`117,
`helper_loop`353 and `moe_fused_kernel`478:

- Each local expert receives `ceil(count/8)` jobs, expert order, contiguous8-row
  chunks. Job metadata stores the expert, entry offset and final partial count.
- Each CTA owns a contiguous proportional slice of `njobs*512` phase-A units,
  then `njobs*256` phase-B units. Every path uses the same job-major map:
  `job=v>>9` / `v>>8`, rowblock `v&511` / `v&255`.
- A unit is36864 B, covering16 weight rows and the full4096-wide K. Therefore
  consecutive jobs of one expert repeat an identical unit after512 logical
  requests in A (18,874,368 B),256 in B (9,437,184 B). These are **index-space
  distances**, not measured cache stack distances or a GPU issue timeline.
- Consumers reload job-dependent activations when the job changes; helper
  credits/readiness are keyed by the original job. A new map must be identical
  in issuing, computing and helper paths and preserve each job's512 phase-A
  credits. Changing only the copy order is invalid.
- Existing weight copies use an evict-first L2 policy. Adjacent logical indices
  are distributed across simultaneously resident CTAs, so the full per-expert
  payload distance alone does not prove current DRAM misses. There are no
  retained L2/DRAM counters demonstrating recoverable repeated requests.

The large path is a different scheduler:64-row jobs, phase-A whole256-row units,
n-tile-count grouping and snake ownership; phase B uses dynamic claims in47.
Source `_CUDA_SRC` begins Python line83, literal SHA256
`bc5b27628ba23f4a2cf0bb339ec42f07ed0711aa8a367cf707022231f79e8ae4`.
`seg_at`, `wproducer`, `xproducer` and setup explicitly use this policy. An8-row
transpose argument does not apply to those loads without a new64-row census.

## What was actually tested

`53c07507b9fd4f8c` is an unchanged-production logical-count diagnostic, PASS5,
control2.201850463x, CI[2.168822706,2.234878220]. Its owning log and JSON are
`muse_moe_w4a16/profiles/weight_count.53c07507b9fd4f8c.{log,json}` in root.
It tested a CPU two-CTA pairing model, not a cache transpose or multicast kernel.

| M/routing | Seed311 duplicates | Seed312 duplicates | Pairable request savings311/312 |
|---|---:|---:|---:|
|16 standalone |0%|0%|0% /0%|
|32 standalone |0%|0%|0% /0%|
|16 layer |0%|11.1111%|0% /0%|
|32 layer |27.2727%|36.3636%|18.1818% /18.1818%|

The preregistered >=10% pairable saving in **all8** cases failed6/8 and parked
the multicast rewrite. Pair alignment explains16-layer312's zero pairing, but
cannot create duplicate payloads in the four standalone cases. These eight
samples are not a census of all official100-repetition inputs. In particular,
`reference.generate_input` has heavy-tailed popularity and busiest-rank rotation;
uniform-router assumptions would be wrong. Current logs do not contain the
official incumbent's per-repetition count arrays.

Other relevant attempts:

- `bfbedf7d48e248a0`: stable token sorting by smallest local expert, all six loads,
  seeds781/782, whole factors0.998626/1.000309 and1.000687/1.000423. It changes
  token order, not this job-by-unit index mapping; it loses even excluding sort.
- `0f478ae5371a4b75`, source9971904: physically interleaved neighbouring units in
  an older nonpersistent kernel,1.683x CI[1.658,1.708], matching configs within
  about1us of old layout. It is not a test of this persistent skinny schedule.
- Phase-A snake ownership and dynamic last-round claims were extensively tested
  on the large path (runs log sections around modes19/20 and62–65). Those target
  load balance/progress. They do not establish a skinny transpose outcome, and
  should not be repeated as a new locality experiment.
- `835253441a214a1c` is the owned multiply decoder trial, not locality: strict
  paired gate failed despite fewer actual instructions. Its current common
  control is not byte-identical to ec565a7. Decoder gains cannot be added to a
  speculative cache gain.

## Score scope, priorities and stopping

The official task uses a geometric mean over six loads, each containing one
standalone cell of weight60. Recomputing from the six `abe765...` medians gives
the exact2.215711562 score; no sum-of-absolute-microseconds ranking is used.

| M | Official adjusted median us | Task floor us | Score factor if this cell alone reached its floor |
|---|---:|---:|---:|
|1|27.000975|11.416508|1.154270764|
|8|67.387859|37.318929|1.103507817|
|16|95.426233|52.519458|1.104649729|
|32|136.746636|70.755396|1.116073631|
|128|192.869429|101.341047|1.113216584|
|256|240.838568|108.944316|1.141353680|

Those floors are unattainable aggregate sensitivity references, not the budget
of the transpose. Reaching both16/32 floors would multiply score by1.232870433;
the retained standalone count samples instead show **zero repeated payload**
for this specific mechanism. A hypothetical10% time cut only at32 would give
1.017715171, or1.035744169 at both16/32. No measured result supports either cut.
M1/8 have no cross-job payload to reuse for any legal top8 routing because each
expert can receive at most M tokens.128/256 are outside this8-row proposal.

Concrete shortlist:

1. **Deferred, distinct:** bounded same-expert8-row-job by output-unit transpose,
   retaining all native tile shapes, precision, fences, credits and launch
   admission. Before any code/check, require retained GPU-generated natural
   standalone inputs with repeated jobs at16/32 across preregistered seeds and
   a meaningful score-covered exposed/traffic opportunity. New report evidence
   alone does not satisfy this trigger. CPU mapping must cover every
   `(phase,job,rowblock)` exactly once, including partial/empty jobs, and agree
   across producer/consumer/helper before a controlled experiment is proposed.
2. **Do not reopen:** paired multicast on old eight-case screen, token sorting,
   old physical interleaved layout, or large-path snake/claim knobs under a new
   name. None acquires new supporting evidence from the locality article.

Stop now before implementation/quota request. If later counts justify reopening,
first distinguish logical request saving from actual L2-to-SM/DRAM traffic and
whole-kernel critical time; then demand matched independent controls, fresh
natural routes, exact BF16/FP32 boundaries, and >=1% whole-six-load benefit beyond
2x null without posthoc narrowing to the positive layer32 samples. No blanket
claim that all cache-aware schedules fail is warranted.
