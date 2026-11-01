# H200 outcomes, retained submission and next checks

November1,06:30 PST. Evidence checkpoint **b2d94fe** includes the final independent
cached-schedule result. Root owns all four job IDs below and verified each against
its receipt. Prefill remains deferred. No public submission occurred in this round.
The top10–50 objective remains active and unmet.

## Assessment against the actual boards

The [final official snapshots](evidence/2026-11-01-final-locality-boards/identity.json)
were fetched after the last H200 result. Overall was queried directly: our rank
is308. Its API provides ranks only. The first500 rank/team pairs match the layer
snapshot, including the top10 and50 rows, but the full lists differ after500.
Thus the speedup cutoffs below are explicitly **problem-board cutoffs**, not an
invented numerical overall score or a universal overall-ranking formula.

| Board | Our rank | Our speedup | Rank50 | Rank10 | Required score increase to50 /10 |
|---|---:|---:|---:|---:|---:|
|Layer|308|2.1261×|2.2894×|2.4028×|7.68% /13.01%|
|Attention|220|2.6043×|2.7119×|2.9331×|4.13% /12.63%|
|MoE|387|2.2157×|2.4549×|2.6077×|10.80% /17.69%|
|Projection|311|1.5278×|1.5850×|1.6087×|3.74% /5.30%|

Gap percentages use rounded CLI speedups and are approximate score ratios, not
projected latency reductions. Deadline: November4,23:59 Pacific, as reported by
the official `okc problems` endpoint. There is **no demonstrated top10 portfolio**
in current measurements. The earlier stage-retirement improvement is already in
the incumbent and gets no future credit. Neither new owned MoE variant produced
a baseline win. Do not extrapolate historical blog results or compiler register
changes into a deadline forecast.

The best submission remains
[`muse_decode_layer/cand/g1b4_stage_release.py`](../../muse_decode_layer/cand/g1b4_stage_release.py),
owned public job **607eb52bdaf24160**, **2.126082676×**,95% CI
[2.094191436,2.157973917]. Its [owning receipt](../../muse_decode_layer/profiles/scored.607eb52bdaf24160.json)
and exact869076-byte SHA256 are retained. SHA256:
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Its public receipt passed13 official tests. Preserve it. FamilyG is the
repository's attributed-fork label; this work does not introduce a different
model or GPU architecture.

## What was implemented and measured

The new pasted *Accelerating MoE Inference with Locality-Aware Kernel Design*
report was read completely and [preserved with identity hashes](evidence/2026-11-01-locality-report/identity.json).
The [primary-source ledger](2026-11-01-moe-locality-primary-ledger.md) corrects
the archived2024 kernel's mapping bug, separates the already-corrected vLLM
integration from that bug, and distinguishes later BF16 grouped-GEMM results.
No historical speedup or the unlinked Twitter rank170→80 claim is used as an
H200 Muse measurement. All standalone prototype code here is owned code.

The first four retained standalone routing samples had no repeated8-row jobs.
A prespecified [64-bank local input-only census](2026-11-01-owned-moe-locality-scope.md#dated-addendum-preregistered-local-input-only-population-census)
found repeats in14/32 of the16-token banks and29/32 of the32-token banks. That
reopened a concrete experiment; it did not measure HBM traffic. RTX-to-H200
seed reproduction remains unverified. The conditional3.07% score scenario was
a hypothesis, never a forecast, and the subsequent H200 comparison did not
realize it.

Required SGLang profiling and the latest owned/public
[hotspot tables](evidence/2026-11-01-owned-moe-locality/hotspots.md) were inspected
before the prototypes. Both targeted loads spend their profiled kernel time in
the owned skinny implementation. Task floor gaps do not attribute that time to
cache misses.

| Owned H200 job | Correctness and score of official judged control | Measured decision |
|---|---|---|
|Attention3f44f891d0ef47db|PASS16;2.490692705×,CI[2.453332314,2.528053095];12 benchmarks/profiles|All12 stamped rows violate the registered timing-distortion gate. No causal wait attribution. Close exact observer.|
|G phasef147d06a64504c1e|PASS5;2.340277798×,CI[2.305173631,2.375381965];6 benchmarks/profiles;216 private checks|All12 detailed readiness/retire rows violate distortion gates. Real route counts retained, no bottleneck or speedup claim.|
|Direct band32a5c7c7e3fe344919|PASS5;2.198221535×,CI[2.165248212,2.231194858];264 private checks;18 timing rows|All6 targeted input rows lose. Four full-score factors0.942199–0.945964. No promotion.|
|Cached band32903e5b03411c4e2d|PASS5;2.190787797×,CI[2.157925980,2.223649614];396 private checks;18 six-arm rows|Cached/control factors0.994470–0.998904;null0.009646;per-seed/regression/aggregate gates fail. No promotion.|

These official scores belong to the controls selected by the diagnostic judged
entries. They are not candidate scores. All four receipts have clean policy
checks. Canonical raw receipts, logs and normalized profiles are committed under
the corresponding problem's `profiles/` directory, including losing runs.

The [direct prototype](../../tools/moe_skinny_band.py) transposes32-unit bands
across adjacent jobs of one expert. Its physical mapping is shared by issuer,
compute and helper; analytic final-occurrence credits retain the original fence,
warp-sync and atomic publication chain. Independent source/progress review and
CPU coverage precede the H200 test. It adds activation reloads, credit calls and
mapping arithmetic. Its mean targeted event latencies increased18.68% at16
tokens and18.11% at32. [Terminal evidence](2026-11-01-owned-moe-band32-terminal.md).

The [cached prototype](../../tools/moe_skinny_cached_band.py) keeps exactly that
physical order/credit sequence but precomputes a350-entry packed shared schedule.
The guarded M≤32/grid≥132 contract proves capacity. Two helper-warp syncs and
the existing CTA join publish it. The final comparison contains two originals,
two cached arms and two direct arms on new held-outs, with private state and
input storage. [Registration](2026-11-01-owned-cached-band32-registration.md).

| Target, averaged over3 inputs and duplicate arms | Original event µs | Cached µs | Direct band µs |
|---|---:|---:|---:|
|16 tokens|102.5547|103.8080|117.4560|
|32 tokens|137.6533|139.4480|160.3680|

Cached scheduling recovers much of the failed direct variant's cost. Its
1.04379–1.04690 full-score factor versus that failed variant is **not a baseline
win**. The experiment does not isolate DRAM/L2 transactions or assign all recovery
to division removal. The cached/control differences fail the preregistered
noise and regression gates. [Final independent recomputation and evidence](2026-11-01-owned-cached-band32-terminal.md).

Actual H200 skinny resources: original/direct130 registers,0local bytes,
155520dynamic shared bytes; cached128 registers,0local bytes,156928shared bytes.
All use288threads and a one-CTA/SM admission limit. Unchanged MAIN47 is96registers,
8local bytes,228736shared bytes,576threads and one-CTA/SM admission. Admission
attributes are not achieved occupancy, bandwidth or utilization measurements.
Cached preparation finished in89.0295seconds. All396 private numerical checks
passed; maximum global/row error0.0044491/0.0047337. Production files were never
changed by these prototypes.

The source-only identity shortcut is separately retained as a compiler-screen
closure:153 registers versus the frozen direct version's130 in that local
compiler. It was not sent to H200. Direct and cached implementations, observers
and source-only losers have explicit rows in `docs/ruled-out.md`.

## Ranked next actions and stopping criteria

| Priority | Concrete action | Evidence and acceptance |
|---|---|---|
|1|At16:00 PST, execute the repaired six-load G incumbent-versus-stage comparison already registered for the reset.|Measure matched natural routing and independent controls; preserve actual precision, ordering and native resources. Apply the frozen full-score/null/regression gates. A public-run difference alone is insufficient.|
|2|Execute the prepared short-path fixed-natural-routing trace.|Only low-distortion, ownership-correct rows can motivate another boundary/fusion change. The losing concurrent pooler/router branch stays closed.|
|3|Consider the850769-byte selected verify32/sliding cross-family comparison after the first checks.|The packaging obstacle is solved, selected native SASS is unchanged, and source ownership is reviewed. Real H200 linking, <300s preparation,24 oracle checks and naturally identical expert IDs remain open. One-cell timing cannot establish a six-load gain.|
|Deferred|G64-row duplicate-weight scheduling or a new owned locality approach.|G's actual census scenario remains about0.72%, not measured savings. Both owned32-band representations failed. Reopen with a materially different, named mechanism and matching-route physical/exposed opportunity; do not relabel token sorting, multicast or old layout tuning.|
|Deferred|Further BF16 projection or attention knob sweeps.|Existing closures remain in force. Require a distinct mechanism and measured score-covered opportunity; do not repeat C3, QK2, ring, weighting or stamp-placement experiments without new evidence.|

Stop this owned scheduling round now. No band-size/cache-hint/identity sweep,
pure-candidate qualification or public submission follows from these results.
Do not weaken the gates to preserve a microbenchmark result. New promotion
requires all official/private math and cache checks, valid cooperative admission,
all four candidate/control score factors≥1.01 and above2×duplicate noise, each
held-out seed passing, and the frozen per-cell regression limit.

After903e, free checks remaining today: layer0,projection0,MoE0,attention4.
Public slots remain unused by this round. The existing16:00 PST reset wakeup
must read this report and run its owned jobs sequentially, retaining exact IDs
and every receipt. The goal is not complete. The next work is already specified;
no additional permission is needed for those private checks.
