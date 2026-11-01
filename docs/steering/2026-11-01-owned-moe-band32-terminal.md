# Owned band32 terminal: correct, repeatably slower, closed

2026-11-01. Independent review of exact job `a5c7c7e3fe344919`, board
`muse_moe_w4a16`, H200. Registration/harness `1bf61d2`, pending row `91a14df`,
patcher `fc28a77`; root owns dispatch and canonical raw archival. Exact artifact:
631897 B, SHA256
`493d1678f7d92ffb2a9908675fcfa33257d235bd3b88cfc0b8696e27108b4c0c`.

## Decision

**Reject the fixed band32 transpose.** All six targeted rows lose, by15.4-31.0us
using duplicate-arm mean differences. All four full six-load factors are below1,
all heldout-seed gates fail, and the regression gate fails. No pure candidate
qualification, public submission, production change or automatic follow-up check.
The existing owned public incumbent remains the submission recommendation for
this track; this diagnostic's control score is not a new candidate score.

The exact job passed5 official tests with no flag: ordinary common-repaired
control score **2.1982215352150547**,95%CI
**[2.165248212186829,2.2311948582432803]**. The diagnostic completed all18 timing
records and264 private normal/fresh/skew/empty/post checks in38.673292553s.
Worst relative global/row errors were0.0044045043/0.0046449970. This establishes
that this schedule lost after the prescribed correctness checks, not that a
logical-byte model predicted its actual memory traffic.

## Paired measured outcome

Values below are private cold graph microseconds, means of the two independently
owned computing arms on each side. The underlying four medians are retained in
the decoded evidence; no best-placement or seed selection is used.

|M|Heldout seed|Control us|Band32 us|Delta us|
|---|---:|---:|---:|---:|
|16|993701|91.424000|109.792002|+18.368002|
|16|993703|102.423999|117.839999|+15.416000|
|16|993709|98.808000|119.680002|+20.872002|
|32|993701|164.271999|195.319999|+31.048000|
|32|993703|131.320000|153.320000|+22.000000|
|32|993709|164.600000|194.904000|+30.304000|

Full-six-load cross factors:
`[0.945964140872,0.944172658054,0.943986837861,0.942199099693]`.
Duplicate absolute-log-derived null:0.006715307314. Per-seed four-factor ranges:

- 993701:0.940629519200-0.944010894372, null0.008200146782.
- 993703:0.947890287963-0.953671255931, null0.006937294569.
- 993709:0.938050229025-0.940197467900, null0.005011043208.

`seed_gate=false`, `regression=true`, `keep=false`. The other four workloads use
the frozen control extension in every arm; their placement noise is included
in the six-load factors. No gain is inferred from isolated unaffected rows.

Actual H200 resource vectors are identical across all four arms:

|Kernel|Registers|Local bytes|Static/shared dynamic bytes|Resident CTA/SM|Threads|maxThreadsPerBlock|
|---|---:|---:|---:|---:|---:|---:|
|skinny|130|0|0 /155520|1|288|288|
|MAIN47|96|8|0 /228736|1|576|576|

Local NVRTC/PTXAS had reported140 control versus130 treatment skinny registers;
that predicted difference did not appear on the actual H200 build. Do not claim
an occupancy/register gain from the local compiler. Local-size bytes above are
runtime attributes, not per-instruction spill-byte counts.

## Independent reconciliation

Read the owning canonical build log under proj
`.build/today-proj/profiles/owned_band32_check.a5c7c7e3fe344919.log` and exact
terminal status JSON. Verified owning job/board, exactly one `BAND32_REPORT`,
source manifest equality with the frozen artifact manifest,18 unique cells and
264 checks. Re-executed the actual host `_bt_summary` against all retained
records and verified every pooled factor, per-seed factor/null, regression and
KEEP value exactly after JSON tuple/list normalization.

Independent decoded report:
`muse_moe_w4a16/profiles/band32.a5c7c7e3fe344919.independent-decoded.json`.
Root archives the canonical receipt, normalized profile, full log and decode;
this independent file is not a different run or extra measurement.

The earlier fixed64-bank local population remains valid as logical distribution
evidence: repeated8-row jobs were common. It does not establish cache misses
or exposed savings. The source/cost audit identified added activation reloads,
division/metadata work and increased gridwide CTA/job publications. Any of those,
or already effective caching, could explain losses; this job did not collect
the counters/phase evidence to distinguish their contributions. Do not label
one as a measured bottleneck or erase the positive population census.

## CPU identity shortcut and stopping criteria

Also retain the offline closure of the `jobs==1` identity shortcut, source
302006 B, SHA256
`8e0161d58604855f56fbecb800f8ecb7a1a2b90878c8f3948474992f27f56819`.
Its actual sm90a skinny compilation used153 registers versus140 local control
and130 frozen band32,0 stack/spill,16 barriers; it failed the prospective
no-worse-register gate and was never dispatched. Proof/logs belong to MoE's
`docs/steering/evidence/2026-11-01-owned-band32/identity-fastpath-closed/`.
Do not turn a hypothetical no-op shortcut into an unmeasured GPU repair.

Close this exact fixed-band32 mapping/last-occurrence-credit implementation and
that CPU identity variant. No automatic band-size, cache-hint, register-cap,
credit or selective-seed sweep. A reopen requires a structurally distinct
mechanism with concrete cost/progress/resource proof, or actual repeated-weight
traffic and exposed-time evidence on the intended owned backend. A new census
alone is insufficient after this paired loss. No broader claim that every
cache-aware schedule is impossible follows.
