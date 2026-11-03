# Cached band32 terminal: faster than direct, no win over control

2026-11-01. Exact root-owned H200 job `903e5b03411c4e2d`, board
`muse_moe_w4a16`; registration/harness `6460509`, pending row `8a71483`, device
patch `e99d848`, frozen direct `fc28a77`. Artifact945013 B, SHA256
`4bf3864b23ef94213794558d86c862cf781acd1d3a0ed5a8667f4b4e223c39d2`.

## Final decision

**Do not promote or qualify a pure/public candidate.** Cached band32 remains
slower than control in all six targeted duplicate-mean timing rows and fails
the preregistered pooled, per-seed, null and regression gates. Close this exact
cached representation and the direct-band implementation. No automatic knob
sweep, additional check or production change. The owned incumbent remains
the recommendation for this track; no new public submission follows.

Official5 tests passed, no flags. The **ordinary common-repaired control**
scored2.190787796885395 with95%CI
[2.1579259799321138,2.2236496138386754]. This is not a cached candidate score.
All18 six-arm timing records and396 private checks completed in89.029513961s:
normal108, fresh108, skew36, remote/empty36, post108. Worst relative global/row
errors were0.00444909185/0.00473365234.

## Same-input measured comparisons

Private cold graph microseconds, means of both independent computing owners
per representation. All preregistered target rows are shown.

|M|Seed|Control us|Cached us|Direct us|
|---|---:|---:|---:|---:|
|16|993727|123.432001|124.384001|142.336000|
|16|993733|95.736001|96.664002|108.792001|
|16|993739|88.496000|90.376001|101.240000|
|32|993727|111.119999|112.456001|127.839997|
|32|993733|152.112000|152.688000|176.576000|
|32|993739|149.728000|153.200001|176.688001|

Across the three banks, target means are M16 control102.554667 -> cached103.808001
-> direct117.456000us; M32 control137.653333 -> cached139.448001 -> direct160.367999us.
No comparison to a different job's input bank is used for these differences.

Four pooled **full-six-load control-versus-cached** factors:
`[0.994470147488,0.997004522138,0.996364377034,0.998903579067]`.
Duplicate null0.009646480237; `seed_gate=false`, `regression=true`, `keep=false`.
Per-seed four-factor ranges, all below the1% requirement:

- 993727:0.992410907778-1.000708744937, null0.008667356612.
- 993733:0.993305935426-1.003440128794, null0.008873038307.
- 993739:0.992290944314-0.997172681174, null0.011401340553.

Same-bank descriptive comparisons:

- Control-versus-direct full-score factors0.952341504719-0.954561987951;
  direct duplicate null0.004422992092.
- Cached-versus-direct factors1.043792220527-1.046898110812.

Thus the cached representation package recovers much of the direct mapping's
loss. That is **recovery against a failed variant**, not a win against baseline
and not evidence of a locality speedup. Direct was deliberately excluded from
the promotion decision; the control-versus-cached gate still failed.

## Actual resources and scope of attribution

Actual duplicate vectors match, with unchanged MAIN47:

|Kernel/representation|Regs|Local B|Static/dynamic shared B|CTA/SM|Threads/maxThreads|
|---|---:|---:|---:|---:|---:|
|skinny control/direct|130|0|0 /155520|1|288 /288|
|skinny cached|128|0|0 /156928|1|288 /288|
|MAIN47, all6 arms|96|8|0 /228736|1|576 /576|

The exact+1408 shared and no-worse admission/register gates passed. Fewer
registers did not increase residency. All non-target loads use frozen control
extensions with private state, so whole-score calculations retain their noise
rather than selectively dropping it.

The same physical schedule/credits with cached lookup was faster than its
direct representation. It changes precomputation, metadata operations, shared
loads and compiler register allocation together; this does not isolate a single
instruction's cost or prove the source of every recovered microsecond. Neither
this check nor the earlier population census measured L2/DRAM traffic or exposed
weight-stall savings. Do not call the logical duplicate fraction a speedup bound.

## Independent audit and closure

Independently read the owning proj canonical build log and exact terminal status;
verified job/board/PASS5/no flags, exactly one `CACHED_BAND32_REPORT`, its full
source manifest equality with the frozen artifact,18 unique six-arm rows and396
checks. Re-executed actual host `_bc_summary` on retained records; after JSON
tuple/list normalization, every pooled/per-seed/null/direct-descriptive/regression
and KEEP field matched exactly.

Independent decoded evidence:
`muse_moe_w4a16/profiles/cached_band32.903e5b03411c4e2d.independent-decoded.json`.
Root owns canonical receipt/profile/log archival; this decode is not another
measurement. Old493d artifact remains unchanged and reproducible.

Close cached band32 as built, direct band32 from `a5c7c7e3fe344919`, and the
offline153-register identity shortcut already closed in its report. No band-size,
height, hint, credit, cache-capacity or register-cap sweep follows. A future
reopen requires a structurally different mechanism with exact lifetime/ownership,
representation/resource cost and whole-score evidence, or actual repeated-weight
traffic plus exposed-time evidence. Another positive count census alone is not
a reopening trigger after these paired losses. No claim that all cache-aware
schedulers are impossible is made.
