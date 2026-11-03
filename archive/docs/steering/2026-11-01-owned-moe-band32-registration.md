# Owned band32: private paired-check registration

2026-11-01. Evidence base `8de9194` (all64 preregistered input-only banks), MoE
patcher commit `fc28a77`. No H200 dispatch/public activity by this worker.
Root owns final review and any check authorization. CPU construction is not an
H200 result, release candidate or measured speedup.

## Scope and source

The corrected population assessment is in
[the locality report addendum](2026-11-01-owned-moe-locality-scope.md).
Root approved one **fixed32-output-unit band** feasibility candidate, not a
band-size sweep. A same-expert group of8-row jobs is walked as
`[band][physical job][32 consecutive output units]`, preserving every
`(phase,job,rowblock)` once. Ring sequence indices remain virtual contiguous
indices; actual weight/activation/output addresses use the mapped physical job.

Original owned source LF SHA256
`c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7`.
This is the current owned source used for prior controlled checks, **not** the
historical235700-byte public `ec565a7` source. Common cooperative launch and
BF16-effective-gamma repairs are applied to all four arms, so this comparison
does not claim pure algorithmic parity with the historical public score.

`../okc-today-moe/tools/moe_skinny_band.py::patch(source,cooperative=True)`
applies the common cooperative wrappers before the skinny transform. Do not
apply `cooperative_source` twice. Raw patcher SHA256
`01db747b7a9d2ba3d9f9a268dc4a42e46d4c46269101c8d0bf18cd7fce9ae694`;
cooperative helper SHA256
`50894dd3f4e522066c916bcd83f2e06dc068f6645e314b66b828fa4638514274`.
The generator independently asserts all original top-level AST outside
`_SKINNY_BLOCK` is unchanged by the raw band patch. Decode/packing are unchanged.

Final private artifact in proj worktree:
`.build/today-proj/owned_band32_check.py`, **631897 B**, SHA256
`493d1678f7d92ffb2a9908675fcfa33257d235bd3b88cfc0b8696e27108b4c0c`.
Native Windows and WSL builds produced the same exact bytes. Manifest is
archived beside this report's evidence. Generator:
`tools/moe_make_band_check.py`; host body:`tools/moe_band_check_body.py`.
Reproduce with original source, output path, and `--moe-tools` pointing at the
MoE sibling's tools from `fc28a77`.

## Activation/publication cost and independent source review

The original `compute_loop` keeps one job's activation tile in `xr`, loading it
again only when job/phase changes. At8 compute warps, `xr[8][2]` of `uint4`
requires64 scalar32-bit register values/thread; retaining a second full job
would add another64 before addresses/accumulators. A full8-row job requests
65536 activation bytes per CTA reload, partial jobs proportionally fewer.
Phase A uses `__ldg`; phase B explicitly uses `__ldcg`, bypassing L1. No measured
L1/L2 hit rate follows from these source intrinsics.

`tools/moe_owned_band_cost.py` executes complete old/new unit maps for the
fixed64-bank cohort at132CTAs, asserts multiset and unique ownership equality,
and counts real live-row activation-load requests. Complete records are in
`evidence/2026-11-01-owned-moe-locality/band32-cost.json`.

|M/phase|Mean old/new job runs, gridwide|Mean old/new activation request B|Mean extra request B|
|---|---:|---:|---:|
|16/A|139.4375 /153.1875|2,569,984 /3,254,528|684,544|
|16/B|139.4375 /145.21875|2,569,984 /2,856,704|286,720|
|32/A|146.5 /195.25|3,654,912 /6,240,768|2,585,856|
|32/B|146.5 /168.625|3,654,912 /4,817,920|1,163,008|

One band contains1,179,648 unique weight bytes per expert, requested again by
its repeated jobs; max jobs/expert in this sample is2 atM16 and4 atM32. These
are logical index/load budgets, not physical cache working sets, bandwidth or
elapsed cost. Other CTAs may access different experts/bands concurrently.

The naive transpose would multiply the old helper's job-change credits/fences.
The patch instead credits each physical job once per CTA, at its last mapped
virtual occurrence, with the exact local contribution count. Importantly this
does **not** imply unchanged gridwide credit count: the mapping changes which
CTAs touch a job. Mean phase-A CTA/job incidences rise139.4375→150.875 atM16,
146.5→184.9375 atM32. This extra publication cost is a real candidate risk.

Independent source audit confirms:

- `S.cnt` is written only by original histogram construction; `S.off` only by
  the original prefix. `position_pass` accepts `const int* off` and does not
  mutate either. Both remain valid for mapping throughout compute/helper.
- Initial issue before the CTA barrier follows warp0's existing `__syncwarp`;
  all relevant job/cnt/off stores are from that same warp. Subsequent accesses
  follow existing routing/position synchronization. No new shared storage.
- Original proxy fences/release credits remain. The last-occurrence predicate
  is uniform in the helper warp and occurs after all that CTA's prior stores
  for the physical job. Total credited count per job remains512. Each CTA
  credits all its phase-A contributions before its first phase-B buffer wait.
  Combined with unchanged all-grid cooperative admission, no new dependency
  cycle was identified. Passing tests alone was not used as ordering proof.

Owner's actual local sm90a compile reports skinny control140 vs treatment130
registers, no stack/spills, same Smem and16 barriers; math instruction counts
unchanged. Actual H200 resources are mandatory. A later identity-fastpath trial
was rejected offline at153 registers and the patch restored to `fc28a77`;
the harness manifest pins the restored source, not that experiment.

## Preregistered check and stopping gates

Before implementation, fixed heldouts **993701/993703/993709**, all six loads
1/8/16/32/128/256. These are disjoint from census65001–65032. All18 records are
mandatory; no seed/shape selection after observing timings.

- Four namespaces, two control/two treatment computing arms, independent packs,
  scratch, graph inputs/output. Duplicate arms share their compiled binary,
  explicitly disclosed in the manifest. All mutable scratch/input storage
  pointers are checked distinct; captures retain their input dictionaries.
- Only M16/32 call the treatment namespace/extension. All four computing arms
  use frozen control extensions at1/8/128/256; MAIN47 is unchanged. Ordinary
  judged `kernel_fn` always returns control, never a selected candidate.
- Common SM90 cooperative admission and effective BF16 gamma in the existing
  float ABI. All supplied packed tensors are byte-equal across four repacks;
  no changed decoder/packing proof is substituted for actual equality.
- Actual resource duplicates must match. Candidate registers/local bytes may
  improve; static/dynamic shared, threads and max-thread attributes must match;
  residency must be>=control>=1. MAIN47's entire actual vector must be equal.
  These revisions were explicitly approved by root.
- Actual FP32 reference with finite output/reference/metrics and strict global
  `.008`, row`.075`; fresh32-replay stress per case, skew/all-remote per load,
  output poisoning, restored normal inputs, post-timing correctness and spin
  flags.32 repeated replays are not32 independently generated inputs.
- Twenty cold rounds with crossed/rotated arm order and256MiB flush; all four
  pooled six-load cross factors AND all four factors for every heldout seed
  must meet>=1% and exceed2x duplicate null. Every case must avoid regression
  greater thanmax(.5us,1% control). Every timing must be finite/positive before
  log evaluation. No posthoc large/small-only promotion.
- Inner and final275s guards, earlier220s build-state guard, required18 rows,
  JSON `allow_nan=False`, fail-closed11KB report guard. Preparation/runtime,
  actual log entropy, CUDA host compilation and graph behavior remain H200
  risks. Failure is not permission to weaken gates or retry automatically.

Seven executed actual TorchCPU host-facade tests pass: complete18-case flow and
target dispatch/common gamma, fresh stale-input negative, nonfinite reference,
inner-budget/TF32 restoration, four-cross/per-seed/null/regression gates, bad
timing values before logs, and realistic full-manifest report transport. Test:
`tools/test_moe_band_host_cpu.py` via the host-validation venv. No native CUDA
host compilation or GPU math was simulated as measured success.

Next action: root and MoE owner independently review the frozen artifact before
any single off-board check. On source/resource/math/budget/score failure, stop;
no band-size, cache-hint or identity-fastpath sweep is authorized here.

## Dispatch addendum

Root dispatched the one authorized private MoE check **`a5c7c7e3fe344919`**
using the exact493d1678 artifact above, source/harness evidence `1bf61d2` and
patcher `fc28a77`. Status/receipt/profile archiving belongs to root. Pending:
no timing, correctness result, score or KEEP decision is asserted here. No
additional job is authorized by this worker; production remains unchanged.

Terminal addendum: `a5c7c7e3fe344919` passed official/private math but band32 lost
all six targeted rows and all four full-score factors; KEEP=false. See the
[independent terminal report](2026-11-01-owned-moe-band32-terminal.md).
No public/pure qualification or knob sweep follows. Production remains unchanged.

Source-attribution correction,2026-11-01: the selected skinny setup is enclosed
by `if(warp==kNW)` (helper warp8), not warp0 as written above. The same-warp
`__syncwarp` publication reasoning remains valid. Fat-main setup caused the
wording confusion; no device modification follows. The later cached-band32
registration uses the corrected identity and adds explicit bounded-cache proof.
