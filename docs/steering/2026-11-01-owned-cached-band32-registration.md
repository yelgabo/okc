# Cached band32 representation: six-arm private-check registration

2026-11-01. Reviewer base `6b55b46`; owned cached patch/report from MoE `e99d848`,
direct band patch `fc28a77`. No GPU dispatch or public activity by this worker.
This is one different representation of the same previously losing schedule,
not another band-size, height, cache-hint or precision experiment.

## Frozen artifact and hypothesis

New independent tools: `tools/moe_make_cached_band_check.py`,
`tools/moe_cached_band_check_body.py`, `tools/test_moe_cached_band_host_cpu.py`.
Old493d builder/body/artifact are unchanged and remain reproducible.

Final proj artifact `.build/today-proj/owned_cached_band32_check.py`:
**945013 B**, SHA256
`4bf3864b23ef94213794558d86c862cf781acd1d3a0ed5a8667f4b4e223c39d2`.
Native Windows and WSL generation produce identical bytes. Transparent three
named source literals plus host body fit the1MB constraint; no compression,
runtime source reads, imported external kernel or hidden binary payload.
Manifest is archived under
`docs/steering/evidence/2026-11-01-owned-moe-locality/cached-check-manifest.json`.

Cache patch API: `moe_skinny_cached_band.patch(source,cooperative=True)` calls
the frozen direct patch and common cooperative adapter exactly once. Original
owned LF source remains `c7854a71...`; control LF hash remains `f5ba8c860...`;
direct LF hash remains `c8ff319f...`. Full hashes are in the manifest.

The treatment precomputes physical unit indices and final-occurrence credit
counts into350 packed shared u32 entries, removing repeated variable-divisor
mapping/credit arithmetic from the hot loops. Physical weight/activation/output
order and each credit's point/count remain identical to direct band32. It
therefore tests whether the losing direct implementation's representation cost
hid a useful schedule. Direct remains a same-input descriptive comparator,
not a promoted candidate. Counters/timing have not established any speedup.

## Capacity and ownership review

At M<=32, at most256 local routes across32 experts imply at most60 eight-row
jobs. For G>=132, each CTA owns at mostceil(60*512/132)=233 phase-A units and
ceil(60*256/132)=117 phase-B units, hence at most350 total. Physical unit
indices are below30720; low16 bits suffice. Credit count is at most512, held in
high16 bits without overlap. The cached launch enforces M<=32 and G>=132, in
addition to the unchanged cooperative admission/SM90/local bounds.

The actual selected skinny routing/setup warp is **warp==kNW (warp8)**, not
warp0. It builds the histogram, job/range metadata and schedule. Its added
pre/post `__syncwarp` publishes immutable metadata among the same32 lanes
before initial copies; the existing CTA join publishes the cache to compute
warps before any consumer read. Entries are never overwritten during the
loops. Original fences, barrier counts, physical address map, math and credit
publication are preserved. Empty routing has no schedule reads. This corrects
the prior registration's warp0 wording, which confused fat-main routing with
skinny; the same-warp ordering reasoning was unchanged.

350u32 adds1400 raw bytes; Smem alignment rounds the actual increase to
**1408 B**. Owner's actual local sm90a compile reports128 skinny registers,
zero stack/spills and16 barriers. The prior local140/control130/direct versus
actual H200130/all discrepancy means actual runtime attributes remain required.
No occupancy benefit is predicted. Root and a separate device reviewer inspect
the source/compile proof; this host review alone does not certify CUDA execution.

## Preregistered six-arm comparison

Before execution, fixed heldouts **993727/993733/993739**, all six loads
1/8/16/32/128/256, all18 rows mandatory. Disjoint from population65001-65032
and direct-band993701/993703/993709. No selection, model weights generated
offline for these seeds, or tuning on census banks.

- Arms0/1 control,2/3 cached,4/5 direct. Three owned binaries; each pair shares
  a binary, with separate Python namespaces, packs, scratch, inputs, outputs
  and graphs. All supplied packed tensors are equal across all six arms;
  mutable pointers are checked distinct and graph objects retain inputs.
- Both treatment representations are called **only at16/32**. Every arm uses
  a frozen control namespace/extension on the other four loads, with its own
  state. Ordinary judged entry is always control. Cached128/256 launch guards
  cannot accidentally fire through an untargeted computing arm.
- Common cooperative launch and BF16-effective-gamma repair in all six arms,
  exactly as the previous diagnostic. Independent FP32 oracle, finite outputs,
  references and metrics; global`.008`/row`.075` gates unchanged.
- Required tally: normal108, fresh108, skew36, remote/empty36, post108:
  **396 checks**. Fresh32-replay stress per case, poisoned outputs, skew/remote
  per load, restored normal input/post-timing math and spin flags. Replayed
  inputs are repeated stress, not32 independent generated inputs.
- Actual duplicate vectors must match. MAIN47 vectors identical for all six
  arms. Direct skinny must match control's vector (as observed in previous
  H200 check). Cached registers/local may improve, static shared and thread/
  max-thread attributes match, dynamic shared is exactly control+1408 and
  admission>=control>=1. Any failure stops the diagnostic.
- Twenty cold rounds,256MiB flush, crossed/rotated six-arm order,275s inner/
  final guards,220s build-state guard. Every timing finite/positive before
  logs. Required18-row JSON report uses `allow_nan=False` and fails above11KB.

Promotion rule remains all four control0/1 versus cached2/3 pooled full-six-load
factors **and** all four factors per heldout seed >=1% and >2x their duplicate
null, with per-case cached regression <=max(.5us,1% control). Direct4/5 has
separate descriptive control/cached factors and its own duplicate null; its
known loss cannot reject or create a cached promotion. Full six-load factors
include untouched-load noise, not a selected16/32-only score.

## CPU validation and stopping

Seven actual TorchCPU host-facade tests pass, including the complete18-case
six-owner396-check flow, target-only dispatch, common gamma, stale captured
input/nonfinite-reference negatives, inner-budget/TF32 restoration, all four
pooled/per-seed/null/regression gates, named bad-timing rejection before logs,
and realistic full-manifest transport. The varied18x6timing report fixture is
**4419 B** including labels, below11KB. A lost direct comparator is explicitly
included in promotion-winner tests. CPU tests do not establish CUDA host build,
math, prepare time or GPU resource/performance success.

Root owns final review and the last proposed private check; this report does
not dispatch. Stop on source/resource/math/report/budget failure or missed
whole-score gate. No additional band/height/hint/credit/cap experiment follows
automatically. A faster cached-versus-direct result alone cannot pass the
control-versus-cached release gate.
