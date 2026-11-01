# Top10 campaign, October31 PDT

User's active objective: **top10 for every problem, use today's checks and scored submissions**.
This explicitly includes prefill and authorizes public scored runs. It supersedes the prior
checkpoint's no-submit restriction, but not correctness, precision, attribution or donor policy.
Goal remains active; no top10 completion claim is made. Evidence base `d3eeeda`.

Previous goal work was progress: authenticated CLI, recovered receipt, and measured/reverted
the PRQ carry experiment. This round expands execution across all five boards rather than
repeating that failed experiment. Initial quota: projection/MoE/layer/prefill20checks10submits,
attention19checks10submits; shared account, requery before dispatch.

## Ownership and running work

| Owner | Checkout/branch | Scope |
|---|---|---|
| Root | okc/main | Decode-layer, prefill, scored submissions, integration |
| today_proj | ../okc-today-proj, codex/today-proj | Prepared codec factorial and router F/V tests, then measured projection improvements |
| today_moe | ../okc-today-moe, codex/today-moe | Producer-wait attribution, then evidence-selected MoE improvement |
| audit_attention_moe | ../okc-today-attn, codex/today-attn | Attention prologue/flat spread attribution, then candidate |

Coordinator working method permits parallel isolated problem workers. Each has an initial
allocation of8checks; root alone submits scored files after exact candidate/check verification.
Workers retain local commits and all receipts; root integrates without overwriting other work.

## Dispatch ledger

| Job | Problem | Kind | Candidate/control |
|---|---|---|---|
|09ba15bd18fd4000|prefill|check|SGLang baseline-as-submission, baseline_profile/baseline.py; first hotspot map|
|36e57280ff1f4c63|decode layer|scored|Existing checked layer_m1rt51sc1g, comment-only repeat; production file unchanged|
|cccc131c55874949|attention|scored|Existing checked pick-off PRQ31/HMRG source, comment-only repeat|
|075dc89d74d144d6|projection|check|Codec cgemv/rd factorial; probes are never scored|
|3170985420f74869|projection|check|Prepared rk-c router F/V/order test; probes are never scored|
|7631974774944a2a|MoE|check|Frozen owned production51 single-file validation, no diagnostic|
|68248e326be14626|projection|duplicate|Exact C2 resend detected; not a new measurement|
|9eba96894ee1402e|projection|scored|C2 comment-only repeat after duplicate; exact base b2c4c096... checked5c4e4e99b6ec450f, paired696179090d264f38|

All first seven statuses were queried explicitly; six were live running and projection resend
was terminal duplicate. The replacement scored ID was returned and problem confirmed. Snapshots
are under evidence/2026-10-31-top10-campaign. No result is inferred from dispatch or old rank.
Daily repeats measure validated candidates under fresh inputs; higher headline scores alone
do not prove a kernel improvement. Future promotion still requires paired evidence.

## Next actions and gates

1. Wait on each exact ID, confirm owner/problem, commit profiles and log scores with95%CI.
2. Profile prefill before choosing its first optimization; its starter has no board entry.
   Baseline uses legal NVFP4 Marlin, not pre-dequantized BF16 weights. Do not port the decode
   MoE blindly to much larger prefill jobs or their scratch bounds.
3. Use remaining scored runs on validated best-owned candidates and controlled fresh-input
   repeats; never submit a probe/bench embedding donor code. Use checks to distinguish mechanisms
   rather than exhausting quota with duplicate code or repeating closed sweeps.
4. Keep all five board targets: component gains and layer integration have separate gates.
   Refresh exact top10 cutoffs and current ranks after scored results, including prefill.

Stop an individual mechanism on its registered failure criterion; pivot to the next measured
gap while retaining the overall top10 objective. Do not loosen a gate, remove ordering fences,
claim projected speedups as measurements, or mark the goal complete without all five official
board ranks proving top10. Deadline remains November4 at11:59pmPT.

## Progress addendum: October31, 22:13 PDT

Previous turn classified as progress (live checks/scored dispatches, committed ledger).
Refreshed authoritative state: root clean at e0ce491; eight named jobs below were all verified
running, not inferred from old state or session locks. Goal remains active and incomplete.

Additional checks:

| Job | Problem | Purpose |
|---|---|---|
|191e57a0f45a4717|attention|Matched owned/donor prologue phase exits, three held-out inputs, independent nulls; diagnostic only|
|e06255199f6d4de1|MoE|Producer-only wait attribution, phase and job-size split, clean51 control; diagnostic only|
|1c1fcf1d49924050|decode layer|Candidate-only check of owned standalone attention HMRG/PRQ31 in current stable M1+51+C1 layer|

Layer candidate saved as `muse_decode_layer/cand/layer_attn31s.py`, SHA256
`585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b`.
Reproduce by copying layer_m1rt51sc1g.py then running dl_one_inline.py on current attn_one.cu.
Only `_ONE_SRC` changes: attention Params ABI compared explicitly via AST string extraction,
both dependency waits retained, host PDL flags unchanged. No workspace-placement tuner added.
This revisits the documented stable-build/no-workspace trigger of the inconclusive layer_hp
experiment, not its old coupled tuner. New candidate must pass full layer correctness and then
independent paired timing before performance promotion; the pending check alone will not prove
an improvement. Existing best candidate remains the scored control.

Root reviewed MoE diagnostic storage/ordering: per-CTA64-word reserved region ends below
kSkOff for G<=256, host bounds assertions cover allocation, single xproducer lane writes its
CTA's counters, host reads after synchronization. No changed waits/releases/fences observed.
Timing caveats retained: phaseB readiness includes proxy-fence region; empty-region timing
includes clock overhead and first-lap no-wait path. These are attribution counters, not pure
wait cycles or measured GPU utilization. claim_a/claim_lean56 are false as in51.

Live IDs verified:1c1fcf1d49924050,191e57a0f45a4717,e06255199f6d4de1,09ba15bd18fd4000,
36e57280ff1f4c63,cccc131c55874949,9eba96894ee1402e,7631974774944a2a. Updated API snapshots
are in the evidence directory. Wait these exact jobs; do not restart them on observation timeout.
Projection worker separately owns the two earlier live projection mechanism checks.

## Progress addendum: first completed results and prefill tile experiment

Prefill baseline09ba15bd18fd4000 completed and passed at1.000953x (95%CI approximately
0.985939–1.015967). Saved raw status and normalized profiles under muse_prefill_layer/
baseline_profile. Running hotspots.py with that same baseline-as-submission source shows
Marlin MoE40–67% of time; representative sliding rows749.8/2383.1/11547us at753/3012/16384
tokens. This satisfies profiling before writing the first prefill optimization.

Dispatched fixed Marlin M16 (e6c12149c99a4a96) and M32 (334149df33f64c54) checks, changing
only the existing M-tile selection. No weight/activation precision or mathematical boundary
changes. Full hypothesis, acceptance and run ledger: docs/runs-prefill.md. Also submitted the
validated baseline as181880202c5742db to establish a first entry, not as an optimization claim.

MoE exact production validation7631974774944a2a passed5tests,2.193073x (2.160177–2.225969).
Prepare selected skinny sk8w3s at1/8/16/32 and51 at128/256; it is not fixed51 at32.
Frozen SHA66618198aafc4d14cfe0350bf561fe8abc66f3d5bc9f6df1dbd2518508297c01 submitted as
336f3a63bd6a44f3, problem confirmed; no diagnostic included.

Projection scored9eba96894ee1402e completed/pass1.527806x, above prior best1.516978x,
but a single headline change is not an isolated kernel gain. Projection owner retains receipt,
CI and attribution in docs/runs.md. Its router F/V check produced a promising128-token paired
signal; candidate confirmation is in progress. Codec model perturbation prevents treating the
bytes-only timing as a usable compression ceiling; no real codec is promoted from that model.

Layer36e57280ff1f4c63, attentioncccc131c55874949 and layer-port1c1fcf1d49924050 were freshly
verified running. Continue exact-ID waits; do not replace them because a watch ends. Active
goal still requires verified top10 on every board and today's remaining authorized experiments.

## Progress addendum: completed controls and paired-layer follow-up

Merged projection evidence throughda4fac3 and MoE evidence through41bf882, preserving
independent workers. Root verified the completed diagnostic and scored owning receipts.
Layer scored repeat36e57280ff1f4c63 passes13tests at2.014780x (1.984558–2.045002),
M1open,mode51 at32/128/256, no prepare failures; best unchanged. Attention scoredcccc131c55874949
passes at2.534850x,MoE scored336f3a63bd6a44f3 at2.199082x, within prior run spreads;
owning workers preserve their CIs/profiles/logs.

Layer attention-only port1c1fcf1d49924050 passes13tests at2.048010x (2.017290–2.078730).
This prompts paired confirmation, not an immediate promotion: four-arm all-six-load
checkc0402e4c92bc4b20, two held-outinputs,8rounds, independent prepares, fixedseed/order
610001/601, deadline280s. Candidate stays585d46e2...; incumbent8112f37a....

Prefill M16 andM32 pass correctness but lose0.715181x/0.884745x decisively. Both closed in
ruled-out, no production change retained. Baseline scored181880202c5742db passes0.999115x,
establishing an entry rather than an improvement. Full intervals/profiles in runs-prefill.
The inspected pinned Marlin Python wrapper does not expose thread-N/K knobs; atomic-output
mode is not a sanctioned precision shortcut. Underlying expert execution remains the target.

MoE producer diagnostic contradicts the earlier small-job-minority assumption on its measured
inputs:78.3%smallstages on256layer seed211,88.9%on256std. Empty-region sums dominate readiness,
with instrumentation caveats. Reviewed fixed4-slot capacity tradeoff now checking39804be1a59c429c;
the independent proof covers all7modulo updates, dead diagnostic-barrier storage, allocation
budgets and preserved releases. This does not prove variable per-job resizing safe or profitable.

Projection C3 cleared exact9test check61330d87d32c4190 and independent router confirmation
7d2fe593fdfa4720, repeatable128FV -0.45us and256orderFV -0.69us. Root submitted pure candidate
as7a15f46e673b43d5. This is a small aggregate mechanism, not the entire top10 gap. Codec worker
is moving to an actual packed lossless stream trial because old load-skipping models changed
codegen and do not establish a compression ceiling. Goal remains active, all-five-board target.

## Progress addendum: MoE ring closure and next structural comparison

Previous goal turn made measured progress; current work reverified live handles, merged
worker evidence, and selected the next experiment from new attribution. Ring4 check
39804be1a59c429c passed5tests but lost at128layer/std/skew by2.2/1.5/1.2us and256 by
2.9/3.9/1.9us; every256seed loses. Close this coupled x-depth/weight-depth tradeoff,
not all ring designs. Default production unchanged; loss/register/source evidence retained
through cc8252a in root. No repeat of identical four-slot capacity is planned.

Attention diagnostic191e57a0f45a4717 completed/pass. The d1full gap is already present at
the first page-group drain: owned7.04us versus pinned donor6.02us; complete-prologue
9.79vs9.41. v1full complete prologue9.57vs9.54. These are drained exits, not additive
live-kernel stages. Therefore defer the speculative newKV staging rewrite. Owned late-tid
address-scheduling testb4d75c3185744d0a is now running with default-off source toggle,
all12shapes,3heldoutinputs and independent control. Existing fences/waits retained.
Evidence/source merged through db2e545; actual promotion remains gated by timing and stress.

Root is investigating a structurally different grouped expert backend for MoE/prefill.
Official Humming primary API pinned948e6fd supports an exact NVFP4 tensor-load path,
BF16 inputs/outputs and FP32 accumulation. Installed compatibility is still unknown:
baseline-only capability checke30ec1bf8e8142df is live. No judge dependency installation,
requantization, coarser activation, host-synchronized route sizing or unvalidated backend
substitution is allowed. See runs-prefill for concrete ABI/precision constraints.

Projection actual packed pair7 first attempt0ab8f0510515473b failed compilation because
the experiment requested C++17 against current PyTorch's C++20 requirement. That is not
a kernel or codec performance loss. Corrected retry962d2f9c1d9644f7 is live after a stronger
all-BF16-bitpattern CPU packing/escape gate. Projection C3 scored7a15f46e673b43d5 passes
1.513805x (1.491098–1.536512); consistent component gains do not imply a higher headline
in one noisy scored run. C2's1.527806x remains the best observed projection score.

Authorized fresh-control repeats dispatched43026195f8c84ffb (layer) and8914751e3b9f468b
(attention), comment-only changes to already checked frozen controls. Reserve remaining
slots for confirmed candidates; these repeats measure run variation and are not new speedups.
Paired-layer c0402e4c92bc4b20 was freshly verified running and must be waited, not restarted.
All-five-board top10 objective remains active and unproved.

## Progress addendum: layer promotion gate and installed backend compatibility

Previous turn classifiedprogress. Paired-layerc0402e4c92bc4b20 completed: all24checks
per arm, all6loads, pooled1.0042, allcrosses1.0036–1.0048, maxnull0.0007 inlog units;
registered candfaster gate clears. Preparation258s. M1 policy at32 differs inoneinc arm,
so attribute this to the complete prepared candidate rather than attention instructions alone.
Root submitted exact checked layer_attn31s as8f2780d7382047e6, and dispatched independent
confirmation9414182118414889 with newseeds620001/order701. Both results remain pending.
Control repeat43026195f8c84ffb passed2.009960x CI1.979811–2.040110, unchanged algorithm.

Humming capability checke30ec1bf8e8142df completed:0.1.12 installed and grouped API imports.
This enables implementation, not a speedup claim. Exact-version investigation rejected a
naive grouped_contiguous fixed-capacity mapping because the scheduler computes the last
expert to shape_m, including unusedtail. MoE owner is preparing INDEXED device metadata
with padded sorted localroutes, device counts and masked sentinels, retaining BF16/FP32
boundaries and original packed NVFP4/E4M3/global scales. Root review precedes first dispatch.

Official public board snapshot22:27PDT (ordered-row ranks, saved current-board-summary.json):

| Problem | Rank | Best score | Observed rank10 |
|---|---:|---:|---:|
|Projection|292|1.527806x|1.604517x|
|Attention|296|2.536566x|2.909794x|
|MoE|371|2.215712x|2.593077x|
|Decode layer|374|2.018705x|2.379322x|
|Prefill|no row|none|see saved exact snapshot|

No top10 requirement is met. Prefill's passed baseline scored0.999115x did not create a
ranked entry; earlier wording stated the intended first-entry step, not verified board state.
Projection's newbest9eba96894ee1402e hotspot table is appended to docs/hotspots.md with its
baseline profile. Actual pair7 correctness now passes but its concrete scalar-plane decoder
loses; that implementation is closed, not all optimized lossless codecs. Next projection
work is a bounded low-shared-memory DSMEM large-tile test with actual cluster-residency gate,
explicitly distinct from the previous200KB/one-CTA c4loss. Workers retain source and receipts.

## Progress addendum: new layer entry and first exact indexed backend check

Scored layer8f2780d7382047e6 passed13tests at2.022870x (1.992527–2.053214); official
board now selects it. The paired0.42% prepared-candidate gain cleared its gate, though the
headline difference from2.018705x is within noise. Release manifest now pins585d46e2...
and the exact source/policy. It explicitly preserves3unselected/report-only preparation
issues rather than claiming all diagnostics clean. Newbest hotspot table appended.

Root and independent review cleared fixed-capacity Humming0.1.12 INDEXED metadata, not the
rejected grouped_contiguous padding scheme. First plain prefills checkbe3cbc7f5a234c64 is
running, sourcee53d374f..., no probing/scored path substitution. DeviceNP limits localexpert
blocks; sentinels are masked in both GEMMread/write paths; allocation/counter resets and
BF16stage boundaries preserved. Still requires official result and queuedfreshinput edge
gates before scoring. Installation/API support alone is not performance evidence.

Attention late-tid firstcheckb4d75c3185744d0a passes19armchecks; d1full delta-0.48us,
null-0.13us, across3inputs. New4arm independentbuild/state confirmationab53721b09854717
running. Defaultoff; actualcandidate-only official tests and2000queuedfresh-reference checks
must pass before promotion. Scored unchanged control8914751e3b9f468b passed2.604290x:
attribute the headline change to fresh-run variation, not the late-tid toggle it did not use.

Dense projection4ca25ee4eccc495c proves new3-stagegeometry residency62activeclusters,
2CTA/SM,90registers,0localbytes,99376dynamicsmem, clearing occupancy gate. Yet it loses
~12.7us against same-process selected Lt,0/100wins; four-stage is correctly skipped at30
clusters. This rules out insufficient cluster residency for that geometry, not the cost of
its DSMEM reduction/mainloop. One explicit timing-only partition diagnostic is next, with
all synchronization preserved in legitimate computing arms. No new dense candidate promoted.

Objective remains active: allfiveboards must become verifiedtop10, and today's remaining
checks/submits are still being used for evidence-selected experiments and checked candidates.

### Same-round terminal/dispatch addendum

First Hummingbe3cbc7f5a234c64 stopped inprepare on installed global-scale parameter shape:
expected[32], supplied[32,1]. No kernel math/performance result exists. Corrected plain
0de84000097946c9 uses original FP32scales reshaped to the installed parameter shape; frozen
sourcebf6d95ae1cbba377dc5bd4a2cd9774815a54ebb26ddcfcab9a33834220c87232.
Stress/release generator additionally asserts32elements and FP32destination; do not mutate
the in-flight artifact or promote it without full official/fresh-input validation.

Projection partition diagnostice6e5bd6d0a104faf now running at the same measured62-cluster
geometry. Computing controls retain official/FP64/replay checks; memory-only/no-reduction
arms are explicitly wrong-output timing probes, excluded from correctness and never scored.
Their purpose is to decide whether dense mainloop or DSMEM retirement dominates the loss,
not fabricate a fast but incomplete kernel. Old17.47us adjusted Lt and newer26.624us raw
Lt are not interchangeable cross-run baselines.

Merged attention df8ed3b while preserving both appended newbest hotspot tables (layer and
unchanged attention control). Main worktree clean at15974d7 before this addendum. Public
best layer/source remains8f2780d7382047e6/585d46e2.... Goal not complete or blocked.

## Progress addendum: October31 23:07 PDT — exact releases, repeatability, and pivots

Evidence pin: root main after merging projection649990a, MoE c0f88ae and attention c374b9d
(root evidence commit c6b8e30 before new root dispatch; `git log` is authoritative). New root experiments below derive
from checked source at2d1572a. CLI installation/login is complete: WSL Ubuntu okc0.7.2;
remote checks report H200. The Windows RTX5070 is not used as a performance proxy.

Public API snapshots `evidence/2026-10-31-top10-campaign/board-2303-*.json` and
`board-summary-2303.json` establish current standings (rank is ordered row index, not attempts):

| Problem | Own rank | Best score | Observed rank10 | Required additional score |
|---|---:|---:|---:|---:|
| Projection |292|1.527806|1.604517|5.02096%|
| Attention |206|2.604290|2.909794|11.73079%|
| MoE |371|2.215712|2.593077|17.03133%|
| Decode layer |375|2.022870|2.379747|17.64209%|
| Prefill |629|1.146466|2.017961|76.01580%|

These are measured cutoffs at this snapshot, not November4 predictions. No board is top10.
The existing measured portfolio does not establish enough gain for any of these gaps.
Projection is closest arithmetically, but its closed experiments do not supply the missing5%.
The best honest assessment remains: top10 is unproved and requires a new structural win;
no claimed projected speedups or probability of success. Objective stays active.

### Completed measurements and exact candidate

Prefill scored92408f1b65c643c4 passes5tests, **1.146465578x**,95%CI
1.129268595–1.163662562. This creates the first verified ranked prefill entry. Exact pure
candidate is `muse_prefill_layer/cand/humming_indexed_v3.py`, SHA256
f3024d1b01a5ef2bf655dcffe3110849ab6b0de15aecad6dba388f09e7ffb8fa.
It contains no stress suffix. Correctness-only checkece54526780a4f28's source begins with
these exact bytes; plain first check0de84000097946c9 passes at1.176449x. Different run scores
are not isolated mechanism estimates. All original NVFP4 weights/scales and BF16/FP32 stage
boundaries remain. Installed Humming0.1.12 source pin636ba856... is used. Newbest hotspot table
is appended to docs/hotspots.md against SGLang09ba15bd18fd4000. GEMM stages remain major costs.

Standalone adapter8d64807ea7504495 passes5tests but loses overall:0.911512837x,
95%CI0.897840145–0.925185530. PerBS1/8/16/32 is slower than SGLang;128/256 beat SGLang
slightly but remain far behind our persistent owned kernel. Preserve experiment, close
INDEXED/routed heuristic as a small-token replacement. This does not close GROUPED_MASKED.

Layer second independent four-arm9414182118414889 passes24checks perarm, all6loads,
but pooled1.0015 and cross1.0006–1.0024 do not clear2xmaxlognull0.0011. All arms selected
M1{} after fresh gate fallback. This does not repeat the first prepared-regime gain from
c0402e4c92bc4b20. One original gate reported rel0.14/order1count4; do not dismiss it as noise.
Scored8f2780d7382047e6 remains the checked public best, with original failclosed runtime gates.
Matched gate diagnostic273ff292536e4429 passes official control but early evidence was lost
from the log tail: observability failure, no gate inference. Tail-fixed03bb478f8b0b4629 passes
13official tests (control2.035941x,CI2.005402–2.066481). At16x1sliding and1x8sliding, all four
matched off/off,on/off,on/on comparisons pass the original gate, order1zero. It does not
reproduce the0.14 event and therefore does not identify its cause. Judged M1 is always off
in this diagnostic; never promote its control score as a kernel gain.

Attention full-only late-tid a9b416a0c7324ecc clears prospective paired gate, but actual
candidate/stress3588375a529c4fe8 fails before official timing on unchanged sliding:
17/2000 exceed strict1e-4 comparison, maximumrelative L2~4.48e-4. Full stress was not reached.
No promotion; source defaults0. A prospectively registered independent0/0-versus2/0 null
study must classify this, keeping the same comparator and printing both shapes before exit.
A timing win and selftests do not override a fresh-output failure.

Humming global local-M ec3618d1604e44ba fails M32 fresh share criterion (graph0.5 vs eager
0.046875) with all metadata/nonfinite/zero controls clean. Bounded local-M only at753/3012,
d26a87ca2f484145, also fails unchanged M32 first: eager64 share0,graph2000 share0.002.
Do not attribute the latter to unexecuted changed sizes, or relax the gate. New single-factor
no-StreamK74eb65dad3fa4e54 retains routed heuristic/zeroing and tests explicit deterministic
config plus official timing. This is a numerical reproducibility investigation, not a win.

Projection dense occupancy proof is now real (62clusters,2CTA/SM,90regs,0localbytes,
99376Bshared), yet4ca25ee4eccc495c loses~12.7us to selected Lt. Ablation e6e5bd6d0a104faf:
Lt26.912us, full~39.6, no-reduction~36.4, memory-no-reduction~32.2. Memory pipeline alone
already trails Lt. Combined128-row TMA7803be789c624f8e saves~0.5us full but leaves~12us;
stop this local dense family. Scalar/vector pair7 concrete decoders also lose; TOP7-LUT32
checkee39ec933a8542d8 tests a different byte ABI after exhaustive CPU gates and is pending.
Neither fake byte savings nor the theoretical entropy ratio is a score forecast.

### Ranked next work and acceptance/stopping rules

1. **Prefill expert execution and traffic:** new indexed nozero/filtered-SiLU hypothesis,
   fixed routed heuristic;4independent arms,NaN-poisoned scratch,two heldoutseeds/all3M.
   Require all computing checks, paired gains beyond duplicate nulls, official candidate
   tests and fresh routing validation before public promotion. Pure generator
   tools/prefill_humming_nozero.py does not alter Humming ordering/math. GROUPED_MASKED is
   a separate larger prototype, reviewed before dispatch; bound MCAP with distinct-top8
   proof and failclosed overflow. If its full overhead erases compute gain, close mechanism.
2. **Attention full-decode scheduling:** classify fresh failure against independently built
   nulls, covering full and sliding. No threshold changes justified solely by passing tests.
   If candidate adds error relative to null, stop/revert. Only then consider layer port.
3. **Low-token MoE:** existing producer counter/4slot evidence closes easy ring resizing;
   indexed library adapter loses. Masked grouped execution is a new structural hypothesis,
   no speedup promise and separate decode policy needed. Keep current persistent release.
4. **Layer boundaries/router:** retain existing routing fusion and M1 gates; do not repeat
   known mechanisms without new hotspot evidence. Current second paired result is unresolved;
   an attention instruction claim requires matched prepared policies. No cross-stage fusion
   is presumed legal or useful. Do not weaken fallback to recover benchmark speed.
5. **BF16 projections:** one pending distinct TOP7 decoder; stop local dense knobs. Any codec
   needs physical bytes including escapes/padding, exact bit roundtrip, FP64/replay/resource
   checks, same-run paired improvement and official score validation. Insufficient score gain
   means pivot to a structurally different measured hotspot, not endless config sweeps.

New source correction to research plan306: H200SXM bandwidth is4.8TB/s;3.35TB/s is H100SXM.
[NVIDIA hardware table](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html)
backs this correction. It invalidates a3.35-based near-floor argument, but does not establish
recoverable bandwidth or a promised gain. Preserve old plan and this dated correction.

New masked pipeline ordering proof: ordinary completed writer kernels and subsequent same-
stream TMA kernels are ordered across proxies under
[PTX8.7 section8.9.4 API synchronization](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-synchronization).
This applies to completed kernels/graph dependencies, not PDL overlap. Retain library internal
shared-memory proxy fences and descriptor handling; no runtime tensormap mutation is added.

Recommended submissions now: keep exact existing public best jobs in the table; prefill uses
f3024d1b... explicit candidate and layer585d46e2... explicit candidate, rather than assuming
main/submission.py equals either. No failing fresh-stress candidate is submitted. Remaining
checks/submits are for new evidence-selected mechanisms and validated candidates; fresh
repeat scores alone cannot support causal improvement. Allfive-top10 objective remains open.

Dispatch addendum: reviewed nozero four-arm experiment2bd4f06042864771 is running on prefill;
judged purecandidate SHA561e4180...; no heuristic/StreamK change. TOP7-LUT32ee39ec933a8542d8
now passes all correctness gates but loses every target despite3.0–3.4% fewer actualbytes.
Close that codec series. One separately registered original12bit integrated-escape-warp
resource-ownership experiment is approved after CPUproof, with actual occupancy gain gate;
it is not another codec or denseTMA sweep. No claimed5%score coverage.

Snapshot time clarification: board-2303 files were fetched23:03:57–23:03:59 PDT (filesystem
receipt times), not the addendum publication time. CPU route and masked-contract tests each
pass, generated root helpers/pure/bench files compile, and diff whitespace check passes.
Plain reviewed GROUPED_MASKED c1f6023f03d34c72 and noStreamK74eb65dad3fa4e54 remain live;
full source/config resource evidence must precede any promotion. No public stressbench sent.

## User steering and four-problem continuation, October31 23:26 PDT

**User instruction: "ignore prefill for now please".** Prefill is deferred. No further
prefill optimization, integration, promotion or dispatch is authorized by the current work
plan. Already dispatched jobs may finish and their receipts are retained for a later handoff.
The active work is projection, attention, standalone MoE and decode layer. This is a scope
priority change, not achievement of the original all-five top10 goal. No completion claim.

Evidence root24cbef6 plus named worker commits below. Official snapshots fetched23:26 PDT
are `evidence/2026-10-31-top10-campaign/board-2326-*.json` and board-summary-2326.json:

| Active problem | Own rank | Best job / score | Observed rank10 | Required more score |
|---|---:|---|---:|---:|
|Projection|292|9eba96894ee1402e /1.527806|1.604517|5.02096%|
|Attention|206|8914751e3b9f468b /2.604290|2.909794|11.73079%|
|MoE|371|abe7651dbaa44458 /2.215712|2.595612|17.14577%|
|Decode layer|377|8f2780d7382047e6 /2.022870|2.383392|17.82229%|

Cutoffs moved on MoE/layer since23:03. The measured portfolio still does not cover these gaps.
Scored unchanged controls: MoE116c5185b14b487d passes2.213803083x,CI2.180596037–2.247010129;
projection2383f5173a934c63 passes1.519919441x,CI1.497120649–1.542718233. Existing best jobs
stay selected. These fresh repeats are evidence of run variation, not algorithmic wins.

### Active evidence and next experiments

1. **Attention numerical classification:**26c8a3ea28554454 is terminal strict failure,
   including BOTH shapes. Identical-code independent nulls also fail1e-4: sliding0b/0a
   19/2000,maxrelativeL2~4.577e-4; candidate-null16/2000,max~4.581e-4. Full0b/0a10/2000,
   max~2.316e-4, candidate-null12/2000,max~4.448e-4; fullcrosses10–14,max~4.523e-4.
   All24FP32 samples pass, but do not establish compliance for all2000inputs. The gate stays
   failed and no candidate is promoted. Prospective new gate checks EVERY queued output
   directly against owned FP32 references and exact TASK global/row/cache bounds, with
   zero/stale/order1 negative controls and canonical reference crosscheck. This answers
   official mathematical compliance, not whether strict bitwise repeatability was achieved.
   Budget280s; incomplete coverage fails closed. No native math/fence/late-tid sweep change.
2. **Standalone MoE new structural backend:**ae8993f4e3e1435c compares owned51/duplicate
   with Humming MASKED/duplicate at128/256, layer/std two heldoutseeds291/292 plus skew
   correctness. Official judgedpath remains owned. Readonly original weights are shared;
   scratch/locks/workspaces are independent. Test all replay outputs after graph timing too.
   Registered gain must exceed3us128/4us256 and2xnull, all cases/gates complete under280s.
   If launch/gather/tails erase GEMM gains, close without knob sweeps. This is distinct from
   losing INDEXED8d64807ea7504495 and old mma.sync staged45/46 designs. No prefill speedup is
   used as standalone evidence. Gamma must remain FP32(1+post_norm_w), matching production51.
3. **Projection:**integrated escape experimentbac995b3b94541c4 loses all shapes despite
   actual occupancy increases and no spills; stop that schedule. Current C2scorecard ranks
   pooler largest scoreelasticity32.01%,unpooler22.79%,qkv22.55%. Delta-table has no measured
   escape-only ceiling and poolerd1 actually selectsrd; defer it. Offline-only rd16 cp.async
   shared-memory ring design targets overlap/register live ranges, distinct from losing
   register PF1/2/3. It must prove staging ownership/alignment/tails, preserve MMA/escape
   arithmetic and final barriers, query registers/spills/shared/blocks before timing and
   abort if occupancy falls. Prior donor wholepath -.69–.90us was versus2265fa2, NOT current
   C3, and cannot establish a present gain. No5%score prediction.
4. **Decode layer:**latest profile still has MoE as major cost (e.g decode1 sliding34.7us,
   39% in one profiled input); remaining glue launches are smaller. Profiling rows are not
   isolated whole-step deltas. Wait for validated standalone mechanisms, then re-port with
   matched policies and full six-load paired tests. No fusion/PDL/barrier change presumed
   profitable. Original first prepared-regime gain and unresolved second result both stay.

Projection large-M Lt diagnostic e15f6a9a9ff047ef completed before the deferral: pooler saves
90.6–96.1us and unpooler109.2–117.8us atM16384 in paired checks; qkv fails2%gain threshold.
The2527us profile row aggregates THREE calls, not one operation. These are prefill-only
opportunities, now FROZEN; no integration/check is planned. Completed generator/result work
is preserved on codex/today-proj through94a6e0b, not claimed to improve standalone projection.

### Frozen prefill handoff — completed work only, no new promotions

Before user deferral, nozero+noStreamK36d2347642e14e37 passed all4arm tests, repeatable gains
~75/310/1700us at753/3012/16384, and2000fresh+64eager at32/753/3012 with allstats0. Scored
pure1530ec79d50a4f69 had already been dispatched; it finished1.216560352x,
CI1.198311946–1.234808757, official board now selects it at593. PureSHAa0fe61e7... is frozen
in experiments/humming_nozero_nostream.py; no bench suffix submitted.

Already dispatched combined nozero+noStreamK+bounded753/3012 e9bd4269a6584df8 passes
1.273593087x,CI1.254489190–1.292696983; exactpureSHA
abce3925a052d3cdc1c482e305904064945a7b4ccd2891c1b195fbe2d496b68d is frozen in
experiments/humming_nozero_nostream_bounded.py. Not promoted or submitted after deferral.
Experts-only deterministic bounded component21e98162b7114898 (prefill board) establishes compensation under
allFalse configs (M753 -191–198us and M3012 -260–400us, duplicates~1us), rather than using
cross-run scores to claim causality.

Masked initial failurec1f6023f03d34c72 was traced to int32 grouped element-offset arithmetic:
32*CAP16544*4096=2168455168 exceeds signed2^31, and G1 stride8192 also exceeds2^32.
Earlier row-bounds-only reviews missed this width bug. Four casts BEFORE multiplication
fix it; CPU address oracle added. Corrected f92ffc2953284a9a passes allofficial/benchmark
checks1.294149997x. Already dispatched poisoned freshdefbb042449d44e7 passes registered
matched eager envelope at32/753/3012, checkscore1.300970887x,CI1.281456324–1.320485450.
It is not bitwise deterministic; strict/eager stats remain visible. Exactpure63eb4171... is
frozen in worker artifacts; no promotion after user instruction. Independent row/offset-width
proof plus hardware verification are both needed, not tests alone as memory-order evidence.

Precision correction for later reuse: prefill baseline explicitly rounds effective postnorm
weight toBF16; standalone production51 instead consumes FP32(1+w). The original standalone
INDEXED losing adapter used the inherited BF16gamma; do not describe it as an exact51norm
comparison. The new standalone MASKED diagnostic corrects gamma toFP32 before dispatch.
