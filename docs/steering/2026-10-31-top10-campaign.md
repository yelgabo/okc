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
