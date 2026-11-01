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
