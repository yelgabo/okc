# Four-track top-10 progress — November 1, 2026

Scope follows the user's latest instruction: **defer prefill**. Active optimization is
projection, attention, standalone MoE and decode layer. The original goal is unachieved;
this report does not claim top-10 placement or a demonstrated path to it.

Evidence source pin `1764d15`, plus exact candidate hashes and owning receipts below.
This supersedes the execution status in the [October 31 campaign](2026-10-31-top10-campaign.md),
while preserving that report's history, failures and corrections.

## Feasibility and current standings

Official board API snapshots fetched November 1 at approximately00:05 PDT are preserved in
`evidence/2026-11-01-four-track/`. Ordered-row ranks and score ratios are:

| Problem | Rank | Best public score | Observed rank10 | Required additional score |
|---|---:|---:|---:|---:|
| Projection |292|1.527806|1.604517|5.02096%|
| Attention |207|2.604290|2.909794|11.73079%|
| MoE |371|2.215712|2.595612|17.14577%|
| Decode layer |377|2.022870|2.383392|17.82229%|

**The existing measured portfolio does not cover any of these gaps.** The new attention
change has a small repeatable paired benefit. Other recent designs lost or failed their
preregistered opportunity screen. A substantial new implementation win remains necessary;
there is no source-backed probability or projected speedup that establishes top10 by the
November4 deadline. This is not a proof that further architecture work cannot succeed.

The [official homepage](https://openkernelchallenge.dev/) confirms kernel challenges close
November4 at11:59pm PT. [Official rules](https://openkernelchallenge.dev/rules) make public
rankings provisional: final placement uses medians of post-challenge reruns, including
held-out data. A lucky single headline is therefore insufficient evidence of feasibility.
Quotas reset at00:00UTC; local midnight did not replenish them.

Competitor per-shape receipts remain an evidence gap. The public board gives aggregates;
our CLI cannot read rank10 attention job5cb3655bf3c9499a. The inspected public frontend asset
exposes aggregate board routes. This scoped search does not establish that such data is
unavailable elsewhere. Do not infer per-shape cutoffs from the aggregate leaderboard.

## Working code and validation

Standalone attention now enables `ONE_LATE_TID=2` in both `submission.py` and `attn_one.cu`.
This rereads the thread identifier after page-copy commit only for L1 SPEC non-one-tile
instances. Math, precision, cache coverage, fences and barriers are retained. The promoted
executable AST matches exact checked pure candidate
`muse_attention_decode/cand/late_full.py`, SHA256
`cbee36a852fbc62f5491e75f957eeb4298dce2db50947984cdbbcd08aa69b356`, apart from CUDA comments.
Known-good default0 remains in source pin1764d15 and earlier1241557; its normalized source
SHA is498161899c637f599d6987f22dcb1c2c2f922fb66f490ce0ef6e926b7ff6f7a4.

| Owning job | Purpose | Result |
|---|---|---|
|a9b416a0c7324ecc|Four-arm prospective performance gate|All12 full-context crosses improve0.29–0.51us; corrected whole-attention sensitivity1.0023928, nulls1.0001715/1.0005802; clears2x larger log-null|
|3588375a529c4fe8 /26c8a3ea28554454|Strict1e-4 fresh-output tests|Remain failures; identical-code nulls also fail. No claim of stricter repeatability|
|80415fe736bb4890|Prospective all-input official-math gate|All4000 candidate and4000 control outputs meet exact TASK global/row/cache bounds; all6 corruption controls reject; workspaces clean; canonical reference crosscheck passes|
|d331aca81c554d70|Clean exact-candidate official check|16/16 pass;2.569431026x,95%CI2.530889561–2.619575175; no diagnostic flag changes|
|c9bc5fbd85b440e2|Scored exact pure candidate|16/16 pass;2.515773722x,95%CI2.478037116–2.553510328; public best remains older2.604290x|

The paired benefit is measured; a higher public aggregate has not been demonstrated.
Do not attribute fresh-run variation to this scheduling change. Affected mode11 retains
114registers,0local bytes,58960B dynamic shared memory,128threads and3blocks/SM; all13
resource vectors matched in the reference diagnostic. The804... aggregate is qualified by
its reference-only TF32-flag handling, which is why the separate clean check was required.

Layer port is deferred. The actual8f2780d7382047e6 decode1 medians are76.753706us sliding
and82.877882us full. Its weighted cost is46*sliding+16*full=4856.716573us. A0.4us change
confined to full decode1 has only0.021980% six-load score sensitivity. This is arithmetic,
not a measured layer gain, and falls below the existing paired noise. Standalone0.24%
benefit does not transfer wholesale into the layer.

## Recent exclusions and corrected recommendations

- Projection rd16 async ring b103af15ba564ef6 passes correctness and resource gates but
  loses0.48–0.61us. Removing24B local spills without increasing occupancy does not make it
  faster. Keep C3 production; no PF/register-cap sweep. Earlier codec and dense TMA losses
  remain closed as built.
- Standalone masked MoE ae8993f4e3e1435c loses145–235us despite all mathematical tests
  passing. At256/std292 its two GEMMs profile380.511us, already exceeding the owned
  whole-call median217.152us. Glue-only fusion cannot rescue that tested schedule.
- Weight-reuse count study53c07507b9fd4f8c passes correctness but fails its>=10% screen
  in6/8 cases. All four standalone-routing cases have zero duplicate expert-weight jobs;
  pairable saving is0 there. Layer32 has18.18% pairable logical requests; that does not
  satisfy the registered standalone16/32 scope. Park the multicast rewrite before GPU
  implementation. Physical HBM/L2 traffic was not measured.
- A VT-only memory compaction sketch does not enable a fourth attention stage at2CTA/SM:
  allocation is max(QST16384,VT), so NS4 needs119392B per CTA, and two exceed233472B.
  The earlier113248B estimate was corrected before an experiment used it. A new query
  staging design would need separate evidence and ordering/resource proof.

## Ranked backlog and stopping criteria

1. **Attention verify/full attribution:** completed8ac6563343f74bcd isolates producer-read,
   producer-zero and no-copy compute behavior for current mode51 at verify16/32 full.
   Preserve all acquires/releases/fences and participant counts. Require107104B shared
   memory and2CTA/SM across arms before timing. Probes are wrong-output diagnostics,
   never submissions. Interpret perturbed producer pace as a bound, not additive costs.
   Stop proposed ring/QK/page changes unless this produces new phase-specific evidence.
   Cold control/read/zero/nocopy-compute times are110.88/110.69/74.24/85.15us at verify16,
   and188.38/197.06/132.51/151.17us at verify32. Removing consumer math alone provides no
   repeatable gain; producer/cache-copy pacing is the next source-audit target. These
   perturbed-stream bounds are not additive recoverable savings or physical-traffic counts.
2. **MoE skinny full-slot wait policy:** [source audit](2026-10-31-moe-skinny-wait-policy-audit.md).
   Test only namespace-local full-slot test_wait versus try_wait, same acquire/scope/parity
   and watchdog; exclude empty, handoff and fat-kernel helpers. Four independently compiled
   arms and private scratch,1/8/16/32 layer/std heldouts. Require original correctness,
   finite/spin gates, resource equivalence,>=1.0% whole-six-load contribution beyond2x
   larger log-null in both candidates, and no individual regression beyond the registered
   threshold. No hint/NS/PF sweep. Independent review precedes the single authorized check.
   First87f22aa71e614460 failed a host formatter: loop variable `round` shadowed the builtin.
   No completed timing verdict. Corrected252df4a54d764ca5 is running after a three-test CPU
   scaffold exercised the actual formatter/gate, including a negative reproduction of the
   error. CUDA strings and gates are unchanged; this is the same hypothesis, not a sweep.
3. **Decode layer:** preserve known-good candidate585d46e2... and failclosed runtime gates.
   The second attention-port paired study9414182118414889 is unresolved, with a different
   M1 policy. Port only a sufficiently large validated component improvement and rerun all
   six loads with matched prepared policies. Do not weaken gates to recover speed.
4. **Projection:** keep current C3 working source and best public C2 entry. No further codec,
   escape-warp or dense TMA sweep without a new measured mechanism. Pooler is the largest
   score elasticity, but that is not proof of recoverable time. No pending projection job.

Recommended checked attention code is the new pure late-full candidate; best public
attention entry remains8914751e3b9f468b. Other best public jobs remain projection
9eba96894ee1402e, MoE abe7651dbaa44458 and layer8f2780d7382047e6. Keep explicit candidate
files and source hashes; main's file is not automatically the highest observed public run.
Prefill is frozen under the user's instruction. No new prefill work accompanies this report.

New attachment correction: the earlier documented Deep Research audit covered a different
report. The user's *Reaching the Top 10...* paste is now preserved and read in full;
[identity and initial triage](2026-11-01-pasted-top10-report-review.md). Do not treat earlier
community-report fact-checking as verification of its fusion estimates or citation markers.
