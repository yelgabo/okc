# Private results versus public releases

November 2, 2026 Pacific. This answers the user's request for other better private results and patches the failure to surface them. The public baseline is a fresh five-board snapshot. These are numerical score leads, not automatic permission or proof of a reproducible, conforming release gain.

The audit found **23** clean passing own receipts with lower elapsed scores than the current public entries, among **181** distinct clean passing own receipts. It inspected **2166** saved JSON files across tracked files, archives and ignored outputs. The current50-job service list adds no better run lacking an owning saved receipt. Three non-JSON diagnostic files are reported as unreadable; the saved receipt corpus is not a guarantee of the service's complete historical job inventory. Original bytes were preserved; legacy Windows JSON was decoded without replacement.

The process failure was the absence of a user-visible promotion queue. Research decisions against newer experimental controls were recorded, while comparison with the actual public entry was left without an owner and completion check. That allowed the5,725.7us MoE result to remain buried. There are also legitimate historical stops, including external-source contract gaps and the user's prefill deferral. This audit exposes both rather than calling every lower point estimate a missed release.

## Largest numerical leads

| Problem | Current public us | Lowest private us | Lower elapsed | Actual role |
|---|---:|---:|---:|---|
| MoE | 6,069.608 | 5,613.592 | 7.513% | external standalone MoE |
| BF16 projections | 3,161.919 | 3,071.044 | 2.874% | external projection source |
| Attention | 1,446.191 | 1,426.842 | 1.338% | external attention source |
| Decode layer | 10,639.426 | 10,058.874 | 5.457% | external G0 layer |
| Prefill | 318,441.991 | 297,780.607 | 6.488% | masked StreamK fresh-gate candidate |

The lowest MoE, BF16, attention and layer points above are external comparator checks. They are not drop-in owned release recommendations. The BF16 comparator rounds split partials to BF16 and needs contract resolution. Raw G0 layer lacks the required native attention BF16 output boundary; its repaired descendant is already public607. Attention's1.34% difference is inside the reported uncertainty. The strongest prefill point uses a stochastic masked backend and was completed after the user's explicit scope deferral. The receipts and their constraints remain visible below.

## The recovered MoE result that should have been presented

Own candidate-only check `46aafddc195c4c57` finished **November1 at03:29:17 Pacific**, about41hours before the user's question. It passed5official tests without flags and reported **5725.692982us**,2.348798659x,95%CI[2.313566679,2.384030639]. Its preceding same-process stage-retirement comparison `63f5a2a8ed2341d1` passed all three banks with six-load factors1.018493–1.028699. That matched result supports the retirement mechanism; the headline difference from the public score is cross-run.

The exact measured file is recovered as [recovered-stage-retire.py](evidence/2026-11-02-promotion-audit/recovered-stage-retire.py),364234bytes,SHA256 `fde6b1ee532f77d4d2d3fc6bae337e2ceeea9d7e07e458a42a382b787100891b`. The generator reproduces the registered hash using the pinned [upstream source](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_moe_w4a16.py), normalized source795ac642, and retained `tools/dl_g1_stage_ab.py --candidate-only`. Attribution and historical check-only markers are preserved in this recovered artifact. It is the historical measured file, not a newly approved public upload.

Its standalone32-token selection differs from the current G1b4 B4FIN control. Thus5,725.7us is a previously tested newer-engine version, not a separate official score for the exact control arm of e044. The current control's stronger matched results still warrant a clean standalone release qualification. The lost queue treatment does not close that release task. Public source-use scope, required numerical/order/lifetime checks and exact release bytes must be resolved and presented before asking for public approval.

## Every lower passing receipt

Finish times below are Pacific. All rows are private checks. `judged arm` identifies what actually produced the official score, which can differ from the experimental treatment discussed in its run log. The full owner, server payload identity and release reason for every row are in [promotion-queue.json](../releases/promotion-queue.json). A server payload hash is not asserted to equal the physical source-file SHA256.

### MoE

| Owning receipt | Finished Pacific | Official us | Lower than public | Judged arm | Release state |
|---|---|---:|---:|---|---|
| [b65c4c878014416c](../../muse_moe_w4a16/profiles/kel_exact.b65c4c878014416c.receipt.json) | 11-01 01:15 | 5,613.592 | 7.513% | external standalone MoE | source-contract-review |
| [46aafddc195c4c57](../../muse_moe_w4a16/profiles/stage-retire-candidate.46aafddc195c4c57.receipt.json) | 11-01 03:29 | 5,725.693 | 5.666% | stage-retirement candidate-only | release-qualification |
| [3c45788127e945c4](../../docs/steering/evidence/2026-11-02-cuda-group-scale/latest-status.json) | 11-02 17:40 | 5,726.825 | 5.648% | scale/group candidate2 | treatment-stopped-base-release-open |
| [3aa3885f6b734f8c](../../docs/steering/evidence/2026-11-02-cuda-issue-groups/status-wait.raw.json) | 11-02 17:11 | 5,744.726 | 5.353% | scale-only candidate2 | treatment-stopped-base-release-open |
| [f147d06a64504c1e](../../muse_moe_w4a16/profiles/g-stage-check.f147d06a64504c1e.receipt.json) | 11-01 05:51 | 5,746.540 | 5.323% | uninstrumented control | observer-invalid-base-release-open |
| [56e0c8b5634545e6](../../muse_moe_w4a16/profiles/expert_top100_pregamma.56e0c8b5634545e6.receipt.json) | 11-02 09:33 | 5,765.743 | 5.006% | pre-gamma candidate2 | treatment-stopped-base-release-open |
| [b38a9ed6dc9743e9](../../muse_moe_w4a16/profiles/gopt-ws-norm-2026-11-02/status-wait.raw.json) | 11-02 19:58 | 5,769.291 | 4.948% | WS norm candidate2 | treatment-stopped-base-release-open |
| [e044121f822d47b1](../../docs/steering/evidence/2026-11-02-moe-bqueue/status-wait.raw.json) | 11-02 20:28 | 5,856.634 | 3.509% | whole-unit queue candidate2 | treatment-stopped-base-release-open |
| [f0b875484960438b](../../muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/status-wait.raw.json) | 11-02 17:48 | 5,864.174 | 3.385% | NS4 candidate2 | treatment-stopped-base-release-open |
| [05e51ddeb7014dd9](../../muse_moe_w4a16/profiles/sc6_compare.05e51ddeb7014dd9.receipt.json) | 11-01 02:36 | 5,952.479 | 1.930% | BYTE control | diagnostic-control-base-release-open |
| [1c6ebc76b91d4b7f](../../muse_moe_w4a16/profiles/frontier.1c6ebc76b91d4b7f.json) | 11-01 02:45 | 5,960.105 | 1.804% | SC6 control | diagnostic-control-base-release-open |
| [63f5a2a8ed2341d1](../../muse_moe_w4a16/profiles/stage-retire.63f5a2a8ed2341d1.receipt.json) | 11-01 03:25 | 6,008.864 | 1.001% | repaired control | candidate-only-check-recovered |

### BF16 projections

| Owning receipt | Finished Pacific | Official us | Lower than public | Judged arm | Release state |
|---|---|---:|---:|---|---|
| [3c6a2baf7b4b4bf3](../../muse_bf16_proj/profiles/kel.3c6a2baf7b4b4bf3.json) | 11-01 01:34 | 3,071.044 | 2.874% | external projection source | source-contract-review |
| [bac995b3b94541c4](../../muse_bf16_proj/profiles/cgemv_ie.bac995b3b94541c4.json) | 10-31 23:11 | 3,116.503 | 1.436% | unchanged C3 control | unchanged-control-variation |
| [b103af15ba564ef6](../../muse_bf16_proj/profiles/rd16_ring.b103af15ba564ef6.json) | 10-31 23:34 | 3,144.612 | 0.547% | C3 control | unchanged-control-variation |

### Attention

| Owning receipt | Finished Pacific | Official us | Lower than public | Judged arm | Release state |
|---|---|---:|---:|---|---|
| [cd6805e7ebce45ee](../../muse_attention_decode/profiles/rohan.cd6805e7ebce45ee.json) | 11-01 02:44 | 1,426.842 | 1.338% | external attention source | source-contract-review |
| [80415fe736bb4890](../../muse_attention_decode/profiles/fp32fresh.80415fe736bb4890.json) | 10-31 23:37 | 1,435.840 | 0.716% | candidate2 validation harness | pure-counterpart-already-scored |

### Decode layer

| Owning receipt | Finished Pacific | Official us | Lower than public | Judged arm | Release state |
|---|---|---:|---:|---|---|
| [3a5f879df0d44825](../../muse_decode_layer/profiles/kel.3a5f879df0d44825.json) | 11-01 01:14 | 10,058.874 | 5.457% | external G0 layer | contract-repaired-descendant-public |
| [17383751c6a24010](../../docs/steering/decode32/evidence/D32-ATTN-01/status-wait.raw.json) | 11-02 16:29 | 10,625.699 | 0.129% | lookahead candidate | confirmation-unresolved |

### Prefill

| Owning receipt | Finished Pacific | Official us | Lower than public | Judged arm | Release state |
|---|---|---:|---:|---|---|
| [defbb042449d44e7](../../muse_moe_w4a16/profiles/humming_masked_fresh.defbb042449d44e7.json) | 10-31 23:20 | 297,780.607 | 6.488% | masked StreamK fresh-gate candidate | user-deferred |
| [f92ffc2953284a9a](../../muse_moe_w4a16/profiles/humming_masked_i64.f92ffc2953284a9a.json) | 10-31 23:15 | 299,350.076 | 5.995% | corrected masked candidate | covered-by-fresh-gate |
| [e9bd4269a6584df8](../../muse_prefill_layer/profiles/nozero_nostream_bounded.e9bd4269a6584df8.json) | 10-31 23:21 | 304,181.849 | 4.478% | deterministic bounded/nozero candidate | user-deferred |
| [2bd4f06042864771](../../muse_prefill_layer/profiles/nozero.2bd4f06042864771.json) | 10-31 23:09 | 306,796.692 | 3.657% | nozero/filtered candidate | pure-counterpart-needs-contract |

## Decisions the raw score alone would miss

- The scale-only, corrected scale/group, pre-gamma, NS4, WS norm and whole-unit B queue treatments failed their frozen matched gates against the newer control. Their exact treatments remain stopped. Their lower official point scores relative to the older public MoE entry still expose a separate base-release gap.
- The frontier and phase observers judged an uninstrumented control. Their headline scores are not wins by the observer or scheduling model.
- BF16 checks bac995 and b103 judged C3 control while the experimental changes lost. Pure C3 was already publicly tested; its later public repeat did not beat C2. These private point estimates are not withheld new algorithm wins.
- Attention80415 used a validation harness with allow_tf32=False. Its exact pure counterpart cbee36a8 was checked as d331aca81c554d70 and publicly submitted as c9bc5fbd85b440e2, which lost to the current best8914751e3b9f468b. This release path was followed.
- Owned prefill e9bd4269a6584df8 scored304181.849us,4.478% below the current public point, and passed its official/fresh deterministic checks. The run log pins pure sourceabce3925. The current `muse_prefill_layer/experiments/humming_nozero_nostream_bounded.py` hashes to6a55bddb instead; the exact old pure file must be reconstructed before release. Its recorded reason for no promotion is the user's explicit prefill deferral. This audit does not resume prefill or dispatch anything there.
- Layer lookahead17383751c6a24010 is only0.129% below the public point, with no valid matched confirmation and unresolved routing equality. It remains unconfirmed and unintegrated. No D32 work is resumed.

## Behavior patch

Root AGENTS now requires separate experiment and public-release dispositions, an owner and user-facing reason for every better clean private receipt, and a concrete release before a public approval question. The rule explicitly prevents a loss to a newer control from erasing that control's lead over the public entry.

`tools/release_audit.py` scans and deduplicates own terminal passing receipts, compares them with saved public snapshots and fails on a missing owner, judged arm, source identity, release state, reason, evidence or disclosure report. It excludes official failures/flags and does not turn lower point estimates into automatic release eligibility. The GitHub workflow runs an offline tracked-receipt check on pushes and pull requests. Tests reproduce the missing-disposition failure, a control-arm lead despite a failed treatment, invalid official results and incomplete dispositions.

The existing hourly tracker will receive this release check without changing its schedule or authority. The user-paused worker reminder remains paused. No new chat, agent, benchmark or public submission is created by the audit. Owners retain their existing implementation scopes.

## Continuing execution

The original user task remains active: measure minimally intrusive B4/MAIN phase/tail exposure, then implement and measure the next qualified MoE candidate. Sparse boundary source `tools/moe_phase.py` has compiled locally for B4/MAIN while the audit proceeds. The source adds boundary timestamps through the existing optional debug pointer and changes no arithmetic, publication or kernel parameter layout. No phase GPU job is accepted yet. Its observer cost must be priced before interpreting phases. The next diagnostic is planned to judge the uninstrumented current control officially, so its complete standalone score is explicit. No old queue treatment is reopened.

Validation in this report is author-checked. No independent reviewer was available under the no-new-agents instruction. Full inputs and coverage limits: [receipt inventory](evidence/2026-11-02-promotion-audit/receipt-audit.json), [public snapshots](../releases/public-snapshots), [queue ledger](../releases/promotion-queue.json).
