# Top100 profile priorities: fresh incumbent, weighted opportunity, missing exposure

November 2, 2026. Research sub-agent; evidence allocation `a705900`, inspected checkout HEAD `71eb71c1a67335d704e53d00632b4bf92ae38845`, branch `t3/top100-profile-priorities`, repository `https://github.com/yelgabo/okc`. Finite evidence pass, no children, GPU work, installs, check/submit POST, default edits or public promotion. Captain retains dispatch ownership. Existing T1 three-launch and hardware-compression work are not duplicated or reallocated.

## Decision and hypothesis

**Prioritize the existing longer-token expert stream for broad layer opportunity, with the existing full-context attention stream as a distinct second opportunity.** This is an allocation recommendation based on selected routes, endpoint weights and activity scale, not a measured new gain. T16 deserves its own decision: it is a skinny MID expert build, not the T32/T128/T256 WGS build. Attention's large full-context activities have only 16/62 of each load's weight; optimizing sliding expert work affects 46/62 as well.

Hypothesis: a distinct mechanism that reduces complete expert-call latency across T16/T32/T128/T256 has more six-load score leverage than an equal fractional reduction of the four full-context attention activities. Neither mechanism's *exposed* saving is measured here. A valid matched complete-call result, route mismatch, implementation-specific correctness blocker, or source/compiler evidence that makes one treatment materially cheaper can reverse this ordering. No measured portfolio currently fills the top100 gap.

## Exact validation and provenance

- Fresh incumbent **`cc474883117947c0`**, check/yelgabo, H200, done/passed/return0, clean checks, **13 tests / 12 distinct benchmarks / 12 matching profiles**. Recomputed score **2.109201357227871**, CI `[2.077563336869453, 2.1408393775862886]`; log retains **56 verified startup replays** and native backend selection.
- Public **`607eb52bdaf24160`**, run/yelgabo, likewise validated 13/12/12; score **2.1260826764231093**. Both owning receipts associate server content `5cf950e1819c8c2bf0edf9bfdc0ccf05b191759c0a8263a45cd962f4e934ff5e` with the same payload. This is a service identity, not a physical file hash.
- Local exact incumbent `muse_decode_layer/cand/g1b4_stage_release.py` is **869076 LF bytes**, SHA256 **`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`**. Source and receipt association agrees with the captain's original upload attestation. This pass did not retrieve the service payload independently.
- Historical SGLang-as-submission **`7d42771c3384449d`**, created **2026-10-28 03:18:17 UTC**, independently validates 13/12/12 and weighted score. All normalized profile reports equal its owning status reports. Normalized benchmark records equal owning records except omitted empty `flags: []` fields; the offline audit checks this explicitly.
- Existing public-to-fresh decomposition independently reproduces own-latency factor **1.0197592983**, reference-baseline factor **0.9728373126**, official score ratio **0.9920598952**. Different seeds, prepared state, node and time prevent causal attribution. Score CIs are not treatment-effect CIs.

Raw source/job inputs remain in their existing owned archives. [Audit JSON](../../tools/top100_profile_evidence/audit.json) pins their file hashes, source policy selections, all12 activities, all6 weights, comparisons and limitations. [Recomputed hotspots](../../tools/top100_profile_evidence/hotspots.md) is the actual output of the supported CLI, not a handwritten substitute. Its `floor` columns are **analytic model values**, not verified physical lower bounds, recoverable latency or proof of hardware saturation. No new best landed, so no shared hotspots/run/ruled-out files were edited.

## Read-only live state

Existing access recipe `docs/steering/2026-10-31-post-login-experiments.md` identifies `wsl -d Ubuntu --exec okc`. Local help confirmed the supported read commands. No token/config contents were read or printed.

At **15:28:26–15:28:29 UTC**, authenticated CLI receipts report:

| Item | Observation |
|---|---|
| Overall | yelgabo rank **356**; rank100 **keell0renz**, solved4; API supplies no numerical overall cutoff |
| Layer | yelgabo rank **356**, displayed2.1261x; rank100 **keell0renz**, **2.2659x**, job `232de027eda1481f` |
| Layer quota | **0/10 private checks left**, **5/5 scored left**; live limits differ from older coordinator defaults |
| Jobs | **50 recent jobs returned**, none queued/running; this is not a claim about unreturned history |

Receipts: [quota](../../tools/top100_profile_evidence/quota.json), [jobs](../../tools/top100_profile_evidence/jobs.json), [overall](../../tools/top100_profile_evidence/overall.json), [layer](../../tools/top100_profile_evidence/layer.json). Each preserves command, UTC start/end, return code and returned JSON. Overall was queried separately; matching ranks at this instant do not establish the overall scoring formula. Using the rounded layer cutoff and exact public score gives **1.065762882**, about **6.5763% score improvement** or **6.1705% uniform endpoint reduction** with fixed baselines. Threshold movement and baseline drift remain material.

## Six-load sensitivity and activity scale

Let `C=(46*t_sliding+16*t_full)/62`. With fixed baselines, score factor is `product(C_old/C_new)^(1/6)`. The derivative for one microsecond removed from both cells of one load is `100/(6*C)` percent log-score; for a single sliding/full cell multiply by46/62 or16/62. These derivatives describe endpoint sensitivity, **not removable work**.

| Load / tokens | Fresh C us | Weighted expert activity us | Full attention activity us | % log-score per 1us whole-load reduction | % log-score per 1us full-cell reduction |
|---|---:|---:|---:|---:|---:|
| D1 /1 |78.765|27.423|11.7|0.21160|0.05461|
| D16 /16 |152.016|69.561|88.6|0.10964|0.02829|
| D32 /32 |191.898|75.390|171.4|0.08685|0.02241|
| V1 /8 |112.682|59.658|14.8|0.14791|0.03817|
| V16 /128 |261.713|146.142|104.9|0.06368|0.01643|
| V32 /256 |351.080|197.339|185.2|0.04747|0.01225|

The activity table is deliberately not an additive latency breakdown. For example, V1full's expert activity is85.7us against an118.2us adjusted endpoint, while its profiler contains other activities too; profile collection and endpoint samples are separate. Overlap, launch effects, route variation and profiler perturbation are not resolved by summing rows.

Conditional arithmetic only: if endpoint reductions actually equaled10% of the named activities, longer-token experts across both contexts would produce **1.03428x**, four long full-attention cells **1.01048x**, and both short-load experts **1.01508x**. These are independent sizing scenarios, not predictions, not upper bounds, and must not be added. The short scenario does not authorize a second T1 worker or expansion to T8.

For a measured1% whole-layer gain, equal absolute endpoint savings would need approximately **3.208us on each of four long loads**, **8.817us on each of V16/V32**, or **2.726us on each of D1/V1**. If only four long loads carried the current layer gap, the corresponding equal-absolute saving would be **19.653us each**; if only V16/V32, **52.041us each**. These equations expose the unfilled objective, not available slack.

## Selected source and all12 profile routes

The offline audit executes only extracted literal policies and three pure selection functions; it never imports the submission or calls CUDA. Source selections are corroborated by all12 `okc::moe_fused` and `attn_kernel` profile signatures and the fresh log's native/PRMT/SC6/Lt-plan evidence. They are not a new binary or resource audit.

| Load | Expert route, both contexts | Expert activity sliding/full us | Attention activity sliding/full us |
|---|---|---:|---:|
|D1|TINY / helper0, NW8, FINPF|29.1 /22.6|5.6 /11.7|
|D16|MID / helper0df, TINY+SPLIT, NW8, FINPF|68.4 /72.9|7.6 /88.6|
|D32|B4 / WGS, NW16, B4FIN, SC6W|73.3 /81.4|9.6 /171.4|
|V1|TINY2 / helper0, NW8, FINPF|50.6 /85.7|7.2 /14.8|
|V16|MAIN / WGS, NW16, SC6W|143.2 /154.6|11.7 /104.9|
|V32|MAIN / WGS, NW16, SC6W|201.7 /184.8|15.4 /185.2|

All select PRMTSC; builder adds DQ7, sm90a/O3, NS3/FNST4/NSF3. WGS stage-retirement is already baked into the exact source (`prev` activation-slot release after `wg_wait<1>`, final drain before last release); it is not another prospective gain. MID's `@moe_helper0df` source marker is explicitly preserved. Whole-module source hash pins source transforms beyond the extracted policy audit.

Four long full-attention cells use **264-CTA flat plans**, nc1/cs0, decode `attn_kernel<1,true,0>` and verify `<8,true,0>`. `_TUNED7` has no override for these four full shapes. Decode uses w0=1024 and noncooperative flat policy; verify uses cooperative flat policy and w0=1135. The common launcher owns scratch explicitly, and `_pdl` is disabled by PDL=False. The fresh native log selects Lt `o/pr/qkv/un` at128/256. The short fused pooler/router signature remains in all four short cells. A longer-token treatment cannot assume all six loads share one native implementation or that PR is all expert work.

## Distinguishing next measurements; no dispatch allocation

1. **Longer-token expert owner:** compare the one bounded new treatment against this exact complete incumbent expert callable, using the same actual layer-derived routing/input/weights and independently prepared duplicate controls. Separate T16 MID from T32 B4 and T128/T256 MAIN; test both contexts, empty/local routes and fresh graph replays. Hold preparation, packing, suffixes, final normalization and consumer order fixed unless the mechanism requires a change; include every added launch/scratch/reset/lookup cost. Measure component and complete layer endpoints separately. A faster component with a flat/slower layer distinguishes hidden work or integration cost from useful saving. Preserve per-expert normalization and BF16 boundaries. Confirm on heldout routes before selecting winning shapes and measure the combined dispatcher. No sum of component deltas certifies it.
2. **Attention owner:** first require a concrete distinct copy-issue/address/lifetime treatment, not another named knob. Use the complete incumbent attention callable and common BF16 output boundary, identical real page mapping/sequence lengths and soft layer distribution; coherent standalone cases remain a separate qualification gate. Freeze grid/split choice in the attribution comparison, retain fences and lifetime/progress proof, and compare independently prepared duplicate controls with fresh retained replay outputs. Use uninstrumented endpoint timings first. If attribution needs a trace, require demonstrably small observer overhead before reading it causally: the earlier leader-wait trace failed every row's perturbation gate. A complete-call gain plus whole-layer gain is the useful result; unchanged copy-only or isolated wait counts cannot establish exposed time.
3. **Captain:** retain existing source/compiler/numerical gates and preregister the future candidate's actual acceptance criterion before dispatch. Suggested new-mechanism decision is all-cross whole-six gain exceeding twice duplicate-control log noise, supported by heldouts and no affected-cell regression greater than max(0.5us,1%) unless explicitly justified beforehand. Report the absolute gain and confirmation cost; this recommendation does not weaken any existing frozen1% gate. Invalid route, replay, ownership or observer evidence stops interpretation. Valid subthreshold/losing treatment stops the exact lane without a sweep or blind repeat.

Specific closures read and retained: G224 `e1271ad5e5e34b24` (unmatched descriptive long regressions); timed-SiLU `dc62cff972934f54` (parked exact fully timed treatment); matched RMW `c5e4f89672f743bb` (frozen gate fail); serial `e4c8f00532b74d15` (not causal family rejection); staged experts `9d94ac75f4724b52` (lost fused phase overlap); finalizer `1c6ebc76b91d4b7f` (sub1% opportunity under its gate); full-attention scope and failed wait trace `3f44f891d0ef47db`; QK two-chain `fe75e566c1ae4d51`; BF16 VP2 `c79ddddb840a4fb7`. Existing owners must name new distinguishing evidence before reopening those exact lanes. The compression build failure `60b38a62a135442c` remains setup-invalid, not a measured compression loss.

## Is a fresh SGLang baseline profile needed?

**Not for this incumbent-priority decision, and not as a duplicate quota spend now.** Fresh `cc474` already includes current SGLang benchmark endpoint times for all12 cells; it profiles only our submission. Historical baseline attribution identifies FA3/Marlin/Lt activity families, but it cannot supply current exposed slack.

The old and fresh owning evaluator receipts have identical hashes for `baseline.py`, `reference.py`, `task.yml`, `challenge/checks.py`, `challenge/nvfp4.py`, and the listed harness files. **`configs/muse-spark-open.yaml` changed** (`98f07230...` to `60e24254...`), as did evaluator code/image identity. This pass has not obtained the old/new config bytes or established the semantic effect; that is a precise gap, not proof that distributions are identical. Different jobs also have different seeds and H200 nodes.

A fresh baseline-as-submission profile is needed before claiming present SGLang per-kernel proportions, a changed reference route, or a baseline mechanism as the basis for a new optimization. First inspect the pinned config delta if available read-only. If that question matters to the next decision, captain may allocate one current baseline profile after capacity/permission reconciliation and archive its complete receipt/source/config identity. It is lower value than a valid new-mechanism matched endpoint measurement when the decision concerns only our kernel. No baseline job is requested or launched here.

## Reproduction, limits and handoff

From this checkout root, Python standard library only:

```powershell
python tools/top100_profile_audit.py
python tools/hotspots.py docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json --baseline muse_decode_layer/baseline_profile/profiles/baseline.7d42771c3384449d.json
wsl -d Ubuntu --exec okc problems --json
wsl -d Ubuntu --exec okc jobs --json
wsl -d Ubuntu --exec okc leaderboard --json
wsl -d Ubuntu --exec okc leaderboard muse_decode_layer --json
```

The audit saves UTF-8 hotspots output directly, avoiding PowerShell redirection encoding differences. API commands above are read-only; rerunning them creates a new dated observation, not a reproduction of the old board snapshot. Archived receipts preserve exact invocations and timestamps.

This pass reconciled AGENTS, steering index/latest research, coordinator Research method, current run ledger and relevant ruled-out entries. No broad public-source survey was renewed: current bounded assignment is measurement evidence, and the latest comparator research/access limits remain in force. Missing evidence: current config delta, causal exposed expert/attention savings, complete matched candidate endpoints, new candidate replay correctness/resources, and a numerical overall threshold (not supplied by its API). No harness was built. Owned output is this report, the narrowly scoped offline audit and its evidence under `tools/top100_profile_*`, plus the requested branch-local steering index entry. Captain owns the next allocation; implementation owners own the bounded treatments; report publication is not qualification or release permission.

## Parent integration and reproduction, November 2

Parent integrated delegate commit `80eef4dd2f4bdce0ed60bd5dc2c21ea67661c56e` into main after reconciling concurrent monitor `f853759`. Independently reran the actual audit with output redirected through its Python OUT variable to a temporary directory, preserving the delegate's frozen evidence. Ten result groups matched exactly: source hash/size, receipt validation, public/fresh comparison, evaluator-file changes, selected full-attention plans, all12 cells, six loads, conditional scenarios and cutoff ratio. Result: PARENT_REPRODUCTION_PASS. This is offline reproduction, not another target profile capture.

Accepted priority is longer-token expert work first and full-context attention second, conditional on actual treatment feasibility and matched endpoints. Both existing implementation delegates received the report pin, workload weights and selected-route caveats without timebox or permission expansion. In particular, T16 MID must not inherit a WGS-only result, and full-attention gains must respect their16/62 weight. Both message deliveries returned steered/running; no new agent or restarted task was created.

The active goal's profile-validation step is complete; candidate implementation/review is now the active step. Rank356 and zero private capacity remain timestamped15:28-15:30UTC observations, not top100 completion. No baseline profile was repeated: the evaluator manifest confirms a config hash change but supplies no old/new config bytes, so its semantic effect remains unresolved. Existing code/reference/task hashes agree. No claim of current baseline per-kernel attribution or equivalent input distributions follows.

Next gate: receive the bounded candidate artifacts, check actual complete callable and numerical/order contracts, and select one eligible matched target experiment when capacity/permission allows. This report does not silently authorize scored substitution. The latest delegated profile entry in the shared index remains the entry point; old evidence and concurrent work were preserved.

## November 2 source-description correction

Parent verified the expert delegate's correction against exact incumbentba9e327c at integration7f0bd5a. In the selected MOE_CUDA_SRC, the inner issue loop uses wg_wait<1>, but the stage-end publication at native lines2191-2195 performs wg_wait<0>, fence.proxy.async.shared::cta, then mbar_arrive1 on BX_E+xs (the current activation slot). The earlier section's parenthetical description of previous-slot retirement after wait1 is incorrect. The pinned source, route/latency data and allocation ranking remain unchanged. No barrier edit or renewed retirement experiment is proposed. Historical reports remain preserved; this dated correction supersedes that parenthetical only.
