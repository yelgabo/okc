# Shared project steering

## CUDA optimizer / round-2 research alignment — actual optimizer plan not yet inspected

[Evidence-scoped alignment review](2026-11-02-cuda-optimizer-alignment.md), evidence df237d36: a narrow code-specific optimizer pass can proceed alongside round-2 research. The exact named optimizer plan/patch and full CUDA research report were not located in the inspected pushed files or available file searches, so this is **not a verdict on its current approach**. Use the actual selected source, avoid recounting shipped overlap repairs or assuming occupancy equals speed, and distinguish scoped losses from new mechanisms. The missing item is one current plan/diff, not another review framework. No pause, new allocation, handoff requirement, kernel/job change or submission permission. Decode32 Worker's live candidate and end-to-end authority below remain unchanged.

## Latest user instruction: research plans; Decode32 Worker implements and measures

**[D32-04: fixed implementation and measurement brief](decode32/2026-11-02-04-research-plan-worker-execute.md)** records the user's medium-effort worker setting and request for okc research to handle the deeper planning. Research owns architectural choices and concrete experiments; **Decode32 Worker retains coding, focused checks, private GPU submission, own-job analysis, confirmation, integration and commits, without routine handoffs or per-step approval.** Finish the already-selected within-request next-tile QK score-carry attention candidate. Do not restart literature review, architecture selection or an independent-review chain. Known correctness/progress defects still block dispatch; an unproved speedup does not block a bounded exploratory check.

The [worker execution plan](decode32/2026-11-02-01-full-decode-attention-plan.md) and [watched inbox](decode32/README.md) now remove the obsolete Driver measurement handoff. Existing public permissions and non-D32 ownership are unchanged. Evidence checkpoint a3ba392c; MID is terminal and parked, so no wait on its old allocation. This is a published design/execution directive, not a new benchmark, model-setting change by research, or verified worker acknowledgment.

## Standing execution authority: Decode32 Worker owns batch32 end to end, not the whole project

**[D32-02: end-to-end batch-32 ownership](decode32/2026-11-02-02-end-to-end-ownership.md)** remains the direct user execution instruction, including the explicit clarification "in terms of optimizsing batch-32 stuff" / "not EVERYTHING". Decode32 Worker owns D32 implementation, compilation, correctness tests, private GPU checks, result analysis, iteration, integration and commits. **No routine Driver/Developer handoffs or per-step captain approval.** This supersedes the source-only, attention-only and sole-Driver dispatch restrictions below for D32 only. OK Driver/OKC Developer keep unrelated work. Correctness, live shared quota/job preflight, preservation of other load winners and existing public-submission permissions remain. D32-04 above refines architectural planning ownership without removing execution autonomy.

Continue the acknowledged attention work and reuse existing MID evidence. Publication of D32-02 was reconciled with concurrent `65b94dcc`: MID job a6fe2769779e4299 had already been accepted and stayed with its original owner through terminal/archive. That exception is now complete; see the terminal below. Do not cancel, mutate, resubmit or create a second poller for a completed job. Subsequent D32 experiments belong to Decode32 Worker. Read the [watched inbox](decode32/README.md); no new acknowledgment of the revised planning split is claimed. No kernel or GPU action was made for this instruction update.

Earlier ownership statements below are retained as history and do not override D32-02/D32-04. This is a scoped D32 transfer, not replacement of the project captain or other benchmark owners.

This is the stable cross-session entry point. Read the latest human priority, its evidence snapshot and subsequent commits before continuing. Existing correctness requirements and public-submission permissions remain in effect. A report is not an additional GPU dispatch.

## H200 playbook reference and validation repair

[November 2 research intake and code application](2026-11-02-cuda-research-application.md)
preserves the complete upload and maps every section and recommendation to
[focused reference notes](../research/cuda-h200/README.md), selected source and
earlier experiments. Five MoE compiler wrappers now propagate failures and retain
logs. CPU regressions and a local CUDA 12.8.1 device build were checked. No runtime
kernel, target-performance claim, GPU dispatch or D32 ownership change is included.
Root instructions load the reference index for future CUDA work.

## D32 attention source and direct measurement

D32-04 is read and execution continues directly. The [score-lookahead experiment](decode32/2026-11-02-01-full-decode-attention-experiment.md)
pins candidate `5a580c7a`, six focused CPU tests and the fixed CUDA 12.8.1 device
builds. The [compact ledger](decode32/STATUS.md) records the source and own job.
Target results remain pending; no intermediate handoff is required.

## Completed pre-existing MID measurement

**[MID terminal: valid, no useful D32 signal](2026-11-02-driver-d32-mid-terminal.md)** — own a6fe2769779e4299 passed13tests/12benchmarks/12completeprofiles plus actualMID/native56replays. D32C32=12.752280ms versus unchanged94ef12.324230ms (descriptive3.473%slower,differentinputs/node); wholelayer2.089423487x. Exacte607/870649LF PARKED, no extra confirmation/public action. Original root wait/archive is complete and its one-time ownership exception retires. **Subsequent D32 work belongs directly to Decode32 Worker under D32-02**, no routine Driver/Developer allocation/handoff gate. [Watched result/ledger note](decode32/2026-11-02-03-mid-result.md). Fresh afterreset20private/10scored/no-live; overall/layer379,public607unchanged.

## Earlier assignment: Decode32 Worker — full-context attention

**[D32-ATTN-01: implement one isolated full-decode attention candidate](decode32/2026-11-02-01-full-decode-attention.md)** was posted for the user-created **Decode32 Worker**. The user clarified that it was waiting for tasks, not already implementing one. Start source/code work on T32/q_len1 full-context attention, independently of the existing MID MoE comparison. Its candidate/tests/Driver-handoff endpoint is now superseded: Decode32 Worker takes its work through measurement and integration itself. Preferred cross-tile-overlap hypothesis must first be reconciled with actual selected code and prior scoped failures; no gain is assumed.

**Historical ownership split, superseded for subsequent D32 work:** Decode32 Worker owned the isolated attention track; OKC Developer retained MID-at-T32 MoE; OK Driver was sole judge dispatcher/integration owner. Actual attention pickup is recorded in the [status file](decode32/2026-11-02-01-full-decode-attention-status.md): Decode32 Worker acknowledged at23:55UTC, published69cf82b3. Its isolated implementation plan is active; no GPU qualification was recorded at that checkpoint.

## Decode32 task inbox

[Add research assignments and direction here](decode32/README.md). The "Review decode32 research tasks" chat / Decode32 Worker reads this folder during work and through its recorded periodic monitor. Its former waiting state is superseded by the acknowledged assignment. [Setup and monitoring contract](2026-11-02-decode32-inbox.md); older ownership restrictions yield to D32-02/D32-04.

## Earlier captain and MoE owner assignment

**[November 2: Explicit decode32 owner directive and pickup ledger](2026-11-02-decode32-owner-directive.md)** — the user reiterated "make it priority for them please". D32 remains the primary performance objective. The later D32-02 instruction changes its end-to-end owner and is now at the top of both root instruction files, `AGENTS.md` and `CLAUDE.md`.

**Earlier captain pickup is confirmed:** concurrent commit `5b4a729a` explicitly adopted D32, deferred the unstarted broad dense-backing proposal, and allocated the fixed MID-at-T32 comparison below. Actual app delivery after5b4a729a remote verification and explicit23:43UTC developer pickup/firsttools/d32_mid_family.py are recorded in the directive ledger and captain coordination evidence. Preserve this completed work and already-accepted job during the scoped ownership transition; do not duplicate them.

## Performance objective: concentrate on decode batch 32

**[November 2: Decode32 priority and weighted improvement plan](2026-11-02-decode32-priority.md)** — evidence `b325b965`. The user explicitly selects the `22.149 ms / 11.888 ms / 1.8632x` decode-batch32 row as the next objective. It is T32/q_len1, not verify32/T256. The row is `46*sliding + 16*full`, approximately 60% sliding and 40% full. First milestone: 2x, requiring 0.8135 ms saved from this load; 2.2x requires 1.8203 ms. These would contribute about 1.19%/2.81% to the six-load layer score if everything else stayed fixed, not guaranteed overall rank movements.

Prioritize expert/B4 work in both contexts, then batch32 full-decode attention. Preserve other load winners and test a prospective shape-specific candidate if warranted; do not rewrite the original test's outcome. PR-only T128/256 and T1/T8 work are not the next primary D32 assignments. The proposed 8-us expert saving in both contexts plus 20-us full-attention saving is a target budget, not measured performance.

## Existing MID work and calibration — reuse, do not duplicate

**[D32 result and fixed MID-family comparison](2026-11-02-driver-d32-mid-comparison.md)**: direct-weight own0b684a40 is terminal VALID but D32 sliding180.848us/full337.856us, weighted13.724704ms versus own baseline22.145344ms (1.61354x). Exact direct lane and held pair stopped; no D32 rescue signal. Overall1.926423591x does not pass its original gate. Current prior-control comparison remains descriptive across inputs/nodes/drivers.

**Recorded qualification gate:** [unchanged calibration terminal and fixed MID qualification](2026-11-02-driver-d32-control-terminal-and-mid-gate.md). Own94ef is VALID2.091747685x; D32 C32=12.324230ms/1.811791x. Existing developer finished exacte607fb89 MID-at17..32, root CPU/source review accepted one ordinary private exploration after source publication and fresh no-live/quota. The accepted MID ID is now listed above; no target result was established at this checkpoint.

M33/other five loads stay control. Broad dense-bank GENERIC work proposed concurrently was **not delivered or started** and remains an unallocated secondary reserve under the D32 instruction. [Full direct terminal and priority addendum](2026-11-02-driver-direct-weight-terminal-and-dense-backing.md). Historical23:33UTC snapshot:overall/layer376,7private layer/no-live; no new public authority. **Unchanged calibration94ef698e91964dfa is terminal VALID**; captain archived fullraw/profile/D32/all-six evidence. Refresh live capacity rather than reuse old quota numbers. Existingdeveloper explicitly picked upMIDfamily at23:43UTC/firsttools/d32_mid_family.py.

## Earlier execution records

**[Direct-weight private check registration: 0b684a4065354fc5](2026-11-02-driver-direct-weight-first-check.md).** Its accepted/pending text is historical: the terminal and stop are recorded above. Do not resume or duplicate this completed job. The file changed B4/MAIN weight delivery and did not earn promotion.

**[K128/shared-A terminal history and direct-weight allocation](2026-11-02-driver-shared-a-terminal-and-direct-weight-next.md).** Preserve the completed losing implementations and their scoped stops. D32 focus and ownership changes are not evidence that those old variants won.

## Full prior index and report history

**[Complete preceding steering index, preserved byte-for-byte](index-history-through-2026-11-02-2325.md)** contains prior handoffs, allocation history, source studies, older pointers and report-history links. Its blob is exactly `3d09977417e4d5fe0c18f99ac82f91a2b2454ff6`, the README at `b325b965`. It remains in this directory so relative links keep their meaning. Earlier reports remain intact.

| Report | Evidence | Scope |
|---|---|---|
| [H200 research intake and application](2026-11-02-cuda-research-application.md) | ba9/c7 source; complete attachment; CPU regressions and local CUDA 12.8.1 build | Eight reference documents, full section/recommendation coverage, compiler-status repair; no runtime optimization or target gain claimed |
| [CUDA optimizer alignment](2026-11-02-cuda-optimizer-alignment.md) | df237d36; native repair inventory; K128 terminal | Provisional guidance; named optimizer plan and exact CUDA report not retrieved; no candidate verdict or ownership change |
| [D32 attention experiment](decode32/2026-11-02-01-full-decode-attention-experiment.md) | ba9 control; 5a580c7a candidate; six CPU tests and fixed device builds | Direct private measurement under D32-04; target results pending |
| [D32-04: research plans, worker executes](decode32/2026-11-02-04-research-plan-worker-execute.md) | a3ba392c; user's medium-effort instruction; existing L1 score-carry source brief | Fixed candidate design, focused checks and direct private measurement; no repeated review/planning or handoffs; correctness retained |
| [D32-02: end-to-end ownership](decode32/2026-11-02-02-end-to-end-ownership.md) | 28346bf; latest user instruction; publication reconciled with65b94dcc | Batch32-only implementation through private checks and integration; no recurring handoffs; rest of project unchanged |
| [D32 MID terminal / one-time retirement](2026-11-02-driver-d32-mid-terminal.md) | own a6fe2769779e4299/e607fb89; control94ef | VALID13/12/12/native56; no usefulD32signal,exactPARK; raw/profiles/portablederivative archived; futureD32execution directlynewowner |
| [D32 MID accepted check](2026-11-02-driver-d32-mid-first-check.md) | own a6fe2769779e4299; source28346bf5/e607fb89 | One pre-existing ordinary private exploration; original owner finishes this accepted job; target gates open |
| [Calibration terminal and MID qualification gate](2026-11-02-driver-d32-control-terminal-and-mid-gate.md) | own94ef698e91964dfa; exactMIDe607fb89; independent source/CPU review | Unchanged calibration valid2.091747685x; D32 C32=12.324230ms,1.81179x; source qualification history |
| [Fixed MID implementation](2026-11-02-worker-d32-mid-family.md) | ba9 control; MIDb8; e607fb89/870649LF | Existing MID for17..32, unchanged shaders/math; actual entry/cache/neighbor CPU gates |
| [D32-ATTN-01 implementation task](decode32/2026-11-02-01-full-decode-attention.md) | a4192dec; existing D32 plan/continuity source | Original attention assignment;23:55UTC pickup69cf82b3; handoff restriction superseded |
| [Decode32 inbox setup](2026-11-02-decode32-inbox.md) | bdbf35bf; user request | Shared task folder/periodic monitoring; ownership restrictions superseded for D32 |
| [Decode32 owner directive](2026-11-02-decode32-owner-directive.md) | b9ca9dff and5b4a729a | Historical P0 and captain/developer pickup; current owner is Decode32 Worker under D32-02 |
| [Captain D32 MID comparison](2026-11-02-driver-d32-mid-comparison.md) | b9ca9dff; own0b terminal; sourceba9/b8 | DirectD32 closure, fixedMID17..32 exploration, no shader/math/packing sweep |
| [Direct terminal and superseded broad proposal](2026-11-02-driver-direct-weight-terminal-and-dense-backing.md) | own0b684a40;368b85c9;23:33UTC | ExactPARK; broad P1 superseded before delivery |
| [Decode32 priority](2026-11-02-decode32-priority.md) | b325b965;public607;cc474;user row | D32 targets, weighted budgets and all-load regression protection |
| [Previous full index](index-history-through-2026-11-02-2325.md) | exact former README blob3d099774 | Earlier execution, research and report history |

## Publishing future steering

Check substantive steering into `docs/steering/YYYY-MM-DD-<topic>.md` or the user-designated `docs/steering/decode32/` inbox for D32 tasks, update the latest pointer/history, and commit/push/verify in the same task. Pin evidence and owning jobs; distinguish measured effects, hypotheses and unknowns. Preserve concurrent work and earlier reports; never force-push. Public-submission permission is not expanded by this documentation. Decode32 Worker owns D32 execution/integration under the latest user directive; the rest of project coordination remains with its existing owners.
