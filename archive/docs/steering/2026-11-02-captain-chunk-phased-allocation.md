# One chunk-phased G2 gather allocation — November 2, 2026

## Decision

Fund ONE fixed source-lifetime implementation and four-entry offline compiler screen, maximum30minutes. This explicitly reopens only a new gather schedule, not the stopped ed082519 implementation, and does not relax the zero-stack/zero-spill gate. No GPU, target/driver query, challenge CLI, packaging, public submission or production/default change is authorized. No subagents/new threads/workflows.

The preceding turn was progress: captain5798c533d23272e3583efeb10f215d3980c9075f independently verified all28 worker publication paths/27 archive entries and9CPU tests, changing the decision from a frontend witness problem to a precise fixed resource STOP. That did not improve the incumbent. This turn starts actual implementation of a different bounded schedule rather than repeating that status.

## Concrete evidence and hypothesis

Actual selected native SHA256 ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f, wgs_gather starting at1889 in the retained source. GB remains8. Existing source declares v[GB][2] and d[GB][2], loads both chunks for eight rows before the store pass. Their dimensional payload and pointer counts equal64+32 scalar32bit slots, excluding predicates, parameters and other live state. That is source accounting, NOT proof all slots overlap in emitted registers or that gather causes every spill. Existing all-four resource failures are real; their dynamic cost and causality remain unknown.

Proposed fixed schedule: outer chunk0..1 around the full q0..q1 batch traversal, j reset0 for each phase, one v[GB]/d[GB] batch at a time, same eight-row batch, identical column=tid+chunk*NT and bit-copy addresses. Suppress outer chunk unrolling to express the intended lifetime. This reduces source-dimensional batch storage to32+16slots. It is not a batch-width, cap, ring or tile sweep. No performance improvement is predicted from this count.

All writes still precede the single existing global proxy fence, CTA synchronization and row-count release block. The producer's publication condition is unchanged; padding is counted but not written. The two chunk column sets and row destinations are disjoint; routing/source activation inputs must remain immutable throughout both phases. No early release or new buffer is permitted. Keep every byte outside the gather function unchanged from the corrected parent.

Captain's independent [source witness](../../../docs/steering/evidence/2026-11-02-captain-chunk-phased/witness.py) passed24count cases1..24, all256compute tids/bothchunks for4096BF16 elements per actual row, plus171partial/empty/multijob/padded range schedules. It verifies equal modeled copy multisets and unchanged padded publication counts; source/gather hashes pinned in [receipt](../../../docs/steering/evidence/2026-11-02-captain-chunk-phased/verification.json). It is not validation of the worker's future transformed function or a full scheduler/progress proof. The worker must bind its own tests to actual generated source and test the missing-j-reset negative.

## Why this investment rather than another lane

The broader two-CTA mechanism is still the proposed performance source; phased gather only seeks a resource-feasible implementation without weakening numerical or ordering properties. Expected benefit remains uncertain. Earlier scenario sizing requires about11.4% net expert reduction on the affected long routes for3% whole-six improvement; it is not a forecast and would not by itself close the last observed top100 gap. A register-cap increase97..102 contradicts the pinned necessary two-CTA register model. Rings/geometry sweeps would disturb the corrected shared fit and require new proofs.

The scalar interleave alternative only rearranges a short epilogue, with equal opcode counts and no demonstrated critical-path improvement. Complete ruwwww adoption remains blocked by precision/progress/adoption eligibility; its legal long-dense policy has no demonstrated endpoint gain. Those are genuine alternatives, not manufactured ready candidates. ONE narrow offline resource discriminator is cheaper and more directly actionable now than redoing those intakes or building another runtime instrument. A second spill failure or lack of a complete qualification route changes the allocation away from this fixed treatment; no inertia-based rescue.

## Delivery, gates and ownership

Existing user-created worker task delivered by message47691dd8-7a07-4c30-a62d-afd53c45daff; T3 returned started ordinal11. [Worker thread](<t3-thread://v2/471a5a44-f247-4d6c-b35d-7f45e6f30552/1c3bc6ca-ab58-48bf-bcfe-f7c2de184b25>). Budget30minutes, concrete-blocker checkpoint15minutes. Worker owns uniquely named append-only transformer/tests/report/evidence, not README/shared ledgers or the stopped parent. Its new report will be2026-11-02-worker-g224-chunk-phased.md. Queue updates to captain, not steer.

1. Exact reversal and unchanged-outside-function proof; actual-generated mapping/negative tests covering partial jobs, padding, empty ranges, both chunks and publication counts.
2. Only the existing fixed NVRTC13/PTXAS13 recipe and attested layout adapter, MAIN/B4 both bools. Same source options, shared geometry, register ceiling, launch bounds and all arithmetic/fences. No compiler or parameter sweep.
3. ANY selected-entry stack/spill failure ends this exact treatment immediately. No second implementation rescue. Record resources, never timing-loss inference.
4. All-four zero-stack/spill is only a resource-screen survivor. Full progress, target numerical/replay/occupancy and payload constraints remain explicit before any separately authorized ordinary candidate check. No four-placement diagnostic harness is being built.

Existing researcher ordinal12 has independent read-only old-binary spill attribution running since08:00UTC, deadline20minutes. Do not duplicate it or claim preliminary notes as its final result. Its evidence can challenge this hypothesis and choose the next decision; no rescue or gate relaxation is authorized by attribution alone.

Goal remains active/unachieved. Incumbent public607 unchanged, zero new performance-qualified winners. This allocation consumes no judge quota and does not establish a new rank or submission result.

## 08:17UTC attribution addendum — hypothesis strengthened, gate unchanged

The researcher's corrected publication afeaa69078160680ad450c5b381cd73325a728aa is now available: [G2 spill attribution](2026-11-02-g224-spill-attribution.md). Captain independently verified all14 publication paths at that corrected tree against fetched origin/main and inspected both bool entry inventories; [receipt](../../../docs/steering/evidence/2026-11-02-captain-chunk-phased/attribution-verification.json). The initial d06b68b classifier had a bool-label binding defect caught and corrected by the researcher; do not use its superseded execution-region fields.

Both bools agree: B4 has76 gather local instruction sites,18 phase-B outer-segment and5 entry/drain,99total; MAIN79/7/2,88total. Six inner WGMMA stage natural loops contain no local instructions in the corrected attribution. PTX has no explicit local declarations/operations; these are assembler spills. Gather dominates the existing static site/operand-width inventory, strengthening the specific lifetime hypothesis, but phase-B state remains a separate pressure source. Static sites/widths are not measured dynamic traffic or latency.

Worker received this evidence as an active-run correction with NO expanded authorization: same one gather-only change, same30-minute/four-entry screen. Persistent phase-B spills still STOP it; no additional phase-B rewrite, cap change or weakened gate. The newly available attribution changes confidence in the source hypothesis, not candidate qualification or the incumbent. Researcher task publication is available while its terminal handoff is finishing; no new task was started.
