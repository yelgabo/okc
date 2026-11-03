# Retained lessons about how we work

November 1, evidence snapshot a9e44b8. The user asks for general working principles rather than an experiment history. This note summarizes the shared operating lessons recorded in AGENTS.md, CLAUDE.md, the coordinator method and dated retrospectives. The active G integration-first objective remains in force.

## General principles

1. Own the discovery work. Search for stronger methods independently of user-supplied names, follow primary sources and inspect beyond the first few results. The user should not have to supply the decisive repository. Record search scope so a failed lookup does not become an assertion that nothing exists.
2. Keep the user's objective above the inherited plan. Revisit the roadmap when evidence changes. Judge effort by its contribution to the requested outcome. A familiar implementation is a starting hypothesis, and a task's presence in the backlog does not prove it is the best next task.
3. Think across the complete system before tuning a part. Ask what work, data movement and dependencies are actually necessary. Estimate the possible end-to-end benefit and cost before investing. A large local measurement or peak-hardware number does not establish a large system opportunity.
4. Use external work as a working comparator and a challenge to our model. Test the complete implementation before decomposing it into familiar ideas. Preserve its provenance and distinguish published claims from reproducible measurements. If our reconstruction loses, the original win remains an open question.
5. Change as little as practical when integrating a stronger baseline. Keep a change inventory, distinguish required properties from our chosen implementation, and measure regressions as we introduce changes. Preserve performance during integration rather than assuming changes are free.
6. Design experiments to decide between explanations. State the hypothesis, control, measurement, expected significance and stop condition first. Hold unrelated variables fixed. Separate a performance observation from its cause and record inconclusive evidence explicitly.
7. Validate the measurement as well as the implementation. Check independent mutable state, fresh inputs, replay behavior, observer overhead, actual selected paths and run ownership. Source intent, successful compilation and passing numerical tests answer different questions.
8. Keep correctness properties explicit. Required precision, ordering, progress and lifetime rules must be proved or tested by the appropriate method. Search for faster implementations preserving those properties. Do not charge every conservative engineering choice to correctness without evidence.
9. Record scoped failures and reopening evidence. A losing implementation closes the tested design under the measured conditions. It should prevent repetitive tweaking while leaving a clear trigger for a genuinely different design or changed workload.
10. Maintain a real candidate funnel. Develop independent, relevant hypotheses and source proofs in parallel. Distinguish ideas, implemented candidates, controls, qualified measurements and deployable wins. More audits or variants do not automatically mean more candidate breadth.
11. Treat quota, time and coordination as resources. Use authorized measurement routes promptly when candidates qualify. An exhausted private quota does not itself prevent a scored measurement. While external execution is unavailable, resolve relevant source, compiler and preparation blockers. Keep a single dispatch owner and exact IDs.
12. Make handoffs usable by the next session. Store decisions, evidence, supersession, unknowns and next actions in the repository. Read the latest report before acting. A durable lesson is effective only when it changes the next decision; retrospective agreement is insufficient.

## Current application and limits

The immediate priority is explaining the G0-to-fork performance discrepancy before unrelated optimization. Maintain multiple directly relevant investigations while respecting current source/CPU-only work and service availability. Stop proposals without sufficient opportunity, a legal implementation, a discriminating experiment or a clear owner. Broader research resumes when the discrepancy is resolved or the user changes the objective.

These lessons are durable repository guidance and evidence that future sessions can load. Their existence does not demonstrate reliable application. Prospective checks should ask whether the agent independently found a useful lead, changed its plan when evidence contradicted it, avoided reopening a scoped loser, prepared a genuinely distinct candidate, and delivered the requested verified outcome.

## Provenance

[Research ownership retrospective](../../archive/docs/steering/2026-11-01-research-ownership.md), [discovery and reasoning review](../../archive/docs/steering/2026-11-01-discovery-and-reasoning.md), [campaign accountability](../../archive/docs/steering/2026-11-01-campaign-accountability.md), [integration-first instruction](../../archive/docs/steering/2026-11-01-g-integration-first.md), [candidate funnel](../../archive/docs/steering/2026-11-01-g-candidate-funnel.md) and [minimum-delta plan](../../archive/docs/steering/2026-11-01-g-minimal-delta-plan.md).

The technical evidence behind those general principles remains in the owning ledgers, including public-comparator discrepancy 3a5f879df0d44825 versus607eb52bdaf24160, paired pipeline result63f5a2a8ed2341d1, locality outcomesa5c7c7e3fe344919/903e5b03411c4e2d, and scored sparse-repair losses2b537518091248f1/dfcfdd01d6f04525. This synthesis launches no experiment or new implementation.

## November 1 addendum — applied in both root instruction files

At evidence snapshot2707984, the user asked to put the general lessons in AGENTS.md and CLAUDE.md, then specifically asked for the lessons from studying public repositories. Both roots now contain the same ten required Research and execution principles. The earlier four-item experiment-selection section was consolidated into them. Existing repository-discovery, publication, run-logging, submission-permission and memory-ordering requirements were preserved, as was CLAUDE.md's parked-family reopening rule.

The public-repository obligations are explicit: independently discover sources, reproduce the complete pinned implementation before extracting mechanisms, inspect packing/preparation/dispatch and active flags, distinguish team-best from exact-file measurements, preserve minimal justified changes, separate required properties from conservative choices, and keep an unexplained stronger comparator open with an owner and experiment. A static check confirms the two new sections are identical, contain all ten principles and retain the pre-existing safeguards. No kernel, judge run or candidate status changed.

## November 1 clarification — lessons from the first public repo, Family F

The user distinguishes the earlier alygg77/okc-muse-spark-kernels study (Family F, external pin6cae49a) from the later keell0renz study (Family G). The preceding general summary merged their lessons. This clarification uses the original October30 audit and its explicit donor-first correction, reconciled at83e54a3.

- Challenge the choice of starting point. Our initial audit prematurely restricted alygg77 to a source of incremental ideas. The subsequent correction required comparing its complete implementation with our release before deciding which base or combination to pursue. The current incumbent must earn its place.
- Similar mechanisms and ancestry do not establish equivalence. Familiar kernel names or a shared vjeux base cannot replace inspection of generated code, layouts, tuning and dispatch. A locally failed version also cannot refute a working version elsewhere.
- Make the next deliverable an implementation decision. Research should answer which code or measured mechanisms to use and why. A new catalog of ideas alone left the main adoption/comparison question unanswered.
- Optimize a portfolio of shapes. One route can win at small loads and lose at another. Compare the exact dispatch choices, preserve independent standalone and layer decisions, and measure the complete combination rather than adding isolated savings.
- Measure resource and dataflow limits instead of relying on hardware intuition. Nominal CTA counts do not establish cluster residency, idle SMs do not establish aggregate throughput headroom, and a faster tensor-core mainloop can lose after reduction and output costs. Determine the limiting stage for the selected shape.
- Count representation costs and interactions. F's lossless weight format and scale-layout work made packing, decoding, output handling and synchronization part of the algorithm under study. Fewer weight bytes or scale instructions are leads that still need complete-call validation.
- Carry useful negative evidence with its scope. Their failed designs could save our quota, but their losses and our losses applied to the implementations measured. Revisit a family only when its recorded reopening evidence appears.
- Apply the user's policy to the action, not the scientific conclusion. The later October31 rule made F a measurement/learning track with owned reimplementation of winning mechanisms. Complete-file comparison remained necessary under that policy.

These lessons are already represented by the new root principles for architecture challenges, whole-implementation comparisons, minimal integration, scoped failures and implementation decisions. This clarification changes no candidate, permission, benchmark status or active G-first priority.

Sources: [original alygg77 audit](../../archive/docs/steering/2026-10-30-alygg77-public-kernel-audit.md), [donor-first correction](../../archive/docs/steering/2026-10-30-donor-first-adoption.md), and [Family F registration and later policy](../../archive/docs/steering/2026-10-30-family-f-donor-baseline.md).

## November 1 addendum — three emphasized Family F lessons

The user selected lessons1/3/5 from the Family F clarification. Both roots now explicitly require our base to earn its place against alternatives, per-workload selection followed by complete-dispatcher measurement, and scope a failed version to its exact implementation rather than excluding a family or a working public reference. Existing principles3/4/9 were strengthened instead of adding another parallel checklist. Verified identical sections and all three obligations. No candidate or permission change.
