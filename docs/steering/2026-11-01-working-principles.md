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

[Research ownership retrospective](2026-11-01-research-ownership.md), [discovery and reasoning review](2026-11-01-discovery-and-reasoning.md), [campaign accountability](2026-11-01-campaign-accountability.md), [integration-first instruction](2026-11-01-g-integration-first.md), [candidate funnel](2026-11-01-g-candidate-funnel.md) and [minimum-delta plan](2026-11-01-g-minimal-delta-plan.md).

The technical evidence behind those general principles remains in the owning ledgers, including public-comparator discrepancy 3a5f879df0d44825 versus607eb52bdaf24160, paired pipeline result63f5a2a8ed2341d1, locality outcomesa5c7c7e3fe344919/903e5b03411c4e2d, and scored sparse-repair losses2b537518091248f1/dfcfdd01d6f04525. This synthesis launches no experiment or new implementation.

## November 1 addendum — applied in both root instruction files

At evidence snapshot2707984, the user asked to put the general lessons in AGENTS.md and CLAUDE.md, then specifically asked for the lessons from studying public repositories. Both roots now contain the same ten required Research and execution principles. The earlier four-item experiment-selection section was consolidated into them. Existing repository-discovery, publication, run-logging, submission-permission and memory-ordering requirements were preserved, as was CLAUDE.md's parked-family reopening rule.

The public-repository obligations are explicit: independently discover sources, reproduce the complete pinned implementation before extracting mechanisms, inspect packing/preparation/dispatch and active flags, distinguish team-best from exact-file measurements, preserve minimal justified changes, separate required properties from conservative choices, and keep an unexplained stronger comparator open with an owner and experiment. A static check confirms the two new sections are identical, contain all ten principles and retain the pre-existing safeguards. No kernel, judge run or candidate status changed.
