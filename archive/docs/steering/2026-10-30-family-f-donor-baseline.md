# Family F — Donor-based baseline and hybrid

Registered: 2026-10-30, at the user's request: "Yes, even label this as a new family."
Evidence snapshot: `7a5965531103f275ac104d340ef30f1508e32353`.
Parent decision: [donor-first adoption](2026-10-30-donor-first-adoption.md).

## Definition and status

**F — Donor-based baseline and hybrid (alygg77)** is a first-class candidate family alongside A–E. Evaluate the complete donor implementation as a replacement baseline, not merely a source of isolated optimizations for our incumbent.

F preserves an independent implementation lineage: preparation, weight packing, dispatch, scheduling and kernels. It is not a claim that every hardware mechanism is novel. Shared ancestry with vjeux does not establish equivalence with our current implementations.

**Initial status: registered / untested against our releases.** No controlled F win, correctness clearance, adoption, kernel import or scored submission is established by this registration. Archived donor results remain external evidence, not our F results.

A–E keep their identifiers and experiment histories. F must not silently be absorbed into A or E because of similar mechanisms. Whole-file replacement is an acceptable outcome; our implementation is a control and rollback, not a mandatory base.

## Pinned donor

- Repository: `alygg77/okc-muse-spark-kernels`.
- Commit: `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`.
- Preserve exact file hashes and compare them with donor `results.json` where available. Retain source provenance and license/NOTICE attributions.

## Variants

| Variant | Definition | Initial status |
|---|---|---|
| **F0 — Exact donor** | The pinned complete file unchanged, including preparation, packing and dispatch; not our reconstruction of its ideas | Registered; controlled comparison pending |
| **F1 — Conforming donor adaptation** | F0 plus only individually documented compatibility or demonstrated correctness repairs, with an explicit diff and separate measurement | Not built by this registration; use only when required |
| **F2 — Measured hybrid** | A donor-based or incumbent-based combination of paths that actually win on their shapes, validated as one complete implementation | Proposed; depends on F0/F1 comparison |

Any edit means an arm is no longer byte-identical F0. A specific correctness defect blocks promotion of the affected path; it does not lower a requirement or reject every F variant. F2 must be measured as a whole, not credited with a sum of isolated gains.

## Independent tracks

| Track | Donor file | Scope and priority | Initial result |
|---|---|---|---|
| **F-PROJ** | `kernels/muse_bf16_proj.py` | Priority 0: all 30 projection cases against our frozen released file | Pending |
| **F-ATTN** | `kernels/muse_attention_decode.py` | Priority 0: all 12 attention cases against our frozen released file; independent owner | Pending |
| **F-MOE** | `kernels/muse_moe_w4a16.py` | Separately owned complete standalone-MoE comparison | Pending |
| **F-LAYER** | `kernels/muse_decode_layer.py` | Separately owned complete-layer comparison, with layer-specific precision and ordering | Pending |

Use result IDs such as `F-PROJ-0`, `F-PROJ-1`, `F-PROJ-2`, and similarly for the other tracks. Until a file/hash and run evidence exist, these identify proposals, not built candidates. Prefill is outside this immediate assignment. Standalone promotion and layer promotion remain separate decisions.

## Comparison and promotion

Follow the full procedure in the [parent report](2026-10-30-donor-first-adoption.md). Start with the actual complete donor and our exact release. Use independently prepared **R1/R2** incumbent arms and **D1/D2** donor arms, matching weights/input values, independent mutable workspaces, cold-L2 graph replay and balanced seeded ordering. Record selected policies, complete-call timings and independent nulls. Test fresh inputs, queued replay, output retention and allocation-layout robustness; do not select favorable allocation layouts to improve a judged score.

Preserve all numerical, precision, asynchronous-ordering, replay and attribution requirements. Specific incompatibilities or correctness defects get explicit findings and, where repairable, an F1 arm. Generic suspicion or a failed similar design is not a reason to skip the complete-file comparison.

| Outcome | Decision |
|---|---|
| Correct F0/F1 consistently wins overall, beyond independent nulls, with acceptable load behavior | Use it as the replacement release candidate; preserve our rollback |
| Each wins different shapes | Build F2 and test the complete combined implementation |
| Difference does not exceed independent nulls | Inconclusive, not a win or universal rejection |
| F is slower in the controlled comparison | Retain the incumbent and record the scoped loss |
| A requirement fails | Identify the exact issue; repair as F1 or reject that path without weakening the gate |

Use each problem's actual weighted score. A useful intermediate winner need not reach a ranking target in one jump. Scored submission remains subject to existing user authorization and coordinator ownership; this registration does not authorize one.

## Required record and next action

Every F result records variant/source/control hashes, donor receipt mapping, eligibility findings, compiler/environment, selected policies, per-case raw and paired timings, both independent nulls, correctness/replay results, weighted standalone result, separate layer result when applicable, and an adoption decision.

The next deliverable is: **Which complete implementation should we use now—ours, F0/F1, or F2—and what evidence supports it?** Complete projection and attention comparisons take priority over new GPU codec/dense rewrites. Existing running work need not be discarded.

Future reviews must carry F forward as a separate family even while its comparisons are pending. Do not regenerate an A–E-only registry from an older tracker snapshot or mark F promoted using an archived donor score. Record cross-family hybrids explicitly and do not add overlapping gains.

## Shared records and scope

This report is F's initial definition and track ledger. The [steering index](../../../docs/steering/README.md) and [dashboard data](../../../docs/architecture-dashboard.json) register F alongside A–E. The dated [A–E guide](../architecture-families-2026-10-29.md) and [performance tracker](../../../docs/architecture-performance-tracker.md) retain their historical evidence snapshots; their earlier five-family statements do not exclude F.

Sources: the [donor-first adoption report](2026-10-30-donor-first-adoption.md) and [alygg77 audit](2026-10-30-alygg77-public-kernel-audit.md), which retain source pins, external results and caveats. This task registers the family; it does not perform a fresh technical audit, query live ranks/quotas, import kernels, run benchmarks, change test tolerances or replace any release.

## Addendum, 2026-10-31: user decision on the use of F results

The user decided, in chat to the coordinator on 2026-10-31 (relayed to the F comparison owner), that **Family F is measurement and learning only.** When a donor path wins a case, an owner re-implements that mechanism in our own code with attribution in comments, as was done for vjeux. Donor files and donor code are never submitted verbatim, and F0 and F2 are not submission candidates. F1 likewise stays a measurement arm.

This narrows the outcome table above: "use it as the replacement release candidate" and "build F2" become "re-implement the winning mechanism in our code and measure that build against the incumbent". The comparisons themselves continue as checks. Each F result names, for every case the donor wins, the mechanism precisely enough for an owner to re-implement it (route or kernel, geometry, stage count, packing, dispatch rule, with the donor file and line).
