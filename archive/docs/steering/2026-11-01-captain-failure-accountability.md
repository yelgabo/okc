# Why recent checks failed: captain accountability

November1 Pacific / November2 UTC, 2026. Evidence: terminal archives26b3b88 (outgoing),339523d (repaired RMW),d756f29 (old paired RMW), and captain decision13e545c. User asks why our code keeps failing.

## Answer

We have repeatedly shipped defects or unproven assumptions in the experimental measurement harness. That is our execution failure, not evidence that every production kernel is broken, and not something to blame on queue delays. I own allocating scarce checks before the relevant integration risks were covered. Documentation and passing CPU mocks did not compensate for missing target behavior.

- The old paired RMW fixture repeated expert0 eight times per token, violating distinct-topk inputs. Source/CPU diagnosis showed T8 could exceed the tiny route's capacity and reach its trap predicate. That explains a concrete defect, but the exact GPU719 instruction was never localized. The CPU mock accepted invalid inputs and therefore did not test the real contract.
- After legal distinct routing was restored, ownc5e4f89672f743bb completed13tests/12benchmarks/12profiles. Its paired gain was only-0.18274%..+0.28321% across bank/cross results, below promotion/noise gates. That is a working experiment with an unsuccessful optimization, not another software failure. RMW promotion remains closed.
- Own9adfd4dbc98f4998 reached native startup/resource checks but failed in our new mixed-graph query binding. We had not established target compatibility of the observer with every graph kernel representation. CPU source/identity mocks and score-oracle tests covered different properties. The exact internal failing API/row/arm remains unknown; runtime/driver identity mismatch is a hypothesis, not a proven cause. No timing result exists.

## Root process problem and correction

We coupled ambitious validation machinery to candidate timing, then treated many narrow CPU/source checks as stronger integration evidence than they were. Each added observer/fixture/private wrapper increased the surface that could abort before the useful question was answered. We also spent too much effort on small speculative gains without producing a qualified winner. Preserving safety is necessary; building ever more preparation machinery is not itself progress toward top100.

Effective now, the existing stop remains: no automatic retry and no extra observer-only GPU allocation. The worker's current15minute source/CPU identity diagnosis must return a minimal justified delta or a specific blocker, not grow into another full harness. Tests must exercise the actual contract and API namespace, including invalid fixtures and foreign graph kernels; CPU coverage must be labelled CPU coverage. Retain exact call/row/arm and resource-prefix diagnostics before failure, rather than depending on a final-only capsule. Do not weaken arithmetic, ordering, lifetime, output or edge gates to obtain a pass.

Before another measurement, captain must compare the repaired fixed candidate's expected whole-score benefit and integration risk against the completed whole-alternative research. New API assumptions must be explicitly enumerated as target-unproven. Prefer existing validated measurement paths where they answer the question. A target run can still fail, but must fail informatively; another unexplained harness failure stops that lane rather than earning another readiness cycle. No performance or rank improvement is claimed here; zero new qualified winners remains the honest status.

References: [old paired failure](2026-11-01-rmw-common-pair-terminal.md), [valid RMW result](2026-11-01-rmw-legal-fixture-terminal.md), [outgoing failure](2026-11-01-worker-outgoing-pair-terminal.md), [bounded next decision](2026-11-01-outgoing-query-stop-decision.md). No kernel, source pin, reader, GPU allocation or public submission changes in this handoff.

