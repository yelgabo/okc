# Independent four-arm score decision audit

November1 Pacific / November2 UTC,2026. Evidence checkpoint `5c64b19` (literal commit prefix). This supplements, not replaces, the [one-check allocation](2026-11-01-outgoing-seam-qualification-decision.md). No GPU, check registration, source treatment, production default or public submission by the captain. Prior goal turn was progress: corrected selected compiler identity and published/dispatched the bounded worker/research allocations. Current live worker ordinal5 and researcher ordinal7 were checked; neither is restarted.

## Decision-relevant finding

The reusable RMW reader's old four-arm decision assumes two raw controls followed by **two treatments**, computes four control/treatment crosses, and estimates duplicate noise using both0/1 and2/3. That is correct for its frozen RMW design but **wrong to reuse unchanged** for the new mapping: raw-OFF1/raw-OFF2/adapter-OFF/adapter-ON. The worker already owns the new reader; captain sent this distinction and supplies an independent arithmetic comparison, not a duplicate harness or another readiness project.

With the new arms, adapter-OFF is a diagnostic control, not a second candidate required to improve by1%. Adapter-ON must beat both raw controls under the published threshold. The2/3 difference is the isolated treatment effect, not duplicate noise. The raw0/1 pair supplies the noise floor. An apparent improvement over an expensive adapter can still lose against the incumbent.

## Independent implementation and adversarial tests

[score_oracle.py](evidence/2026-11-01-captain-outgoing-qualification/score_oracle.py) is a small pure CPU function. It consumes36 rows with explicit four-arm order,3 fixed banks,6 loads and2 contexts; rejects incomplete/duplicate/mislabeled cells, invalid arm count and nonpositive/nonfinite/boolean times. It computes per-load costs with46:16 context weights **before** log/geometric aggregation; compares arm3 against0and1; reports adapter2/raw and3/2 separately; retains short-only contributions. Duplicate-null is the six-load average of absolute log0/1 ratios, so opposite noise directions cannot cancel.

The decision is exactly the published arithmetic gate: everybank and both raw controls must pass >1%whole-six and >2xduplicate null plus everycontext regression tolerance max(0.5us,1%ofcontrol). No extra performance prerequisite is added. The function explicitly does not establish source, owning-job, graph, numerical, progress, cleanup, resource or measurement validity.

**11/11 synthetic tests PASS**, [test source](evidence/2026-11-01-captain-outgoing-qualification/test_score_oracle.py), [transcript](evidence/2026-11-01-captain-outgoing-qualification/score-oracle-tests.txt). Tests establish:

- Short-only improvement receives only its2/6 log contribution, not full-score credit.
- Adapter120 -> ON110 with raw100 is rejected despite an isolated adapter improvement.
- Adapter-OFF equal to raw is allowed when ON has a valid net improvement; a no-op and a subthreshold improvement fail.
- One failing bank, excessive duplicate noise, or a full-context regression hidden by a sliding win rejects the aggregate.
- Context weights are46:16, not an arithmetic mean of speedups, and opposite-sign control differences cannot cancel the noise floor.
- Wrong mapping, missing/duplicate rows and invalid timing values fail closed.

These are invented CPU inputs for testing the decision rule, **not experimental latencies or candidate performance**. No historical RMW rows are relabeled as this candidate. The new owning reader may compare its recomputation against this function on the same synthetic fixtures; it need not import this evidence tool into the upload.

## Next action, ownership and stopping criteria

Worker remains sole harness/GPU owner under the existing one-check allocation. No extra approval or dispatch is needed to consume this audit. Finish exact readiness or return the named fit/API/source blocker within the existing timebox. On accepted job, preserve exact source/mapping/receipt and let valid terminal results choose promotion or closure; no fixed-seam rescue sweep. Researcher independently pursues the broader multi-load expert mechanism, not another small overlap variation.

No rank refresh or ranking-formula inference was made. Existing overall API is ranks-only; matching own overall/layer rank is not proof of a universal formula. Top100 remains unachieved and zero new measured winners exist at this checkpoint. Repository publication/tests are progress on experiment validity, not performance progress.
