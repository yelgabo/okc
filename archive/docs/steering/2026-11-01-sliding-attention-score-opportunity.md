# Sliding attention score opportunity: bounded source audit

November1,2026; attention worker branch after5697703. No new GPU run or kernel
change. Official calibration receipt is `scored.8914751e3b9f468b.json`; source
was comment-only repeat of the earlier control. These timings characterize
that receipt, not a new candidate or an immutable hardware floor.

For each load, define D=46*s+16*f from its official corrected `median_ns`
(sliding/full), in microseconds. A sliding-only reduction delta changes the
six-load aggregate by `(D/(D-46*delta))**(1/6)` when all other rows are fixed.
Full-only uses16*delta. Baseline ratios cancel in this sensitivity calculation.
This is not the arithmetic mean of per-shape speedups or a layer speedup.

|load|sliding us|full us|sliding share of weighted load|score gain for1us sliding|score gain for1us full|
|---|---:|---:|---:|---:|---:|
|decode1|5.1834|12.3896|54.603%|1.87256%|0.62409%|
|decode16|7.8333|85.0455|20.937%|0.45254%|0.15579%|
|decode32|10.0921|162.5517|15.146%|0.25234%|0.08727%|
|verify1|6.7698|14.8193|56.773%|1.47051%|0.49461%|
|verify16|11.4538|101.5550|24.486%|0.36081%|0.12447%|
|verify32|15.6026|183.5644|19.638%|0.21133%|0.07315%|

Thus one microsecond at batch1 sliding matters most; score relevance alone
is not evidence that it can be removed. Fullverify improvement is diluted
per microsecond, but a genuinely large gain still matters:10us atverify32full
would imply0.74889% sensitivity, whereas0.61us atverify32sliding implies0.12854%.
Neither example is a measured candidate win.

## Matched comparison versus theoretical opportunity

Receipt `bench_o.0519d0b15e9c4583`, generated froma70192d/evidence589fb00,
compared three inputs and independent release/donor nulls. Its sliding gaps
were d1−0.11us(null),d16+0.00(null),d32−0.05, v1−0.10(null),v16−0.12(null),
v32−0.61(real). Donor use is measurement/learning only. Transplanting the
v32 gap onto891 calibration yields0.12854% standalone score sensitivity;
this is a transfer scale for that matched gap, NOT a universal ceiling on
all future designs. Lowbatch sliding offers high sensitivity but currently
lacks a measured repeatable donor lead from this comparison.

Phase attribution `1c861d3944f149a0` measured after HMRG/PRQ: v16 sliding loop
4.35us ours versus3.93 public; v32 sliding8.13 versus7.26. Its phase totals
are perturbed phase-exit differences, not removable independent costs. Both
prologues were already close (8.93/8.77 and9.18/9.06). These data motivated
address carry, which was subsequently tested and LOST its aggregate gate.
Do not propose the phase-gap observation as a new experiment without reading
that follow-up.

`c77ce71c81af4fb1` implemented per-request QK descriptor carry and Vtranspose
address toggles. All19 checks passed each arm; no aggregate win beyond null.
Both-carry v16sliding0.00us pooled, with inputs−0.19/+0.19; v32sliding−0.19
versusctrl2+0.03, inputs−0.16/−0.13. Holding unchanged rows fixed, corrected
sensitivity was only1.000438; no shape-selective promotion. Sourcefdaaa11
was reverted, patch preserved. This remains closed absent a larger mechanism.

Other overlaps in `docs/ruled-out.md`: per-request scheduling/order was
already implemented in PRQ31 (a7a4c23e98ca484a/409996a7214f411a); changing
verify cooperative finalize to ticket ce92d32e26024bd2 lost+1.70/+1.41us
v16/v32sliding. Branch-free in-loop KVFAST82e52f7b23ec49da was small and not
resolved by judged422f81a11f384deb. Early stage releasebf6b8ca9e37341fb lost
sliding/full. General late-tid1ab53721b09854717 failed because sliding benefit
was reproduced by null. None is reopened by the six-load weighting table.

## Decision and next justified measurement

No new source change is justified by this bounded audit, so no paired GPU
check was spent. The largest score sensitivity lies in already fused
batch1sliding prologues/finalize; a specific removable dependency or duplicate
traffic there has not been demonstrated. Highbatchverifyloop differences are
small after PRQ and the obvious carry mechanism was unsuccessful.

Next justified experiment is a fresh matched, independent-build/state phase
attribution of sliding batch1 and verify32 on the current pure checked source,
only after the coordinator's layer timeline/counter feasibility result. It
must identify a repeatable stage difference beyond both nulls and a concrete
owned dependency/layout change, with unchanged fences/math/resources, before
an optimization check. Do not sweep stage counts, address carries, ticket
finalize, placement, or reorderings already closed. Any eventual candidate
needs affected-row/all-input correctness and fresh replay gates, resource
checks, repeatable six-load aggregate improvement and a separate measured
layer transfer gate. The0.13% matched v32sliding scale cannot close the
~11.73% saved snapshot score gap by itself.
