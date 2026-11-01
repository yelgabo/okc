# MoE architecture screen for top10–50

Evidence pin b8b9ecb plus this report's receipt commit. November1 PDT. User resumed
the goal as top10–50; prefill remains deferred. One off-board comparison authorized;
no public submit or production adoption. Parent owns publication/index.

The fresh official board snapshot has our team at rank374,2.2157x; rank50 is
2.4534x (48494e439a5749ff), rank10 is2.6077x (1084b630433041f9).
Relative to exact owned published2.215712x (abe7651dbaa44458), rank50 requires
1.107274x score. If only two loads change, each needs1.35758x; if four change,
each needs1.16515x. These are scoring algebra, not proposed speedups.
Snapshot: `muse_moe_w4a16/profiles/architecture.board.2026-11-01.json`.

The newly recovered [keell0renz source](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_moe_w4a16.py)
is materially different from the earlier alygg77 comparator already measured in
3d8902381a194d56. Commit8e30e1be994db4310794b8d04e90d18d895deb84, file SHA256
2d931f09133d058053dda833b51f1867ea06cdcac0c912521ad37e8f7853185c.
Its inherited license/author notices remain intact in the private frozen scratch
`.build/today-moe/kel_exact.py`. No donor implementation was copied into production.
Root explicitly authorized donor-only measurement, then an attributed layer fork
only after a separate ordering/precision/license audit.

## Exact external check, not a claimed owned implementation

Check b65c4c878014416c ran that file byte-identically, without a harness, source
patch, concurrent GPU prepare, flag modification or forced routing. All5 official
tests pass on H200; official score2.395702934x,95%CI2.359767390–2.431638478.
This is1.081234x our published score across different runs/inputs. It is NOT a
same-input owned-vs-external causal speedup. It remains2.408% below the observed
rank50 score, and alone does not establish top50, let alone top10.

| tokens | owned published47 median us | owned51 verified763197 median us | exact external b65c median us |
|---|---:|---:|---:|
|1|27.001|27.980|25.114|
|8|67.388|72.369|63.252|
|16|95.426|96.161|91.149|
|32|136.747|133.309|122.315|
|128|192.869|192.829|176.505|
|256|240.839|241.826|211.115|

These rows have independent fresh routing; do not select their minimum as a
measured hybrid candidate. Each receipt pairs its own timings against SGLang.
Complete per-shape profiles, normalized result, log and owning receipt are at
`profiles/kel_exact.b65c4c878014416c.*`. Preparation reports modules built53s;
SC6 applies to WGS, not skinny. Actual regs/local/static bytes: MAIN96/0/0,
TINY/TINY2/MID149/0/0; dedicatedM8 issuer151/0/0. No measured occupancy or
physical HBM/L2 throughput is inferred from those attributes.

## Which architectural families remain credible

The new source combines warp-parallel stream-K cost-prefix setup, per-expert
completion counters/generation management (MFIN), lossless6-bit block scales
with a fallback, BF16 ybuf, data-flag A-to-B handoff at16, and a dedicated ring
issuer at8. The owner's [LESSONS](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/LESSONS.md)
attributes late gains to event/ramp/handoff/tail behavior rather than a lower
steady DRAM rate. These are author claims; our check verifies only the combined
implementation's result. It does not isolate which mechanism pays.

Already tested alternatives have weaker evidence: Humming0.1.12 INDEXED check
8d64807ea7504495 passed0.911513x (also had an extra BF16 gamma round, so not an
exact owned norm comparison). Exact-FP32-gamma GROUPED_MASKED ae8993f4e3e1435c
lost145–184us at128 and203–235us at256; its G1+G2 alone380.51us exceeds owned
217.15us in the matched256 case. No tile retune can be assumed to fix that.
Old separate W4/Marlin decomposition c4b1b5366bc54857 was1.689820x, with
medians46.658/88.048/107.401/145.527/250.388/366.482us at1/8/16/32/128/256.
The current-components staged pair9d94ac75f4724b52 lost24–32us at32–256;
it does not prove a newly designed full-output-owner decomposition impossible,
but supplies no reason to re-run that family unchanged.

Contiguous active-route reuse at16/32 fails the direct53c07507b9fd4f8c screen:
all four standalone cases had one job/expert, hence zero repeated logical weight
jobs. Literal one CTA/expert would use at most32/132SMs (at1 at most8/132), so
independent output-tile ownership/output-K parallelism is essential. Prior
full-output/no-K-split W4 GEMM was0.55–0.75xMarlin; any new full-owner design must
first prove enough active CTAs without reintroducing a costly cross-tile norm.
Do not repeat the failed shared-pair multicast, ring4, claims/fixup, token sorting,
polling, or single-warp position-map experiments without a new mechanism.

## Next decision and stopping criteria

The new architecture now has a credible several-percent *combined* screen and
supersedes more local tuning. Before reuse, audit the exact layer version: standalone
and layer sources are different. Prove MFIN release/acquire, flag parity/reuse,
grid residency/progress, SC6 lossless fallback and BF16 boundaries; preserve notices.
Repair or route around unproven paths. Passing this check does not prove ordering.

Smallest causal follow-up after that audit: one same-weight/control build and its
exact source-level twin disabling ONLY per-expert completion, with SC6/packing and
all GPU math held equal, compare all six loads and private workspaces. The external
author's existing no-MFIN twin also disables SC6, so it is a two-factor comparison
and cannot attribute MFIN alone. Alternatively a SC6-only byte-preserving twin
with identical scheduling distinguishes compression from completion scheduling.
No follow-up GPU check is dispatched here: this task's one check is spent. A fork
must pass fresh/replay official-math gates and matched whole-score evidence before
promotion. If exact combined alternative is insufficient after safe paths, top50
still needs the remaining observed2.4%; do not predict additivity or rankings.
