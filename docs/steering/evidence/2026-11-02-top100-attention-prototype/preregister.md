# Bounded attention continuity screen, before CPU implementation

2026-11-02. Evidence base a705900; supplied checkout HEAD
71eb71c1a67335d704e53d00632b4bf92ae38845. Source/CPU only; captain owns dispatch.

Hypothesis screened: retain the layer flat attention K/V producer ring across
request segments, overlapping the next request's gather with the current request's
merge. Affected cells would be D16/D32/V16/V32 full; no sliding or batch-one change.
This changes producer/consumer lifetime, not tile dimensions. Before implementing
a CUDA treatment, determine whether it is materially new, isolates that mechanism,
and has evidence sufficient to justify its integration cost.

Controls: exact public607 layer source, independent identical preparation, and
the complete current owned standalone attention (already carries its ring across
segments). Do not treat their different inputs/output boundaries as matched timing.
For a later isolated treatment keep current decode half-merge, page-index pipeline,
partition, flags, precision and finalization fixed. Merely enabling OKC_CONT does
not meet that isolation requirement.

Proof obligations: every physical page/key exactly once in its assigned segment;
new-token patch and causal masks unchanged; original FP8 Q/K/P and BF16 attention
output boundary; each full/empty phase uses the cumulative tile ordinal; no raw
ring alias during next-segment prefetch; page-index barrier reuse only after all
readers and async copies retire; writing-side proxy fences, WGMMA drains, CTA/WG
joins and cooperative admission/progress retained. Private mutable state and graph
output/cache lifetime remain necessary. CPU models cannot prove GPU ordering.

Prospective acceptance: two independent controls and two treatments, identical
fresh soft-distribution inputs, three held-out seeds, all twelve layer cells,
official numerical/cache checks plus separate BF16-boundary checks, captured
outputs checked after fresh replays. Whole-layer six-load geometric mean of
46:16 sliding/full endpoint ratios must improve >=1% on every seed and exceed
twice the larger independent log-null; no affected-cell regression beyond the
larger of 1% and its measured null. Preserve smaller effects without promotion.
Compile exact selected sm90a instances with source/build/resource identities and
no unexplained spills/residency change before any captain-allocated target test.

STOP if this is an existing complete mechanism without new favorable matched
evidence, if the switch bundles unrelated arithmetic changes, or if lifetime/
resource obligations cannot be established in this source/CPU allocation. A STOP
does not declare continuous pipelines generally slow. No GPU, challenge CLI,
install, donor adoption, default edit, or public promotion is authorized.
