# RMW setup failure: reference work and exact reuse opportunities

2026-11-01. Bounded source/receipt analysis for root-owned H200 job
`3fd1317143304bd5`, exact914572B candidate SHA256
`97935e7dc45d9fa69f5b4965bd70a5c2946cd8b829cedffb7a2aefc598cb53c7`.
No new GPU, judge/service call, kernel change, benchmark or prototype here.
Owner today_moe handles the separate qualifier-budget epoch; root owns index.

## What the actual failure establishes

The owning raw envelope is root
`docs/steering/evidence/2026-11-01-rmw-wait-2258/persistent-wait-result.json`.
It reports setup failure, no score, and `GQUAL_PREPARE_BUDGET` at the entry
of `_gq_empty`. Original production preparation had printed its successful
native56-replay startup marker. The qualifier's state clone/resource pass
and that cell's two input/front-reference banks precede this entry guard.

**The stack does not identify the cell index.** Every cell has this same
call site; no retained per-cell progress marker distinguishes first from
later. It is therefore not established that the first empty check timed out.
Nor can compile, tuner, input generation, oracle or clone durations be
reconstructed from this log. Public `prepare_s` is null. Do not assign invented
stage timings or attribute the loss to the RMW kernel.

Normal G preparation includes parallel native builds, packing, the base
projection tuner, pinned Lt planning, chain-policy tuning and the original
56 checked replays. The wrapper's outer295s clock includes all that work.
Later static policy overrides do not prove those earlier tuners are dead;
removing them would require a separate selector/normal-behavior proof. No
compile/tune removal is recommended from this receipt alone.

## Reference work before an empty check

Exact payload `_gq_inputs:17967` generates one full published-distribution
input and evaluates `_ab_ref_front` twice, with BF16-Q and FP8-Q semantics.
`_gq_private:18014` calls it for base and fresh, then deep-copies the base
model input, before entering `_gq_empty`. Thus four front evaluations and two
full input banks precede each cell's empty guard. The model's fresh KV and
other mutable input copies are intentional owners, not removable aliases.

`_ab_attention:17539` repeats QKV FP32 projection, normalization/RoPE, new KV
encoding, FP8 cache gather/dequantization and attention for the two Q formats.
Only Q-format conversion differs; QKV/norm/RoPE and K/V preparation are common
pure work. Factoring that shared prefix is possible in principle but would be
a distinct source rewrite requiring oracle-equivalence tests. It is not mixed
into the owner's present minimal change.

Each `_ab_ref_front:17581` also recreates immutable QKV/O FP32 weight views
and router FP64 weights. Conversion outputs are72MiB+32MiB+16MiB per call,
480MiB across four calls. These are source-derived **logical allocation/output
sizes**, not measured DRAM traffic, allocation peaks or exposed time. A scoped
immutable weight-cast cache could reuse them without arithmetic changes if
the raw tensors cannot mutate and the cache is cleared on prepare exit. No
such extra mechanism is implemented or needed to interpret this failure.

## Back-oracle repetition and the selected minimal opportunity

`_qual_oracle:17708` recomputes both `_ab_ref_back` results for every model
validation:72 validations mean144 back evaluations. Each cell uses only base
and fresh immutable front dictionaries. **If** exact routing IDs stay the
same per bank, only four back results per cell are distinct,48 total;96
evaluations repeat. This count is conditional, not a runtime speed forecast.

Owner today_moe is implementing only a per-cell exact-ID back-result memo
plus flushed stage/cell breadcrumbs. Routing validation must still execute
before lookup. Keys must distinguish front/bank identity and complete ID
contents, snapshot mutable IDs, and keep results private/immutable. Different
IDs or front must recompute; all candidate output, reference-finite, cache,
error, routing and fresh-state checks still run on hits. Clear on cell exit
and exceptions; do not persist oracle/input states in returned production.
This reuses a pure reference computation, not candidate answers or tolerance
calibration from another input. It cannot repair a genuine first-cell
pre-empty timeout, so breadcrumbs are essential before interpreting the next
outcome. No blind cache or relaxed deadline/gate is justified.

`_ab_deq:17593` already caches each expert's immutable FP32 w13+w2, keyed by
local expert during this prepare. They occupy192MiB/expert, up to6GiB for32
experts. Adding another identical dequant cache is redundant. New dequant
does not occur before the current cell's empty entry: it is called later by
`_ab_ref_back`; earlier cells may already have populated the existing cache.
Back-result reuse also avoids repeated pooler/unpooler float conversions and
expert math on hits, without changing that dequant policy.

## Decision, proof gates and stop

Prioritize the owner's isolated exact-ID memo/breadcrumb change, not a new
dense kernel, tuning removal or simultaneous frontend/cast rewrite. CPU tests
should cover same-front/different-ID misses, distinct-front/same-ID misses,
mutation of a caller's IDs after keying, reference-finite rejection on first
evaluation, stale output/cache rejection despite cache hits, cell/exception
cleanup, and exact old/new oracle outputs. Preserve all72 model/36 fresh/
60 empty checks and actual dispatch/resource witnesses.

The next runtime interpretation must use real emitted breadcrumbs to locate
setup cost; source work counts alone cannot promise a295s fit or a score.
Stop this analysis without dispatch. Machine evidence in
`evidence/2026-11-01-g-reference-budget/source-cost.json` pins function hashes,
source anchors, counts and byte arithmetic; no elapsed measurements are claimed.
