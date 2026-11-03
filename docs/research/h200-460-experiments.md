# H200 kernels: the useful lessons from ~460 experiments

**Source:** user-supplied "Lessons from ~460 experiments", summarized November 2, 2026. H200 SXM (`sm_90a`), Muse-style BF16 projections, gated paged-FP8 attention and NVFP4 W4A16 experts.

**Evidence:** all numbers below are reported by that account, not independently reproduced here. No source URL, author, code pins or raw receipts accompanied it. Gains use different paired baselines and are not additive.

> Once steady-state streaming reaches its ceiling, attack bytes, startup, handoffs and tails—not another bandwidth knob. The big wins were structural; hundreds of tweaks to one lineage bought only low-single-digit gains.

## 1. Benchmark correctly or optimize noise

- **Cold L2 each replay:** roughly 1 GiB flushed before timing. Match this conditioning; warm local tests reportedly mis-measured small shapes by ~5 us. Cold starts do not eliminate reuse **within** a replay.
- **Adjusted timings:** roughly 4.1 us of graph/event overhead subtracted, estimated as `one - (two - one)` from empty-kernel graphs. Reproduce the accounting; do not assume that constant applies everywhere.
- **Noise matters:** ~0.63% score SD per run; separate checks differed by ~1.5–1.9%. The source recommends 6–8 runs per arm for ~1% changes, or 30+ matched local repetitions per shape for smaller effects. These are empirical budgets, not significance guarantees.
- **Use fresh inputs and check replay outputs.** Routing changes between repetitions/jobs. Best-of-many leaderboard rows are selection-biased. Equal-weight geometric means reward equal relative improvements equally, making microseconds especially valuable on ~5 us shapes; use the actual score weights.

## 2. Three hardware findings that change priorities

**The practical streaming ceiling was 4.2–4.45 TB/s, not 4.8.** A bare TMA ring with 132 CTAs, four slots and ~36 KB units reached it. More CTAs, deeper rings and different address orders did not help that skeleton. This does not prescribe four slots for every production kernel.

**Low average bandwidth did not mean a slow steady stream.** The MoE reportedly matched the bare ring in steady state; its apparent ~3.3 TB/s came from a ~3 us ramp, A→B handoff, slow batch-1 phase-B startup and a 6–9 us tail. Even a bare 55 MB read averaged ~3.5 TB/s from entry. Separate those events before tuning; their durations are not automatically removable savings.

**Synchronization gets expensive under HBM load.** A 132-CTA flag hop rose from ~1.4 us idle to ~6.3 us under a ~4.3 TB/s stream. Whole-layer fusion can lose to a kernel chain despite eliminating launches. That is not an argument against persistent MoE itself.

## 3. Wins worth retaining

| Mechanism | Reported result / qualification |
|---|---|
| Lossless 12-bit BF16 weights, exact escape corrections | Pooler M=1: 13.0 us versus a 14.0 us **uncompressed-byte** floor—not a violation of physical bandwidth limits. |
| Lossless 6-bit NVFP4 block scales | ~2% MoE gain at 32–256 tokens. Tested tensors had 33 distinct E4M3 bytes; validate the value set and retain an exact fallback. |
| PDL release after the producer's last weight issue | ~0.9% layer gain by overlapping the next prologue with the previous tail. Preserve dependent waits and memory ordering. |
| Separate MoE builds for 1 / 8 / 16 / 32+ tokens | Several percent per shape; specialization and code generation matter. |
| Stream-K WGMMA with per-unit cost tables | A 12.5% scheduling-cost error reportedly cost 7–9% performance. |
| Data-flag A→B handoff; warp-parallel job-cost scan | Avoid exposed grid synchronization/serial setup. Scan gained ~0.9% at 32 and ~0.5% at 128 tokens. |
| Per-shape kernel dispatch plus pinned cuBLASLt plans | Compare complete alternatives, then measure the combined dispatcher rather than adding isolated gains. |

## 4. Do not recycle these failed variants without new evidence

| Approach | Reported loss / mechanism |
|---|---|
| Persistent whole-layer megakernel | Loaded cross-SM synchronization lost to chain + PDL. |
| Fused A/B per intermediate slice | Six variants, 5–13% slower; 16-wide slices made GEMM2 K=16 and doubled instruction work. |
| Prefetch MoE weights during the predecessor's tail | 0.99–1.00x; even phase A fully in L2 saved only 0.4–0.65 us from ~22 us. |
| Four-CTA clusters splitting QKV GEMV K | 0.87x at decode1: GPC placement forced two waves. |
| Dual-lane WGMMA / per-unit barriers | 8–20% slower: extra synchronization on the lane that needed to progress. |

Speculative prefetch, persistence windows, lookahead rings and ring/unit/helper knobs also failed in the tested configurations. Attention truncation failed routing checks; dropping weak expert routes failed the residual bound. These are scoped negatives, not universal bans or permission to approximate required math.

## 5. Methods and traps to keep

**Measure a matching bare skeleton first.** Establish remaining headroom with the same bytes, shape and cache conditions.

**Use same-codegen controls.** Unrelated hot-loop edits reportedly shifted nvcc output by ~5%. Compare against the same edit with the treatment disabled, inspect SASS/resources, and keep the original baseline too.

**Batch experiments, not assumptions.** Compile multiple variants in untimed preparation, time matched inputs in one process, and use CPU-only compilation/shared caches where supported. Respect preparation limits and isolate mutable state.

**Plot per-SM timelines.** `%globaltimer` + `%smid` phase stamps exposed idle edges and tails hidden by averages. Measure observer overhead; uninstrumented full-call timing decides promotion.

**Graph/runtime gotchas:** captured `torch.zeros` executes every replay; pooled streams may already be capturing; captured allocations need live graph/pool ownership. Bind the CUDA context per calling OS thread—fresh-thread TMA descriptor creation reportedly failed otherwise.

**Harness integrity:** fresh inputs with every output checked defeated answer caching in the author's private replica. Discrete top-8 routing checks caught shortcuts beyond a downstream norm check. Preserve these protections, not the shortcuts.

## Use in OKC

Reference only: no kernel, job, ownership or permission changes. Consult [current steering](../steering/README.md), [scoped failures](../ruled-out.md) and [source corrections](cuda-h200/source_corrections.md) before adoption. Do not assume a technique is missing from our selected implementation. A new experiment needs an exact source, genuinely different mechanism, matched control and whole-score acceptance/stopping gate.
