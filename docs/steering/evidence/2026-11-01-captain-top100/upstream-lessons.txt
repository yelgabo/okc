# Lessons from ~460 experiments

Hopper (H200 SXM, `sm_90a`) inference kernels for one decode layer of a Muse Spark-style MoE model: BF16 projections, gated
paged-FP8 attention, NVFP4 W4A16 experts. Every number below was measured on an H200 (the challenge judge or a Modal H200
that matched it within ~1%). "x1.010" means 1.0% faster than the base it was paired against.

## 1. How the board measures (know this before optimizing anything)

- **Cold L2 before every timed replay.** The judge reads a buffer much larger than L2 (~1 GiB, ~260 us) before each CUDA
  graph replay. Weights never stay in L2 across calls; every kernel pays its full HBM traffic and its first-byte ramp.
- **A fixed ~4.1 us per replay is subtracted** (graph launch + events, measured as `one - (two - one)` empty-kernel graphs).
  Local harnesses that skip the flush mis-measure small shapes by ~5 us (idle clocks, warm caches).
- **Fresh inputs every repetition**, new seeds every job, so routing changes each run.
- **Noise:** about 0.63% SD per run on the board speedup, ±1.5–1.9% for one check against another. A real 1% change needs 6–8
  runs per arm on the judge. Decide sub-1% changes on your own paired A/B (30+ reps per shape) and use the judge only as a
  conformance check. A board row keeps a team's best draw, so rows are biased upward by roughly 1 SD for everyone.
- **Tiny shapes are amplified.** A shape whose net time is ~5 us moves the geomean as much as one that takes 300 us.

## 2. Hardware facts that shaped every design (H200)

- **HBM read ceiling with a TMA bulk ring: 4.2–4.45 TB/s** (spec 4.8). 132 CTAs × 4 slots × 36 KB in flight is enough;
  more CTAs, deeper rings or other address orders (contiguous per CTA vs interleaved) did not move it.
- **Small reads cannot reach peak.** The first 36 KB unit lands ~4 us after launch because ~19 MB is requested at once and
  served round-robin (Little's law, not latency). A 55 MB read measured from kernel entry averages ~3.5 TB/s even for a bare
  ring. Per-SM steady state: one 36 KB unit every ~1.09 us.
- **Cross-SM synchronization costs 3–6× more while HBM is saturated** (a 132-CTA flag hop: 1.4 us idle, 6.3 us under a
  4.3 TB/s background stream). Grid-wide flags and barriers inside a stream are expensive; this killed every "persistent
  layer / megakernel" plan.
- SM clock drops to ~1.81 GHz under dequant + MMA math (1.97 GHz on a bare ring), but the steady stream rate did not change.

## 3. What worked

| Change | Gain | Notes |
|---|---|---|
| Lossless 12-bit re-encoding of BF16 weights (exact escape corrections) in TMA-ring GEMVs | beats the BF16 byte floor (pooler m=1: 13.0 us vs 14.0 us floor) | the decode layer and bf16 board are weight-bandwidth bound at small m |
| Lossless 6-bit NVFP4 block scales in the wgmma MoE builds | ~+2% on the MoE at 32–256 tokens | scales were 33 distinct E4M3 bytes per tensor |
| PDL chain, `griddepcontrol.launch_dependents` right after the producer's last weight issue | +0.9% layer | the next kernel's prologue overlaps the previous tail |
| A separate MoE build per scored batch size (1, 8, 16, 32+) | several % per shape | the judge times discrete sizes; dispatch on shape is allowed |
| Stream-K wgmma MoE (4 consumer warpgroups) with per-unit cost tables | the 32–256 token MoE | tables tuned once; a 12.5% cost error costs 7–9% |
| Data-flag A→B hand-off instead of a grid barrier in the MoE | small-batch MoE | the hop cost above |
| Warp-parallel stream-K prologue (shuffle-scan of job costs instead of one lane) | +0.9% at 32 tokens, +0.5% at 128 | the serial prologue was the critical path before the first weight byte |
| Per-shape dispatcher over the best public kernels + pinned cuBLASLt plans | bf16 board | judge profiles pick the kernel per (layer, m) |
| Per-thread CUDA-context bind | robustness | a state first used from a fresh OS thread failed TMA descriptor creation |

## 4. What failed (measured dead; do not re-open without a new idea)

- **Megakernel / persistent whole layer**: the cross-SM hop cost under load caps it below the chain + PDL.
- **Fused-slice MoE** (A and B per I-slice in one CTA, no A→B wait): six variants, all 5–13% slower. With 16-wide slices the
  down-projection has K = 16, so accumulators cannot be shared across tiles and instruction count doubles.
- **"The MoE streams at 3.3 TB/s vs 4.4"**: a myth. ncu PM sampling shows the MoE's steady DRAM rate equals the bare ring.
  The loss is in events: a ~3 us ramp, the A→B hand-off dip, phase B starting slowly at bs1, and a silent 6–9 us tail.
- **Prefetching the MoE's first weights into L2 during the previous kernel's tail**: x0.99–x1.00. With phase A fully
  L2-resident the MoE gains only 0.4–0.65 us of 22. It is paced by its own skeleton, not by supply.
- **More CTAs for the 96-CTA qkv GEMV via 4-CTA clusters splitting K**: x0.87 at decode 1. Clusters at 1 CTA/SM do not all fit
  in the GPCs at once, so the kernel runs in two waves; with 8-row granularity, 2-CTA clusters fall back to 96 CTAs.
- **Speculative expert prefetch, L2 persistence windows, lookahead rings, twin L2 policies, ring depth / unit size / helper
  warp knobs**: all at optimum or worse.
- **Dual-lane wgmma MoE (WGS2) and per-unit barriers (PUB)**: 8–20% slower; they add sync to the lane that must not wait.
- **Approximations inside the correctness bound**: the router check (picked experts within 0.05 of the true top-8 score)
  pins everything before the router to near-exact; attending only the last 2–16k tokens on full layers failed it on every
  shape. Dropping low-weight expert routes after the router broke the residual-update bound (0.17–0.7 vs 0.15).

## 5. Method lessons

- **Measure the ceiling first.** A bare-ring skeleton of the same bytes and shape tells you what is left before any tuning.
  Most of our late work found that we were already at the skeleton's limit.
- **Same-codegen controls.** nvcc codegen shifts ~5% from unrelated edits in a hot loop. Compare a variant against a build
  with the same edit switched off, not against the original binary.
- **Time many variants in one process.** Build N variants in the untimed `prepare()` and time them back to back on the same
  inputs; one GPU session then covers 10–40 experiments.
- **Compile on CPU.** JIT-building CUDA extensions on a CPU-only container into a shared cache cut GPU time to timing only.
- **Draw per-SM timelines.** `%globaltimer` + `%smid` stamps per CTA (start, phase marks, PDL wait, end), plotted one row per
  SM, showed every idle edge, wait and tail at a glance. That beat tables of averages.
- **CUDA graph capture gotchas we hit:** `torch.zeros` inside a capture becomes a graph node and re-runs every replay;
  `torch.cuda.Stream()` hands out pooled streams that may already be capturing; memory used by a captured body must stay
  owned by a live graph or pool.
- **Tweaking one lineage gives marginal gains.** Hundreds of tweaks of one starting kernel bought low single-digit percent;
  the large steps came from structural changes (weight re-encoding, per-size builds, stream-K MoE).

## 6. For benchmark designers

From our own testing against our own judge replica (never against the real judge):
- **A harness that times fixed seeds, or reuses a small pool of inputs, can be fooled by an answer cache.** The cache lives
  inside the timed graph, outputs are bit-exact, and output checks pass. Fresh inputs for every repetition, with every
  output checked, defeat it. The real judge already does this.
- **A tight routing check made lossy shortcuts impossible.** Checking the discrete decision (top-8 experts within a score
  margin), not just a relative L2 bound on the output, removed all the slack we tried to exploit.
