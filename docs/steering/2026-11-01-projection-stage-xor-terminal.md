# Stage-local sparse-XOR terminal loss

2026-11-01. Source/evidence commits `e27c86d` and `fd21fdf`; analyzer
correction `ee972f1`; root preregistration `a08f501260ea7e9625c01aa4eeecda0934afa7f9`.
Owning public projection job **dfcfdd01d6f04525** is DONE/PASS: nine official
tests, thirty benchmarks and thirty unique profiles, `checks.ok=true`, no flags.
Root owns canonical receipts/logs/profiles. This worker used only the supplied
terminal files, independently reconstructed the score, and made no official call.

**Close the exact stage-local BEFORE-A variant. It loses to C2; do not
promote, retry automatically, port to layer or tune capacity/geometry.**

## Result and scope

| Owning run | Official score | 95% CI | Role |
|---|---:|---|---|
| C2 `9eba96894ee1402e` | 1.527806482956 | [1.504889385711431, 1.5507235802001038] | Primary archived incumbent comparator |
| Global records `2b537518091248f1` | 1.394120070870 | [1.3732082698071773, 1.4150318719332842] | Prior failed representation |
| Stage records `dfcfdd01d6f04525` | 1.419146839388 | [1.397859636797322, 1.440434041978966] | New failed representation |

Stage score is -7.112% versus archived C2.
All ten targeted cells regress in both adjusted and raw latency: adjusted
range +6.274% to +40.738%, raw range
+5.860% to +30.694%. Setup took 129.385967127 seconds.
Qualification completed all ten overrides, zero packing fallbacks, 110 pick
keys including 100 neighbors, 160 math checks, ten captures and thirty fresh
replay checks. No local-memory or admission gate was bypassed.

These are separate official jobs, with different input seeds, baseline
measurements, overhead correction and C2 tuner/environment state. Weight
identity was not shared or verified bit-identical across the jobs.
The fixed source is preserved, but the C2 tuner can select different plans.
The direction and magnitude do not justify promotion; they do not isolate a
causal stall cost. In particular, the higher score than failed global SX is
not a win over C2 or proof of a specific memory-latency saving.

The exact thirty-cell/six-load formula reconstructs both official scores.
Inserting only the ten stage adjusted medians into archived C2 other twenty
cells and baselines gives 1.415281145448, factor
0.926348435641. This is an explicitly unpaired
sensitivity scenario, not a scored hybrid or a measured paired effect.
The fully raw six-load baseline/candidate ratio is
1.297622225525 versus archived C2
1.371349040292; it uses raw baseline
and raw candidate medians on both sides and is not the official adjusted score.

## All thirty cells

Times are microseconds. Untargeted rows retain C2 code and are included in
the actual official score; their cross-job variation receives no optimization
credit. Full profiles, seeds, raw baselines and offsets are in `rows30.json`.

| Layer | M | Selected policy | C2 adjusted | Stage-run adjusted | Change | C2 raw | Stage-run raw |
|---|---:|---|---:|---:|---:|---:|---:|
| qkv_gate | 1 | C2 code | 9.058 | 8.591 | -5.16% | 12.960 | 12.832 |
| o | 1 | C2 code | 5.036 | 5.393 | +7.09% | 9.232 | 9.280 |
| pooler | 1 | C2 code | 13.516 | 13.327 | -1.40% | 17.408 | 17.472 |
| unpooler | 1 | C2 code | 3.734 | 3.541 | -5.18% | 7.616 | 7.584 |
| router | 1 | C2 code | 3.838 | 3.714 | -3.21% | 7.696 | 7.744 |
| qkv_gate | 8 | stage | 9.771 | 12.789 | +30.90% | 13.680 | 16.944 |
| o | 8 | C2 code | 5.476 | 5.320 | -2.86% | 9.504 | 9.472 |
| pooler | 8 | C2 code | 13.419 | 13.610 | +1.42% | 17.472 | 17.568 |
| unpooler | 8 | stage | 4.535 | 4.819 | +6.27% | 8.464 | 8.960 |
| router | 8 | C2 code | 4.468 | 4.288 | -4.03% | 8.096 | 8.288 |
| qkv_gate | 16 | stage | 9.741 | 12.946 | +32.90% | 13.920 | 17.088 |
| o | 16 | stage | 5.366 | 6.701 | +24.87% | 9.504 | 10.720 |
| pooler | 16 | stage | 13.800 | 16.956 | +22.87% | 18.240 | 20.960 |
| unpooler | 16 | stage | 13.557 | 15.568 | +14.83% | 17.856 | 19.776 |
| router | 16 | C2 code | 4.449 | 4.219 | -5.17% | 8.432 | 8.576 |
| qkv_gate | 32 | stage | 10.557 | 14.858 | +40.74% | 14.752 | 19.280 |
| o | 32 | stage | 5.869 | 6.819 | +16.18% | 9.920 | 11.168 |
| pooler | 32 | stage | 15.084 | 18.664 | +23.74% | 18.992 | 22.672 |
| unpooler | 32 | stage | 14.721 | 16.520 | +12.23% | 18.416 | 20.832 |
| router | 32 | C2 code | 5.490 | 5.072 | -7.63% | 9.168 | 9.312 |
| qkv_gate | 128 | C2 code | 14.664 | 14.521 | -0.98% | 18.656 | 18.592 |
| o | 128 | C2 code | 6.812 | 7.089 | +4.07% | 10.976 | 11.456 |
| pooler | 128 | C2 code | 19.994 | 20.528 | +2.67% | 23.808 | 24.640 |
| unpooler | 128 | C2 code | 18.987 | 18.924 | -0.33% | 22.896 | 22.976 |
| router | 128 | C2 code | 6.639 | 7.175 | +8.08% | 10.832 | 11.136 |
| qkv_gate | 256 | C2 code | 16.941 | 17.437 | +2.92% | 21.536 | 21.472 |
| o | 256 | C2 code | 8.338 | 8.772 | +5.21% | 12.928 | 13.120 |
| pooler | 256 | C2 code | 24.434 | 24.254 | -0.74% | 28.112 | 28.192 |
| unpooler | 256 | C2 code | 23.441 | 23.503 | +0.26% | 27.392 | 27.632 |
| router | 256 | C2 code | 9.261 | 9.568 | +3.31% | 13.440 | 13.792 |

The six full load sums, including all five projection types:

| M | C2 candidate sum | Stage-run candidate sum | C2 SGLang sum | Stage-run SGLang sum |
|---:|---:|---:|---:|---:|
| 1 | 35.182617 | 34.565951 | 59.936000 | 60.064001 |
| 8 | 37.668069 | 40.825728 | 75.808000 | 75.647998 |
| 16 | 46.914076 | 56.389937 | 74.479999 | 75.727997 |
| 32 | 51.720988 | 61.932763 | 77.039998 | 76.896001 |
| 128 | 67.095785 | 68.236245 | 85.904000 | 85.472000 |
| 256 | 82.416369 | 83.533056 | 100.976000 | 101.471999 |

## Actual resources and capacity

All forty CUDA resource vectors pass independent parsing and source-derived
layout checks. Every vector has zero local bytes and zero static shared
bytes. Stage admission equals its corresponding C2 control for FULL and NF;
the values below are residency limits, not measured active occupancy or SM
utilization. Actual capacity is the maximum actual per-stage record count;
no cap or target subset was chosen from results. All record arenas fit the
existing SX SST alignment slack for these weights, so stage allocation equals
the previous global-SX formula. This is not proof of zero transport cost:
16/32 record bytes are still added to each weight-stage bulk copy.

Registers column order is C2 FULL / stage FULL / C2 NF / stage NF. Shared and
thread/admission columns show C2 FULL / stage FULL; all forty complete vectors
and NF values are retained in `resources.json`. C2 pmax is inferred from the
actual control allocation and checked even, >=2 and identical FULL/NF.

| Route | Capacity | Arena B/stage | C2 pmax | Registers | Dynamic shared B | Threads | Blocks/SM admission |
|---|---:|---:|---:|---|---|---|---|
| qkv_gate:8 | 10 | 32 | 30 | 78/77/83/78 | 220992/218352 | 288/160 | 1/1 |
| qkv_gate:16 | 8 | 16 | 30 | 66/68/71/76 | 182096/180480 | 288/160 | 1/1 |
| qkv_gate:32 | 8 | 16 | 30 | 112/101/112/101 | 219200/210160 | 288/160 | 1/1 |
| o:16 | 4 | 16 | 12 | 54/58/61/58 | 69248/65760 | 256/160 | 3/3 |
| o:32 | 4 | 16 | 12 | 83/74/92/83 | 100160/90336 | 256/160 | 2/2 |
| pooler:16 | 9 | 32 | 50 | 60/61/63/67 | 174080/169200 | 288/160 | 1/1 |
| pooler:32 | 9 | 32 | 50 | 85/85/111/84 | 186480/176352 | 256/160 | 1/1 |
| unpooler:8 | 5 | 16 | 12 | 60/55/60/55 | 118384/120048 | 416/288 | 1/1 |
| unpooler:16 | 6 | 16 | 28 | 54/58/61/58 | 69376/65760 | 256/160 | 3/3 |
| unpooler:32 | 6 | 16 | 28 | 83/74/92/83 | 100288/90336 | 256/160 | 2/2 |

The before/after machine snapshots identify NVIDIA H200, 1980 MHz SM and
3201 MHz memory clocks for both this run and C2, on different nodes. They
do not provide in-kernel utilization, HBM throughput, cache misses, stalls
or causal attribution. No such counters were measured. Lower thread count,
removed correction readers and spill-free compilation did not guarantee speed.

## Reproducibility and stop/pivot

Source `muse_bf16_proj/experiments/sk_stage_xor_release.py` is 450781 bytes,
SHA256 `de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229`. The owning receipt SHA256 is
`fea1189cf02a3648690ef0b0dced398e6640864d40d3d49522561e3a24fb5285`. Root canonical archives are under
`muse_bf16_proj/profiles/stage-xor-release.dfcfdd01d6f04525.*`; input path,
byte length and hash manifest plus decoded data are under
`docs/steering/evidence/2026-11-01-projection-stage-xor-terminal/`.
The original global result remains preserved separately.

The reader rejects foreign/running receipts, nonfinite or failed benchmarks,
duplicate/missing cases or profiles, score mismatch, invalid stage extent,
incorrect arena size, malformed C2 pmax, FULL/NF table mismatch, admission
regression and incomplete qualification. Its six synthetic tests pass;
terminal-resource/counterfactual reconstruction is independently checked.

MoE independent terminal review `91f0204` reconstructed both owning scores
before invoking this reader, checked all forty resource formulas/capacities,
then matched ten summary metrics and all thirty median pairs. Its six owner
synthetic tests and three additional independent contract tests pass. The
worker's three archived-data tests also reconstruct official, raw and
counterfactual ratios and verify coverage, resource admission and input pins.

Retain C2 rather than this variant for the owned comparison. Both exact
global-record and stage-local BEFORE-A representations have now failed their
performance gate. Do not retry by changing capacity, tiles, stage count,
placement or selected targets. A reopening needs a genuinely different
dataflow/representation with new instruction/dependency evidence and a
credible full-thirty-cell >=1% paired-score opportunity; locality intuition
or recovering an already losing variant is insufficient. Existing FP64,
BF16 boundary, fence/lifetime, admission, fresh-state and score gates remain
mandatory. No next experiment or quota request is made in this closure.
