# Performance review, November 2 at 18:33 Vancouver time

Evidence snapshot `77023d44130f293acbd5f8f2d8165ae6e0eab2c2`, following `2ffc98c1`. Review completed at 2026-11-02 18:33 MST. This review launches no jobs and modifies no running experiment. The separately user-assigned General Optimizer experiment below has its own source and execution record.

## Current decision

Public G1b4 remains2.126083x. Attention E has no isolated confirmed gain. Both matched attempts stopped in setup. Constant WGS scales pass correctness but fail the performance gate. The corrected scale/group combination and current skinny NS4 comparison are pending. A still leads the staged B implementations actually tested at128/256; these new G-native changes do not compare A against B. Component factors are not added to whole-layer scores.

## Attention confirmation failures

`4fece18b20a2447b`, payload109cdb07/sourcec7e13600, ended 2026-11-03T01:02:18.318079+00:00 with an unsupported FP8 finiteness operation. `62a4f6b27ed74b50`, corrected payloadf397bca8/sourcef90da774, ended 2026-11-03T01:27:50.544016+00:00 with `EXPERT_FRESH_ROUTING_MISMATCH`. Each owning receipt contains one failed setup, return112 and no tests, benchmarks, profiles or matched timings. Neither supplies an attention timing or correctness verdict. The existing exploratory full-context D32 reduction5.035us remains unresolved because unchanged sliding moved more. [First failure](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/decode32/evidence/D32-ATTN-02/status-wait.raw.json), [second failure](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/decode32/evidence/D32-ATTN-03/status-wait.raw.json).

The strict routing gate stays. A bounded diagnostic payload9e5e5d64,929890 bytes, records load, seed, replay and differing placements before raising the same failure. It is prepared, with no accepted diagnostic job at this snapshot. The failed receipt cannot identify whether controls, treatment or instrumentation caused the mismatch. Decode32 Worker owns the investigation. [Owner decision](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/decode32/2026-11-02-11-matched-routing-mismatch.md). Read-only owner cursor `1cba8e71-65d3-4a48-b3c2-f00e5feaafa6:5` agrees. No message was sent to that chat.

## Constant WGS scale multiplier, component gate failed

Own job3aa3885f6b734f8c, sourceca588999, payloadccef8a76/342733 bytes, native881874d4 versus c7c201ae, changes only B4/T32 and MAIN/T128/T256 in the standalone adapter of current G. It passes5 official tests,6 benchmarks and6 profiles. All54 private cases,378 observations,216 captures,1152 timed replays and54 restorations pass; worst global/row relative error0.000138487/0.000372057. The official2.341016864x versus SGLang, CI2.305901611–2.376132117, is a separate comparator from the matched treatment/control factors. [Owning receipt](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/evidence/2026-11-02-cuda-issue-groups/status-wait.raw.json), [source association](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/evidence/2026-11-02-cuda-issue-groups/accepted.json).

The24 whole-six matched factors span0.992411–1.004932. All six bank/distribution groups fail the preregistered1% and duplicate-null gate. Their descriptive geometric mean0.997953 does not establish a causal slowdown because unchanged small paths drift too. Three affected bank/load cells cross the1% regression threshold. The exact scale-only treatment is parked, without confirmation or runtime promotion. Removing local B4 spills did not establish a speedup. [Reader analysis](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/evidence/2026-11-02-cuda-issue-groups/analysis.json), [summary](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/evidence/2026-11-02-cuda-issue-groups/summary.json).

The following numbers are microseconds. Control/candidate columns average the two placement medians. Delta and percent are descriptive differences of those medians, while the decision uses the paired-log crosses. Each placement has8 cold replays. There is no treatment confidence interval in the receipt. The census lists local route entries / occupied local experts / maximum entries per expert. Duplicate columns are mean absolute paired-log nulls, expressed in percent. All36 rows come from the linked reader analysis.

| Seed | Tokens | Routing | Census | Control us | Candidate us | Delta us | Delta % | Four paired factor range | Duplicate log nulls |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1102031 | 1 | normal | 2/2/1 | 31.080 | 31.584 | +0.504 | +1.622% | 0.975452–1.006536 | 2.529/3.595% |
| 1102031 | 8 | normal | 13/10/2 | 89.416 | 89.144 | -0.272 | -0.304% | 0.999950–1.005477 | 0.388/1.765% |
| 1102031 | 16 | normal | 33/15/7 | 125.360 | 124.704 | -0.656 | -0.523% | 0.991427–1.007654 | 0.699/0.782% |
| 1102031 | 32 | normal | 56/18/11 | 149.168 | 149.432 | +0.264 | +0.177% | 0.990264–1.002189 | 1.000/1.163% |
| 1102031 | 128 | normal | 238/25/40 | 202.032 | 202.320 | +0.288 | +0.143% | 0.994086–1.001333 | 0.761/0.165% |
| 1102031 | 256 | normal | 443/26/84 | 230.984 | 230.768 | -0.216 | -0.094% | 0.993485–1.004278 | 0.608/0.484% |
| 1102031 | 1 | skewed | 2/2/1 | 31.208 | 31.712 | +0.504 | +1.615% | 0.975709–1.007475 | 1.504/2.622% |
| 1102031 | 8 | skewed | 17/3/8 | 38.248 | 38.176 | -0.072 | -0.188% | 0.990607–1.000545 | 2.158/1.689% |
| 1102031 | 16 | skewed | 33/3/16 | 55.728 | 56.864 | +1.136 | +2.038% | 0.976458–0.999442 | 1.215/2.802% |
| 1102031 | 32 | skewed | 69/4/32 | 60.864 | 61.320 | +0.456 | +0.749% | 0.981910–1.011477 | 1.690/2.933% |
| 1102031 | 128 | skewed | 281/4/128 | 98.928 | 98.880 | -0.048 | -0.049% | 0.991235–1.001910 | 1.428/1.132% |
| 1102031 | 256 | skewed | 556/5/256 | 151.232 | 150.328 | -0.904 | -0.598% | 0.999801–1.011805 | 0.370/0.973% |
| 1102037 | 1 | normal | 3/3/1 | 39.264 | 39.560 | +0.296 | +0.754% | 0.983976–1.004681 | 1.253/1.182% |
| 1102037 | 8 | normal | 14/8/3 | 74.504 | 74.576 | +0.072 | +0.097% | 0.989915–1.008084 | 1.950/1.178% |
| 1102037 | 16 | normal | 23/11/6 | 95.672 | 96.016 | +0.344 | +0.360% | 0.996736–1.003289 | 0.385/0.744% |
| 1102037 | 32 | normal | 42/9/22 | 88.208 | 88.832 | +0.624 | +0.707% | 0.995786–1.008696 | 1.470/1.745% |
| 1102037 | 128 | normal | 198/22/84 | 189.144 | 188.352 | -0.792 | -0.419% | 0.996462–1.004274 | 0.330/0.935% |
| 1102037 | 256 | normal | 441/26/173 | 242.648 | 242.720 | +0.072 | +0.030% | 0.995353–1.001594 | 0.530/0.478% |
| 1102037 | 1 | skewed | 2/2/1 | 31.120 | 31.240 | +0.120 | +0.386% | 0.978727–1.024672 | 3.653/2.046% |
| 1102037 | 8 | skewed | 16/3/8 | 37.920 | 38.312 | +0.392 | +1.034% | 0.982145–1.002464 | 2.310/2.180% |
| 1102037 | 16 | skewed | 32/3/16 | 55.552 | 55.616 | +0.064 | +0.115% | 0.987804–1.007779 | 1.370/1.368% |
| 1102037 | 32 | skewed | 63/3/32 | 55.512 | 56.064 | +0.552 | +0.994% | 0.995638–1.006620 | 1.864/1.877% |
| 1102037 | 128 | skewed | 255/3/126 | 90.696 | 90.480 | -0.216 | -0.238% | 1.000448–1.004536 | 0.610/1.080% |
| 1102037 | 256 | skewed | 523/3/253 | 136.216 | 136.112 | -0.104 | -0.076% | 0.992462–1.005408 | 0.541/0.806% |
| 1102049 | 1 | normal | 2/2/1 | 31.192 | 31.384 | +0.192 | +0.616% | 0.992073–1.017564 | 2.900/2.775% |
| 1102049 | 8 | normal | 14/8/4 | 73.800 | 74.360 | +0.560 | +0.759% | 0.983811–0.999031 | 1.405/0.802% |
| 1102049 | 16 | normal | 32/15/9 | 130.464 | 130.376 | -0.088 | -0.067% | 0.998107–1.002150 | 0.173/0.634% |
| 1102049 | 32 | normal | 67/17/17 | 143.592 | 143.584 | -0.008 | -0.006% | 1.000054–1.005798 | 0.999/0.759% |
| 1102049 | 128 | normal | 239/20/55 | 170.472 | 169.872 | -0.600 | -0.352% | 0.997839–1.006824 | 0.445/0.509% |
| 1102049 | 256 | normal | 491/23/125 | 228.936 | 228.696 | -0.240 | -0.105% | 0.998667–1.006191 | 0.380/0.438% |
| 1102049 | 1 | skewed | 3/3/1 | 39.336 | 39.744 | +0.408 | +1.037% | 0.977990–0.997222 | 1.801/2.750% |
| 1102049 | 8 | skewed | 26/4/8 | 45.400 | 46.408 | +1.008 | +2.220% | 0.972408–0.988860 | 0.908/1.980% |
| 1102049 | 16 | skewed | 51/4/16 | 78.032 | 77.496 | -0.536 | -0.687% | 0.994903–1.020813 | 1.601/2.089% |
| 1102049 | 32 | skewed | 106/4/32 | 64.072 | 64.256 | +0.184 | +0.287% | 0.989377–1.014361 | 1.700/1.296% |
| 1102049 | 128 | skewed | 406/4/128 | 119.768 | 120.528 | +0.760 | +0.635% | 0.983493–1.004679 | 1.022/1.583% |
| 1102049 | 256 | skewed | 832/4/256 | 192.096 | 191.456 | -0.640 | -0.333% | 0.996922–1.012595 | 0.740/0.619% |

## Pending exact candidates and corrected compiler scope

Corrected constant-scale plus larger-WGMMA-group job3c45788127e945c4, source9669ee15, payloadc4e6d108/342999 bytes, native7db5ec77, is accepted. Local B4/MAIN compilation has96 registers and no spills; matching collective/publication counts support a bounded target test. No target result is recorded. The initial group-only and initial combined sources skipped the deferred flush because their four-operation group could not equal the original ten-operation threshold. Their spill counts remain real for those flawed sources, but cannot reject the corrected mechanism. The new threshold is the first completed group at or after the original boundary. This is a combined G experiment, not isolated scale evidence. [Registered source and compiler correction](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/2026-11-02-cuda-part2-experiments.md), [accepted job](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/evidence/2026-11-02-cuda-group-scale/accepted.json).

General Optimizer's separately authorized current-helper skinny NS4 versus NS3 jobf0b875484960438b, source21db3bc3, payloadcf1dbda3/343611 bytes, is accepted. Only tiny/tiny2/MID at T1/T8/T16 change depth; B4/T32 and MAIN/T128/T256 remain identical. Three held-out banks, normal/skewed EP8,8 repeats/placement/cell and duplicate controls are registered. Six local builds pass without spills, with unchanged214784-byte allocation. No GPU result is recorded. The old75bca owned-skinny result is a different implementation. [Exact scope and gates](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/docs/steering/2026-11-02-general-optimizer-pipeline-depth.md), [accepted job](https://github.com/yelgabo/okc/blob/77023d44130f293acbd5f8f2d8165ae6e0eab2c2/muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/accepted.json).

## Checkpoint and next evidence

Inspected new commits, run-log and ruled-out changes through77023d44. Parsed5 new profile JSON documents; one pending empty own-job status file is an explicit placeholder, not a result. Parsed all18 untracked historical local profile JSONs without staging or modifying them. Scale reader factors and source hashes were recomputed against the raw receipt and registered source. Both attention failure event counts were verified. No independent reviewer was used under the current no-new-agents direction.

Next evidence is the owned routing diagnostic, corrected grouping result and skinny depth result. Retain the incumbent until their respective gates pass. Do not relax natural-routing equality to obtain a timing. No isolated A/B or whole-layer improvement follows from pending source or compiler results.
