# Performance review, November 2 at 18:54 Vancouver time

Evidence snapshot `93ed0192d3a60f199c87b9b4771c54fdefc6179f`. Review completed at 2026-11-02 18:54 MST, following77023d44. Public G1b4 remains2.126083x. Both new G-native component checks pass correctness and fail their frozen performance gates. The attention routing diagnostic5909739e8a4445b2 is accepted, without a recorded result. [D32 owning registration](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/docs/steering/decode32/evidence/D32-ATTN-04/accepted.json).

## Corrected scales plus larger WGMMA groups

Job3c45788127e945c4, source9669ee15, payloadc4e6d108/342999 bytes, native7db5ec77 versus c7c201ae, completed2026-11-03 01:40:22 UTC. It passed5 tests,6 benchmarks,6 profiles and all54 private cases/378 observations/216 captures/1152 cold timings/54 restorations. Worst global/row errors0.000135982/0.000171862; private preparation276.366447 seconds. Official2.348334366x,95% CI2.313109350–2.383559381, is versus SGLang. All24 whole-six factors0.995560–1.008180 miss1.01 and the duplicate-null gate; all6 groups fail, with3 affected bank/load regression cells. Park exact combination, with no confirmation or ingredient-level causal claim. Positive compiler changes did not establish an endpoint gain. [Raw receipt](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/docs/steering/evidence/2026-11-02-cuda-group-scale/status-wait.raw.json), [full analysis](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/docs/steering/evidence/2026-11-02-cuda-group-scale/analysis.json).

## Current skinny NS4 versus NS3

Jobf0b875484960438b, source21db3bc3, payloadcf1dbda3/343611 bytes, completed2026-11-03 01:48:03 UTC. Only T1/T8/T16 helper depth changes; T32/128/256 compiler requests are identical. It passed5 tests/6 benchmarks/6 profiles and the same54-case private qualification. Worst global/row errors0.0000754371/0.0001989355; private preparation279.219903 seconds. Official2.293332488x,95% CI2.258932501–2.327732475, is versus SGLang. All24 affected aggregate factors0.966414–0.990167 are below1; descriptive mean0.977832, with17/18 affected cells crossing the1% regression threshold. An unchanged256/skewed cell at bank1102049 also exceeds its frozen drift threshold, so the precise causal slowdown remains unresolved. Mean affected changes are+0.675..+1.539us. RetainNS3, park exactNS4, no confirmation or layer gain. [Raw receipt](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/status-wait.raw.json), [all36 rows and gate](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/muse_moe_w4a16/profiles/gopt-skinny-depth-2026-11-02/analysis.json), [terminal decision](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/docs/steering/2026-11-02-general-optimizer-pipeline-depth.md).

## Complete cell ledger

For each job: EP8 normal/skewed, three held-out banks1102031/1102037/1102049,8 cold repeats per placement/cell, two independent control and two candidate preparations. The columns average the two placement medians, then the three banks within each routing distribution. Delta percentages are descriptive. Paired-log crosses and duplicate-control uncertainty are retained in each linked full analysis; no treatment confidence interval was recorded. Correctness passes for every listed group.

| Job | Tokens | Routing | Control us | Candidate us | Delta us | Delta % | Four-cross range across banks |
|---|---:|---|---:|---:|---:|---:|---|
| 3c457881 | 1 | normal | 33.840 | 34.075 | +0.235 | +0.693% | 0.969064–1.022224 |
| 3c457881 | 1 | skewed | 33.984 | 33.992 | +0.008 | +0.024% | 0.988088–1.023295 |
| 3c457881 | 8 | normal | 79.248 | 79.013 | -0.235 | -0.296% | 0.988368–1.012082 |
| 3c457881 | 8 | skewed | 41.032 | 40.605 | -0.427 | -1.040% | 0.986211–1.020585 |
| 3c457881 | 16 | normal | 116.973 | 117.419 | +0.445 | +0.381% | 0.990599–1.006729 |
| 3c457881 | 16 | skewed | 63.387 | 63.061 | -0.325 | -0.513% | 0.989693–1.015079 |
| 3c457881 | 32 | normal | 126.995 | 126.349 | -0.645 | -0.508% | 0.989918–1.021387 |
| 3c457881 | 32 | skewed | 59.709 | 59.683 | -0.027 | -0.045% | 0.986870–1.011880 |
| 3c457881 | 128 | normal | 187.680 | 187.176 | -0.504 | -0.269% | 0.989638–1.013371 |
| 3c457881 | 128 | skewed | 103.517 | 103.285 | -0.232 | -0.224% | 0.992379–1.010255 |
| 3c457881 | 256 | normal | 233.227 | 233.752 | +0.525 | +0.225% | 0.994426–1.005527 |
| 3c457881 | 256 | skewed | 159.859 | 159.592 | -0.267 | -0.167% | 0.992381–1.012330 |
| f0b87548 | 1 | normal | 34.003 | 35.541 | +1.539 | +4.525% | 0.923102–0.986413 |
| f0b87548 | 1 | skewed | 34.224 | 35.267 | +1.043 | +3.047% | 0.967369–1.002856 |
| f0b87548 | 8 | normal | 79.264 | 80.248 | +0.984 | +1.241% | 0.959355–1.008782 |
| f0b87548 | 8 | skewed | 40.792 | 41.467 | +0.675 | +1.654% | 0.955772–1.009459 |
| f0b87548 | 16 | normal | 117.579 | 118.965 | +1.387 | +1.179% | 0.975043–1.002516 |
| f0b87548 | 16 | skewed | 63.669 | 64.965 | +1.296 | +2.036% | 0.959137–0.994563 |
| f0b87548 | 32 | normal | 126.064 | 126.363 | +0.299 | +0.237% | 0.989489–1.001918 |
| f0b87548 | 32 | skewed | 57.349 | 57.531 | +0.181 | +0.316% | 0.970926–1.009390 |
| f0b87548 | 128 | normal | 187.003 | 186.675 | -0.328 | -0.175% | 0.991055–1.005572 |
| f0b87548 | 128 | skewed | 102.616 | 103.387 | +0.771 | +0.751% | 0.982400–1.002804 |
| f0b87548 | 256 | normal | 233.579 | 233.936 | +0.357 | +0.153% | 0.990820–1.004343 |
| f0b87548 | 256 | skewed | 158.691 | 159.765 | +1.075 | +0.677% | 0.982387–1.000636 |

## Historical transfer audit

The user asked which older successes were lost in the F/G rewrites. The [published source audit](https://github.com/yelgabo/okc/blob/93ed0192d3a60f199c87b9b4771c54fdefc6179f/docs/steering/2026-11-02-historical-wins-transfer-audit.md) finds current counterparts for DCARRY/PK16/VHOIST/HMRG/PRQ, small live code groups and final-pass sums. Large-token G instead uses stage cuts/fixups and weighted static ranges, so old whole phase-A ownership and dynamic phase-B claiming are changed choices, not current retained mechanisms. Their historical gains do not measure a G transfer. C2 O8 register-direct is a real missing small operator, with a conditional old-delta sizing of0.050–0.055% whole-six, not a forecast. FRT/PR fusion and required arithmetic/order boundaries prevent blind pooler/router or PDL restoration. Workspace picking was deliberately user-removed. No new restore experiment is allocated.

A remains ahead of the staged B implementations actually tested at128/256. These are G component results, not isolated A/B comparisons. No component scores are added. The old next-stage hoist and sparse-XOR projection losses remain closed.

## Checkpoint

Inspected new commits, run logs, ruled-out entries and8 changed/new profile JSON documents since77023d44 through93ed0192. Parsed all18 untracked historical local profile JSONs without staging or modification. Recomputed source hashes, record counts and both frozen decisions against archived rows. An existing read-only reviewer independently recomputed the skinny table, factors, regression/drift counts and historical projection/source conclusions. No new reviewer or GPU job was created by this review. D32's separate owner keeps diagnostic5909739e8a4445b2; no job-service poll or message was sent to that owner.
