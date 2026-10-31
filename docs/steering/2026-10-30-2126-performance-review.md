# Hourly review: F comparisons measured; combined layer gain and unresolved attribution

Reviewed 2026-10-30 21:26 PDT. Evidence through `d73c4fd`. New measurements supersede the pending labels from the prior
review. The latest recorded decision makes F a measurement and learning track only.
Winning mechanisms must be re-implemented in owned code with attribution; F0/F1/F2 are
not submission candidates. Historical adoption recommendations below are superseded by
this [registry addendum](2026-10-30-family-f-donor-baseline.md#addendum-2026-10-31-user-decision-on-the-use-of-f-results).

### Layer: combined M1 + 51 gain, isolated A still unresolved

All four-arm comparisons below independently prepare the release and candidate twice,
use two inputs and eight rounds per shape, and pass 24/24 checks per arm. Ratios are
weighted score ratios, not elapsed-time percentages. No paired confidence intervals are
recorded; independent nulls determine the written decision.

| Candidate / job | Source or file hash | Pooled ratio | Release / candidate null | Result |
|---|---|---:|---|---|
| layer_g51b / `64691c2e128247f4` | acd2385c | 0.9991 | 0.9993 / 1.0071 | Inconclusive; isolated mode 51 gain remains unresolved |
| layer_m1rt51c / `4d4ea5db46db4a10` | 1eda8ec9 | 1.0137 | 0.9884 / 1.0012 | Inconclusive; incumbent null too large |
| layer_hp / `6001272c2b1c4852` | 0efd553c | 1.0003 | 0.9987 / 1.0064 | Inconclusive; PRQ + HMRG + workspace pick bundle |
| Stable A/A / `e582fde557114062` | layer_stable 752123dd | 0.9989 | 0.9988 / 0.9997 | Expected null; stable-picker test passes |
| layer_m1rt51s / `3aee22ba4a404c8a` | f1235356 vs release_s a45fae28 | 1.0052 | 1.0008 / 1.0024 | Combined candidate faster under written rule |

The stable M1 + 51 comparison uses seeds 250001 onward and order seed 73. Its four cross
ratios span 1.0036-1.0068. At 128, sliding control medians 271.2/270.4 us have paired
cross deltas -3.9 to -4.7 us, or -1.44% to -1.74%; full controls 346.0/347.4 have
-4.4 to -5.8 us, or -1.27% to -1.67%. At 256, sliding controls 347.2/344.8 have
-0.7 to -1.6 us, or -0.20% to -0.46%; full controls 517.5/516.0 have -0.2 to
-4.4 us, or -0.04% to -0.85%, with full nulls -2.6/+1.6 us. The weighted 256
result remains inconclusive. Decode 1 regresses, ratio 0.9887. This is one measured
combined M1 + A gain, not an isolated family-A improvement.
[Layer evidence and per-case tables](../runs-decode-layer.md),
[stable paired log](../../muse_decode_layer/profiles/bench9_m1rt51s.3aee22ba4a404c8a.log).

The earlier comparisons also cover 128/256. g51b's weighted ratios are 1.0164/1.0041;
m1rt51c's are 1.0089/1.0071; hp's are 0.9964/1.0015. Their overall decisions remain
inconclusive. The workspace-pick hp build is withdrawn, rather than still awaiting a test.
M1 reproduction `9d00f194ed29471b` invalidated stream capture and produced no stage
records. M1 was off in the timed path, so it establishes no new M1 failure attribution.
[Layer run log](../runs-decode-layer.md).

Three scored repeats of layer_m1rt51s at `0bc16c6` pass: `ad42a066bd3a4e01`
2.006935x, CI 1.976831-2.037039; `138ae1a402934c44` 2.001632x,
CI 1.971607-2.031656; `0f58d5d8223f45af` 2.014815x,
CI 1.984593-2.045037. The best is within noise of the old 2.0143x receipt.
This does not establish a cross-run score gain.
[Scored receipt](../../muse_decode_layer/profiles/scored.0f58d5d8223f45af.json),
[repeat 1](../../muse_decode_layer/profiles/scored.ad42a066bd3a4e01.json),
[repeat 2](../../muse_decode_layer/profiles/scored.138ae1a402934c44.json).

**Unresolved source conflict:** the run log calls `c8d1e57261dd4901` a second M1+51
confirmation at 1.0106, while the [M-B plan](../plan-2026-10-31-m-b-pdl-chain.md) identifies
it as layer_mb1s. No local receipt or raw log exists for that job. Do not count it as a
replication or assign its gain to either candidate yet. Release-manifest headings saying
these candidates are untested/unsubmitted are stale against the receipts above.

### F-LAYER: repeated measurement win with eligibility caveats

Jobs `20abe6291eb04ed6` and `00607fc6f8c84c77` compare pinned donor `6cae49a`
layer hash 8f17613b with release `2cee446`, hash b9c23bda. Builder source is `bc10cb1`
plus a scratch comparison patch. Each uses four arms, two inputs and 12 rounds per shape,
new seeds, preparation order and allocation pads; all 24/24 checks pass per arm.
Pooled score ratios are 1.0897/1.0949, with null pairs 1.0054/1.0009 and
0.9835/1.0011. Both clear the rule, but donor copies share module scratch, weakening that null.

At 128, first-run sliding deltas are -20.3 to -20.9 us against 293.3-295.0 us,
about -6.9% to -7.1%; full -23.9 to -26.8 against 379.9-383.1, about -6.3% to
-7.0%. At 256, sliding -32.6 to -34.8 against 362.0-362.3, about -9.0% to -9.6%;
full -16.6 to -20.3 against 512.5-515.1, about -3.2% to -4.0%. The confirmation
retains the sign; weighted 128/256 ratios are 1.0664/1.0893 versus 1.0734/1.0792
in the first run. No paired CI is recorded.
[Full cases and methods](../runs-decode-layer.md),
[first log](../../muse_decode_layer/profiles/benchF0a.20abe6291eb04ed6.log),
[confirmation](../../muse_decode_layer/profiles/benchF0b.00607fc6f8c84c77.log).

These measurements do not clear the donor for adoption. Recorded findings include a T1
route reading PART/PO before its wait, an unresolved q8 nonunit-kv-scale precision question,
and shared growing scratch lifetime concerns. Stage attribution is provisional because
kernel names were lost. Do not assign the full gain to A, D or E. M-A MoE and M-B PDL
chain work are next; M-C requires an ordering repair before use.
[Eligibility review](../runs-decode-layer.md), [M-B plan](../plan-2026-10-31-m-b-pdl-chain.md).

### E and attention context: ring variants lose; workspace selection withdrawn

Round-4 benches use two inputs, 200 cold rounds and pooled medians. Every real arm passes
11/0 checksum checks; all five receipts pass 16/16. No paired CI is recorded. These switch
16/32-token full attention, not the 128/256 full paths, whose code is unchanged.

| Job / revision | Candidate versus control | 16 / 32 full delta us | Result |
|---|---|---|---|
| `bce37d22a0ec4045` / d12b20e | HMRGF merge, controls 97.89/173.25 | -0.29/-0.13 (-0.30%/-0.08%) | Small gain below 0.20% weighted promotion gate; separate merge context |
| `c8e99e038d9f4575` / 2bf3967 | QK2, controls 100.58/179.36 | +0.12/+0.32 (+0.12%/+0.18%) | No gate pass; PE16 -0.04/+0.03 is null |
| `4b60264a276c4805` / 0b326a4 | HMRGF repeat, controls 100.45/179.39 | -0.22/-0.19 (-0.22%/-0.11%) | Below gate; own-state -0.29/-0.35 with null -0.06/-0.22 |
| `42c4739cf6b247c4` / 5064e2b | E NS4/5/6 versus NS3, controls 100.58/179.49 | NS4 +0.80/+0.29 (+0.80%/+0.16%); NS5 +1.24/+0.99 (+1.23%/+0.55%); NS6 +1.72/+2.24 (+1.71%/+1.25%) | Losses, null -0.07/-0.07; NS6H +1.85/+2.27 also loses |
| `a7e5331d89ba4f77` / 5cb2be0 | E NS2 versus NS3, controls 100.61/179.26 | +3.65/+8.87 (+3.63%/+4.95%) | Loss; NS2H +3.39/+8.80 also loses; null +0.00/+0.20 |

NS3 remains best of these tested depths. PRQ's earlier win and XPIPE losses remain.
Producer-only zero-data and no-DRAM probes are diagnostics, not production gains.
[Round-4 records, all variants and profiles](../runs-attention.md).

Workspace pick was removed, with kernel SASS unchanged. Own-code `dcda02e` check
`3250ab04126b4865` passes 16/16, 2.53674x, CI 2.49869-2.57479. Scored
`68d3b88fbc8b40da` passes, 2.53657x, CI 2.49852-2.57461, replacing the recorded
2.521 entry. Cross-run comparisons do not isolate the effect of removing the picker.
This is no new E pipeline gain. [Scored receipt](../../muse_attention_decode/profiles/scored.68d3b88fbc8b40da.json).

### F-ATTN: controlled overall win, some large-token cells unresolved

Job `0519d0b15e9c4583`, source `589fb00`, compares release `a70192d` with pinned
attention donor `6cae49a`, hash e6d929d4. Four independent prepares, three inputs,
150 cold rounds across 12 shapes, 19/0 arm checks and 16/16 judge checks. Donor score
ratios 1.0097/1.0083 exceed the written gate and sum-null uncertainty of 0.50%.
No paired CI is recorded. The judged score is the release's, not a donor score.

| Tokens / context | Mean release us | Mean donor us | Delta us / percent | Decision |
|---|---:|---:|---|---|
| 128 sliding | 15.630 | 15.505 | -0.125 / -0.80% | Inconclusive against nulls |
| 128 full | 113.170 | 111.040 | -2.130 / -1.88% | Donor win |
| 256 sliding | 19.935 | 19.330 | -0.605 / -3.03% | Donor win |
| 256 full | 193.860 | 193.345 | -0.515 / -0.27% | Inconclusive against nulls |

Deltas subtract means of the displayed arm medians. Other resolved wins are d1 full,
d16 full, d32 full and d32 sliding. These belong to F, not isolated E. Two separate donor
checks `45adaeddef9d464e`/`ab38543d2c8642fb` score 2.57384/2.60532 versus release
check `b90af2a967b946b1` 2.54194, all pass 16/16; the cross-run difference alone is
inconclusive within recorded 1-3% spread. `cb99c110c9cd4c01` duplicates an old receipt,
not a new run. [F-ATTN methods and results](../runs-attention.md),
[controlled log](../../muse_attention_decode/profiles/bench_o.0519d0b15e9c4583.log).

### F-PROJ: repeated measured gain; documented host comparison patches

Jobs `434c01d39a1b4fa6`/`1ea00e9b5bbd49b0`, `bc10cb1` plus scratch bench,
compare release `2265fa2`, hash 6fe7b146, and donor `6cae49a`, original hash 290a86ab.
The comparison removes threading and serializes builds; kernel code is preserved, but this
is not a byte-identical whole-file F0. Four independent preparations use three inputs x40
rounds, then four x60, with new pads/seeds and reversed preparation order. All 129/129,
then 159/159 checks pass per arm, including fresh inputs, FP64, replay and retention.
Pooled weighted ratios 1.0363/1.0396 clear independent nulls 0.9968/1.0001 and
0.9999/1.0009. Stub judge scores are meaningless for donor performance. No paired CI.

| Case | First control us / paired delta | Confirmation control us / paired delta |
|---|---|---|
| qkv 128 | 14.94 / -0.86 (-5.76%) | 14.68 / -0.80 (-5.45%) |
| pooler 128 | 21.64 / -1.15 (-5.31%) | 21.61 / -1.15 (-5.32%) |
| router 128 | 7.15 / -0.82 (-11.47%) | 6.87 / -0.83 (-12.08%) |
| router 256 | 9.70 / -2.30 (-23.71%) | 9.39 / -2.40 (-25.56%) |

These are D1-R1 pairs. Full four-cross tables retain other cases and nulls. No resolved
single-token gain; several small-token qkv/pooler/o paths improve. Other 256 cases are mixed
or unresolved. The router's grid join relies on co-residency and needs redesign before a
layer port. Measured donor gains require reimplementation, not import.
[Full 30-case evidence](../runs.md),
[first log](../../muse_bf16_proj/profiles/bench_f4a.434c01d39a1b4fa6.log),
[confirmation](../../muse_bf16_proj/profiles/bench_f4b.1ea00e9b5bbd49b0.log).

### Other projection diagnostics and hypotheses

Source for BG/RG/LTS is recorded as branch `proj-r6`, evidence committed in `5ef830b`;
the measured exact source SHA is not supplied. Receipts pass nine tests each. They do not
establish isolated candidate score gains; no paired confidence intervals are recorded.

- BG `cab546f978974515`: 60 cold paired rounds at 256, four independent weights.
  Fixed qkv config costs 17.47 us singly, 18.77/24.10 per instance at batch2/4.
  A different batched algorithm reaches 12.46 per instance. That changes workload and is
  no single-call win. o costs 8.70 singly, 7.26/7.20 per instance batched; nulls -0.08/-0.02.
  No 128 test; missing profiler names limit attribution.
- RG `97869fa8302041b5`: 100 cold paired rounds. At 128, control 7.07 us:
  A+S spin -0.22 (-3.11%, 84/100 faster), G -0.13 (-1.84%, 69/100), A watchdog
  +0.48 (+6.79%, 8/100). At 256, control 9.57: G/A+S +0.03, within null;
  A watchdog +0.19 (+1.99%, 24/100). Nulls -0.03/+0.00. Spin arm is not shippable;
  G+A lacks required twins, donor control failed CUDA13 build. No production change.
- LTS `8b9e8b7c1ecd4c3e`: 60 cold paired rounds after screening 1,680 configs/shape
  with FP64 checks. qkv256 large-tile splits +2.74 to +3.10 us versus 17.68,
  +15.50% to +17.53%, zero of 60 faster. qkv128 +1.18 to +1.46 versus 14.98,
  +7.88% to +9.75%, two to five of 60 faster. o256 +0.13 to +0.54 versus 9.01,
  with null +0.18, no winner. Same-tile qkv256 -0.18 with 36/60 faster is unresolved.

[Diagnostic run records and profile links](../runs.md),
[BG log](../../muse_bf16_proj/profiles/bench_bg.cab546f978974515.log),
[RG log](../../muse_bf16_proj/profiles/bench_rg.97869fa8302041b5.log),
[LTS log](../../muse_bf16_proj/profiles/bench_lts.8b9e8b7c1ecd4c3e.log).
Codec pair7/quad12 remain CPU models, not GPU wins. Predicted +1.44%/+2.58% depends on
hidden decode cost. Dense whole-tile sketch was declined without a run. Neither is family C.

### A component follow-ups: modes 76-78

Mode 76 pairs scale loads; 77 changes the small-token final pass; 78 combines them on 51.
Job `5b2db8029f094a66` at `45f990d` failed to build its donor comparison arm, so it
produced no A/B evidence. Its passing judged score 2.205x is not a candidate gain.
[Run record](../runs-moe.md#donor-moe-m-a-against-mode-51-2026-10-31).

Job `64254f49676c4ce6` at `5f19bc1` tests six seeds 121-126 per cell. Job
`c8fc5f669fa34d06` at `95c9848` retests 78 with seeds 131-136. Both use cold-L2
median-of-20 graph replays per arm and paired seed means; no paired confidence intervals.
Layer-like routing is a component workload, not a whole-layer test. Negative deltas mean faster.

[bench_m76.64254f49676c4ce6 raw log](../../muse_moe_w4a16/profiles/bench_m76.64254f49676c4ce6.log)

| Tokens / routing | Arm | Control us | Paired delta us | Percent | Seeds faster |
|---|---|---:|---:|---:|---:|
| 32 / layer | o76 | 90.7 | -0.1 | -0.11% | 3/6 |
| 32 / layer | o77 | 90.7 | -3.1 | -3.42% | 6/6 |
| 32 / layer | o78 | 90.7 | -3.5 | -3.86% | 6/6 |
| 32 / std | o76 | 149.3 | -0.1 | -0.07% | 4/6 |
| 32 / std | o77 | 149.3 | -1.8 | -1.21% | 6/6 |
| 32 / std | o78 | 149.3 | -2.4 | -1.61% | 6/6 |
| 32 / skewed | o76 | 69.0 | -0.0 | -0.00% | 3/6 |
| 32 / skewed | o77 | 69.0 | -1.9 | -2.75% | 6/6 |
| 32 / skewed | o78 | 69.0 | -2.9 | -4.20% | 6/6 |
| 128 / layer | o76 | 157.3 | -0.6 | -0.38% | 5/6 |
| 128 / layer | o77 | 157.3 | +0.7 | +0.45% | 2/6 |
| 128 / layer | o78 | 157.3 | -0.4 | -0.25% | 4/6 |
| 128 / std | o76 | 202.9 | -0.0 | -0.00% | 3/6 |
| 128 / std | o77 | 202.9 | -0.0 | -0.00% | 4/6 |
| 128 / std | o78 | 202.9 | -0.2 | -0.10% | 4/6 |
| 128 / skewed | o76 | 94.5 | -0.3 | -0.32% | 4/6 |
| 128 / skewed | o77 | 94.5 | +0.4 | +0.42% | 1/6 |
| 128 / skewed | o78 | 94.5 | -0.3 | -0.32% | 5/6 |
| 256 / layer | o76 | 223.0 | -0.8 | -0.36% | 6/6 |
| 256 / layer | o77 | 223.0 | +0.1 | +0.04% | 3/6 |
| 256 / layer | o78 | 223.0 | -0.9 | -0.40% | 6/6 |
| 256 / std | o76 | 247.3 | -0.5 | -0.20% | 3/6 |
| 256 / std | o77 | 247.3 | +0.5 | +0.20% | 3/6 |
| 256 / std | o78 | 247.3 | -1.0 | -0.40% | 5/6 |
| 256 / skewed | o76 | 149.6 | -1.1 | -0.74% | 6/6 |
| 256 / skewed | o77 | 149.6 | +0.0 | +0.00% | 3/6 |
| 256 / skewed | o78 | 149.6 | -1.3 | -0.87% | 6/6 |

[bench_m78.c8fc5f669fa34d06 raw log](../../muse_moe_w4a16/profiles/bench_m78.c8fc5f669fa34d06.log)

| Tokens / routing | Arm | Control us | Paired delta us | Percent | Seeds faster |
|---|---|---:|---:|---:|---:|
| 32 / layer | o78 | 90.8 | -2.6 | -2.86% | 6/6 |
| 32 / std | o78 | 145.0 | -1.5 | -1.03% | 5/6 |
| 32 / skewed | o78 | 70.5 | -3.1 | -4.40% | 5/6 |
| 128 / layer | o78 | 161.6 | -0.6 | -0.37% | 4/6 |
| 128 / std | o78 | 191.3 | +0.6 | +0.31% | 1/6 |
| 128 / skewed | o78 | 99.7 | -0.8 | -0.80% | 5/6 |
| 256 / layer | o78 | 218.5 | -0.6 | -0.27% | 6/6 |
| 256 / std | o78 | 236.7 | -0.4 | -0.17% | 5/6 |
| 256 / skewed | o78 | 163.3 | -1.3 | -0.80% | 5/6 |

Both receipts pass, with spin flags zero. In the first gate, all arms match 51 exactly at
128/256; at 32 the maximum difference is 4.7e-5. Mode 76 misses its performance threshold
and its written 1e-5 numerical bound at 32, where the run record attributes variation to
51's atomics. Mode 77 misses the +0.5 us regression limit at 128 layer-like routing.
Thus neither, nor their conditionally eligible combination, was kept. The separate 78 gate
passes output checks, exact at 128/256 and at most 3.8e-5 at 32, but loses +0.6 us at
128 standalone against the +0.5 limit. It is not kept. Its 32-token gains remain measured.
The next fin4 idea restricts dispatch to below 64 tokens; it has not been tested.
[Gate decisions and raw receipts](../runs-moe.md), [rejection register](../ruled-out.md).

### F-MOE: complete-file measurement, not an eligible replacement

Exact donor `6cae49a` versus release `ec565a7` in jobs `40b8c798289a4258` and
`218ff300de7d4c22` gives pooled weighted score ratios 1.0387 and 1.0297. Independent
nulls are 0.9996/0.9993 and 1.0017/0.9977. Each uses four independently prepared arms,
12 standalone seeds, six layer-like seeds, four robustness seeds, median-of-20 cold replays.
Fresh-input replay passes and no numerical failures occur. At 1 token ours wins; at 8 the
donor wins; 16 is inconclusive across routings; 32/128/256 favor the donor.

At 128 standalone, first-run R1/R2 means 197.36/197.40 us versus D1/D2 182.63/183.16,
paired deltas -14.73/-14.24 us (-7.46%/-7.21%). At 256, 255.85/256.77 versus
228.85/228.81, -27.00/-27.97 us (-10.55%/-10.89%). All 12 seeds improve in each pair.
The confirmation preserves the sign; raw means and all routing ratios remain in the source.
No paired confidence interval is recorded.
[Full F-MOE evidence](../runs-moe.md#f-moe-0-family-f-donor-comparison-2026-10-31-utc-2026-10-30-pdt).

The donor fat path lacks the writer/reader global proxy fences required by its generic-to-async
handoff. Passing numerical tests does not establish safe ordering. F1 fence costs remain
unmeasured. These wins guide an attributed re-implementation with the required fences, not
verbatim submission. Both-phase stream-K with the donor walk order is a proposed coherent
alternative; modes 72/74 did not test that full combination. A-versus-B's staged gate is unchanged.


Validation: independent reviewers checked layer, attention, projection and F evidence; all 36 MoE76-78 table rows were checked against raw logs. Review corrections changed control means to medians and tightened one elapsed-time range. Source links resolve; dashboard filters and 30 comparison rows pass executable checks, with four context cards. Chrome inspection found no collisions in the updated E/F cards and pending section. The builder verifies timestamps and source fingerprint. The c8d1 receipt/attribution gap remains unresolved. No kernels, benchmarks, submissions or pushes were changed by this review.
