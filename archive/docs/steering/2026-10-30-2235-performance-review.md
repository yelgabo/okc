# Performance review, 2026-10-30 22:35 PDT

Evidence snapshot: `9bf7f05`. Documentation review only.

Evidence through `9bf7f05`, reviewed 2026-10-30 22:35 PDT. C1 and C2 projection
reimplementations win successive paired comparisons. C1's layer port is inconclusive.
The recovered M-B receipt measures the existing M1 + 51 policy because the added slice
was rejected. Both-phase stream-K mode 79 loses; attention PBULK hangs before timing.
These results keep A ahead of tested B at 128/256 and leave the full D scheduler untested.

### Layer: attribution resolved; PDL extensions and projection port remain inconclusive

The missing `c8d1e57261dd4901` receipt and log are now present. They confirm that the
file was `layer_mb1s`, hash `16feec52`, against `release_s`. Both candidate prepares
rejected the added unpooler PDL mode and selected the same M1 policy as `layer_m1rt51s`:
o+front at 8 tokens, o at 16/32. Thus pooled 1.0106 is another reading of that combined
policy, not a gain from the new M-B slice. The earlier 1.0052 reading and this 1.0106
reading differ by more than their within-run nulls. Keep run-to-run uncertainty visible.
The prepare-time acceptance rule requires savings above max(0.3 us, twice MAD)
in at least eight of nine rounds. The extension did not pass; exact per-mode times
were truncated from the log.
[Receipt](../../../muse_decode_layer/profiles/bench10_mb1s.c8d1e57261dd4901.json),
[raw policy and timings](../../../muse_decode_layer/profiles/bench10_mb1s.c8d1e57261dd4901.log).

All timed four-arm benches below use two inputs, eight rounds and 16 observations per arm.
There are no paired confidence intervals; cross ratios and nulls provide the recorded
uncertainty. Ratios are weighted whole-layer time ratios, not isolated family gains.

| Candidate / source hash | Job / order seed | Control | Preparation | Pooled ratio / null R, D | Decision |
|---|---|---|---|---|---|
| layer_mb1s / 16feec52 | `c8d1e57261dd4901`, 79, inputs 260001+ | release_s | Independent | 1.0106 / 0.9985,1.0013 | M1+51 policy faster; added slice rejected |
| layer_mb2s / 3aee22fd | `a3760918c13c4e40`, 83, inputs 270001+ | layer_m1rt51s | Shared duplicate state | 1.0015 / 1.0013,1.0053 | Inconclusive |
| layer_mb3s / 76da38c1 | `6e51f2c9dc774a79`, 97, inputs 290001+ | layer_m1rt51s | Shared duplicate state | 1.0002 / 0.9975,1.0006 | Inconclusive |
| layer_m1rt51s78 / 96f69ff0 | `a03039679e304d95`, 101, inputs 280101+ | layer_m1rt51s | Shared duplicate state | 0.9987 / 0.9994,0.9995 | Identical selected policies; apparent loss is not a valid decision |
| layer_m1rt51sc1 / 9373b976 | `b70cfa48c0264d20`, 103, inputs 300001+ | layer_m1rt51s / f1235356 | Independent, both with bench-only fixed M1 policy | 1.0007 / 1.0008,1.0023 | Projection port inconclusive |

The C1 label here is a projection implementation name, not architecture family C.
Shared duplicate state omits module/preparation variation from the null. The 0.13% apparent
loss with identical selected policies demonstrates that these shared-state readings cannot
support small whole-layer gain/loss decisions. The independent C1 comparison fixes M1's
policy and uses 500 gate replays in a bench-only patch so four prepares fit the time limit;
this patch is not part of the candidate. [Run methods and decisions](../../../docs/runs-decode-layer.md),
[M-B plan and stop rules](../plan-2026-10-31-m-b-pdl-chain.md).

The 128/256 table uses cand-inc paired deltas against the inc median. Other three crosses
remain in each raw log; these representative values do not override the aggregate decisions.

| Job | Tokens / context | Control median us | Paired delta us | Percent |
|---|---|---:|---:|---:|
| `c8d1e57261dd4901` | 128 / sliding | 245.0 | -5.5 | -2.24% |
| `c8d1e57261dd4901` | 128 / full | 367.9 | -7.2 | -1.96% |
| `c8d1e57261dd4901` | 256 / sliding | 340.0 | -5.3 | -1.56% |
| `c8d1e57261dd4901` | 256 / full | 513.2 | -6.3 | -1.23% |
| `a3760918c13c4e40` | 128 / sliding | 272.3 | -1.2 | -0.44% |
| `a3760918c13c4e40` | 128 / full | 365.3 | +0.9 | +0.25% |
| `a3760918c13c4e40` | 256 / sliding | 347.8 | -0.7 | -0.20% |
| `a3760918c13c4e40` | 256 / full | 502.5 | -0.7 | -0.14% |
| `6e51f2c9dc774a79` | 128 / sliding | 250.4 | -0.1 | -0.04% |
| `6e51f2c9dc774a79` | 128 / full | 335.8 | +0.3 | +0.09% |
| `6e51f2c9dc774a79` | 256 / sliding | 359.4 | +0.1 | +0.03% |
| `6e51f2c9dc774a79` | 256 / full | 500.2 | +0.4 | +0.08% |
| `a03039679e304d95` | 128 / sliding | 266.1 | +1.2 | +0.45% |
| `a03039679e304d95` | 128 / full | 367.9 | +0.3 | +0.08% |
| `a03039679e304d95` | 256 / sliding | 344.2 | +0.5 | +0.15% |
| `a03039679e304d95` | 256 / full | 516.8 | -0.6 | -0.12% |
| `b70cfa48c0264d20` | 128 / sliding | 266.5 | -1.3 | -0.49% |
| `b70cfa48c0264d20` | 128 / full | 359.7 | -0.4 | -0.11% |
| `b70cfa48c0264d20` | 256 / sliding | 347.5 | -0.1 | -0.03% |
| `b70cfa48c0264d20` | 256 / full | 505.2 | -0.4 | -0.08% |

Raw logs: [c8d1e57261dd4901](../../../muse_decode_layer/profiles/bench10_mb1s.c8d1e57261dd4901.log), [a3760918c13c4e40](../../../muse_decode_layer/profiles/bench11_mb2s.a3760918c13c4e40.log), [6e51f2c9dc774a79](../../../muse_decode_layer/profiles/bench13_mb3s.6e51f2c9dc774a79.log), [a03039679e304d95](../../../muse_decode_layer/profiles/bench14_m78.a03039679e304d95.log), [b70cfa48c0264d20](../../../muse_decode_layer/profiles/bench15_c1.b70cfa48c0264d20.log).

Correctness and limits: all five successful bench receipts pass 13 judge tests; each reports
24/24 arm checks. mb1 has a replay-class report on unpicked MoE main-d2. mb2 reports no
gate failures. mb3 rejects unpicked sk8w4s-er on its replay rule and reports unpicked
main-m2-ar2-ub sensitivity at 256. The mode78 rerun reports skf-r@16 sensitivity. The
C1 prepare reports a route side-stream reference check of 1.21e-4 against 1e-4 and falls
back; that route is not used by the selected skf-r front. Passing selected-arm checks does
not erase these findings. Judge scores, all control-path checks rather than isolated effects:
mb1 1.9837x, CI 1.9539-2.0134; mb2 1.9681x, CI 1.9386-1.9976; mb3 2.0104x,
CI 1.9802-2.0405; mode78 rerun 1.9950x, CI 1.9651-2.0249; C1 2.0095x,
CI 1.9793-2.0396. [Receipt-linked run record](../../../docs/runs-decode-layer.md).

M-B small-token tuner evidence is more specific than the unresolved whole-layer ratios.
In mb2, blanket sk entry trigger loses 1.00 us at 8 tokens, 6.06 us at 16 and 5.46 us at
32 relative to preferred whole-step controls 124.5/180.1/240.4 us: +0.80%/+3.36%/+2.27%
elapsed time. Adding unpooler PDL and MoE's post-barrier trigger gives 1.39 us savings at
8 versus the preferred mode, -1.12%, nine of nine rounds. In mb3, unpooler-only trigger
saves 0.80 us against 127.6 at 8, -0.63%, nine of nine. At 16 all extensions lose 1.67-2.54 us
against 194.2, +0.86% to+1.31%. At 32 unpooler+attention saves 1.25 us against 202.2,
-0.62%, but o alone saves 1.10 us, so the added chain contributes only 0.15 us. These are
single-preparation nine-round decisions, not replicated whole-layer gains. Slices 1-3 are
parked at their tested scope; slice 4, the one-token chain, remains designed and unbuilt.
[M-B measurements](../plan-2026-10-31-m-b-pdl-chain.md#results-of-2026-10-31-five-checks).

Mode 78 at 32 is now layer-tested, not merely a pending port. Initial bench
`36f93b13d4c14d5b` exits after an illegal memory access in the candidate's32-token front
tuner. The responsible MoE variant is absent from the truncated tail. A one-layer diagnostic
`18d48c5676fa4461`, hash a17a51df, passes 13/13 with 78 selected, and the rerun above also
passes but selects 47 in both files. The pre-registered selection condition fails, so 78 is
not shipped. Neither passing rerun proves the original fault fixed. No 128/256 mode78 gain
exists because the candidate restricts 78 to below 64 tokens.
[Failure log](../../../muse_decode_layer/profiles/bench12_m78.36f93b13d4c14d5b.log),
[diagnostic](../../../muse_decode_layer/profiles/diag78.18d48c5676fa4461.log),
[rerun](../../../muse_decode_layer/profiles/bench14_m78.a03039679e304d95.log).

Projection C1's layer test has weighted load ratios v1=1.0034,d16=1.0027,d32=1.0013,
v16=1.0001,v32=1.0005,d1=0.9959. The pooled 1.0007 is unresolved. C2 rd paths and
C1's128-token pooler library pin were not ported because the layer uses different paths.
The next structural M-C one-token fusions are proposed, not built or measured; their C-1/C-2
slice labels are also not architecture family C. Their donor fusion mapping is inferred, and
must not be treated as an established causal attribution.
[M-C plan](../plan-2026-10-31-m-c-t1-fusion.md).

### A/C/D-related MoE: both-phase stream-K loses its gate

Job `2a4fb39964aa448c`, source `587da5d`, tests mode 79 against mode 51 and mode 72.
Mode 79 retains 72's stream-K phase A and replaces claimed phase B with static cost-weighted
ranges and a stage-0 designated finisher. This is an alternative to C's ownership and D's
partial dynamic claiming inside A, not staged family B or a full ready-only queue.
The combined structure is now tested. Six seeds 141-146 per routing use cold-L2 graph
replay, median of 20 replays per arm and means of paired seed deltas. No paired CI is
recorded; percentages below divide the paired mean delta by the rounded control mean.

[Method and gate](../../../docs/runs-moe.md),
[raw log](../../../muse_moe_w4a16/profiles/bench_m79.2a4fb39964aa448c.log).

| Tokens / routing | Mode 51 us | Mode 79 delta us / percent | Seeds faster | Mode 72 control-arm delta us |
|---|---:|---|---:|---:|
| 32 / layer | 88.0 | +16.4 / +18.64% | 0/6 | +13.1 |
| 32 / std | 139.3 | +17.5 / +12.56% | 0/6 | +10.3 |
| 32 / skewed | 69.4 | +6.0 / +8.65% | 0/6 | +9.1 |
| 128 / layer | 166.2 | +19.4 / +11.67% | 0/6 | +11.3 |
| 128 / std | 205.6 | +24.8 / +12.06% | 0/6 | +9.8 |
| 128 / skewed | 101.7 | +27.9 / +27.43% | 0/6 | +20.1 |
| 256 / layer | 215.7 | +21.4 / +9.92% | 0/6 | +6.1 |
| 256 / std | 250.3 | +11.3 / +4.51% | 0/6 | +4.3 |
| 256 / skewed | 159.2 | +31.4 / +19.72% | 0/6 | +18.8 |

The judged path passes, 2.19646x, 95% CI 2.16351-2.22941; that score is not isolated
mode-79 performance. Spin flags are zero. Mode 79 fails its timing rule and its written
5e-4 output-comparison bound on one skewed 32-token seed, 5.1e-4 versus mode 72's 5.0e-4.
The gate's numerical failure remains recorded even though the judged receipt passes.
Mode 79 is not kept and its shippable follow-up was not run. At 128/256, replacing
mode 72's claimed phase B costs +8.1/+15.3 us on layer-like routing and +15.0/+7.0 us
standalone. Fixup costs, chained waits and code size are hypotheses, not separately measured
causes in this run. Reopen only after stamps show sub-1-us critical-path fixups per share
in both phases. [Receipt](../../../muse_moe_w4a16/profiles/bench_m79.2a4fb39964aa448c.json),
[scoped rejection](../../../docs/ruled-out.md).

### F/MoE diagnostics: stage-cost attribution revised

Job `43c35d18b5504eb5` at `1d42779` stamps a bench-only donor MAIN build beside mode 55,
mode 51's stamped twin. The donor is `6cae49a` with its MID helper line removed, not a new
whole-file F0 comparison. Six seeds 151-156, cold replays, 32/128/256 on layer/standalone
routing. Donor minus 51 is -3.0/-6.2, -9.8/-12.1 and -21.8/-22.4 us respectively;
all six seeds improve except 32 layer, five of six. The receipt passes, 2.19680x,
CI 2.16385-2.22976, spin flags zero. No paired CI.

At 128/256 layer-like routing, donor start is 3.4/3.8 us earlier; mean phase A is
3.4/10.3 us shorter; A end spread is 5.4/2.7 us narrower. Mean phase B differs by
0.0/-2.1 us. The stamped end gaps, 9.5-21.7 us across 128/256 routings, are mostly
phase A. These phase quantities overlap and must not be added as independent savings.
Donor fixups cost 10.4-12.8 us/CTA in A and 5.4-8.5 in B; doneA releases cost 2.5-7.7.
Share counts differ between instrumentation counters and remain unreconciled. Stamped donor
code has 16/20 B spill outside stage loops; stamped end times are within 1-3 us of unstamped
replays. Donor missing proxy fences remain unchanged and ineligible for direct adoption.
[Stamped experiment and per-phase table](../../../docs/runs-moe.md),
[raw log](../../../muse_moe_w4a16/profiles/bench_pubstamp.43c35d18b5504eb5.log).

A closer static review revises the proposed instruction-count explanation. The earlier
estimate of about 60 excess fast-path instructions becomes about 40; our dequant already
has the donor's 7 ALU + 2 IMAD.SHL form. Proposed mode 80 was not built because its expected
cut fell below its own gate. This is a revised hypothesis, not a measured mode-80 loss.

Job `1d767886e86d46e1`, source `2921d50`, tests timing-only mode 83, which omits the
per-thread weights-slot proxy fence, and repeats scale-pair mode 76. Mode 83 is explicitly
bench-only and cannot enter policy. Six seeds 161-166, cold median-of-20 replays; no paired CI.

| Tokens / routing | Mode 51 us | Mode 83 delta / percent | Seeds faster | Mode 76 delta / percent | Seeds faster |
|---|---:|---|---:|---|---:|
| 128 / layer | 154.8 | -0.1 us / -0.06% | 4/6 | +0.2 us / +0.13% | 0/6 |
| 128 / std | 206.2 | +0.2 us / +0.10% | 2/6 | +0.3 us / +0.15% | 2/6 |
| 256 / layer | 212.7 | -0.3 us / -0.14% | 4/6 | -0.9 us / -0.42% | 4/6 |
| 256 / std | 241.9 | -0.5 us / -0.21% | 5/6 | -0.8 us / -0.33% | 5/6 |

The receipt passes, 2.19253x, CI 2.15964-2.22542, spin flags zero; both arms are bit-identical
to 51 in every tested case. That does not license removing mode 83's fence. Its small
savings fail the diagnostic rule for the fence as the main cost, and the fence stays.
Mode 84's decode ablation was built but not run because it serialized WGMMA; mode 85 was
built but not run because it compiled to 51's loop. Neither has a GPU timing result.
[Raw ablation log](../../../muse_moe_w4a16/profiles/bench_abl.1d767886e86d46e1.log),
[receipt](../../../muse_moe_w4a16/profiles/bench_abl.1d767886e86d46e1.json).

Re-reading earlier mode-41 counters suggests waits account for the residual stage gap,
rather than cheap instruction cuts. The next proposed diagnostic is a counter twin of 51
that separates copy latency, producer delay and consumer skew. It is not built or measured.
This does not overturn B's staged losses or establish a new D scheduler win.
[Revised interpretation and next experiment](../../../docs/runs-moe.md).

### E: page-index bulk-copy implementation hangs before timing

`ONE_PBULK` in mode 49 replaces page-entry loads with a bulk copy into shared windows,
retaining existing ordering and adding a producer barrier. It is a producer-side E candidate,
not a measured win. Job `fcf904b32d9a427b` at `03675e8` hangs on `pb` at decode16 full
during ABCHECK; the guard stops the process. Control and matched control passed that shape.
The planned three-input, 200-round full-context bench never reaches timing or judging.
Job `61069efbf90d4c44` at `c811908` is a one-input, 60-round diagnostic plan; `dbg2`
hangs before timing even though it performs no bulk fills or window waits.
[First failure log](../../../muse_attention_decode/profiles/bench_p.fcf904b32d9a427b.log),
[diagnostic log](../../../muse_attention_decode/profiles/bench_q.61069efbf90d4c44.log).

Offline source/SASS bisection found the producer drain and EXIT missing for the unbraced
`if constexpr ... else for` form. Bracing the else restored producer EXITs. The default
build remains instruction-identical, and the fixed PBULK build changes only mode 49 with
no spills. That is a compile-level fix, not a GPU correctness clearance. The fixed build
still awaits a repeated check under the original gate. There are no new 128/256 timing
results, no scored result and no change to the pick-free 2.5366x attention entry.
PRQ wins and prior ring-depth/XPIPE losses remain. [Failure diagnosis and repeat plan](../../../docs/runs-attention.md).

### F-PROJ reimplementations: C1/C2 win paired tests; C1 holds scored entry

These are owned-code reimplementations of donor mechanisms, not verbatim donor candidates,
and belong to F-PROJ plus separate projection context, not A-E. C1 at `05286d9`, SHA256
f4141848, combines per-thread fail-closed NOVOTE waits, FULL addressing twins at M=TOKR
except qkv32, and a pooler128 cuBLASLt pin. C2 at `3044ee7`, SHA256 b2c4c096, adds a
register-direct 12-bit kernel offered at pooler1/8 and o8. [Run record](../../../docs/runs.md).

| Comparison / job | Method and uncertainty | Result |
|---|---|---|
| C1 versus release2265fa2 / `5ec16ca4d42a4e13` | Four independently prepared arms, RDRD order, three inputs x50 cold rounds,129/129 checks per arm | Pooled weighted ratio 1.0092, crosses 1.0082-1.0102, nulls 0.9981/1.0002; candidate faster |
| C2 versus C1 / `696179090d264f38` | Same four-arm method,129/129 checks per arm | Pooled1.0066, crosses 1.0060-1.0072, nulls 1.0001/0.9990; candidate faster |

These are successive paired comparisons, not a directly measured C2-versus-release aggregate;
do not add the two gains. No paired CI is recorded. Stub judged scores are not candidate scores.
At 128, C1 pooler D1-R1 is -1.33us against 21.93us, -6.06%, but the other cross deltas
range -0.26 to -0.86us with R/D nulls -0.64/+0.32us. The whole-candidate decision is stronger
than this individual cell. C2 changes small-token paths: pooler8 D1-R1 -1.06us against 15.21,
-6.97%; o8 -0.35 against 6.40,-5.47%; pooler1 -0.32 against 14.43,-2.22%. All four crosses
favor these paths. No isolated C2 gain is established at 128/256, whose candidate code is unchanged.
[First paired log](../../../muse_bf16_proj/profiles/bench_c4a.5ec16ca4d42a4e13.log),
[second paired log](../../../muse_bf16_proj/profiles/bench_c4b.696179090d264f38.log).

Clean C1 check `a4ce752c0f434ac5` passes 9/9,1.49480x,CI 1.47238-1.51723.
Clean C2 `5c4e4e99b6ec450f` passes 9/9,1.51721x,CI 1.49445-1.53997.
Coordinator scored C1 three times, all pass: `2d897770a75a4f3f`1.49694x,
`e1d3d341549c484d`1.51698x, `8a9c8ca810f142dc`1.51590x.
C2 scored repeats `e745eb8ca902471e`1.51522x and `355fc7894f6546ed`1.50824x both pass.
Their 95% intervals overlap. Recorded board entry is C1's `e1d3d341549c484d`,
CI 1.49422-1.53973; the lower C2 samples do not overturn its paired gain against C1.
[Scored C1 receipt](../../../muse_bf16_proj/profiles/scored.e1d3d341549c484d.json),
[C2 receipt](../../../muse_bf16_proj/profiles/scored.e745eb8ca902471e.json), [all repeats](../../../docs/runs.md).

The exploration preserves losers and failed builds. f1-f5 at source 79f9043 use cold paired
rounds, with FP64 and fresh-input replay checks on surviving arms; no paired CIs. Raw logs
retain controls and every variant. Numbers below are paired timing deltas, not isolated score gains.

| Job | Scope and result |
|---|---|
| `a63fa0c391fd40fc` f1 | Global-x escapes +0.13us qkv8,-0.13 slice8,+0.4 to+1.2 at 16+; donor geometries +0.02 to+1.17. Pooler128 donor pin -0.78,56/60 faster; qkv8 null0.45 |
| `6d5554de901143d0` f2 | Plain escape loads qkv32 -0.13, others within 0.13; escape-off attribution pool32 -0.34,pool16 -0.10. FULL arms dropped because checked at unsupported sizes. Pooler128 pin -1.36,59/60 |
| `a8de7b162c9f41d5` f3 | FULL/no-watchdog qkv32 -0.77,qkv16 -0.58,pool32 -0.46,pool8 -0.30. No-watchdog is diagnostic. Router GRED counter join loses+3.5 to+6.9; pooler128 pin -1.22 |
| `8712eb94a71e4075` f4 | LITEWD retains vote and recovers little: qkv32 +0.10 versus no-watchdog -0.96. Router memset-zeroed GZERO loses+0.83 at 256 to+1.92us |
| `eebd764690d5470f` f5 | Fail-closed NOVOTE+FULL pool32 -0.45,pool16 -0.34, several 8/16 cases -0.18 to-0.19; NOVOTE alone qkv32 -0.45 versus combined -0.22. This supplies C1 |
| `bcabe48caa884e67` f6 and `1759290da58047ab` f7 | Both fail pooler m7 through a late-bound runner-closure k before 3044ee7. Timing observations favor pooler8/o8 but are not clean candidate validation. Pooler16 loses+0.70 to+1.1,qkv8/16+1.7/+2.9,slice0 to+1.5; corrected C2 clean check follows |

The GRED join uses module-global accumulator/counters and cannot be ported into overlapping
layer launches without per-call state or a wait. Its fence-SC ordering argument was reviewed,
but this review makes no new ordering proof. [Port guide](../../../docs/proj-port-guide.md#global-split-k-join-wv_gred-f-proj-1-candidate-2026-10-31),
[exploration records and profile links](../../../docs/runs.md).

### Projection codec and router diagnostics: no new promoted router/codec candidate

Codec factorial `077b210c2087470d`, source recorded as proj-r8, compares C2 small-batch sk
picks with timing-only byte/decode perturbations.100 cold paired rounds; receipt passes 9/9,
1.51107x with CI 1.48841-1.53374, but byte-truncated arms are not correct codec implementations.
No 128/256 codec experiment. At qkv32, extra 16ALU instructions cost+1.81us and the combined
pair7/q12 models cost+3.04us. Other modeled combined deltas mostly improve. With qkv32
excluded, calculated pair7 weighted effect is+0.85%, below the+1.0% build trigger, so pair7
was not built. q12's+1.35% omits its 35-fold escape count and 32 KB LUT conflicts. rd/cgemv
codec savings remain unmeasured hypotheses. [Factorial log](../../../muse_bf16_proj/profiles/bench_f8.077b210c2087470d.log), [decision](../../../docs/runs.md).

Router rk-a `3fd0c7a7cea74ab4`, source proj-r8, and rk-b `e9d9499a465d4548`, source
`6d7f6a1`, use 100 cold paired rounds; both receipts pass9/9, including FP64/replay checks
for H arms. No paired CI. rk-a's standalone fill-node time around 1.1us includes a graph
launch and must not be treated as the zeroing cost inside the router graph. rk-b measures
only 0.16-0.25us saving from moving the clear into the kernel.

| rk-b case | Control median | Paired delta / percent | Rounds faster |
|---|---:|---|---:|
| 256 H+O, first-ticket clear + rk block order |9.49us|-0.61us/-6.43%|90/100|
| 256 H+O+J |9.49us|-0.70us/-7.38%|90/100|
| 256 duplicate H+O |9.49us|-0.67us/-7.06%|88/100|
| 256 memset+O |9.49us|-0.45us/-4.74%|81/100|
| 256 H without O |9.49us|+0.05us/+0.53%|46/100|
| 128 H+O, r64c16s8 |6.85us|+0.26us/+3.80%|27/100|
| 128 H+O, r64c12s8 |6.85us|+0.54us/+7.88%|13/100|
| 128 H+O, r64c8s8 |6.85us|+1.28us/+18.69%|0/100|

The duplicate H+O's -0.67 is its delta against production, not a -0.67 null between copies.
Production duplicate nulls are+0.06 at 128 and-0.06 at 256. The donor rk still saves 0.88/2.27us
versus production in the same run, leaving about1.6us versus H+O at 256 unattributed. The large
identified difference is block order; no extra gain should be attributed to barriers or spin.
H/WV_GTICKET is a new cross-CTA protocol awaiting PTX-chain review, not a promoted candidate.
Do not credit these diagnostics as an A-E or whole-layer gain.
[First stamp](../../../muse_bf16_proj/profiles/bench_rk.3fd0c7a7cea74ab4.log),
[clear/order test](../../../muse_bf16_proj/profiles/bench_rk2.e9d9499a465d4548.log), [decision](../../../docs/runs.md).

## Whole-layer outcome and attribution

Recorded scored layer remains `0f58d5d8223f45af`, 2.0148147x, CI 1.9845925-2.0450369,
from the M1 + mode-51 candidate. Its scored gain over the prior release is unresolved.
Independent paired readings of the combined policy are 1.0052 and 1.0106. The second is
`layer_mb1s`, whose added unpooler slice was rejected. It establishes no M-B slice gain.
Shared-state M-B slices 2/3 are inconclusive; the mode-78 rerun selected 47 in both arms.
Projection C1's independent layer comparison is inconclusive at 1.0007. No isolated A
or projection gain can be inferred from those whole-layer readings.
[Manifest](../../../docs/release-manifest.md), [layer evidence](../../../docs/runs-decode-layer.md).

## Decision and next evidence needed

Preserve the release. The missing c8d1 receipt is resolved, but the spread between its
1.0106 combined-policy result and the earlier 1.0052 exceeds within-run nulls. Future
small-gain decisions need independent preparations, duplicate controls and replication.

A-versus-B at 128/256 remains unchanged: B needs its documented phase-B/phase-A-tail trigger.
C ownership beats tested modes 72/74/79. Mode 79 is the now-tested both-phase alternative;
its loss does not reject every stream-K implementation. D's full ready-only queue remains
untested. Modes 76-78 are not promoted. Mode 80 was not built after static analysis revised
its expected benefit; the proposed mode-51 wait counters are the next diagnostic, not a gain.

For E, repeat the corrected PBULK check under its original gate before drawing timing
conclusions. The compile fix alone cannot clear the hang. Keep PRQ and NS3; the old hp
workspace-pick bundle stays withdrawn. For F, retain measurement-only donor policy and
measure owned reimplementations with required ordering repairs. Projection C1 and C2 win
successive component tests, but C1's layer port is inconclusive. Router H awaits PTX-chain
review; codec and M-C one-token fusion ideas remain unbuilt. Never add component scores.
[MoE decisions](../../../docs/runs-moe.md), [attention evidence](../../../docs/runs-attention.md),
[projection evidence](../../../docs/runs.md), [F policy](2026-10-30-family-f-donor-baseline.md).

