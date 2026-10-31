# Architecture performance tracker

Last review: 2026-10-30 14:05 PDT. Evidence snapshot: `bc10cb1`.
First review reconciles progress since `aab7d75`, the snapshot in the
[architecture-family guide](architecture-families-2026-10-29.md).
This is a review of recorded experiments, not a new benchmark run.

## Current family status

Negative timing deltas mean faster. Percentages below are elapsed-time changes against the
named control, not changes in aggregate score. Families overlap; their gains cannot be added.

| Family | Current result | What has actually been measured | Visibility gap |
|---|---|---|---|
| A. Hybrid persistent/legacy MoE | Component wins; mode 51 layer gain inconclusive | Four-arm mode 51 comparison reports 1.0057 pooled, below its noise gate. Old three-arm deltas remain observations, not established isolated gains. | Board stays mode 47. Stabilize prepare-time picks and resolve arm bias before promoting 51. |
| B. Staged grouped MoE | Retested loss | Modes 45 + 46 on mode 37 components lose 30.4/24.4 us at 128/256 versus 37, six seeds each. | Parked until phase B alone drops below about 50 us at 128 or the fused phase A tail disappears. |
| C. Owner-completes tiles | Retained; stream-K alternative retested and loses | Corrected designated-finisher mode 74 loses +4.6/+2.0 us at 128/256 layer-like routing versus 51; earlier mode 72 losses remain. | Designated finishing is tested; setup and fixup costs still exceed its timing-spread benefit. Revisit triggers remain scoped to this implementation. |
| D. Dependency-aware scheduling | Dynamic claiming retained; new phase-A claim layouts lose | Phase-A modes 62/63 and K=0 controls 66/69 lose; lean claimed mode 68 is not timed. Earlier split-tail 52/53 losses stand. | Full ready-only queue remains untested. Reopen phase-A claims only after the no-work control ties mode 51 within 1 us at 128/256. |
| E. Attention producer-consumer pipeline | PRQ per-request win; XPIPE cold losses retained | PRQ31 saves 0.48/0.96 us at 128/256 sliding in matched-null confirmation. Earlier DCARRY/PK16/VHOIST remains integrated. | layer_hp combines PRQ + HMRG + placement and awaits GPU testing; full-context XPIPE losses stand. |
| F. Donor-based baseline/hybrid | Registered, untested | F0 exact donor, F1 conforming adaptation, F2 measured hybrid; no controlled result. | Projection, attention, MoE and layer comparisons pending. [Registry](steering/2026-10-30-family-f-donor-baseline.md). |

## Latest review: designated finisher tested; ownership stays ahead

Corrected mode 74 at `4187b8b` loses against mode 51 in job `e02f60fec94f45cb`.
This is a stream-K alternative to C inside A, retaining D-related phase-B claims.
It uses a designated finisher instead of mode 72's last-arrival fixup. The finisher is
chosen during setup from the cost walk. It is now tested, rather than an untested idea.
[Design and gate](runs-moe.md#designated-finisher-for-stream-k-phase-a-modes-74-and-75-2026-10-30).

Six seeds 107-112 per size/routing use cold-L2 graph replay, medians of 20 replays per arm.
The following are means of paired component deltas, not whole-layer results. Percentages
use the logged control means; rounding can differ from subtracting the displayed arm means.
No paired confidence interval is recorded.

| Tokens / routing | Mode 51 control | Mode 74 delta | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 80.3 us | +7.9 us | +9.84% | 0/6 |
| 32 / standalone | 143.6 us | +7.6 us | +5.29% | 0/6 |
| 128 / layer-like | 163.2 us | +4.6 us | +2.82% | 0/6 |
| 128 / standalone | 201.4 us | +7.4 us | +3.67% | 1/6 |
| 256 / layer-like | 219.9 us | +2.0 us | +0.91% | 1/6 |
| 256 / standalone | 233.1 us | +4.1 us | +1.76% | 1/6 |

The receipt passes, 2.1974x with 95% interval 2.1645-2.2304. Its judged path is the
unchanged board dispatch, so this score is not mode 74's standalone score. First-seed
output checks against 51 are 1.3e-4 to 3.4e-4 relative L2; spin flags are zero.
The diagnostic requires a negative mean at 256 on both routings. It fails, so no held-out
or skewed gate ran and mode 74 remains parked.
[Raw log](../muse_moe_w4a16/profiles/bench_sk74fix.e02f60fec94f45cb.log),
[receipt](../muse_moe_w4a16/profiles/bench_sk74fix.e02f60fec94f45cb.json).

The earlier build `5f35018`, job `eaa881a934aa471e`, used slow SiLU because its predicate
excluded modes 74/75 and used serial finisher designation. Preserve its results as that
build's loss, not the corrected design's result. Six seeds 95-100, same replay method:

| Tokens / routing | Mode 51 control | Earlier mode 74 delta | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 85.1 us | +8.2 us | +9.64% | 0/6 |
| 32 / standalone | 142.1 us | +9.0 us | +6.33% | 0/6 |
| 128 / layer-like | 159.9 us | +8.3 us | +5.19% | 0/6 |
| 128 / standalone | 209.3 us | +7.2 us | +3.44% | 0/6 |
| 256 / layer-like | 222.0 us | +2.1 us | +0.95% | 0/6 |
| 256 / standalone | 248.9 us | +3.8 us | +1.53% | 1/6 |

That receipt passes, 2.2023x with interval 2.1693-2.2354. First-seed output differences
are 1.5e-4 to 2.0e-4, spin flags zero. No paired interval is recorded.
[Earlier log](../muse_moe_w4a16/profiles/bench_sk74.eaa881a934aa471e.log),
[earlier receipt](../muse_moe_w4a16/profiles/bench_sk74.eaa881a934aa471e.json).

The corrected stamped twin reduces phase-A end spread to 6.6-8.3 us at 128/256,
versus 10.9-19.5 for the control. Fixups cost 3.9-4.5 us per CTA, but route plus setup
still costs 3.4-3.6 us more. Mode 72's larger fixup cost and the public kernel's similar
end spread come from other processes/seeds. They are diagnostic comparisons, not paired
speedups. The remaining setup cause is unresolved. Reopen this implementation only with
setup under 1 us above 51, per-share fixup under 1 us, or a refitted cost table, then apply
the paired gate. These findings retain C for this comparison, not reject stream-K generally.
[Measured stamps and hypotheses](runs-moe.md), [revisit triggers](ruled-out.md).

### Separate projection context: one-wave QKV cluster split loses

Job `ad6c673d9a444c1e`, source `90ee4b9`, tests qkv8c4/qkv16c4/qkv32c4 and wider
c4w variants, 72 rows per CTA and four CTAs per cluster. All configurations passed FP64,
2-ulp twin checks and 22 identical one-hot columns. The dedicated paired timing loop failed
with a `torch.sum(out=)` error. These results are prepare-time tuner observations only:
cold-L2 graph replay, 24 alternating repetitions, one preparation, overhead included.

| Tokens | Unsplit qkv reference us | c4 us / delta / percent | c4w us / delta / percent |
|---|---:|---|---|
| 1 | 13.8 | 23.0 / +9.2 / +66.67% | 24.9 / +11.1 / +80.43% |
| 8 | 13.9 | 23.7 / +9.8 / +70.50% | 25.5 / +11.6 / +83.45% |
| 16 | 14.7 | 24.1 / +9.4 / +63.95% | 25.7 / +11.0 / +74.83% |
| 32 | 15.6 | 27.9 / +12.3 / +78.85% | Not tested |

At one token the tuner selects cgemv at 12.8 us; the table uses qkv8 at 13.8 us
as the unsplit geometry reference, not the selected incumbent.
No paired confidence interval exists, and 128/256 were not tested by this experiment.
Raw twins also lose. Receipt passes 9 tests, 1.5089x with interval 1.4863-1.5315, but the
judge selected production picks, not c4. The experiment is parked; production is unchanged.
Cluster residency and decode geometry are hypotheses for the loss, not established causes.
Reopen with measured active clusters at least 32, or a configuration at most 113 KB shared
memory whose raw twin beats qkv16raw.
[Run record](runs.md), [raw log](../muse_bf16_proj/profiles/proj_c4_bench.ad6c673d9a444c1e.log),
[receipt](../muse_bf16_proj/profiles/proj_c4_bench.ad6c673d9a444c1e.json), [scoped rejection](ruled-out.md).

### Family F registration and next decision

Family F is now registered separately as the donor-based baseline/hybrid, pinned to
alygg77 source `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`. F0 is the exact donor,
F1 a documented conforming adaptation, F2 a measured hybrid. Projection, attention, MoE
and layer tracks are pending. No controlled F gain, correctness clearance or promotion is
established. Archived donor scores are external evidence only.
[Registration](steering/2026-10-30-family-f-donor-baseline.md).

The shared decision now prioritizes complete donor-versus-release projection and attention
comparisons before new GPU rewrites, with independent preparations and null controls.
A coherent replacement or measured hybrid is an allowed outcome. A-versus-B at 128/256
remains unchanged: retain the working baseline, and reopen staged B only under its stated
phase-B/phase-A-tail trigger. E's PRQ gain and XPIPE losses stand; layer_hp and revised
mode-51 layer candidates remain pending. Whole-layer release stays 2.014x. No gains are
added across families or projection context.
[Donor-first decision](steering/2026-10-30-donor-first-adoption.md), [layer log](runs-decode-layer.md).

## Historical review, 12:44: projection split exit barrier wins paired router tests

This is separate projection context, with no new A-E or whole-layer result. Job
`4f3ff774981a494c` at `db0f8ad` compares relaxed split exit barriers (wd KLUEYZ,
rt mode 3) with a copy of `ce25f67`'s conforming tail (wd KLUET, rt mode 1).
The arrive moves after the receive wait and before reduction/output stores; the exit wait
stays. Router workloads use 8/16/32/128/256 tokens, not MoE routing distributions.
All arms passed FP64 checks on two router matrices, adjacent sizes and two graph replays.
[Run record](runs.md), [raw paired log](../muse_bf16_proj/profiles/proj_tail_bench.4f3ff774981a494c.log).

One same-process experiment uses 100 seeded cold-L2 rounds. Deltas are paired medians;
percentages divide that delta by the logged control median, rather than subtracting marginal
medians. No paired confidence interval is recorded.

| Router tokens | Control median | Relaxed split delta | Percent | Rounds faster | Production-copy delta |
|---|---:|---:|---:|---:|---:|
| 8 | 4.64 us | +0.00 us | +0.00% | 45/100 | -0.06 us (-1.29%) |
| 16 | 4.96 us | -0.22 us | -4.44% | 76/100 | -0.18 us (-3.63%) |
| 32 | 5.79 us | -0.03 us | -0.52% | 51/100 | +0.00 us (+0.00%) |
| 128 | 7.71 us | -0.45 us | -5.84% | 94/100 | -0.32 us (-4.15%) |
| 256 | 10.21 us | -0.32 us | -3.13% | 86/100 | -0.26 us (-2.55%) |

The wd relaxed duplicate saves 0.42/0.29 us at 128/256 (5.45%/2.84%; 88/87 rounds
faster), while duplicate-control nulls are +0.00/-0.03 us. The rt 8/32 results remain
inconclusive. The release-arrive split loses +0.35/+0.26/+0.40 us at 8/16/32
(+7.54%/+5.24%/+6.91%; 7/16/9 rounds faster); its wd savings are only 0.14/0.16 us
(1.82%/1.57%; 71/72 rounds faster). The no-exit-barrier arm is a diagnostic bound,
not a candidate. [Scoped rejection](ruled-out.md).

The bench receipt passes 9 tests and scores 1.4972x (95% CI 1.4747-1.5196).
Clean check `25ea2b62ff8a47a7` at `2265fa2` passes 9 tests, 1.4931x
(1.4707-1.5155). Coordinator scored run `d85d0081fa8a4a43`, same source, passes
and scores 1.4976x (1.4751-1.5200). These overlap the previous 1.495/1.496 runs:
aggregate improvement remains inconclusive across runs. The paired gain is relative to
`ce25f67`, not proof of a gain over the older projection baseline at 256.
[Bench receipt](../muse_bf16_proj/profiles/proj_tail_bench.4f3ff774981a494c.json),
[clean receipt](../muse_bf16_proj/profiles/submission_tail1.25ea2b62ff8a47a7.json),
[scored receipt](../muse_bf16_proj/profiles/scored.d85d0081fa8a4a43.json).

The recorded ordering review calls the split sound with notes. Its relaxed form relies on
arrive executing after the receive-wait loop; the guide notes that the ISA does not explicitly
state the barrier-speculation property. Passing tests do not establish ordering safety.
A future CLR>1 PUSH configuration needs a fresh review. The o128 library pin is still
unresolved because six preparations select different winning configurations; P3 stays parked.
[Ordering argument and caveats](proj-port-guide.md), [run record](runs.md).

Keep standalone promotion separate from layer integration. Projection still needs a layer
port and paired layer test. A-versus-B, C ownership, D's untested full ready queue and E's
PRQ win/XPIPE losses are unchanged. `layer_hp`, `layer_g51b` and `layer_m1rt51c` still
await their recorded tests; the layer release remains 2.014x. Do not add projection savings
to any family score. [Layer run log](runs-decode-layer.md), [release manifest](release-manifest.md).

## Historical review, 05:39: stream-K phase-A retest loses

Job `098949b28fe84bd1`, source `86a633e`, tests mode 72 against mode 51. It replaces
whole-owned phase-A units with mode 16's cost-weighted stream-K partition and last-arrival
fixup while retaining 51's current stage loop, epilogue and dynamic phase-B claims.
This is the split-K alternative to family C inside A, with D-related phase-B claiming retained.
It is not staged family B or a full dependency-ready scheduler.
[Experiment and ordering review](runs-moe.md#stream-k-phase-a-on-51s-components-modes-72-and-73-2026-10-30).

Same-process cold measurements use six seeds 83-88 for each size and routing distribution,
20 graph replays per arm per seed after L2 flush. Layer-like routing is a component workload,
not a whole-layer test. Percentages divide the logged paired mean delta by the control mean.

| Tokens / routing | Mode 51 control | Mode 72 minus control | Percent | Seeds faster |
|---|---:|---:|---:|---:|
| 32 / layer-like | 88.7 us | +13.5 us | +15.22% | 0/6 |
| 32 / standalone | 151.0 us | +9.4 us | +6.23% | 0/6 |
| 128 / layer-like | 166.1 us | +12.6 us | +7.59% | 0/6 |
| 128 / standalone | 195.3 us | +7.1 us | +3.64% | 0/6 |
| 256 / layer-like | 220.3 us | +6.6 us | +3.00% | 0/6 |
| 256 / standalone | 227.9 us | +9.2 us | +4.04% | 0/6 |

No paired confidence interval is recorded. Individual seed deltas remain in the
[raw log](../muse_moe_w4a16/profiles/bench_sk72.098949b28fe84bd1.log).
The [receipt](../muse_moe_w4a16/profiles/bench_sk72.098949b28fe84bd1.json) passes 5/5,
2.193x (2.160-2.226); that judge score is not an isolated mode 72 gain. First-seed output
checks in all six cells differ from 51 by 1.5e-4 to 2.4e-4 relative L2, consistent with the
changed FP32 summation order. Spin flags are zero. The diagnostic condition fails at 256 on
both routings, so the held-out/skewed gate was not run and nothing ships.

Stamped mode 73 versus 55 shows phase-A end spread widening to 15-19 us at 128/256 from 8-15 us.
Fixups are estimated at 9.4-10.9 us per CTA, with about 1.85 cut segments per CTA. The per-share
count path is about 3.3 us and finishing read-back 6-9 us. ABFIX's total subtracts a mean
pre-count time and is an estimate; the per-share and per-finish columns are measured directly.
Unlike the previous claiming builds, this kernel compiles no duplicate loop. Global fixup
round trips and an outdated cost table are hypotheses for the loss, not separately measured
causes. A shorter gather in the stamped twin does not help: the last first-x-ready is later.

C's ownership choice remains favored for this comparison. Preserve the loss at its tested
scope: mode 72's partition plus last-arrival fixup on current components at 32-256 tokens.
Revisit with a fixup well below 1 us per share on the critical path, or a stage-cost table
refitted to the current loop, then compare in-process with ABFIX. At that checkpoint, designated finishing was untested. Mode 74 has since tested it and lost,
as recorded in the latest review; other cut placements remain open. [Scoped rejection](ruled-out.md).

A-versus-B is unchanged: the staged pair still needs its phase-B or phase-A-tail trigger.
D's full ready-only queue remains untested. PRQ's E win and previous XPIPE losses stand.
`layer_hp` still has no GPU check; the whole-layer release remains 2.014x. No new attention,
projection or layer measurement appears in this checkpoint. [Layer manifest](release-manifest.md).

## Historical review, 04:43: PRQ wins within E; claiming and compression implementations lose

Family E now has a per-request pipeline win: `ONE_PRQ=31` combines transpose overlap,
Q.K-first issue, fast copies, split P.V issue and an offset-test change on q_len=8 modes
1/3/11/15. This is a coherent five-change design, not an isolated gain for each ingredient.
It leaves flat full-context paths unchanged, so it does not invalidate the XPIPE losses.
MoE runtime phase-A claiming variants lose; their no-work controls also lose. The evidence
rejects these code layouts, not runtime claiming itself. B's staged comparison remains unchanged.
[Attention round 3](runs-attention.md), [MoE evidence](runs-moe.md).

### E: PRQ standalone win and untested layer bundle

Three cold benches, each two held-out inputs and 120 randomized rounds, compare PRQ against
HMRG-on attention with PRQ off. All pass 16/16 judge tests and ABCHECKSUM 19/0 for every arm.
Routing distribution is not applicable. Deltas below are logged pooled arm-minus-control
values. No paired confidence intervals are reported.

| Job / source | Verify-1 full (8 tokens) | Verify-16 sliding (128 tokens) | Verify-32 sliding (256 tokens) | Gate |
|---|---:|---:|---:|---|
| `a7a4c23e98ca484a` / `1baad66` | -0.64 us | -0.45 us | -0.96 us | Modeled score contribution G +0.86%, duplicate null -0.11% |
| `3c6e1a7f1cc644b6` / `300b915` | -0.70 us | -0.35 us | -0.87 us | Missed written null clause: G +0.55% vs required +0.59%; retained as failed confirmation |
| `409996a7214f411a` / `8a6c8bb` | -0.64 us (-3.26%) | -0.48 us (-3.02%) | -0.96 us (-4.59%) | Confirmed with matched-state null; G +0.72%, null about 0%; independent-state G +0.80% |

Confirmation controls are 19.65/15.87/20.93 us. Both inputs improve on all three targeted rows.
G is a sensitivity-weighted estimate over four changed rows, not a measured whole-layer score.
The second bench's gate was not relaxed: the third bench adds a same-state null and an
independent-state comparison. Placement picking also failed its original row clause in the
second bench, then passed the third bench's own-state controls: about 0.3-0.4 us at decode-1
full when the initial allocation is slow. That is a separate change, with no guaranteed saving
when the first allocation is already fast.
[First log](../muse_attention_decode/profiles/bench_g.a7a4c23e98ca484a.log),
[failed confirmation](../muse_attention_decode/profiles/bench_h.3c6e1a7f1cc644b6.log),
[matched-null confirmation](../muse_attention_decode/profiles/bench_i.409996a7214f411a.log).

Final-file checks `640ee49b691b45ac` and `8b00100bf85d4ef5` at `57f57dd` (comment-only
repeat) pass 16/16 at 2.588 (2.549-2.627) and 2.545 (2.506-2.583). The file combines
PRQ31, HMRG and workspace picking. Cross-run variation exceeds PRQ's modeled gain; unchanged
full-context rows move too. Scored `1896e72c11ae4b0f` at `a70192d` passes 16/16 at
2.521 (2.484-2.559), replacing the 2.504 bench-file attention entry with a clean file.
This is standalone attention, not the 2.014x layer board.
[Check](../muse_attention_decode/profiles/sub_prq.640ee49b691b45ac.json),
[repeat](../muse_attention_decode/profiles/sub_prq_r.8b00100bf85d4ef5.json),
[scored receipt](../muse_attention_decode/profiles/scored.1896e72c11ae4b0f.json).

`layer_hp.py`, SHA256 prefix `0efd553c`, ports PRQ31 + HMRG + workspace picking onto the
release, retaining MoE47 and no M1. Its source/compile checks are recorded, but it has no GPU
check or measured layer gain. A four-arm bench is prepared. It is an E-plus-merge/placement
bundle; a future gain cannot be assigned entirely to E. Existing layer_g51b/layer_m1rt51c
candidates still await tests. [Release manifest](release-manifest.md), [layer log](runs-decode-layer.md).

### A/D-related MoE claiming: code-layout losses

Five new MoE receipts pass 5/5. Detailed experiments use six seeds per routing distribution
(layer-like and standalone), normal routed workloads at 32/128/256; 128/256 outputs are
bitwise and spin flags zero. All deltas below are versus mode 51. No paired confidence intervals.

| Job / source / seeds | Implementation | Layer-like 128 / 256 | Standalone 128 / 256 | Interpretation |
|---|---|---|---|---|
| `48b9bf9d77d146ac` / `003be4b` / 47-52 | 60, remove per-thread gather threadfence, retain proxy fence and release chain | -0.4 / 0.0 us (-0.25% / 0.00%) | -0.5 / -0.4 us | Gate fails at 256; parked |
| `b7e43ee6a517487a` / `d5bafad` / 53-58 | 62, claim last phase-A round | +7.9 / +9.2 us (+4.86% / +4.35%) | +7.5 / +8.6 us | 0/6 faster everywhere |
| Same job | 63, claim last two rounds | +7.0 / +8.5 us | +6.9 / +7.9 us | 0/6 faster everywhere |
| `ed4591a2477041c2` / `a727b12` / 65-70 | 66, same merged-loop build with nothing claimed | +7.0 / +8.8 us (+4.38% / +4.15%) | +7.9 / +8.7 us | Code-shape penalty; 62 vs 66 ranges -0.3 to +1.6 us |
| `b2c92a71600d488b` / `30c5a58` / 71-76 | 69, lean separated-loop build with nothing claimed | +4.5 / +5.2 us (+2.65% / +2.36%) | +5.0 / +3.9 us | Null fails; claimed mode 68 not timed |

Mode 60 also gives -0.3/-0.3 us at 32 layer/standalone. Mode 62 loses +3.9/+6.1 us at 32;
mode 69 loses +3.3/+3.7 us. The mode 69 build has a second compiled phase-A copy and longer
stage loops, with phase-B slowdown too. Narrower phase-A end spread does not establish a
faster kernel. Reopen only with no second compiled copy and a K=0 null within 1 us of 51
at both 128/256 before timing claims. These are partial phase-A scheduling attempts inside A,
related to D, not a tested full dependency-ready queue.
[Gather log](../muse_moe_w4a16/profiles/bench_gnf60.48b9bf9d77d146ac.log),
[claims](../muse_moe_w4a16/profiles/bench_ca62.b7e43ee6a517487a.log),
[merged-loop null](../muse_moe_w4a16/profiles/bench_ca66.ed4591a2477041c2.log),
[lean null](../muse_moe_w4a16/profiles/bench_cl69.b2c92a71600d488b.log).

The fifth job, `3d8902381a194d56`, is a standalone scorecard diagnostic, source logged as
"this branch (tool only)" without an exact revision. Ten standalone-routing seeds 301-310,
20 cold replays and duplicate preparations compare current sk8w3s/mode 51 against the public
kernel. Public-minus-ours gaps at 128/256 are -9.60/-14.92 us (-4.83%/-6.39% of ours'
198.88/233.515 us means), public faster 10/10. Across-seed standard deviations 3.92/3.47 us
are not confidence intervals. At 1 token ours wins 1.37 us; at 8 public wins 0.80; at 16 the
0.04 us difference is a tie; at 32 public-minus-ours -1.54 us has SD 8.02 and only 4/10 faster,
so it is inconclusive. The scorecard does not revive B without its phase-B or phase-A-tail trigger. [Scorecard and controls](runs-moe.md).

### Separate projection context: P3 tested and parked

`4922f366a0c142ca`, source `f8855b7`, passes 9/9. Pooler 128 compressed 12-bit weights
are reconstructed into shared-memory WGMMA tiles. Two independently compiled/packed copies
use 100 seeded cold rounds, synthetic randn weights x0.01, 5107 escapes (up to 6/block);
escape-heavy correctness covers up to 39/block, bitwise raw-twin checks and FP64 checks.
CW6d3 loses +12.90/+12.67 us to its raw twin in A/B, 0/100 faster; controls 35.55/35.63 us
make this +36.29%/+35.56%. Duplicate nulls are 0.03-0.27 us. Other buffering choices also lose.
Against Lt 20.45 us, the paired CW6d3-A loss is 27.95 us (+136.67%); the raw twin itself loses
15.01 us (+73.40%). The memory-only compression saving is only 1.01 us, below the modeled 5 us.
No paired confidence interval is reported. No 256-token test occurred.

The tuner rejects CW. Its 1.496 (1.474-1.519) judged score measures the incumbent path,
not successful compression. Code is parked under experiments; production returns to ce25f67.
A decoder-release ordering question remains open for this unshipped experiment. Reopen only
a raw geometry within about 1 us of Lt and a measured decode cost below the traffic saving.
This is projection context, not a new B or E result.
[Run and model](runs.md), [bench log](../muse_bf16_proj/profiles/proj_p3_bench.4922f366a0c142ca.log),
[receipt](../muse_bf16_proj/profiles/proj_p3_bench.4922f366a0c142ca.json), [stop rule](ruled-out.md).

### Receipt reconciliation and next decision

Previously missing scored receipts now verify: attention `57b0bed35f694a23` passes 16/16,
2.479 (2.442-2.516); projection `1d277e7fe9c44712` and `d59bc7af342a465d` pass 9/9,
1.482 (1.460-1.504) and 1.488 (1.466-1.510). Their cross-run interpretation is unchanged.
[Attention receipt](../muse_attention_decode/profiles/scored.57b0bed35f694a23.json),
[projection P1](../muse_bf16_proj/profiles/scored.1d277e7fe9c44712.json),
[conforming projection](../muse_bf16_proj/profiles/scored.d59bc7af342a465d.json).

The priority is now the prepared layer_hp controlled comparison. Preserve the working layer
release and the A-versus-B gate. After repeated claiming-layout losses, the next claim design
must remove the duplicate compiled work rather than repeat a larger kernel. P3 needs a new
raw geometry before another compression test. Do not sum PRQ, HMRG, placement or projection gains.

## Historical review, 03:37: HMRG attention merge wins; projection ordering correction

A-E outcomes and the A-versus-B decision at 128/256 remain unchanged. `ONE_HMRG` is an
attention key-half merge before global reductions, not the family E cross-tile pipeline or
MoE family C. It is a standalone component win awaiting a separate layer port/test.
The XPIPE cold losses remain valid. No new MoE or whole-layer result is recorded.
[Attention round 2](runs-attention.md#standalone-round-2-10-30-a3-merge--finalize-and-the-batch-1-full-prologue).

### Separate attention context: HMRG

`ce92d32e26024bd2` at `d8d43f3` first reports HMRG cold ABSCORE 1.0180 against the
board-code control (duplicate null 0.9993). Per-input rows were truncated, so the sign gate
could not be checked from that log alone. Confirmation `d0819bb26907426b` at `9eafafc`
reports 1.0214 with the control's workspace, 1.0232 with an independent workspace, versus
null 1.0037. Both benches pass 16/16 judge tests and ABCHECKSUM 19/0 for real arms.
The bench uses 120 randomized rounds, two held-out inputs, seeds 1000/1001; routing distribution
is not applicable. Logged pooled cold deltas and control medians from the confirmation:

| Decode shape | Control | HMRG delta | Delta / control | Per-input deltas |
|---|---:|---:|---:|---|
| 1 token sliding | 9.89 us | -0.64 us | -6.47% | -0.54 / -0.70 us |
| 1 token full | 18.69 us | -1.54 us | -8.24% | -1.50 / -1.50 us |
| 16 tokens sliding | 14.34 us | -1.44 us | -10.04% | -1.47 / -1.41 us |
| 32 tokens sliding | 16.61 us | -0.83 us | -5.00% | -0.86 / -0.86 us |

Pooled deltas differ from per-input summaries. No paired confidence intervals are reported.
The independent-state result supports the gain despite the known decode-1 full placement
spread. HMRG changes only q_len=1 modes 1/3/11/13; 128/256-token verify paths have identical
code, so their timing movement is not an HMRG gain. It halves the q_len=1 global reductions
by merging key halves in shared memory first. The recorded independent ordering review finds
the added waits/barriers sound within its source scope.
[First bench](../muse_attention_decode/profiles/bench_e.ce92d32e26024bd2.log),
[confirmation](../muse_attention_decode/profiles/bench_f.d0819bb26907426b.log).

Candidate-only checks `5d309f9d13924dca` (`dc6b409`, SHA256 prefix `93297101`) and
`82b1bf0fad3244b9` (same code plus a comment) pass 16/16, scoring 2.512
(2.475-2.550) and 2.582 (2.543-2.620). These are cross-run observations; unchanged verify
paths also accelerate in the repeat. The coordinator's scored run `57b0bed35f694a23`, same
source, is logged as passed at 2.479x; its receipt, interval and test count are absent locally.
The attention board remains `e9085c86d89449c2` at 2.504x, a bench-file entry. The scored HMRG
result is the best clean candidate-only scored run recorded, not a board replacement.
[Check receipt](../muse_attention_decode/profiles/hmrg.5d309f9d13924dca.json),
[repeat receipt](../muse_attention_decode/profiles/hmrg_r.82b1bf0fad3244b9.json),
[scored log](runs-attention.md#scored-57b0bed35f694a23-2479x-2026-10-30).

Preserve rejected alternatives: HMRG plus last-CTA ticket (`ONE_T9`, `hmtk`) adds 8.67 us
at verify-1 full and 1.79/1.70/1.41 us at verify-1/16/32 sliding versus control in
`ce92d32e26024bd2`. Its additional 0.22 us saving at decode-1 full is inconclusive under
the placement null. Removing cooperative launch gains only 0.38 us at decode-1 full and
retains no promotion claim. Phase comparisons close the batch-1 full prologue as the
public kernel's advantage: both pay nearly the same cost. Cluster/DSMEM merge and overlapped
finalize remain unbuilt. The next meaningful HMRG gate is a complete-layer comparison after
porting, not another isolated pipeline attribution. [Rejection register](ruled-out.md).

### Separate projection context: conforming push paths

The current `ce25f67` candidate uses exit cluster barriers on every push path and local
stores for rt's own-rank share. It no longer relies on the two ambiguous behaviors discussed
in the previous review. This is a correction to the current candidate status, not a claim
that the old behavior was proven sound. [Port-guide decision](proj-port-guide.md).

`3abb1cd974e2477d` and `d7c662296eaa4413` at `ce25f67` (header-only difference) pass
9/9 and score 1.495 (1.473-1.517) / 1.496 (1.473-1.518). Lt picks agree; router32
switches between tied rt variants. The following are cross-run means of the two conforming
checks versus the two earlier confirmations, not paired isolated barrier costs:

| Router tokens | Earlier mean | Conforming mean | Change | Change vs scored d0dc80f |
|---|---:|---:|---|---|
| 8 | 4.26 us | 4.43 us | +0.17 us (+3.99%) | -1.25 us (-22.01%) |
| 16 | 4.54 us | 4.80 us | +0.26 us (+5.73%) | -0.44 us (-8.40%) |
| 32 | 5.48 us | 5.49 us | +0.01 us (+0.18%) | +0.04 us (+0.73%) |
| 128 | 6.61 us | 7.19 us | +0.58 us (+8.77%) | -0.30 us (-4.01%) |
| 256 | 9.60 us | 9.81 us | +0.21 us (+2.19%) | 0.00 us (0.00%) |

Percentages use the rounded logged means; d0dc80f controls are reconstructed from the logged
deltas (5.68/5.24/5.45/7.49/9.81 us). Two runs per build, no interval for these per-shape
changes. The previously measured non-conforming KLUE/rt wins remain historical; current
cross-run data shows no retained gain at 32/256. Do not transfer the old paired percentage
to the corrected candidate. [Run log](runs.md#conforming-push-paths-coordinator-decision-2026-10-30),
[receipt 1](../muse_bf16_proj/profiles/submission_r2t1.3abb1cd974e2477d.json),
[receipt 2](../muse_bf16_proj/profiles/submission_r2t2.d7c662296eaa4413.json).

Coordinator submission `d59bc7af342a465d` is logged at 1.4879x, source `ce25f67`, versus
previous projection board entry `1d277e7fe9c44712` at 1.4818x. Receipt/interval/correctness
count for this submission are absent locally; do not interpret the score difference as a
controlled gain. Projection remains unported to the layer. P3 remains unbuilt.

Next evidence: HMRG layer port and controlled layer test, receipts for the two new scored
runs, and a paired conforming projection comparison if isolated retained gains matter.
Standalone and layer promotion are separate decisions; do not add these component scores.

## Historical review, 02:38: E pipeline losses; separate projection router gains

Attention roadmap labels A1/A2 are experiments within family E, not family A (MoE).
The earlier DCARRY/PK16/VHOIST wins and their combined layer integration remain valid.
The new cross-tile designs lose on the cold-cache workload and have no layer integration.
A-D and the 128/256-token A-versus-B decision are unchanged. No new whole-layer result
appears in this checkpoint. [Attention evidence](runs-attention.md#standalone-round-10-30-a0-scorecard-and-a1-cross-tile-overlap-in-mode-51).

### E: real overlap tested, cold-cache gate failed

`5575e7d15d3e49a8`, kernel `86f2bec`, generator `8a0a4a0`: attention mode 51 with
`ONE_XPIPE=2` issues P.V before the next stage wait and overlaps Q.K/V transpose. The
control is the board attention code, not MoE mode 51. Each arm passes ABCHECKSUM 19/0;
judge receipt passes 16/16. The bench uses 120 randomized rounds and two held-out inputs
(seeds 1000/1001), with independent control states. Full attention verify 16/32 corresponds
to 128/256 tokens; routing distribution is not applicable.

| Candidate vs control | 128 tokens | 256 tokens | Status |
|---|---|---|---|
| XPIPE=2, cold pooled medians | 117.31 vs 113.41 us: +3.90 us (+3.44%) | 193.25 vs 192.45 us: +0.80 us (+0.42%) | Measured loss; 128-token gate fails |
| XPIPE=2, cold per-input deltas | +3.32 / +4.55 us | +0.54 / +0.99 us | Both inputs slower |
| XPIPE=2, warm per-input deltas | -1.8 / -1.9 us | -3.5 / -3.3 us | Warm-cache gain only; does not pass cold gate |
| Mode 115, one CTA/SM vs mode 51, cold | 137.79 vs 110.27: +27.52 us (+24.96%) | 244.93 vs 187.26: +57.67 us (+30.80%) | Measured loss |
| Mode 115 + XPIPE=1 vs mode 51, cold | 138.37 vs 110.27: +28.10 us (+25.48%) | 239.87 vs 187.26: +52.61 us (+28.09%) | Measured loss |

The first row subtracts pooled medians, not medians of paired differences. Logged pooled duplicate
nulls are -0.38/-0.06 us; no confidence interval for these deltas is reported. Warm percentages
are omitted because the summary does not supply the corresponding control times.
[First bench log](../muse_attention_decode/profiles/bench_a0.5575e7d15d3e49a8.log).

The mode-115 rows come from `318ae6fc3c3746ce`, source `682aeaa`: two held-out inputs,
120-round default, 16/16 judge tests, ABCHECKSUM 19/0 for real arms. Nulls are +0.06/+0.42 us.
XPIPE=1 saves 5.06 us (-2.07%) inside mode 115 at 256, but leaves it 52.61 us slower than
mode 51. At 128 it adds 0.58 us (+0.42%) inside mode 115. Q.K-first XPIPE=1 at two CTAs/SM
is compile-infeasible in this build (C7512 register serialization); the one-CTA variant compiles
without C75xx but loses residency. These results reject the tested configurations, not all
cross-tile pipelines. [Second bench log](../muse_attention_decode/profiles/bench_b.318ae6fc3c3746ce.log).

### Attention diagnostics and next target

`a92aa9c43f064c49` (`df4060d`) tests one-allocation workspace carving: 16/16 and ABCHECKSUM
19/0, but the control remains 0.54 us slower at decode-1 full on both inputs. Carving fails
its +-0.2 us gate. `7144310358564fa3` (`a0a78ce`, generator locally edited) passes 16/16
judged tests but has no usable bench data: undefined `_AB_CAP` breaks every bench row.
It establishes no capture-order result. Fixed repeat `b7762d764d474eae` (`46df563`), six
held-out inputs, 16/16 and ABCHECKSUM 11/0, finds a 0.5-0.7 us spread tied to workspace state
rather than capture order. L2 placement is the proposed mechanism, not a measured cause.
Cross-state decode-1 full differences below 0.7 us are inconclusive under this evidence.
[Carve log](../muse_attention_decode/profiles/bench_c.a92aa9c43f064c49.log),
[failed bench](../muse_attention_decode/profiles/bench_d.7144310358564fa3.log),
[fixed repeat](../muse_attention_decode/profiles/bench_d2.b7762d764d474eae.log).

The standalone scorecard puts most of the remaining gap in short/sliding shapes. Diagnostic
phase exits in `318ae6fc3c3746ce` assign about 2.4-4.0 us to merge/finalize on decode-1 sliding,
decode-1 full, decode-16 sliding and verify-1 full. These are diagnostic budgets, not achievable
savings. Attention A3 reduction ownership and prepare-time allocation selection remain untested;
A3 is not evidence for MoE family C. Next decisive test is a fixed-state, null-controlled
short-shape reduction comparison, with the log's gate of 0.5 us at decode-1 sliding or 1 us
at decode-16 sliding/decode-1 full. Reopen the one-CTA design only with a resource model that
recovers its residency cost. [Scoped failures and triggers](ruled-out.md).

### Separate projection context: rt and KLUE router gains

Five new receipts pass 9/9. R2a `a91f2152d8974458` (`113d513`) scores 1.505
(1.482-1.527), R2b `ba85985b8c5e4969` (`01ad9d9`) 1.484 (1.461-1.506), and
R2c `6cc72e840403402b` (`b50dcb6`) 1.499 (1.477-1.522). Confirmations
`f2373dedd4f44648` and `b80b817c67d043ad`, both `1e12506` with a header-only difference,
score 1.484 (1.462-1.506) and 1.508 (1.485-1.530). Their spread is not a controlled gain;
the run log attributes most of it to unchanged verify-1 shapes. The prior P1 submission
`1d277e7fe9c44712` at `d0dc80f` is logged at 1.482 (1.460-1.504), but its receipt is absent
locally, as are `44fa08c6d2f44b44` and `6cbd8b981a5d4c3f`.
[Projection round and receipt identifiers](runs.md#2026-10-30-round-2-router-attribution-rt-stable-cublaslt-picks-branch-worktree-agent-a7d5e0782cac00376).

R2c records 100 seeded cold rounds, paired against the pre-round production router picks:

| Variant / tokens | Paired delta | Control median | Delta / control median |
|---|---:|---:|---:|
| rt8w8 / 8 | -1.25 us | 5.38 us | -23.23% |
| rt16w8p8 / 16 | -0.88 us | 5.33 us | -16.51% |
| rt16w8p8 / 32 | -0.40 us | 5.74 us | -6.97% |
| wd KLUE / 128 | -0.80 us | 7.63 us | -10.48% |
| wd KLUE / 256 | -0.74 us | 9.97 us | -7.42% |

Percentages use paired deltas divided by control medians after the logged 4.24 us replay
overhead subtraction. They are component results, not layer savings. Nulls are about
0.02-0.13 us for rt and -0.13/-0.06 us for KLUE at 128/256; no paired confidence intervals
are recorded. FP64 checks cover two router matrices, outliers/tiny values, adjacent sizes and
fresh-input graph replays. Routing distribution is not applicable to these projection operators.
[R2c log](../muse_bf16_proj/profiles/bench_r2c.6cc72e840403402b.log).

Push reduction is the strongest single tested wd change (-0.46 to -0.66 us across R2a/b).
KLUE combines packing, promotion, push reduction and prologue changes; do not assign its full
win to one ingredient. Removing the producer warp with its barrier loses +1.74/+1.25 us at
128/256; XA orientation is a null, and rt8w8 loses +1.50 us at 32 in R2c. Stable Lt choices
now agree across confirmation preparations except o128 at the margin. P3 compressed WGMMA
remains unbuilt and parked for budget, with a raw-BF16 twin required before testing decode.
[Projection evidence](runs.md), [scoped losses](ruled-out.md).

The [port guide](proj-port-guide.md) still records unresolved outgoing `st.async` sender-exit
and rt self-rank questions. Passing correctness checks does not resolve them; TAILSYNC is the
recorded fallback, costing about 0.1-0.2 us for rt and 0.3 us for wd. No projection layer port
is present. These wins therefore stay separate from A-E and from layer promotion.

## Historical review, 01:31: standalone projection gains, no new A-E layer result

The new projection round is separate context, not a gain attributed to A-E. The
[standalone roadmap](steering/2026-10-30-attention-projection-standalone-roadmap.md)
separates standalone promotion from layer integration. A-E standings and the 128/256-token
A-versus-B decision remain unchanged. Direct cuBLASLt and router geometry changes have not
been ported to the layer. The full layer still needs its own controlled comparison.

The [projection run log](runs.md#2026-10-30-standalone-round-direct-cublaslt-and-router-geometry-p1)
records these jobs. Routing distribution is not applicable to these standalone dense operators.
Scores below compare against each job's SGLang sample, not a same-process release control.

| Job | Source revision / candidate | Score (95% CI) | Evidence status |
|---|---|---|---|
| `bf6c825bd5cf4659` | `eb374e4`, LATE_ARRIVE baseline, submitted by coordinator | 1.457 (1.436-1.479) | Receipt verified, 9/9 |
| `1b00244922654709` | `d61deea`, P1a | 1.465 (1.443-1.487) | Receipt verified, 9/9; Lt build failed, router changes only |
| `047bc3a4168e42d6` | `33daf8b`, P1b, Lt heuristic search | 1.463 (1.442-1.485) | Receipt verified, 9/9 |
| `d6eaf1e73db047a5` | `62a6ae0`, P1c, tile/cluster sweep | 1.484 (1.461-1.506) | Receipt verified, 9/9; prepare-time picks vary |
| `44fa08c6d2f44b44` | `d0dc80f`, P1d, cross-layout search and longer screening | 1.479 (1.457-1.501) | Run-log evidence; receipt absent from checkout, correctness count unverified here |
| `6cbd8b981a5d4c3f` | `d0dc80f`, clean candidate, file SHA1 `03389e0d` | 1.482 (1.460-1.504) | Run-log evidence; receipt absent from checkout, correctness count unverified here |

Verified receipts: [baseline](../muse_bf16_proj/profiles/proj_eb374e4.bf6c825bd5cf4659.json),
[P1a](../muse_bf16_proj/profiles/bench_p1a.1b00244922654709.json),
[P1b](../muse_bf16_proj/profiles/bench_p1b.047bc3a4168e42d6.json),
[P1c](../muse_bf16_proj/profiles/bench_p1c.d6eaf1e73db047a5.json).

P1d's log reports 100 seeded rounds, medians against preparation A's pre-Lt pick.
Independent preparations newA/newB save 1.87/2.99 us on pooler 128 and 2.98/2.46 us on
pooler 256. Null deltas are -0.13/-0.03 us respectively. Qkv 256 saves 1.92/1.86 us;
unpooler 128 saves 1.34/1.41 us. Percent deltas are unavailable from this table because
absolute paired control times are omitted. These are component comparisons, not layer gains.
Pooler 128 picks vary by about 1.1 us between preparations; no paired confidence interval is
recorded in the table. Full per-shape deltas, including near-ties and router gaps, remain in
the linked run log.

The clean candidate's cross-run verify-16/verify-32 totals (128/256 tokens) are
73.29 -> 70.22 us and 91.84 -> 84.29 us: -3.07 us (-4.19%) and -7.55 us (-8.22%).
Decode totals vary from -0.45 to +0.62 us with unchanged decode code. These observations
cannot establish an isolated change or be added to A-E gains. Preparation rises from about
70 to 105 seconds. [Port status and cost](proj-port-guide.md#direct-cublaslt-plans-lt_cpp-m--128--256-dense-2026-10-30).

Preserve the losses: Lt qkv 128/o 128 are near-ties in the searched space; several router
geometries lose outside their winning bucket; heuristic-only Lt search misses the pooler-256
winner. CUTLASS pooler 256 remains a tuner candidate but is superseded in this comparison.
P2 compressed qkv split-K is declined on a work model, without a new run; its reopening
condition is a one-wave geometry or persistent schedule that preserves startup and escape
cost. P3 compressed WGMMA is untested; its predicted gains are hypotheses. Neither is family
B or E evidence. [Scoped losses and reopening conditions](ruled-out.md).

Next evidence: obtain the two missing receipts, stabilize the remaining pooler-128 pick
variation, and use the proposed raw-BF16 twin before judging P3's compression mechanism.
Standalone gains can qualify on their own; any layer port still needs a full-layer A/B.

## Historical review, 00:32: scored candidates do not resolve M1 gains

Board remains `3fbb3ad027e84c6b`, 2.014x. Two new scored readings are cross-run observations,
not controlled timing gains, and do not change the unresolved mode-51 layer conclusion:

| Job / file | Score and correctness | M1 status | Interpretation |
|---|---|---|---|
| `b68bdcd7a1ba464c`, layer_m1rt, SHA256 `d6958449`, byte-identical to `dd1777e` | 2.013x (1.983-2.043), 13/13, prepare 143.1 s | Unknown: SUMMARY and policy lost from log tail | Cannot credit this score to enabled M1 |
| `d6c30f88c617438b`, layer_m1rt51 at `7cdb755`, SHA256 `65264f31` | 2.013x (1.983-2.043), 13/13, prepare 154.8 s | Off for the entire timed run, policy empty | Measures mode 51 in the SK_PDL/ONE_TRIG build with M1 off |

For `d6c30`, the first gate shape, decode 1 sliding, has one replay in 2,000 with relative
hidden/stream error 3.2e-4/1.9e-4 versus a bitwise control. Only the maximum-error envelope
fails; order-1 count is zero and the other criteria pass. At this token count both arms use
cgemv for o and fp32 front, so M1 changes no launch. This is not an isolated M1 ordering
failure; rare rounding noise is the report's hypothesis, with a reproduction prepared but
unrun. All 11 MoE variants pass their gates, no variant is rejected, mode 51 is bitwise at
128/256, and the picker chooses 51 at 32/128/256. Whole-layer times at 128 are
268.6/351.6 us, at 256 345.2/513.6, sliding/full; no paired control supplies a gain delta.
[Run log](runs-decode-layer.md), [raw gate results](../muse_decode_layer/profiles/layer_m1rt51.d6c30f88c617438b.log).

Two route negative controls fail sensitivity at 32: the row injection replaces a zero row
with another zero row and changes no output. This is a test-injection defect, not a measured
route correctness failure. The new tool chooses differing rows/values; synthetic CPU tests
reproduce old misses and remove row misses. GPU validation of the revised files is pending.
[Failure analysis](steering/2026-10-30-m1-1x1s-gate-failure.md).

The run log records a user decision to judge M1 only on shapes whose captured candidate graph
actually launches an M1 PDL secondary, with unchanged bounds on those shapes. If the launch
census fails, the gate still judges the shape. This review reports that record and changes
no validation. Prepared files, all without a new GPU check:

- `layer_m1rt51c`, SHA256 `1eda8ec9`: runtime M1 + 51, census skip and negative-control fix;
  rebuilt at `a48edef`, no device-code change from layer_m1rt51.
- `layer_g51b`, SHA256 `acd2385c`: release + 51, current gate and W4 early-exit wait fix,
  no M1. The previous four-arm mode-51 result remains unresolved.
- `layer_stable`, SHA256 `752123dd`: diagnostic build, not a candidate; release with nine interleaved tuner rounds, a paired
  margin and 8/9 winning-round rule before replacing the receipt's preferred pick. CPU
  tests retain a true tie in 1991/2000 draws and accept the synthetic 2 us win in 500/500;
  these are rule tests, not GPU speed or stability results. Four-arm A/A is prepared.
- Targeted 1x1 reproduction diagnostic: compiled/CPU-smoke-tested, not run on the judge.

[Manifest](release-manifest.md), [tuner-stability design](plan-2026-10-30-tuner-stability.md).
A-E component standings, B's retest loss and the four-arm uncertainty remain unchanged.

## Historical review, 23:35: mode 51 layer gain unresolved after arm-bias checks

The earlier three-arm bench timed the candidate twice using the same prepared state, but the
incumbent only once. It could not distinguish a candidate gain from an incumbent-arm penalty.
The [arm-bias audit](steering/2026-10-30-layer-bench-arm-bias.md) and new four-arm runs therefore
supersede the previous claim of an established isolated mode-51 layer gain. Preserve the old
numbers, including `29fd004fd58e40ef`'s 1.0084 and its 128/256 deltas, as historical observations.
Mode 47's old three-arm attribution shares this limitation. Its release and component checks
remain valid; the isolated layer magnitude needs the same caution.

Four-arm benches use independent prepare() per copy, random prepare/capture/replay order,
two held-out inputs per shape, eight rounds and both same-file nulls. The heuristic requires
all four cross ratios on one side of 1 and the pooled log ratio above twice the larger null.
It is not a formal significance test. Source bench builder `a877401`:

| Job / candidate | Seed start / order seed | Pooled speed ratio | Nulls | Decision |
|---|---|---|---|---|
| `724aa0dd6ed140ca`, release versus itself, file `9473029f` | 180001 / 43 | 1.0045 | 0.9965 / 1.0030 | A/A not resolved |
| `2dc566298be1498f`, mode 51 `dacd4e0d`, bench `e95f3218` | 160001 / 37 | 1.0057 | 0.9898 / 0.9972 | Mode 51 not resolved |
| `40b9b3f795f84073`, runtime-order M1 + 51 `d57b5e38`, bench `824e1e45` | 170001 / 41 | 1.0162 | 0.9928 / 0.9970 | Combined candidate faster by heuristic |

Each compares against release `b9c23bda`. These runs' judge scores are 2.009x (1.979-2.039),
2.000x (1.970-2.030), and 1.992x (1.962-2.022), respectively. These are check readings,
not new releases. All 96 checks pass (24 per arm); prepare-time policy differences contribute
to the nulls. A/A retains a decode-1 offset: pooled 1.0262 there, with incumbent 3-6 us slower
than the other copies. Exclude decode 1 from attribution. [Layer run log](runs-decode-layer.md).

At 128/256, mode 51's pooled load ratios are 1.0080/1.0107, equivalent to about 0.79%/1.06%
less weighted elapsed time. Weighted arm times are 283.6/285.7 us for the two incumbents
versus 282.7/282.1 for candidates at 128; 375.7/374.2 versus 371.8/370.1 at 256. The overall
cross range 0.9991-1.0123 and larger null about 1.03% leave the gain unresolved. Do not invent
a single paired microsecond delta from these independently summarized arms. Policy flips
include front/boundary/W4 choices even between copies of the same file. Raw four-cross paired
deltas at 128 range -1.6 to -3.2 us sliding and -0.4 to -4.7 full; at 256, -4.4 to -7.6
sliding and -3.0 to +0.5 full. These are ranges, not a resolved pooled microsecond gain.
[Raw mode-51 four-arm bench](../muse_decode_layer/profiles/bench4_g51.2dc566298be1498f.log).

Runtime M1 + 51 crosses 1.0111-1.0214 and clears its larger null about 0.72%. Per-load pooled
ratios are d1 1.0320, d16 1.0074, d32 1.0065, verify 1 (8 tokens) 1.0166, verify 16 (128)
1.0257 and verify 32 (256) 1.0096. Verify-1 sliding paired differences range -2.1 to -2.9 us.
The aggregate combines M1, mode 51 and the decode-1 offset. Dividing by A/A suggests about
1.2% combined gain; the report's approximate 0.4% M1 share is cross-run arithmetic, not an
isolated measurement. M1 operates only at <=32; 128/256 improvements cannot be credited to
it. The same mode-51 code reads 1.0080 at 128 in one run and 1.0257 in another. Do not add
component scores. [Raw combined bench](../muse_decode_layer/profiles/bench4_m1rt.40b9b3f795f84073.log).

The earlier runtime-order measurement `f0ee07c4d4904511`'s 0.87% three-arm result is also
unestablished. It motivated the corrected harness, not a separate accepted gain. The new
`cand/layer_m1rt.py` (SHA256 `d695844971f2bc7ab85720ad3a464c3464fc7efb01182746b4a96bcf14be7269`)
is **mode 47 + runtime M1**, with enforced 2,000-replay fresh-input gates and negative controls.
The measured combined build used mode 51 and report-only 1,000-replay gates. Those builds are
not interchangeable: the new candidate has no check or submission. Board remains 2.014x.
[Release manifest](release-manifest.md).

### Attention history resolved for the tested mode; ordering review completed

Attention diagnostic `2b93c6ecf4f641c6`, source `389da5c`, passes 16/16, 2.462x
(2.425-2.499), with production SASS unchanged. This is a diagnostic score, not a speedup.
Mode 49, decode 16/32 full, context 80000, page 1, four qkv inputs: production gives four/five
distinct outputs in nine calls on one input (1-3 elements, one BF16 ulp). A fixed-order debug
merge gives bitwise outputs across repeated, changed and separate-case histories, flushes and
interleaved calls. All 283 workspace scans are clean. This supports merge-order rounding,
not carried input values, for these tested cases. The earlier bitwise repeated-input reading
was a sample, not a general property. Do not claim every attention configuration is proved.
[Attention diagnostic](runs-attention.md#input-history-dependence-at-decode-16--32-full-merge-order-not-a-carry-10-30).

The [independent M1 ordering review](review-m1-ordering-2026-10-30.md) and
[other-PDL audit](review-pdl-secondaries-2026-10-30.md) replace the "unreviewed argument" status.
The review supports the fenced runtime design under its stated assumptions. The other-PDL
audit finds no issue on the shipped scored path; a latent W4 GEMM2 early return before its
wait is off that path and is fixed in the new unchecked candidate. Neither review substitutes
for checking that exact candidate with enforced gates.

### New MoE losses and limits

`7063f86352344efa`, source `e866c8a`, layer seeds 1-6 at 128/256, selects x gather as the
larger lead: instrumented gather 6.8/8.3 us versus public 3.6/4.9, while the trailing-warpgroup
skew bound is only 0.45/1.07 us. Those are attribution bounds, not recoverable layer savings.
`e7e75c9510994c08`, source `fca2be7`, layer seeds 41-46, tests mode 57 (early x loads plus
16-byte stores) and 59 (early loads only), versus 51 at 32/128/256. Mode 57 loses
+1.2/+1.1/+1.2 us, zero of six faster at each size. Mode 59 is inconclusive/tied:
-0.0/+0.6/-0.3 us, 2/1/5 of six faster. Checks pass with spin flags zero; a mode-59 normalized
output difference at 32 is not separated from rounding or a defect, and it is not kept.
Neither is gated or shipped. Shorter per-CTA gather does not materially advance the first
job's readiness. These are scoped A-descendant results; B's staged retest and D's full
ready-only status do not change. [MoE run log](runs-moe.md).

## Historical review, 22:34: mode 51 layer gain and measured M1 limits

Mode 51 closes its pending layer test. Job `29fd004fd58e40ef`, bench source `d9b7d67`, compares
`cand/layer_g51.py` (SHA256 prefix `dacd4e0d`, sent comment-stripped) with clean mode-47
release `b9c23bda`. Two held-out inputs per shape, seeds 140001+, eight cold-L2 rounds with
seeded random order give 16 paired samples per shape and 72/72 reference/routing/cache checks.
Candidate/release speed ratio is **1.0084**, null ratio 0.9997. At 128/256 only the MoE mode
differs in the timed path; gate changes affect prepare. Other shapes stay within null spread.

| Tokens | Sliding delta | Full delta | Control sliding/full | Null-minus-release sliding/full |
|---|---|---|---|---|
| 128 | -5.1 us (-1.92%) | -5.3 us (-1.52%) | 265.0 / 349.5 us | -3.7 / -5.2 us |
| 256 | -6.5 us (-1.86%) | -1.1 us (-0.21%) | 348.9 / 524.3 us | -4.4 / +0.3 us |

Percentages use paired delta divided by the logged control, not a subtraction of arm medians.
No per-delta confidence interval is reported. Modes at 32 are 47 in both arms, while fronts
differ at 16 through prepare-time selection. This supports the targeted layer gain, not a
new gain at every size. [Raw mode-51 layer comparison](../muse_decode_layer/profiles/benchG51.29fd004fd58e40ef.log).

The first gate-version candidate, SHA256 prefix `c8a843cf`, passes check `a438b82ff2c141c2`
at 2.022x (1.991-2.052), 13/13, with mode 51 selected at 32/128/256. The later `dacd4e0d`
file has the bench result above but no unpatched scored check. It is not released. Board entry
remains `3fbb3ad027e84c6b`, 2.014x. Mode 51 is bitwise at 128/256 in both arms of the
route-PDL checks. All 11 MoE variants pass; route gates pass 8/8; negative controls reject
tile, 1%, 0.1%, row and single-element injections in all 19 gates, and all uninjected
observations pass. [Release/candidate manifest](release-manifest.md), [layer runs](runs-decode-layer.md).

### M1: diagnostic gain within its build, little net release gain

Job `6ce5be0ba29642ea`, source `8f545ae`, runs M1 + mode 51 with a fresh-input gate and a
reader-side proxy fence. Judge 13/13 passes (1.988x, 1.959-2.018), but M1 is blocked on the
control at decode 1 sliding: off replay error 2.7e-3/2.5e-3, on bitwise. M1 changes no launch
at one token, so this does not isolate an M1 effect. Its timing gate did not run.

Stage diagnostic `60cf58b69b75491e` uses the same file through `dl_m1_stage_diag.py`, five
fresh-input pairs, 1,000 replays each for off/off2/on plus 200 eager calls. Attention relative
error is 1e-5 to 2e-4 in all arms; larger errors first appear at routing/MoE. No M1-specific
stale-read signature is found. Report-only timing now runs: at 8 tokens sliding, whole-step
off 105.9 versus o+front 100.5 us; at 16 sliding/full 153.4/229.1 versus 150.0/226.6; at
32, 217.7/360.1 versus 214.7/357.1. These are within-build diagnostic timings, not an accepted
candidate's release gain. Its judge score 1.965x (1.936-1.995) uses M1 off.

Measurement build `fdf75ba5a8124544`, generated with `dl_m1_bench_build.py`, compares M1 + 51
with the mode-47 release, seeds 130001+, eight rounds, 72/72 checks. It scores 2.007x
(1.977-2.038), paired aggregate 1.0040, null 0.9984. At <=32 tokens, candidate-minus-release
sliding/full is d1 +0.2/-0.3, d16 +0.6/+3.0, d32 -0.1/-0.2, verify 1 -1.3/+0.2 us.
Although turning M1 on inside this build saves 5.1 us at 8-token sliding, the build changes
plain projection issue order too, so net release gain is about zero at small sizes. Front
policies also differ at 16. At 128/256, mode 51 explains the targeted timing differences,
not M1. Gate is report-only: this build is not a validated candidate. No isolated E or D
family gain follows. [Layer run log](runs-decode-layer.md).

The same build's 32-token MoE prefetch probe finds off-minus-on +0.22/+0.00 us sliding and
-0.16/-0.48 full across two inputs, comparable to null -0.06/+0.26 and -0.32/-0.67. This
probe gives no measurable prefetch effect; it does not explain the earlier in-layer/component
gap. A prepared --no-pf-probe build and history diagnostic have no new recorded run.

### Validation changed; attention history remains an open lead

The [run log](runs-decode-layer.md#validation-policy-change-fresh-input-replay-gate-user-decision-2026-10-30)
records user-approved fresh-input gates for M1 and MoE/route replay checks, with later changes
to relative-L2 order-1 rejection, control envelopes, element checks and negative controls.
This review reports that recorded decision; it does not change validation or infer permission
for new work. The released file still uses the old gate; the candidate uses the newer one.
Attention `_one_stress`, norm and route side-stream bounds remain unchanged.

A [PTX ordering argument](steering/2026-10-30-m1-ptx-ordering.md) now exists and the M1 build
adds reader-side `fence.proxy.async.global`. The argument states that it has not had independent
review; runtime passing alone does not prove ordering. The former "argument missing" status
is superseded by "written, independently unreviewed".

New open lead in `5fefe98`: fresh-input attention differs from its eager reference by about
1e-4 at decode 1 sliding and 2e-4 at verify 1 sliding, while repeated-input stress is bitwise.
Previous-call dependence is a hypothesis, not a measured causal result or confirmed stale
read. `tools/dl_history_diag.py` is prepared to distinguish noise, previous-call dependence
and separate-case state; it has no run yet. This limits the earlier explanation that all
variation was atomic merge order. Do not clear M1 or reject E on this unresolved evidence.
[Open lead](runs-decode-layer.md#open-lead-2026-10-30-unproven-input-history-dependence-in-the-attention-output).

## Historical review, 19:31: mode 47 layer attribution and mode 51 component gain

Current layer release `3fbb3ad027e84c6b`, source `2cee446`, scores **2.014x (95% interval
1.984-2.045), 13/13 passed**. Its file SHA256 `b9c23bda16bb9d6311f025062e415841c67b16284737172badaa190fdc506a49`
matches main's layer file. It adds mode 47 plus LATE_ARRIVE/o16/unp32 projections to the prior
release, with M1 code absent. Check `c2ba945496ab47b4` was 2.006x (1.976-2.037).
These are cross-run scores, not an isolated gain estimate. [Release manifest](release-manifest.md).

### Mode 47's isolated layer contribution

Job `720079090ee44cc3`, bench source `fa7200c`, compares the scored file with the same file
holding mode 47 out of the picker (SHA256 prefix `84d8507a`). Both arms compile the same
kernels; prepare picks 47 versus 37 at 32/128/256. Two held-out inputs per shape, seeds
110001+, eight seeded-random cold-L2 rounds give 16 paired samples per shape; 72/72 checks
pass. Candidate-minus-held deltas, sliding/full: **128 tokens -4.9/-5.5 us (-1.81%/-1.43%);
256 -5.7/-3.6 us (-1.58%/-0.70%)**. Logged held controls are 270.2/383.7 and 361.2/514.8 us.
At 32 tokens -1.3/-2.6 us is inside the arm spread. This is one paired check, without a
per-delta confidence interval. Null-minus-held at 128 is -3.8/-4.0 and at 256 -4.8/-1.0 us.
[Raw attribution](../muse_decode_layer/profiles/benchInc_47.720079090ee44cc3.log).

The 1.0155 overall ratio is contaminated by a decode-1 arm-position artifact: the held arm is
about 5 us slower even though mode 47 never runs there. Do not report that aggregate as an
isolated mode-47 gain. Full-package comparison `50c4c1328ab742f4`, source `2cee446`, seeds
80001+, eight rounds, 72/72 checks, similarly prints 1.0182 but about 1.009 excluding decode 1.
Its paired deltas at 128 are -5.4/-6.1 us, at 256 -2.1/-3.8; projection share inferred by
subtracting across these two benches is only indicative. [Layer run log](runs-decode-layer.md).

### Mode 51 advances A; tail-split descendants lose

Mode 50 moves sum-of-squares work to a two-pass final pass. Job `8a7051fa926e43e1`, source
`a03ca00`, seeds 23-28, passes correctness but fails its 32-token layer-routing performance
gate (+0.7 us against a +0.5 limit), despite -5.9/-6.6 us at 128/256. It is not shipped.
Mode 51 uses that final pass only at 64+ tokens and retains mode 47 below. This is an A
implementation change retaining C ownership/D claiming, not a new isolated C or D result.
Gate `c6b0e500c3a84c9a`, source `69c06ed`, six held-out seeds 29-34 per routing:

| Routing | 32-token delta vs 47 | 128-token delta | 256-token delta | Seeds faster, 32/128/256 |
|---|---|---|---|---|
| Layer | +0.0 us | -3.9 us (-2.41%) | -3.9 us (-1.71%) | 2/6/5 of 6 |
| Standalone | -1.0 us | -3.8 us | -4.4 us | 5/6/6 of 6 |
| Skewed | -0.1 us | -4.0 us | -7.7 us | 2/6/6 of 6 |

Layer controls are 161.9/228.0 us. All 54 output comparisons pass (maximum 4.6e-5), self-test
3.88e-3, spin flags zero, judge 5/5. Shippable check `a5a2bbd1a18b4b89`, source `6bbb252`,
repeats -3.7/-3.0 us at 128/256, six of six faster, scoring 2.210x. Submission
`bcd2d7a3506a4b7d` passes at 2.202x, below mode 47's 2.216x board entry, which stays.
Those cross-run scores do not negate the paired gain. No mode-51 layer test is recorded.
[MoE runs](runs-moe.md), [raw gate](../muse_moe_w4a16/profiles/bench_g51.c6b0e500c3a84c9a.log).

Modes 52/53 split the last dynamically claimed phase B units into K halves via fixup, an
A/C/D descendant rather than a ready-only queue. Job `209cf29bf91c4ea2`, implementation
recorded in `260d791` (exact submitted source hash not pinned in the row), seeds 35-40:
52 minus 51 at 32/128/256, layer/standalone, is +3.3/+3.5, +5.9/+4.9, +8.5/+5.3 us;
53 is +8.1/+5.2, +7.3/+6.5, +9.2/+7.7. At 128/256 layer controls 159.2/216.0 give
52 +3.71%/+3.94%, 53 +4.59%/+4.26%. Output differences are at most 9.7e-5, spin flags zero,
zero to two of six faster. Tail does not shrink (7.9 versus 7.1 us at 32). Both lose; no
layer port or further gate. This does not reject every split-K design.
[Raw tail test](../muse_moe_w4a16/profiles/bench_tail52.209cf29bf91c4ea2.log).

### Corrections and diagnostic limits

M1 remains blocked and untimed, but the failure is not isolated to M1. Diagnostic
`6372154fcd7a4fa9` on source `6377b5a` finds repeated-input M1-on/off bitwise equal at verify
1 full; the M1-off control itself varies up to 6.7e-4 at 32x1 full. Fresh-input diagnostic
`3e2aa83358dd4f60` finds 217-459/2000 off replays and 288-307/2000 on replays above 1e-4 at
verify 1 full, maximum 1.2e-3 in both arms; no replay exceeds 1e-2 or changes ids. The
source report attributes this to layer run-to-run variation, consistent with atomic reduction
order. The first divergent intermediate and proxy-ordering argument remain unverified.
The 1e-4 bound is unchanged; no timing or safety conclusion follows. Validation-policy changes
require explicit user approval under the recorded test plan; this review changes none.
[Layer diagnosis](steering/2026-10-30-layer-release-attribution.md),
[raw fresh-input test](../muse_decode_layer/profiles/m1fresh.3e2aa83358dd4f60.log).

Public comparisons `86ced49f092a48d8` and `f5ee8632e1f64bf2`, current release against pinned
blob `936fefff`, each pass 72/72 with 16 paired samples per shape; raw null deltas are within 1.1 and 1.8 us,
respectively.
Public advantages at verify 1 sliding/full are 8.7/10.5 then 10.2/10.8 us; decode 32
15.7/10.8 then 12.2/15.9 us. Profiled boundary overlap suggests a mechanism, not measured
recoverable time. Component test `1fbb9fbf524f4ed7`, source `b3509a0` plus tool changes,
layer/standalone seeds 1-6, measures only 1.5/5.9 us public advantage at 32 tokens (47 controls
90.9/132.9 us); the follow-up tail test finds 1.9 us on layer routing. This does not reproduce
the 10-14 us in-layer profile gap. Routing, prefetch and profiling effects remain hypotheses.
sk8w3s loses 19.2 us versus 47 at 32 on layer routing, zero of six faster; do not generalize
its 1/8/16-token win. ABPUB stage columns are invalid without --pubranges; use ABWORK.
[MoE attribution](runs-moe.md), [layer attribution](steering/2026-10-30-layer-release-attribution.md).

Diagnostic `ed3b4fdc76b54263` (source `64cec1f`) crashed after 128-token timing; no 256 result.
Rerun `d47c78bdc130424b` (source `54318bd`) completes, public minus 47 -14.7/-22.6 us at
128/256; diagnostic mode 49 adds 5.8/9.4 us. Its mislabeled ABSKEW columns must be read in
skew_cols order. The estimated 72-74% wait contribution from warpgroup skew is attribution,
not recoverable layer savings. [MoE run log](runs-moe.md).

## Historical review, 18:25: combined layer gain, staged loss and mode 47

The A/E plus projection package now has same-process whole-layer evidence. Check
`aa4276e91c924dad` (bench source `6c6038b`, candidate SHA256 prefix `4ed09046`, incumbent
`a923f11`) reports incumbent/candidate load-time ratio **1.0158**: 1.58% speedup, about
1.56% less elapsed time. It tests 12 shapes, two held-out inputs each (seeds 50001+), eight
cold-L2 graph rounds with seeded random arm order. All 72 reference/routing/cache checks
pass. The identical-candidate null gives about 1.0006 relative to the candidate; per-load
spread is about +/-0.5%, not a formal confidence interval. [Raw layer comparison](../muse_decode_layer/profiles/benchInc.aa4276e91c924dad.log).

At 128 tokens, paired candidate-minus-incumbent is -6.2 us sliding (-2.20%) and -13.3 us
full (-3.56%); at 256 it is -3.5 us (-0.98%) and -22.5 us (-4.22%). Percentages use logged
control means 281.2/373.1 and 356.6/533.2 us. Other loads: decode 1 +0.2/-1.3 us,
decode 16 -1.9/-2.1, decode 32 +0.9/-1.0, verify 1 (8 tokens) -0.7/-0.7, sliding/full.
Those smaller effects are similar to the null spread. The package combines A mode 37/sk8w3s
(retaining C ownership and D dynamic claiming), E DCARRY/PK16/VHOIST, and qkv8/cgemv changes.
It does not isolate a family gain or justify adding component savings.

Release `edd71d3152a946fe` uses the confirmation file from `261a76ac7b364091` (SHA256 prefix
`1355d39f`): **1.982x, 95% interval 1.953-2.012, 13/13 passed**. Confirmation was 1.989x
(1.959-2.019); six candidate checks ranged 1.960-2.012, averaging 1.982. These cross-run scores
support repeatability but the paired comparison supplies the gain claim. FIN8 mode 11 with
eight copies now passes the asserted 2,000-replay tests, replay/eager difference zero in
`3e2b5771d7254fd2`. [Layer run log](runs-decode-layer.md).

### A versus B and the next A candidate

B's retest `9d94ac75f4724b52`, source `54257b5`, uses modes 45 + 46 (variant 301), mode 37's
components split into owned phase A then claimed phase B/final pass. Layer-routing seeds 1-6
lose +30.3/+30.4/+24.4 us at 32/128/256, zero of six faster at each size. At 128/256 this is
+18.4%/+10.7% versus logged mode 37 means 165.5/227.1 us. All arms pass output checks,
spin flags zero; no delta confidence interval is reported. B remains parked under the new
trigger in the status table. [Raw retest](../muse_moe_w4a16/profiles/bench_gw1.9d94ac75f4724b52.log).

Mode 43's next-stage hoist improves timing but does not reduce measured wait cycles. Repeat
`304f17f3e8704992` (source `a4687c8`, layer seeds 7-12) finds mode 47, code groups without the
hoist, faster at 128/256. This scopes the failed hypothesis to the one-group hoist; mode 47
retains the code-group change. Gate `5f381b8b6946492d`, source `20c295e`, seeds 13-18:

| Routing | 32-token delta | 128-token delta | 256-token delta | Seeds faster, 32/128/256 |
|---|---|---|---|---|
| Layer | -1.0 us | -2.3 us (-1.32%) | -3.7 us (-1.65%) | 6/5/6 of 6 |
| Standalone | -2.4 us | -2.8 us | -3.4 us | 6/6/6 of 6 |
| Skewed | -2.0 us | -2.2 us | -2.8 us | 5/6/6 of 6 |

All deltas are paired mode 47 minus 37; layer controls are 173.6/223.8 us at 128/256.
Outputs differ by at most 4.4e-5, self-test 3.89e-3, spin flags zero; 5/5 judge tests pass.
Shippable check `b50351288b504e81`, source `ec565a7`, repeats gains (-2.5/-3.4 us means at
128/256, seeds 17-22). Component release `abe7651dbaa44458` scores 2.216x. Mode 47 also runs
at 128/256 in layer check `e2d1d81447cc48d9`, but that file changes several mechanisms and
scores 1.963x (1.933-1.992); it does not establish an isolated layer gain for 47.
[MoE runs](runs-moe.md), [raw gate](../muse_moe_w4a16/profiles/bench_g47.5f381b8b6946492d.log).

### Invalidated comparisons and separate layer context

- `2cd4e3b292904662` (source `d006531`) passes 13/13 but trips the attention setup test and
  falls back to SGLang, scoring 1.785x. This is not the intended bundle's performance.
  Restoring original qkv rows for the fallback comparison fixes the test path in `3e2b`.
  Later FP32 diagnostics show the fresh-input discrepancy was larger in SGLang's path.
- `135ab2a9f22b49a5` has an invalid incumbent arm: comment stripping removed dispatch markers.
  Discard its apparent 23-240 us gains. `121dff3ab17a4245` never runs its public bench because
  Triton cannot retrieve the arm source. The repaired comparisons are `aa4276` and `d9e07`.
  Oversized/static-rejected attempts produced no job and no measurement.
- Public comparison `d9e07c554f614d83`, same bench source, seeds 60001+, passes 72/72.
  Public/candidate speed ratio is 1.0464, null/candidate 0.9989. Weighted public advantages
  are 9.9 us at verify 1 and 16.5 us at decode 32; candidate leads decode 1 by 1.9 us.
  This replaces the older cross-run gap inference. The public blob is pinned in the run log;
  its identity with the earlier public score's submitted file is not established.
- M1 (SK_PDL + ONE_TRIG), source `6377b5a`, job `e2d1d81447cc48d9`, fails forced-on replay
  correctness at verify 1 full: after 2,000 replays hidden 2.0e-4 and stream 1.6e-4 exceed
  the 1e-4 bound; ids match. M1 is disabled before timing, so no M1 timing gain/loss exists.
  This is an attention-to-projection chain failure, not rejection of E or D. Reopening needs
  an ordering explanation and clean replay checks. [Layer results](runs-decode-layer.md),
  [rejection register](ruled-out.md).

Projection context outside A-E: jobs `01703396ea1a409e`, `3678bfd3ef1c4ce6`,
`2048b59099f3435d` test o16/unp32 geometries (bench sources `dd67954`, `977afb3`, then
resulting geometry change `4d53658`), about 0.3 us gains across
six passes, cold L2, 200 randomized rounds per pass, ABCHECK passes and null within 0.10 us.
`c6e13cc974a24294` repeats against old geometries/public. `d2c9cb6997794859` rejects NACC2
(+0.00 to +0.78 us); `807464e240ec4f6c` finds fence/vote probes within noise and late release
losing +0.26 us on pool32. Fences stay. `f14c880c19a34369` and `1093e22ca2534d7f`, source
`eb374e4`, test LATE_ARRIVE: at 32 tokens qkv saves 0.38-0.48 us, o 0.22-0.26, unpooler
0.16-0.32 over nolate, across two passes each; module-null spread is -0.06 to +0.16.
The mixed M1-off layer check includes these changes but does not isolate their layer gains.
Detailed quartiles, control timings and negative geometry results remain in the
[projection run log](runs.md) and its linked profiles.

## Historical review, 16:27: staged retest and assembled layer bundle

No new scored run or paired timing appears in `dbd99c3..f61d8d9`. The
[rejection register](ruled-out.md) now parks the staged design with an explicit revisit trigger:
cheaper phase B/setup or a changed overlap tail. Modes 23, 26 and 28/37 landed after the old
staged comparison. That trigger is met, so repeat A versus B at 128/256 with current components
in one process. Preserve the old +27.9/+24.2 us losses as historical measurements; they do not
establish the result against today's A. No new winning B implementation has been measured.

The [layer breakdown](steering/2026-10-29-v1-d32-breakdown.md) reports an assembled bundle at
`b694b4a`: mode 37 + sk8w3s (A, retaining C/D mechanisms), DCARRY/PK16/VHOIST (E), plus qkv8
and single-token cgemv projection changes outside A-E. It is unjudged. Its branch log and
source were inspected locally; FIN8 expected-plan assertions and fresh-qkv replay checks are
implemented there, but no completed runtime result closes that coverage gap. Whole-layer
score remains 1.964x from `9b921febc7ad4140`. Do not add standalone savings into a layer gain.

Separate layer context: `3da50e5` adds default-off SK_PDL projection-weight prefill and
ONE_TRIG attention launch-dependent signaling. This is a proposed attention-to-projection
chain, not D's ready-job scheduler or a measured E gain. Recorded compile checks show the
disabled paths SASS-identical (58/58 projection kernels, 15/15 attention kernels); enabled
builds have no recorded spills. The host launch path was not compiled in that gate, and
runtime correctness, replay stress and timing remain untested. No job ID or timing delta
exists for this proposal. [Compile evidence and test plan](steering/2026-10-29-v1-d32-breakdown.md#6-built-sk_pdl-and-one_trig-compile-gated-off-by-default).

The new score-weighted hotspot tool in `f61d8d9` ranks existing profile evidence. Its rankings
are analysis, not fresh benchmark measurements. The layer breakdown's bundle projections
and remaining-gap estimates are also hypotheses or cross-run arithmetic, not isolated family gains.

## Historical review, 14:25: family E improves again

PK16 changes probability packing and VHOIST carries transpose addresses/descriptors per
segment. The combined attention candidate improves full verification beyond DCARRY.
Job `025ccaf9c57a4ba2`, switches introduced in `a50fc0c`, records cold same-process control
116.26 / 204.22 us versus candidate 110.91 / 194.59 at 128/256 tokens. That is **-5.35 / -9.63
us**, or **-4.6% / -4.7%**. Warm deltas are -7.52 / -13.88 us. ABCHECKSUM is 19 passed and
zero failed for every arm. These full-attention cases use batch 16/32, q_len 8, context
80,000 and TP8; MoE routing does not apply. The run-log table supplies no per-delta
confidence interval or independent repeat count. [Raw bench](../muse_attention_decode/profiles/bench_pk.025ccaf9c57a4ba2.log).

Candidate-only run `569b233dfe08414f`, source `c37d381`, passes 16/16 at 2.449x,
95% interval 2.413-2.502. Full verify times 103.65 / 187.67 us are 7.1 / 10.7 us below the
mean of the two previous DCARRY checks. Those are cross-run observations. Repeat
`e9085c86d89449c2`, source `61e3fd4`, uses the same candidate as its judged path and passes
16/16 at 2.504x, interval 2.467-2.542, with 100.83 / 182.24 us. Its machine is faster on
unchanged rows too; the extra headline-score increase is not another code improvement.
[Attention report](runs-attention.md).

Family E's pending package is now **DCARRY + PK16 + VHOIST**. No new layer result exists in
the reviewed records; layer score stays 1.964x. A-D standings and the mode 37/sk8w3s layer
gaps remain unchanged. The no-fence timing probe in the repeat is diagnostic only and is not
an eligible winning candidate; the production fence stays.

The [community audit](steering/2026-10-29-community-audit.md) identifies two validation gaps:
the intended q_len8 FIN8 stress selected mode 15 instead of mode 11 with eight copies, and
the whole-layer comparison lacks enforced per-seed acceptance, reliable arm-order balancing
and independent mutable cache state. These are open coverage/measurement gaps, not recorded
correctness failures of the incumbent. They should be resolved before relying on the next
combined layer comparison. Existing 13/13 layer judge passes remain recorded.

Decode-32 full attention's apparent +8.7 us regression in the older DCARRY comparison is
now attributed to SGLang's bimodal baseline statistic and machine variation on unchanged mode
49. Preserve the observed cross-run statistic but do not label it a demonstrated candidate
regression. [Rejection register](ruled-out.md), [attention analysis](runs-attention.md).

## Historical review, 13:27: projection context

No new A-E or whole-layer result appeared since the 12:24 review. The three family candidates
awaiting layer tests remain mode 37, sk8w3s and ONE_DCARRY. New projection results belong
outside A-E; they do not change the winning-family badges.

Job `f6c80b4ba11048f3`, source `a35709c`, compares the new single-token cgemv with the
pre-change file `b79a627` in one process. Two passes use different inputs, each with 300
randomly ordered rounds and cold L2. New minus control, pass 0 / pass 1:

| Single-token projection | Paired change |
|---|---:|
| qkv_gate | -0.54 / -0.45 us |
| pooler | -0.86 / -0.96 us |
| unpooler slice | -0.35 / -0.22 us |
| o | +0.08 / -0.11 us, inconclusive |

ABCHECK passes on each reported shape/pass. The same-code null stays within 0.13 us.
The implementation combines row-major planes, a single-load escape start and a larger
register allowance for two chunks per thread. It is a combined projection change, not a
MoE ownership or scheduling experiment. Absolute paired control means and intervals are
not quoted here, so no elapsed-time percentage is assigned.

The candidate-only submission `81254e8aed794eab` records **1.444x, 95% interval 1.422-1.466**.
This is the projection problem's score, not the layer's. Layer integration remains pending;
current layer score stays 1.964x. The earlier qkv8 projection port is also still pending in
the port instructions. [Projection report](runs.md),
[paired raw log](../muse_bf16_proj/profiles/bench.f6c80b4ba11048f3.log),
[candidate result](../muse_bf16_proj/profiles/submission.81254e8aed794eab.json).

The proposed 128-token wide concat front was declined without a new run. Its estimated
24.4 us best-case time exceeds cuBLAS's recorded 24.0 us, before accounting for a second CTA
wave. This is an analytical rejection of that geometry, not a measured candidate loss and
not evidence against staged MoE B. [Rejection register](ruled-out.md).

## Historical review, 11:20: corrected MoE wait measurements

Job `ca43908f02874886`, code `3f67a45`, corrects a diagnostic defect. The timer previously
started after the first potentially blocking try-wait, which could hide waiting entirely.
The earlier claim that fat-path consumers never wait is invalidated. Production instructions
are unchanged, so this is corrected attribution, not a new production speedup or regression.

| Diagnostic observation | 128 tokens | 256 tokens |
|---|---:|---:|
| Phase A weights + activation waits, per consumer warp | 15.73 us | 23.26 us |
| Phase A waits / stage-loop time | 19.6% | 20.2% |
| Phase B waits / stage-loop time | 20.9% | 20.8% |
| Instrumented mode 41 minus production mode 37 | +7.0 us, +4.3% | +6.1 us, +2.7% |

Six paired layer-routing seeds; production controls 164.6 / 227.2 us. The run passes 5/5
judge tests. Counter overhead is material, and per-warp waits cannot be summed into layer
savings. The producers still spend about 70% of phase A waiting for free ring slots.
[Raw log](../muse_moe_w4a16/profiles/bench_fw1.ca43908f02874886.log),
[measurement report](runs-moe.md#fat-path-waits-with-the-first-try-timed-2026-10-29).

A/C/dynamic-claiming production wins remain supported. B remains a tested loss; no new C,
ready-only D, E or whole-layer candidate result appeared in the reviewed changes. The same
three candidates await layer evidence. A proposed per-group load/hoist experiment remains
unbuilt in the [steering report](steering/2026-10-29-fat-moe-waits.md); its predicted savings
are hypotheses. This correction motivates a test, not a new winning-family badge.

## Large-token comparison ledger

| Family / candidate | Control | 128 tokens | 256 tokens | Evidence and scope |
|---|---|---:|---:|---|
| B: staged 19 then 21 | Mode 16 | +27.9 us, +11.2% | +24.2 us, +8.4% | `e221858b020c4dee`, six same-process seeds, complete standalone MoE. Controls 249.4 / 287.3 us. Correctness passed. [Staged gate](runs-moe.md#staged-gemm1-gate-2026-10-29). |
| C inside A: mode 20 | Mode 16 | -2.1 us, -0.8% | -3.8 us, -1.3% | Same job and controls. Other paired checks support the sign. This isolates owned phase A within a fused launch. [MoE log](runs-moe.md). |
| D-related inside A+C: mode 23 | Mode 20 | -16.2 us, -6.6% | -16.1 us, -5.7% | `a9dbec1d98334943`, source `10375cb`, six paired normal seeds; controls 246.2 / 283.7 us. Self-tests pass, spin flags zero. [Dynamic phase B](runs-moe.md#phase-b-claimed-at-run-time-mode-23-2026-10-29). |
| Mode 23 held-out repeat | Mode 20 | -22.9 us, -8.7% | -18.7 us, -6.1% | `36812632dae242ba`, source `ec219f0`, six held-out normal seeds; controls 263.3 / 304.6 us. Skewed routing also improves, -4.8 / -22.9 us. [MoE log](runs-moe.md). |
| A+C+dynamic claiming: mode 26 routing/gather changes | Mode 23 | -5.8 us | -9.5 us | `817eab6401d54a84`, source `ba9a457`, six held-out seeds; self-tests pass. Skewed -4.2 / -7.7 us. [MoE log](runs-moe.md). |
| Same combined design: mode 28 consumer/ring and sorting changes | Mode 26 | -10.6 / -11.6 us | -11.6 / -12.5 us | Two six-seed checks `f928f8a802014e4f` and `13fe68e7ca2e423f`, source `ec62b5a`; correctness passes. [Consumer and ring investigation](runs-moe.md#consumer-stage-and-the-ring-slot-releases-2026-10-29). |
| Same combined design: mode 37 phase A epilogue | Mode 28 | -3.2 / -4.3 us | -4.9 / -5.3 us | `987d024620e44eb7` and `6307d0c91c8d4ada`, source `96f2c43`, held-out seeds 17-22. Correctness passes. Reciprocal and publication-order changes, not a new ownership experiment. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29). |
| E: ONE_DCARRY, cold paired bench | Same attention without DCARRY | -3.65 us, -3.0% | -7.33 us, -3.5% | `4ff339332fef4aff`, full verify only, controls 120.03 / 210.59 us; all arms pass checksum checks and judge 16/16. [Attention bench](runs-attention.md). |
| E: ONE_DCARRY, judged repeats | ONE_PFAST pair without DCARRY | -3.6 us, -3.2% | -7.9 us, -3.8% | Cross-run mean versus mean, `f64e8680b3e441be` / `bf49eceb15a64f81`, source `0d76acc` / `503c0f4`; 16/16 both. Component evidence, not layer savings. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, cold paired bench | DCARRY control | -5.35 us, -4.6% | -9.63 us, -4.7% | `025ccaf9c57a4ba2`; controls 116.26 / 204.22 us, candidate 110.91 / 194.59. ABCHECKSUM 19/0 for every arm. [Attention log](runs-attention.md). |
| E: PK16 + VHOIST, candidate-only judged | Mean of previous DCARRY pair | -7.1 us, -6.4% | -10.7 us, -5.4% | `569b233dfe08414f`, source `c37d381`; 16/16 passed, cross-run component observation. Same-code repeat `e9085c86d89449c2` also passes but has machine-speed variation. [Attention log](runs-attention.md). |
| B: staged 45 + 46 retest | Mode 37 | +30.4 us, +18.4% | +24.4 us, +10.7% | `9d94ac75f4724b52`, six layer-routing seeds, outputs pass; zero faster. [MoE log](runs-moe.md). |
| A+C+D: mode 47 code groups | Mode 37 | -2.3 us, -1.32% | -3.7 us, -1.65% | `5f381b8b6946492d`, held-out layer seeds 13-18, 5/6 and 6/6 faster; checks pass. [MoE log](runs-moe.md). |
| A/E + projection: whole-layer bundle | Incumbent a923f11 | Sliding -6.2 us (-2.20%); full -13.3 us (-3.56%) | Sliding -3.5 us (-0.98%); full -22.5 us (-4.22%) | `aa4276e91c924dad`, 2 inputs per shape, 8 rounds, 72 checks pass; combined gain only. [Layer log](runs-decode-layer.md). |
| A: mode 47, historical three-arm picker observation (arm-bias caveat) | Same file with 47 held out | Sliding -4.9 us (-1.81%); full -5.5 us (-1.43%) | Sliding -5.7 us (-1.58%); full -3.6 us (-0.70%) | `720079090ee44cc3`, 16 paired samples/shape, 72 checks pass; overall ratio invalidated by d1 arm artifact. [Layer log](runs-decode-layer.md). |
| A: mode 51 final pass | Mode 47 | -3.9 us, -2.41% | -3.9 us, -1.71% | `c6b0e500c3a84c9a`, layer seeds 29-34, 6/6 and 5/6 faster; no layer test. [MoE log](runs-moe.md). |
| A/C/D: mode 52 split tail | Mode 51 | +5.9 us, +3.71% | +8.5 us, +3.94% | `209cf29bf91c4ea2`, layer seeds 35-40, output checks pass. Mode 53 also loses +7.3/+9.2 us. [MoE log](runs-moe.md). |
| A: mode 51, historical three-arm observation (gain unestablished) | Clean mode-47 release | Sliding -5.1 us (-1.92%); full -5.3 us (-1.52%) | Sliding -6.5 us (-1.86%); full -1.1 us (-0.21%) | `29fd004fd58e40ef`, 16 paired samples/shape, 72 checks pass, 1.0084 aggregate with null 0.9997. Candidate not released. [Layer log](runs-decode-layer.md). |
| A: mode 51 four-arm layer | Clean mode-47 release, two copies each | Weighted ratio 1.0080; unresolved | Weighted ratio 1.0107; unresolved | `2dc566298be1498f`, overall 1.0057 fails null rule; prepare policies differ by copy. [Layer log](runs-decode-layer.md). |
| E: XPIPE=2, P.V-first cross-tile pipeline | Attention mode 51, XPIPE=0 | +3.90 us, +3.44% | +0.80 us, +0.42% | `5575e7d15d3e49a8`, kernel `86f2bec`, two held-out inputs, 120 rounds, cold pooled median subtraction; controls 113.41/192.45. 16/16 and ABCHECKSUM 19/0. Warm improves; cold gate fails. [Attention log](runs-attention.md). |
| E: mode 115, one CTA/SM | Attention mode 51 | +27.52 us, +24.96% | +57.67 us, +30.80% | `318ae6fc3c3746ce`, source `682aeaa`, two inputs, cold pooled medians; controls 110.27/187.26, nulls +0.06/+0.42. 16/16 and real-arm ABCHECKSUM 19/0. [Attention log](runs-attention.md). |
| E: mode 115 + XPIPE=1 | Attention mode 51 | +28.10 us, +25.48% | +52.61 us, +28.09% | Same job, controls and correctness; overlap helps within mode 115 at 256 but cannot recover residency loss. [Attention log](runs-attention.md). |
| E: PRQ31 per-request tile pipeline | HMRG-on attention, PRQ off | Sliding -0.48 us, -3.02%; full unchanged code | Sliding -0.96 us, -4.59%; full unchanged code | `409996a7214f411a`, `8a6c8bb`, two inputs, 120 rounds, controls 15.87/20.93 us, same-state null +0.03/+0.09 us; 16/16 and 19/0 arm checks. Combined five-bit design. [Attention log](runs-attention.md). |
| Alternative to C inside A: mode 72 stream-K phase A | Mode 51 owned phase A | +12.6 us, +7.59% | +6.6 us, +3.00% | `098949b28fe84bd1`, `86a633e`, six seeds 83-88, layer-like routing, controls 166.1/220.3 us; 0/6 faster, receipt 5/5, first-seed L2 checks 1.5e-4 to 2.4e-4 across cells. Standalone +7.1/+9.2 us. No gate run. [MoE log](runs-moe.md). |
| Alternative to C inside A: mode 74 designated finisher | Mode 51 owned phase A | +4.6 us, +2.82% | +2.0 us, +0.91% | `e02f60fec94f45cb`, `4187b8b`, six seeds 107-112, layer-like routing, controls 163.2/219.9 us; 0/6 and 1/6 faster. First-seed L2 checks 1.3e-4 to 3.4e-4; spin flags zero. Standalone +7.4/+4.1 us. No gate. [MoE log](runs-moe.md). |

Percentages use the logged paired delta divided by the logged control mean, except the explicit pooled-median subtraction rows.  Paired summaries use logged paired deltas rather than subtraction of separately summarized
arm timings. Layer deltas are medians of per-round differences; these can differ by several
microseconds from differences of arm medians. Rows without a comparable control mean report
microseconds only. Seed counts are reported instead of inventing confidence intervals for the
paired differences.


## Small-token results and rejected changes

For A's new `sk8w3s`, job `75bca9775f6f4b0b`, source `03887be`, uses six seeds per size and
routing distribution. This measures the MoE component using layer-like or standalone routing;
"layer-like" does not mean a whole-layer benchmark.

| Tokens | Layer-like control sk8w4s | Change with sk8w3s | Standalone-routing change |
|---|---:|---:|---:|
| 1 | 35.6 us | -0.9 us, -2.5% | -0.6 us |
| 8 | 66.2 us | -2.4 us, -3.6% | -1.8 us |
| 16 | 84.8 us | -3.1 us, -3.7% | -2.9 us |

All 36 paired seeds favor three slots. Every arm matches the control within 2.1e-5 and spin
flags are zero. The candidate entered the standalone variant list at `d0c062b`.
[Evidence and port status](runs-moe.md#skinny-weight-stream-against-the-public-kernel-2026-10-29).

Losses matter as much as wins:

- A's unhinted skinny weight copies lose 1.2-3.3 us across the six size/routing rows in that
  job. Early refill with three slots also loses in every row. Scale-decode and helper-handoff
  changes in `e937013292b24906` were inconclusive and not adopted.
- Phase A mode 35 reduces spills but loses +3.0 / +4.7 us at 128/256 versus mode 28.
  Mode 39's paired stores lose +1.0 / +0.2 us versus mode 37 in `473d87d03de94e96`.
  A shorter epilogue alone does not establish a faster kernel. [Phase A report](runs-moe.md#phase-a-against-the-public-kernel-2026-10-29).
- E's run-time pair weighting loses +9.0 / +12.0 us at full verify 256 for weights 1.40 / 1.50
  in `4ff339332fef4aff`. The arrival counter fails to predict scheduler priority. This is an
  attention scheduling loss, not evidence against MoE family D. [Attention log](runs-attention.md).

## Whole-layer outcome and attribution

Board remains `3fbb3ad027e84c6b`, 2.014x (1.984-2.045), mode 47. Mode 51 has component wins,
but its layer gain is unresolved by the four-arm/null rule; earlier three-arm magnitudes are
not established isolated gains. Runtime M1 + 51 shows a combined heuristic win with arm-offset
and cross-run attribution limits. The enforced-gate M1 + 47 file has a 2.013x scored run with unknown M1 status; the M1 + 51
file scores 2.013x with M1 off. Revised census-gate candidates are unchecked.
[Manifest](release-manifest.md), [arm-bias audit](steering/2026-10-30-layer-bench-arm-bias.md).

## Decision and next evidence needed

Preserve the release. Check the revised census-gate candidate before promoting it, and stabilize
prepare-time policies so layer comparisons can resolve small effects. The tuner-stability build
has CPU tests but no GPU measurement. Keep mode 51's component gains separate from its unresolved layer
result. The tested mode-49 history lead supports merge-order rounding; ordering reviews are
complete within their scope. B remains parked, and the new gather variants do not revive it.
D's full ready-only queue is still untested; phase-A claim layouts and their K=0 controls lose. E's XPIPE variants fail their cold gates, while PRQ31 wins on per-request paths.
The combined layer_hp (PRQ + HMRG + placement) is built but has no GPU result. Projection
retains its paired split-barrier gains at 128/256; aggregate cross-run gains remain inconclusive.
The new QKV cluster split loses in tuner observations. Corrected mode 74 also loses, so C stays.
Family F complete-file comparisons now take priority over new projection/attention rewrites. [Rejection register](ruled-out.md).

## Review history and checkpoint

- 2026-10-30 14:05 PDT: inspected through `bc10cb1`, including jobs `eaa881a934aa471e`,
  `e02f60fec94f45cb` and `ad6c673d9a444c1e`. Added corrected mode 74 loss, earlier build
  caveat, QKV tuner loss and Family F registration. No new attention or layer result.

- 2026-10-30 12:44 PDT: inspected through `c98018f`, including projection round 4
  jobs `4f3ff774981a494c`, `25ea2b62ff8a47a7` and `d85d0081fa8a4a43`.
  Paired router gains at 16/128/256, nulls at 8/32, release-split losses retained.
  Four run logs, ruled-out entries, port guide and release manifest reconciled;
  18 untracked receipts remain historical provisional evidence. A-E and layer status unchanged.

- 2026-10-30 11:49 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. All 18 untracked profile JSONs parse and are already logged.
  Family outcomes and pending layer tests remain unchanged. Review metadata refreshed
  locally; no new-evidence commit or push.

- 2026-10-30 10:45 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. All 18 untracked profile JSONs parse and are already logged.
  Family outcomes and pending layer tests remain unchanged. Review metadata refreshed
  locally; no new-evidence commit or push.

- 2026-10-30 09:44 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. All 18 untracked profile JSONs parse and are already logged.
  Family outcomes and pending layer tests remain unchanged. Review metadata refreshed
  locally; no new-evidence commit or push.

- 2026-10-30 08:44 PDT: checkout remains `5a8e7f9`; fetched origin/main has no
  remote-only commits. Four run logs, rejection register, release manifest and project
  instructions are unchanged. Parsed all 18 untracked profile JSONs, all already logged.
  No new result or correction; family outcomes and pending layer tests remain unchanged.
  Review metadata refreshed locally; no new-evidence commit or push.

- 2026-10-30 07:49 PDT: checkout and fetched origin/main remain `5a8e7f9`. Four run
  logs, rejection register and release manifest are unchanged. All 18 untracked profile
  JSONs parse and are already logged. Existing local edits contain only the prior review's
  metadata refresh. Family outcomes and pending layer tests are unchanged. Review metadata
  refreshed locally; no new-evidence commit or push.

- 2026-10-30 06:46 PDT: checkout remains `5a8e7f9`; only the previous review commit
  follows evidence snapshot `15cc6b9`. Four run logs, rejection register and release manifest
  are unchanged. Fetched origin/main with no remote-only commits. Parsed all 18 untracked
  profile JSONs; all are already logged jobs. Accessible chat inventory provides no newer
  frontier result. Family outcomes and pending layer tests are unchanged. Review metadata
  refreshed locally; no new-evidence commit or push.

- 2026-10-30 05:39 PDT: inspected `b5b8a34..15cc6b9`, mode 72 job
  `098949b28fe84bd1` and its raw receipt/log, ruled-out entry and source/ordering notes.
  Receipt 5/5; stream-K alternative loses all six size/routing cells. Attention, projection
  and layer logs unchanged; layer_hp still untested. All 18 untracked profile JSONs are
  already logged. Fetched origin/main with no remote-only commits at inspection. No
  benchmark, submission or push by this review.

- 2026-10-30 04:43 PDT: inspected `16573a6..b5b8a34`. Checkpointed five PRQ checks,
  scored attention `1896e72c11ae4b0f`, five MoE scorecard/claim checks and P3 check `4922f366a0c142ca`
  listed above. Three previously missing scored receipts now verified. layer_hp is built only.
  E gains PRQ evidence; A/D code-layout losses and P3 loss preserved. Whole-layer release and
  B decision unchanged. Existing 18 untracked profiles are already logged. No benchmark,
  submission or push by this review.

- 2026-10-30 03:37 PDT: inspected `8838272..16573a6`, including projection changes that
  landed before the previous documentation commit. Checkpointed attention `ce92d32e26024bd2`,
  `d0819bb26907426b`, `5d309f9d13924dca`, `82b1bf0fad3244b9` and projection
  `3abb1cd974e2477d`, `d7c662296eaa4413`; all six receipts parsed. New scored jobs
  `57b0bed35f694a23` and `d59bc7af342a465d` are log-only. Prior three missing projection
  receipts remain absent. E pipeline losses stand; HMRG is separate merge context. MoE/layer
  logs unchanged; 18 untracked profiles already logged. Origin/main fetched, no remote-only
  commits at inspection. No benchmark, submission or push by this review.

- 2026-10-30 02:38 PDT: inspected `1e42dbe..8838272`; checkpointed five attention jobs
  (`5575e7d15d3e49a8`, `318ae6fc3c3746ce`, `a92aa9c43f064c49`,
  `7144310358564fa3`, `b7762d764d474eae`) and five R2 projection jobs above.
  Scored P1 job `1d277e7fe9c44712` remains log-only. Three projection receipts absent locally.
  All ten new check receipts parsed; 18 untracked profile JSONs are already logged jobs.
  MoE/layer logs unchanged; fetched origin/main with no remote-only commits. E's new tested
  losses are scoped to cold-cache implementations; A-D and layer release unchanged.

- 2026-10-30 01:31 PDT: inspected `45a12fe..87bc0b8`, the standalone roadmap and new
  projection P1 round. Six jobs above are now checkpointed; four receipts verified and
  two available only through the run log. MoE, attention and layer run logs are unchanged.
  The 18 existing untracked profile JSONs remain already logged, provisional local files.
  A-E outcomes, layer release and pending layer candidates are unchanged. No benchmarks,
  submissions or pushes by this review.

- 2026-10-30 00:32 PDT: reviewed `5b5571e..a48edef`, scored runs `b68bdcd7a1ba464c` and
  `d6c30f88c617438b`, gate failure analysis, updated manifest and unrun census/stable-tuner
  builds. No new paired comparison; latest four-arm checkpoint remains `40b9b3f795f84073`.
  Other component streams unchanged. Existing 18 untracked JSON profiles are already logged.
  Origin/main fetched, no remote-only commits. Board and family timing conclusions unchanged.

- 2026-10-29 23:35 PDT: reviewed `a9a6381..dd1777e`, four-arm calibration/comparisons
  `724aa0dd6ed140ca`, `2dc566298be1498f`, `40b9b3f795f84073`, older newly incorporated runtime
  bench `f0ee07c4d4904511`, attention `2b93c6ecf4f641c6`, MoE `7063f86352344efa` and
  `e7e75c9510994c08`, ordering reviews and candidate manifest. Board unchanged; isolated
  mode-51 layer claim downgraded to unresolved. All 18 untracked profile JSONs already logged.
  Origin/main fetched, no remote-only commits. No benchmark, submission or push by this review.

- 2026-10-29 22:34 PDT: reviewed `22264ae..6cb0796`, five new layer jobs
  `6ce5be0ba29642ea`, `60cf58b69b75491e`, `fdf75ba5a8124544`, `a438b82ff2c141c2`,
  `29fd004fd58e40ef`, manifest, validation changes and unrun history diagnostic. Other
  component checkpoints unchanged. Existing 18 untracked JSON profiles are already logged
  jobs. Origin/main fetched with no remote-only commits. Board release unchanged; mode 51
  now layer-tested, final-file confirmation pending.

- 2026-10-29 21:29 PDT: checkout remains `22264ae`; fetched origin/main with no
  remote-only commits. Four run logs, rejection register and steering index are unchanged.
  All 18 untracked profile JSONs are already logged jobs. No new measurement or correction;
  mode 51 still awaits layer testing. Review metadata refreshed locally, no commit or push.

- 2026-10-29 20:31 PDT: inspected through `22264ae`; fetched origin/main with no
  remote-only commits. Four run logs and rejection register are unchanged since `b41cb8d`.
  All 18 untracked profile JSONs are already logged jobs. Accessible chat inventory supplies
  no newer relevant evidence. Run checkpoints, family standings and mode 51's pending layer
  test are unchanged. Review metadata refreshed locally; no new-evidence commit.

- 2026-10-29 19:31 PDT: reviewed `b7da8bb..b41cb8d`, release manifest, four run logs,
  rejection updates and raw layer/MoE diagnostics. Latest layer release `3fbb3ad027e84c6b`,
  attribution `720079090ee44cc3`; latest MoE submission `bcd2d7a3506a4b7d` (board stays 47),
  tail test `209cf29bf91c4ea2`. Projection/attention standalone checkpoints unchanged.
  All new run IDs and invalidated diagnostics are listed above. Existing 18 untracked profile
  JSONs already correspond to logged jobs. Origin/main fetched, no remote-only commits.

- 2026-10-29 18:25 PDT: reviewed `b75e2f9..283e348`, four run logs, rejection updates and
  raw MoE/layer comparisons. Latest layer `edd71d3152a946fe`, MoE `abe7651dbaa44458`,
  projection `1093e22ca2534d7f`; attention component remains `e9085c86d89449c2`.
  All new jobs are listed above, including invalidated attempts. All 18 untracked profile
  JSONs are already logged jobs and were preserved. Fetched origin/main, no remote-only
  commits. Dates on source logs use 10-30; this review is 10-29 in America/Vancouver.

- 2026-10-29 17:22 PDT: inspected through `b75e2f9`; fetched origin/main with no
  remote-only commits. All four run logs and the rejection register are unchanged since
  `f61d8d9`. All 18 untracked profile JSONs correspond to already logged jobs. No new
  measured result or layer confirmation; run checkpoints and A-E standings are unchanged.
  Refreshed review metadata locally; no commit because no new evidence was found.

- 2026-10-29 16:27 PDT: inspected `dbd99c3..f61d8d9`, all four run logs, the updated
  rejection register, compile-gate reports and bundle `b694b4a`. Fetched origin/main with
  no remote-only commits. All 18 untracked profile JSONs are already logged jobs.
  No new timing jobs; newest attention remains `e9085c86d89449c2`, layer remains
  `9b921febc7ad4140`. B retest trigger is met; bundle assembled but unjudged.

- 2026-10-29 15:24 PDT: inspected through `dbd99c3`; fetched origin/main with no
  remote-only commits. Run logs and rejection register are unchanged since `efddc84`.
  All 18 untracked profile JSONs correspond to already logged jobs; no new provisional
  measurement found. Accessible chat inventory shows no newer relevant activity.
  A-E standings, pending layer tests and run checkpoints are unchanged. Refreshed review
  metadata and generated output locally; no commit because there is no new evidence.

- 2026-10-29 14:25 PDT: reviewed `615539c..efddc84`, including attention jobs
  `025ccaf9c57a4ba2`, `569b233dfe08414f`, `e9085c86d89449c2` and community-audit validation
  findings. Fetched origin/main; no remote-only commits. New E gains, no new layer result.
  Newest incorporated attention run is `e9085c86d89449c2`; other stream checkpoints remain.
  Existing untracked profiles preserved. Local commit only under the no-push rule.

- 2026-10-29 13:27 PDT: reviewed through `615539c`; fetched origin/main, no remote-only commits.
  Incorporated projection runs `5ff76a62b0044378`, `f6c80b4ba11048f3` and
  `81254e8aed794eab`, sources `803178f` / `a35709c`, reported in `754549d`, plus
  `f88e278`'s analytical concat rejection. A-E logs and layer outcomes unchanged. Untracked
  profiles remain untouched. Updated dashboard context without assigning projection gains
  to a family. Commit is local under this automation's no-push rule.

- 2026-10-29 12:24 PDT: no new experiment evidence since the 11:20 review. Fetched origin/main;
  no remote-only commits. Run logs and tracked profiles unchanged; all 18 untracked result
  JSONs are readable and their run IDs already appear in existing reports. Family outcomes,
  pending layer tests and latest incorporated run IDs remain unchanged. Refreshed review
  metadata only; no new evidence commit under the automation's commit policy.

- 2026-10-29 11:20 PDT: reviewed `d0c062b..14d6da1`, fetched origin/main and found no remote-only
  commits. Incorporated diagnostic run `ca43908f02874886`, source `3f67a45`, reported by
  `eced63d`. No new layer result. Prior candidate timing ledger retained. Dashboard summaries
  and review metadata updated; no kernel, benchmark or submission action. Commits remain local
  because this automation explicitly forbids pushing.

- 2026-10-29 10:24 PDT: initial reconciliation through `d0c062b`. Read current MoE, attention
  and layer run records, architecture baseline and layer build selection. Recorded wins and
  losses above without running GPU work or modifying implementation files.
- Latest incorporated streams: MoE skinny `75bca9775f6f4b0b`; MoE fat `6307d0c91c8d4ada`;
  attention `4c12b0199efd4b8c`; layer `9b921febc7ad4140`. These streams have independent
  histories; use run IDs and source revisions rather than treating this list as time order.
- Untracked profile JSONs were inventoried and readable. Their corresponding historical runs
  already appear in committed logs; they were not staged. This review uses the committed
  experiment reports, not an independent recomputation of every raw sample or memory-ordering
  proof. Later reviews should inspect changed reports and new run IDs, including uncommitted
  results, and explicitly flag any disagreement.
- Superseded "pending" and "not submitted" paragraphs remain in historical logs. Later named
  result rows control the current status. The old MoE footer still names mode 23's board entry;
  the later mode 37 submission row supersedes it. No live leaderboard was queried.

## Visual dashboard

Open the [local dashboard](http://127.0.0.1:4387/session/3de14ef757fbea5b) while Lavish is running,
or open the [standalone HTML](../.lavish/architecture-performance/index.html). It uses the
existing gap-analysis palette and typography stack. Filters separate measured wins, tested
losses, partial designs and candidates awaiting layer evidence.

Hourly reviews update this tracker and `docs/architecture-dashboard.json`, then run
`python3 tools/build_architecture_dashboard.py`. The builder derives the comparison table
from this tracker and embeds its fingerprint. Keep JSON summaries, pending candidates and
review timestamp aligned with the evidence. Layout changes belong in
`tools/architecture-dashboard-template.html`; rebuild and check them in the browser.
