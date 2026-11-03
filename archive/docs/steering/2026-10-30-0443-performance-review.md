# Hourly review: PRQ gain, claiming-layout losses and P3 loss

Evidence snapshot `b5b8a34`. Reviewed 2026-10-30 04:43 PDT.

Family E now has a per-request pipeline win: `ONE_PRQ=31` combines transpose overlap,
Q.K-first issue, fast copies, split P.V issue and an offset-test change on q_len=8 modes
1/3/11/15. This is a coherent five-change design, not an isolated gain for each ingredient.
It leaves flat full-context paths unchanged, so it does not invalidate the XPIPE losses.
MoE runtime phase-A claiming variants lose; their no-work controls also lose. The evidence
rejects these code layouts, not runtime claiming itself. B's staged comparison remains unchanged.
[Attention round 3](../../../docs/runs-attention.md), [MoE evidence](../../../docs/runs-moe.md).

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
[First log](../../../muse_attention_decode/profiles/bench_g.a7a4c23e98ca484a.log),
[failed confirmation](../../../muse_attention_decode/profiles/bench_h.3c6e1a7f1cc644b6.log),
[matched-null confirmation](../../../muse_attention_decode/profiles/bench_i.409996a7214f411a.log).

Final-file checks `640ee49b691b45ac` and `8b00100bf85d4ef5` at `57f57dd` (comment-only
repeat) pass 16/16 at 2.588 (2.549-2.627) and 2.545 (2.506-2.583). The file combines
PRQ31, HMRG and workspace picking. Cross-run variation exceeds PRQ's modeled gain; unchanged
full-context rows move too. Scored `1896e72c11ae4b0f` at `a70192d` passes 16/16 at
2.521 (2.484-2.559), replacing the 2.504 bench-file attention entry with a clean file.
This is standalone attention, not the 2.014x layer board.
[Check](../../../muse_attention_decode/profiles/sub_prq.640ee49b691b45ac.json),
[repeat](../../../muse_attention_decode/profiles/sub_prq_r.8b00100bf85d4ef5.json),
[scored receipt](../../../muse_attention_decode/profiles/scored.1896e72c11ae4b0f.json).

`layer_hp.py`, SHA256 prefix `0efd553c`, ports PRQ31 + HMRG + workspace picking onto the
release, retaining MoE47 and no M1. Its source/compile checks are recorded, but it has no GPU
check or measured layer gain. A four-arm bench is prepared. It is an E-plus-merge/placement
bundle; a future gain cannot be assigned entirely to E. Existing layer_g51b/layer_m1rt51c
candidates still await tests. [Release manifest](../../../docs/release-manifest.md), [layer log](../../../docs/runs-decode-layer.md).

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
[Gather log](../../../muse_moe_w4a16/profiles/bench_gnf60.48b9bf9d77d146ac.log),
[claims](../../../muse_moe_w4a16/profiles/bench_ca62.b7e43ee6a517487a.log),
[merged-loop null](../../../muse_moe_w4a16/profiles/bench_ca66.ed4591a2477041c2.log),
[lean null](../../../muse_moe_w4a16/profiles/bench_cl69.b2c92a71600d488b.log).

The fifth job, `3d8902381a194d56`, is a standalone scorecard diagnostic, source logged as
"this branch (tool only)" without an exact revision. Ten standalone-routing seeds 301-310,
20 cold replays and duplicate preparations compare current sk8w3s/mode 51 against the public
kernel. Public-minus-ours gaps at 128/256 are -9.60/-14.92 us (-4.83%/-6.39% of ours'
198.88/233.515 us means), public faster 10/10. Across-seed standard deviations 3.92/3.47 us
are not confidence intervals. At 1 token ours wins 1.37 us; at 8 public wins 0.80; at 16 the
0.04 us difference is a tie; at 32 public-minus-ours -1.54 us has SD 8.02 and only 4/10 faster,
so it is inconclusive. The scorecard does not revive B without its phase-B or phase-A-tail trigger. [Scorecard and controls](../../../docs/runs-moe.md).

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
[Run and model](../../../docs/runs.md), [bench log](../../../muse_bf16_proj/profiles/proj_p3_bench.4922f366a0c142ca.log),
[receipt](../../../muse_bf16_proj/profiles/proj_p3_bench.4922f366a0c142ca.json), [stop rule](../../../docs/ruled-out.md).

### Receipt reconciliation and next decision

Previously missing scored receipts now verify: attention `57b0bed35f694a23` passes 16/16,
2.479 (2.442-2.516); projection `1d277e7fe9c44712` and `d59bc7af342a465d` pass 9/9,
1.482 (1.460-1.504) and 1.488 (1.466-1.510). Their cross-run interpretation is unchanged.
[Attention receipt](../../../muse_attention_decode/profiles/scored.57b0bed35f694a23.json),
[projection P1](../../../muse_bf16_proj/profiles/scored.1d277e7fe9c44712.json),
[conforming projection](../../../muse_bf16_proj/profiles/scored.d59bc7af342a465d.json).

The priority is now the prepared layer_hp controlled comparison. Preserve the working layer
release and the A-versus-B gate. After repeated claiming-layout losses, the next claim design
must remove the duplicate compiled work rather than repeat a larger kernel. P3 needs a new
raw geometry before another compression test. Do not sum PRQ, HMRG, placement or projection gains.


The [tracker](../../../docs/architecture-performance-tracker.md) preserves earlier evidence and the comparison ledger.

Validation: independent source review checked PRQ, MoE and P3 figures and attribution.
Local links, derived percentages, source fingerprint, 23 comparison rows and dashboard
filters passed checks. The family guide links the tracker once. No layout changes were made.
No benchmarks, submissions or pushes were performed by this review.
