# Fat MoE: repaired wait accounting and the next mechanism

Evidence: code commit `3f67a45` (counter fix, modes 41 and 42) on branch
`worktree-agent-a074aed40d591d983`, check `ca43908f02874886` (2.174x, 2.142-2.207, 5/5 passed;
profiles `muse_moe_w4a16/profiles/bench_fw1.ca43908f02874886.*`), SASS record
`muse_moe_w4a16/profiles/sass_fatwait.2026-10-29.txt`. Full tables are in
[docs/runs-moe.md, "Fat-path waits with the first try timed"](../../../docs/runs-moe.md). This report closes
section 2 of the [overnight steering report](2026-10-29-overnight.md). It recommends one
section 5 mechanism and builds nothing.

## What was wrong and what changed

`mbar_wait_cyc` (phase A diagnostic modes 31, 33, 38 and 40, and the producers' phase A slot
waits) returned 0 when the first `mbarrier.try_wait` succeeded and started its clock only after
a failed first try. try_wait is potentially blocking (PTX ISA 9.4), so the "0.00 us" weights
waits did not show that the consumers never wait. The clock now starts before the first try.
try_wait stays; it was not replaced with test_wait. The audit found no other try_wait counter
with this mistake: the is_diag stamps, the skinny stamps (fixed in 8272efe, 03887be) and the
public-kernel patches in `tools/moe_make_bench_ab.py` already time the whole wait.

Production code is unchanged. Modes 0, 16, 26, 28 and 37 and the skinny kernels have the same
SASS text, registers and spills before and after the change. Only the diagnostic modes differ.

## Measured (mode 41 = 37 with counters, layer routing, seeds 1-6)

| per consumer warp, us | 128 tokens | 256 tokens |
|---|---|---|
| phase A waits, weights + x | 8.4 + 7.3 | 11.8 + 11.5 |
| phase A stage loops without waits | 64.6 | 91.7 |
| phase A epilogue | 6.1 | 9.2 |
| phase B waits, weights + x + claim | 4.5 + 3.9 + 1.2 | 6.4 + 5.4 + 1.2 |
| phase B stage loops without waits | 31.7 | 45.0 |
| phase B epilogue (per unit) | 4.4 (2.2) | 10.1 (3.8) |
| producers waiting for a free slot in phase A (weights / x) | 65.5 / 64.3 of 93.0 | 94.0 / 92.0 of 131.7 |
| average cycles per wait (A / B) | 209 / 222 | 227 / 231 |

- Waits are about 20% of the stage-loop time in both phases at both sizes. The earlier statement
  that the fat consumers never wait for weights or x is withdrawn, with a dated note in
  `docs/runs-moe.md`.
- The ring is still full for about 70% of phase A from the producers' side. The waits average
  210-230 cycles each, nearly the same in every phase and at both sizes. So the stream is not
  falling behind on average. The cost comes at the stage boundary.
- Instrumentation overhead is measured, not assumed. 41 minus 37 is +7.0 / +6.1 us at 128 / 256,
  42 (light stamps) minus 37 is +1.1 / +2.3, and both are 0 of 6 lower. The counters add
  1.3-2.2 us to phase A, too little to explain the waits: at least 13.5 / 21.9 us remain if the
  whole overhead sat inside them.
- Same-process gap to the public kernel, by stamped phase (42 against the public): phase A
  +3.6 / +11.1 us, phase B +2.7 / +7.3, launch end +12.3 / +24.6.

## Hypotheses (not separated by this run)

H1: every stage starts with a serial path on each warp. The try_wait round trip comes first, then
the shared loads of the stage's codes and only then the first dequant. These run while at most one
wgmma group of the previous stage is in flight.
H2: the 16 warps of a CTA drift apart. A slot refills only after all 16 release it, and there
are 3 weights slots at both sizes, so a leading warp waits on its slowest peer.

Per-warp waits are not SM time lost one for one, because three other consumer warps share each
scheduler. The SM-level effect of removing a warp's waits is unknown until it is tested.

## Recommendation: one mechanism

**Mode 43: per-group code loads with the next stage's waits and first code group issued under the
current stage's last wgmma group.** This is the "smaller code groups per stage" item of
section 5, used here for latency hiding rather than register relief:

- Each warp loads one 4-step group at a time (one `uint4` of codes and its scale word, 4-6
  registers) instead of the stage's 16 code registers up front. That frees the registers to
  hold the next stage's first group.
- During the current stage's last group, the warp waits on the next stage's `wfull` and `xfull`
  and loads that stage's first code group. The barrier round trip and the shared loads then
  overlap tensor-core work instead of preceding it.
- The weights slot is released after the stage's last group load. The ordering chain is mode 28's
  per-thread `fence.proxy.async.shared::cta` then arrive, at a later point, and every PTX step in
  docs/runs-moe.md "Ordering of the ring slot releases" still applies. This is the public kernel's
  stage structure.
- Mode 44 is the same with the mode 41 counters, so the run shows whether the waits moved.

This is not the losing mode 35 (volatile shared reloads of per-unit values). Smaller groups
alone, without the hoist, are not recommended. Stage-loop spills are 9 accesses in mode 37, and
register pressure has not been shown to limit the loop. Holding the slot longer without
overlapping anything could make H2 worse.

Predicted saving (hypothesis): -3 to -6 us at 256 tokens and -2 to -4 us at 128 on layer
routing. The basis is the public kernel's 11.1 us phase A lead at 256 with this structure, about
4 us of which its 4% shorter stage loop explains. The ceiling of the per-warp waits, 35 us at
256, is not an SM-level saving.

## Validation and stopping rule

1. Local gate, no GPU: 96 registers, no more LDL/STL inside the stage loop than mode 37, no
   ptxas C7512/C7513 (wgmma serialization), `tools/moe_sass_counts.sh` showing one `wait1` per
   group, and production SASS unchanged. Any failure: stop and record, with no check spent.
2. One check after the 00:00 UTC reset: `tools/moe_make_bench_ab.py --base=37
   --sizes=32,128,256 --routings=layer o43=43 o44=44 o42=42`, seeds 1-6. Adopt toward the gate
   only if 43 minus 37 has a mean of -2 us or lower at 256 with at least 5 of 6 seeds lower, a
   mean of +0.5 us or lower at 128 and 32, and ABCHECK within 1e-4 of 37.
3. Mechanism check, from the same run: 44's average cycles per wait must fall by at least 30%
   against 41's 209-231. If timing wins but the waits do not fall, the gain is unexplained:
   repeat it before any adoption. If the waits fall but timing does not improve, H1 is right
   but not on the SM's critical path. Record that and stop this line.
4. A loss or tie ends the mechanism. Record it in docs/ruled-out.md, scoped to this hoist depth
   (one group ahead), with 44's counters. One follow-up variant is allowed only if 44 shows the
   waits gone and the stage loop slower, which would point to a register or issue cost of the
   hoist.
5. Budget: at most 2 MoE checks after the reset, one for step 2 and one for a repeat or the
   shippable gate run (GATE with held-out and skewed seeds). Never `okc submit`.

## Next in line, not recommended now

- The phase B epilogue costs 3.8 us per unit at 256 (10.1 us per warp) against phase A's 1.7.
  It is the largest non-loop cost in phase B and is worth its own attribution (sum-of-squares
  atomics, ybuf stores, the release) after mode 43 is settled.
- The natural-k intermediate layout remains a separate, higher-risk option and needs the
  traffic estimate and layout proof that section 5 asks for.

## Gaps

- The counters cannot separate H1 from H2 or give the try_wait's intrinsic cost on a phase
  that has already completed.
- Only layer routing at 128 and 256 tokens was measured. Standalone, skewed and 32-token
  routings were not.
- The public kernel's waits were not measured: its DEBUG build stamps phases, not waits.
- No whole-layer effect is claimed.

## Addendum 2026-10-30: mode 43 built and checked

Evidence: code a4687c8 (modes 43, 44, 47; staged pair 45 + 46), checks 9d94ac75f4724b52 (layer
seeds 1-6) and 304f17f3e8704992 (held-out layer seeds 7-12), both passed 5/5. Details:
docs/runs-moe.md, "Code groups with the next stage hoisted" and "Staged pair on current
components"; SASS record muse_moe_w4a16/profiles/sass_grpcodes.2026-10-29.txt.

- Local gate passed: production SASS unchanged, no C75xx, one wait1 per group, 43 spills 8 / 100 B
  against 37's 156 / 416 B, nothing local in the stage loops.
- Measured: 43 minus 37 is -2.7 and -2.1 us at 256 tokens (6 of 6 in both checks), -1.4 / -1.2
  at 128 and -1.3 / -0.9 at 32. The timing half of step 2 passed twice.
- Measured: 44's average cycles per wait did not fall (phase A 239 / 235 against 41's 223 / 225 at
  256 tokens; up 3-7% at every size). Step 3's mechanism check failed twice, so H1 as stated (the
  serial try_wait and code loads at the stage start) is not the cause of the waits. H2 (warp drift
  within a CTA) or the stream's own pace at the boundary remain; neither was separated.
- Measured: mode 47, the same code groups without the hoist, beat 43 by 1.5 us at 256 (6 of 6)
  and 37 by 3.6 (6 of 6), 1.6 at 128 (6 of 6), 0.4 at 32 (3 of 6), on the held-out seeds. The gain
  is the register relief this report's recommendation called secondary ("smaller groups alone
  ... are not recommended"); that sentence is withdrawn by this evidence.
- Decision under the stopping rule: 43 is not kept, and the hoist is recorded in
  docs/ruled-out.md. 47 is a candidate that was not pre-registered, with one check behind it.
- Coordinator request, same checks: the staged two-launch pair rebuilt on 37's components lost
  again by 24-30 us and stays parked with a new trigger.

Next actions (not done; the 2-check budget is spent):
1. A gate run for 47 against 37: GATE with held-out standalone seeds 17-22 and skewed seeds 11-13,
   plus layer seeds, before any adoption. Pre-register the rule: -2 us or better at 256, no worse
   than +0.5 at 32 and 128.
2. If 47 is adopted, port it: MAIN_MODES and MAIN_CANDIDATES in muse_moe_w4a16/submission.py,
   LAYER_MODES and MAIN_NAMES (47: "main-lean-grp-nohoist") in tools/dl_moe_inline.py.
3. The waits themselves: to separate H2, stamp per-warp stage starts within a CTA (skew) rather
   than per-warp wait totals.

## Addendum 2026-10-30 (2): mode 47 gate

Next actions 1 and 2 of the addendum above are done. The rule was committed (28fc668) before
check 5f381b8b6946492d; 47 passed every condition on held-out seeds 13-18 (256 tokens: -3.7 /
-3.4 / -2.8 us against 37 on layer, standalone and skewed routing, 18 of 18 seeds lower). The
shippable build (ec565a7, check b50351288b504e81, 2.190x) carries 47 in MAIN_MODES and the policy
picks it at 128 and 256. Layer port steps are in docs/moe-port-guide.md; they were applied to a
scratch copy of the layer branch only. Next action 3 (per-warp stage-start skew for H2) is open.

## Addendum 2026-10-30 (3): warp skew measured, phase gap re-attributed, mode 51

Evidence: branch worktree-agent-ad7a25ed576556191, code 64cec1f (stamped twins 48, 49), a03ca00
(mode 50), 69c06ed (mode 51); checks d47c78bdc130424b (skew and phase stamps, layer seeds 1-6),
8a7051fa926e43e1 (50's gate), c6b0e500c3a84c9a (51's gate). Details: docs/runs-moe.md, "Warp skew
and the gap to the public kernel after mode 47".

- Measured (next action 3, H2): the four warps inside a warpgroup stay within 50-160 cycles; the
  four warpgroups of a CTA are 750-1,200 cycles apart at each stage start (about half a stage at
  256 tokens) and their slot releases 950-1,400 apart. A slot is refilled 300-550 cycles after its
  slowest release and lands about 5,000 cycles later, while a warp returns to the slot 5,800-7,200
  cycles after its own release, so the leading warpgroups wait. Up to 72-74% of the weights-wait
  cycles come from that skew (a bound). x waits are 1.2-1.8 times the weights waits; the x slot
  is released a stage later than the weights slot. Not shown: that the skew costs SM time; the
  trailing warpgroup sets the pace.
- The public kernel could not be stamped per warp: any consumer stamp makes its 96-register
  DEBUG build spill and three of four subsets serialize its wgmmas (C7512).
- Measured, gap to the public kernel by phase (48 against its stamps, us at 128 / 256): setup
  +4.3 / +4.4 (3.4 / 3.5 of it in the x gather and release), phase A +2.3 / +7.7, phase B +2.9 /
  +7.3, tail +3.7 / +3.6, barrier and final pass +4.0 / +3.6. 32 tokens was not stamped (no check
  left); in the gate runs public minus 47 at 32 tokens on layer routing is -4.5 us (graph replay).
- Built for phase B: 50 forms each route's sum of squares in the final pass (the public kernel's
  OKC_WGSFINSS) instead of shuffles and atomics in the phase B epilogue. Its gate passed at 128 and
  256 (-4.1 to -8.3 us, 36 of 36 seeds lower) and failed at 32 tokens on layer routing (+0.7
  against +0.5). 51 does it from 64 tokens up and runs 47 below; its own pre-registered gate on
  held-out seeds passed every condition: -3.8 to -4.0 us at 128, -3.9 to -7.7 at 256, -1.0 to +0.0
  at 32. Shippable build handed to the coordinator; not judged as a whole file (budget spent).

Next actions (not done):
1. A shippable check of the build with 51, then the layer port (docs/moe-port-guide.md, verified
   on a scratch copy of origin/main).
2. Phase A and the skew: per-warpgroup weights slots (four 9 KB bulk copies per stage, each with
   its own full and empty barriers) would let a leading warpgroup refill without waiting for the
   slowest one. It changes the ring's ordering chain (needs its own PTX chain and review) and only
   pays if the leaders' idle time is on the critical path, which the stamps do not show.
3. Setup: 3.4-3.5 us of the gap sits in the x gather and its release at every size (it also
   applies at 32 tokens); the gather's per-thread `__threadfence` before the barrier is a
   candidate under mode 37's chain, unmeasured.
4. The 32-token gap (10-14 us in the layer) has no phase stamps yet.

## Addendum 2026-10-30 (4): gather or skew, and the gather change

Evidence: branch worktree-agent-a24ef7a48cd7155d3, code e866c8a (twins 55, 56 of 51) and fca2be7
(modes 57-59); checks 7063f86352344efa (re-measurement, 2.210x) and e7e75c9510994c08 (diagnostic,
2.207x). Details: docs/runs-moe.md, "The x gather or the warpgroup skew".

- Measured, next action 2 (skew): on 51 the warpgroup that starts a stage last is the same one in
  79-87% of a CTA's stages, and only 0.45 / 1.07 us per CTA of its weights waits (128 / 256
  tokens, A + B) come from the other warpgroups' releases. Per-warpgroup slots would shorten the
  leaders' waits, not the trailing warpgroup's; not built (docs/ruled-out.md).
- Measured, next action 3 (setup): the gather is +3.2 / +3.4 us behind the public kernel's; in
  thread 0's stamps the loads and stores take 1.7 / 5.6 us and the fences and barrier 4.3 / 2.1.
- Built and lost: 57 (rows loaded under the routing work plus 16-byte stores, all fences kept)
  +1.1 to +1.2 us against 51 at 32, 128 and 256 tokens (0 of 18 seeds lower); 59 (early loads
  alone) tied. Each CTA's gather got 0.8-1.8 us shorter, but a job's x readiness follows its
  slowest gathering CTA and moved 0.2-0.6 us.

Open, needs a decision outside the MoE line: the remaining gather difference from the public
kernel is the per-thread `fence.proxy.async.global` and `__threadfence` before the gather barrier
(the public publishes with a barrier and one red.release per job). Changing it is a fence change:
it would rest on mode 37's chain, need an independent review, and was excluded from this brief.
Two of the four checks given to this line are unspent.
