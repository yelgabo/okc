# Layer report: clean release, M1 diagnosis, verify 1 / decode 32 attribution

Evidence commit: this report's commit on branch `worktree-agent-a39812d9e0cf11e26` (after
fa7200c). Run rows: `docs/runs-decode-layer.md`, 2026-10-30, jobs c2ba945496ab47b4,
86ced49f092a48d8, 50c4c1328ab742f4, 6372154fcd7a4fa9, 3e2aa83358dd4f60, 3fbb3ad027e84c6b,
f5ee8632e1f64bf2, 720079090ee44cc3. Public file pin: vjeux/okc-2026 blob 936fefff, SHA256
cf4a75ee6e4a5cd9e33b82b5347c47abcdc7b81d80dc6a8b5262dea7388f849c. Answers items 1-4 of the
[post-reset steering](2026-10-29-post-reset-release-steering.md).

## 1. Release state (measured)

The board entry is scored run **3fbb3ad027e84c6b, 2.014x** (1.984-2.045), file SHA256
`b9c23bda...`, byte-identical to `muse_decode_layer/submission.py` at 2cee446 on this branch.
It is the 1.982x release (`1355d39f`, now archived at
`muse_decode_layer/release/edd71d3152a946fe.py`) put through the two inliners, adding MoE mode 47,
LATE_ARRIVE and the o16 / unp32 geometry, with SK_PDL, ONE_TRIG and the M1 host code absent.
Merging this branch makes main's layer file the scored file again. The M1 experiment stays at
`git show 6377b5a:muse_decode_layer/submission.py`. Full manifest with policies:
[release-manifest.md](../release-manifest.md).

Validation before the first check: inliner output byte-identical on a second run; against the
release only `_MOE1_SRC`, `_MOE1_VARIANTS`, `_SKP_SRC`, `_SKP_CONFIGS` differ; SASS opcode counts
(CUDA 12.8.1, sm_90a) identical on the nine shared MoE kernels and on 35 of 39 projection
instances, the four LATE_ARRIVE instances differing only in instruction totals; no new local
memory.

Readings of this file: 2.006, 1.991, 2.003, 1.974, 1.987 (checks) and 2.014 (scored); the
release's six check readings averaged 1.982.

## 2. Isolated attribution (measured, one paired check)

720079090ee44cc3 compared the scored file with the same file holding mode 47 out of the MoE
picker (`MOE1_HELD`), so both arms share every other compiled kernel and pick their own
policies at prepare time. Mode 47 is worth about 4-5 us at verify 16 (128 tokens), 2-5 us at
verify 32 and 1.3-2.6 us at decode 32. The decode 32 figure is inside the spread between arms.
The projection package's share, by difference from 50c4c1328ab742f4 (release against the full
package), is about 1-2 us at decode 32 and under 1 us elsewhere. That is cross-bench and only
indicative. A bench artifact matters for every inc-arm comparison: the second arm runs about
5 us slow at decode 1 whatever it contains, so decode 1 ratios from inc benches are not evidence.

## 3. M1 (measured; still parked)

- With repeated inputs, M1 off and M1 on were bitwise equal at verify 1 full in every trial
  (6372154fcd7a4fa9).
- The M1-off layer breaks the stress's own 1e-4 bound. Eager calls at 32x1 full differ by up to
  6.7e-4, and at 16x1 full by up to 7.8e-4 (3e2aa83358dd4f60). Two MoE replay self-tests dropped
  unused variants at 1.2e-4 and 1.4e-4 with M1 not involved.
- With fresh inputs each replay, checked on the device against that input's M1-off eager result,
  M1 off goes over 1e-4 on 217-459 of 2000 replays at verify 1 full. M1 on goes over on 288-307.
  The maximum is 1.2e-3 in both arms. No replay in either arm exceeds 1e-2, the size a read of
  the previous replay's rows would produce.

Conclusion: the e2d1d81447cc48d9 failure fits the layer's own run-to-run variation (the
merge order of FP32 atomics). This diagnostic run shows no stale-read signature at the M1
boundaries. Not done: localizing the first divergent intermediate with a retained-output ring,
and the written proxy-ordering argument for TMA reads after `griddepcontrol.wait`. The 1e-4
bound is unchanged. M1's timing gate needs a validation policy that the M1-off layer can pass,
for example the fresh-input stress with an order-1 discriminator and matching on/off
distributions. That **needs the user's explicit approval**.

Risk in the shipped code: the MoE replay self-tests use the same 1e-4 replay bound and can drop
a variant on atomics alone. They dropped unused variants twice today. A dropped sk8w3s would
cost the small-T MoE pick in a scored run.

## 4. Verify 1 / decode 32 gap (measured whole-step, profiled attribution)

Whole-step paired deltas, public minus ours in us, over two held-out input sets: 86ced49f092a48d8
and f5ee8632e1f64bf2. Each set covers both routings, with 16 paired samples per shape. Nulls are
within 1.2 us on every shape.

| shape | set 1 | set 2 |
|---|---|---|
| v1 sliding | -8.7 | -10.2 |
| v1 full | -10.5 | -10.8 |
| d32 sliding | -15.7 | -12.2 |
| d32 full | -10.8 | -15.9 |

The attribution comes from f5ee8632e1f64bf2's profiled median replay. Each kernel's exposed
time is how far it moves the latest end, so the exposed times sum to the span. Gaps where no
kernel runs are charged to the boundary they precede. These are profiler timings, not decision
metrics.

| boundary or kernel (ours - public, exposed us) | v1 full | d32 sliding | d32 full | d16 sliding |
|---|---|---|---|---|
| attention -> o projection | +2.8 | +1.7 | +2.4 | +3.0 |
| attention boundary -> front (pooler, router, route, pre-norm) | +4.0 | +1.7 | +4.5 | +5.5 |
| MoE kernel | +2.2 | +9.8 | +3.9 | +0.5 |
| MoE -> unpooler | +1.7 | +1.0 | +2.2 | +0.1 |
| qkv + attention | +1.0 | +3.6 | -0.8 | +2.1 |
| final boundary | -0.7 | +0.4 | +0.3 | -0.3 |
| span | +11.1 | +18.4 | +12.6 | +10.8 |

Mechanism, read from the public source. Hypothesis for the causal part: at T <= 32 every public
kernel is a PDL secondary. The sk2 projections issue two weight stages before
`griddepcontrol.wait`. Attention and sk2 trigger their dependents after their main loops, and the
Triton glue triggers at entry. In the timelines, the public o projection starts 2.6 us before
attention ends, with 4.7 us exposed of a 7.4 us lifetime. The pooler shows 17.4 of 19.6, and the
unpooler at decode 32 shows 15.8 of 20.9. Our chain launches these three projections plain and
shows 0.1-1.2 us gaps at eight boundaries (4.2 us at v1 full). The public MoE waits at entry, so
its PDL buys launch latency only. Its 4.8 us early start at v1 full is waiting, not work.

Causal hypothesis with budgets (profiled; to be confirmed on whole-step time):

1. The two sk weight-prologue boundaries M1 targets (attention -> o, boundary -> front):
   **about 7-8.5 us at v1 full, d16 sliding and d32 full; 3.4 at d32 sliding**. M1 builds exactly these and has
   never been timed. This is the largest layer-owned item at verify 1.
2. MoE -> unpooler, the same mechanism on the slice / unp projections: 1-2 us.
3. MoE work at 32 tokens: **about 10-14 us at d32 sliding** (mode 47 or 37 at about 102 us
   exposed against 88). This belongs to the MoE owner and is the largest single item at
   decode 32.
4. Inter-kernel launch gaps: up to 4.2 us at v1 full. These are profiler gaps, possibly
   inflated by instrumentation. The earlier blanket-PDL test, "PDL across our own chain" in
   ruled-out.md, found no whole-step gain from early launches alone, so do not count this
   budget until a whole-step test shows it.

## Priorities and next actions

1. MoE owner: MoE work at 32 tokens (d32 sliding, about 10-14 us exposed). At 128 / 256 tokens
   86ced49f092a48d8 measured +19 / +26 us of kernel lifetime against the public fused MoE.
2. Layer: time M1's two boundaries once the validation policy is decided. Cheapest decisive
   test: M1's own `_m1_chain` and whole-step tuner at T = 8 / 16 / 32. Stopping rule: under
   1 us saved at 8 tokens on whole-step paired time closes it.
3. If M1 times well, extend the same prologue to the unpooler (slice8 / unp16 / unp32).

## Gaps

- The first divergent intermediate for the run-to-run variation is not localized.
- The PTX proxy argument for TMA reads after `griddepcontrol.wait` is still owed.
- The exposed-time attribution comes from one profiled replay per shape.
- The P share comes from cross-bench differences.
- Routing agreement and job sizes per arm were not printed in the tail. All 72 checks per bench
  passed, which includes routing acceptance.
