# layer-b2 lane checkpoint (decode 16 / 32 row boundaries)

Updated 2026-11-04 12:35 PT. Base: muse_decode_layer/cand/ly_stk3.py, SHA256
a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6 (judge-valid 4a2bf53969b44e2f, 10,109.9 us).

## Step 1: stamps (done)

Instrument: `tools/dl_ly_rbst_patch.py` writes `muse_decode_layer/cand/ly_rbst.py` (SHA256
801adafc717a8a6990f9cc4a1f34c5bfc91bb9cc6bc9a18c6aad8e5170cd8639; never a candidate). Harness
`tools/ly_box_rbst.py` (plain + stamped graph per cell, 20 fresh inputs, cold L2). Box log `.build/rbst/st1.log`,
copied to `muse_decode_layer/profiles/box/rbst/st1.log` and `st1_summary.txt`.

Medians over 20 inputs, us after the producer's last CTA end stamp (o for b1, unpooler for b2); steps are
per program from its own wait return (thread 0 stamps; a reduction step includes every thread's loads landing).

| item | b1 d16 (glue, 128 CTAs) | b2 d16 (Gluon, 16) | b1 d32 (Gluon, 32) | b2 d32 (Gluon, 32) |
|---|---:|---:|---:|---:|
| first / last program entry | -1.15 / -0.74 | -1.07 / -0.99 | -1.31 / -0.14 | -1.39 / -0.96 |
| pre-wait loads landed (last) | 0.19 (sync) | -0.32 | **+0.38** | -0.16 |
| wait return (last) | 0.45 | 0.48 | 0.51 | 0.48 |
| branch landed (thread 0) | 0.03 | 0.08 | **0.42** | 0.06 |
| first reduction | 0.90 (cluster) | 0.54 | 0.26 | 0.58 |
| stream landed | (pre-wait) | 0.35 | 0.29 | 0.32 |
| NewStream store issued | - | 0.22 | 0.51 | 0.26 |
| second reduction | 0.74 (cluster) | 0.48 | 0.50 | 0.45 |
| Out store issued | 0.06 | 0.13 | 0.13 | 0.13 |
| wait to end, p50 / max | 1.73 / 1.87 | 1.78 / 1.86 | 2.15 / 2.51 | 1.78 / 1.92 |
| last program end | 2.30 | 2.30 | 3.04 | 2.37 |

Reading: (1) every boundary waits 0.45 to 0.51 us after its producer's last store for griddepcontrol.wait;
(2) b1 at T = 32 enters late (o's CTAs hold the SMs; 1024-thread programs cannot co-reside), so its 80 KB of
pre-wait constants land 0.38 us after the wait and the branch load queues behind them (0.42 vs 0.06 in b2);
(3) the glue's DSMEM cluster reductions cost 0.9 / 0.74 us, more than the Gluon CTA reductions (0.5), so more
programs per row with a cluster reduction is not supported; (4) b2's stream (s1, written by b1, which completed
before the non-programmatic MoE launched) is loaded after the first reduction, while b2's programs idle about
0.6 us at the wait with their constants landed.

## Step 2: candidates (next)

1. ly_b1g32: b1 at T = 32 on the PDL8 graft glue b1 (8-CTA clusters of 128 threads, stream and constants before
   the wait, as b1 at T = 16 already runs). Needs bitwise check against the Gluon b1 at T = 32.
2. ly_b2sp: b2 Gluon loads the Stream row (s1) before gdc_wait (legal: not the unpooler's output; b1 completed
   before the MoE's non-programmatic launch, the ly_finb1 / b2pre argument).

Next action: build both patches, bitwise check, gate + two-round A/B vs ly_stk3.
