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

## Step 2: candidates (built, bitwise, A/B running 12:38 PT)

`tools/dl_ly_b2_patch.py --mode g32|sp` (base pinned to a1c48810):

1. ly_b2g32 (`muse_decode_layer/cand/ly_b2g32.py`, SHA256
   c5d276b13adcdd1d2b649efeaf6ece0a725d5104c5fddd3279f34c324567fd86): b1 at T = 32 on the PDL8 graft glue b1
   (8-CTA clusters of 128 threads, stream and constants before the wait, as b1 at T = 16 runs); glue host row
   limit 16 -> 32 in `_GRAFT_GLUE_CUDA` b1(). Bitwise equal to the Gluon b1 at T = 32 (8 seeds x 4 scales,
   NewStream and Out, `tools/ly_b2_bitwise.py`, `profiles/box/rbst/bitwise.log`).
2. ly_b2sp (`muse_decode_layer/cand/ly_b2sp.py`, SHA256
   c97fe37063504a951ad304ce273392302e273308d1070b15899f1cd23944f4b5): b2 (POST) loads the Stream row s1 before
   gdc_wait (s1 is b1's output; b1 completed before the MoE's non-programmatic launch). Moving the load changed
   LLVM's FMA contraction (first build: 15% of NewStream elements differ, `bitwise_first_sp_contraction.log`), so
   ns is spelled out as the base contracts it: `where(c < 4096, fma(a, s, beta*b), fma(beta, b, a*s))`; now
   bitwise equal at T = 16 / 32. SASS (POST): stream LDG.E.128.STRONG.GPU at 0x190 / 0x1a0 before ACQBULK 0x1b0
   (base: 0x6d0 / 0x710 after the first BAR.SYNC), 51 registers (base 44); `profiles/box/rbst/sass_*.txt`.

Gates: five PDL8H guards pass for both; check PASS 12/12, startup OK, control's CG_GATE line.

## Result (12:55 PT): no candidate, lane closed

Box A/B vs ly_stk3, two rounds, 24 pairs: ly_b2sp decode 16 sliding +0.09 (10/24), decode 32 sliding +0.25 (10/24),
null (kernel timeline: b2 -0.16 to -0.23 us); ly_b2g32 decode 32 sliding / full +1.04 / +1.04 (0/24, 2/24), loss.
Run log "Decode 16 / 32 row boundaries (box)" in docs/runs-decode-layer.md; two ruled-out rows. Nothing registered.
