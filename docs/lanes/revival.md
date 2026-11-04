# Revival lane checkpoint: per-T knob sweep on ly_stk3

Updated 2026-11-04 13:20 PT. Branch worktree-agent-a034ce431fea8011b.

Base: muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6.
Instrument: tools/ly_arms.py (one prepare, one graph per arm, rotated replay, reference check per arm), router seed
106218864389449120, judge .build/judge/g1b4_pdl8d.f931abf568544018.json, GPU lock /root/okc/.build/box/gpu.lock.

## Knobs skipped (already swept on a recent base, docs/ruled-out.md)

Unpooler P at T = 16 / 32 (row 769) and T = 8 P = 4 (row 731); o P = 1 at T = 1 (row 792); attention splits at
decode 1 / verify 1 / decode 16 / 32 (rows 789, 812, 820, 821); workspace copies (787, 827); pooler L2 prefetch
extension and alternatives (772, 814, 817); RT_CFG row boundary (ly_rbc); MoE ring depth OKC_NS (678); MoE two CTAs
per SM (696); verify 16 / 32 PDL edges (768).

## Sweep 1: sk2 pre-wait stages P per role at T = 1 / 8 / 16 (sliding, 48 runs, all checks PASS)

Paired median difference against base in us (negative is faster), faster pairs out of 48.

| arm | decode 1 (base 64.19) | verify 1 (base 100.78) | decode 16 (base 125.89) |
|---|---|---|---|
| qkv1 | -0.14 (26) | +0.11 (23) | -0.53 (36) |
| qkv3 | +0.26 (21) | -0.24 (31) | +0.14 (21) |
| qkv4 | +0.56 (17) | +0.43 (15) | +0.66 (12) |
| o1 | +0.11 (20) | +0.77 (17) | +0.38 (17) |
| o3 | +0.16 (21) | -0.58 (40) | +0.13 (20) |
| o4 | +0.66 (11) | -0.27 (29) | +0.66 (14) |
| pool1 | +0.66 (15) | +0.40 (17) | -0.24 (27) |
| pool3 | +0.03 (24) | +0.11 (23) | -0.40 (27) |
| pool4 | +0.98 (8) | +0.85 (12) | +0.40 (18) |
| router1 | +0.00 (23) | -0.35 (32) | -0.13 (27) |
| router4 | +0.35 (21) | +0.00 (22) | +0.11 (21) |
| base2 (control) | +0.02 (22) | -0.34 (28) | +0.03 (24) |

P = 4 loses for every role. Leads: o P = 3 at T = 8 (verify 1 -0.58, 40/48, control -0.34) and qkv P = 1 at T = 16
(decode 16 -0.53, 36/48, control +0.03). Logs: muse_decode_layer/profiles/box/rev/arms_pcx.{log,json}, script pcx.sh.

## In flight

Sweep 2 (muse_decode_layer/profiles/box/rev/pcx2.sh, box /root/okc/.build/rev/arms_pcx2.log): single-T arms o3_8,
qkv1_16, pool3_16, router1_8 and T = 32 P = 2 for qkv / o / pool, loads verify 1 / decode 16 / decode 32, both
layers, 96 runs, duplicated control.

## Next

If a lead holds at 96 runs: build ly_stk3 + one PCX_CFG line (after line 10194), gate it (reference check plus five
PDL8H guards), paired A/B with tools/ly_prtl_ab.sh two rounds against CTL=ly_stk3.py. Then sweep SL_CLUSTER_CONFIG
(line 15865) and SL_TILE_CONFIG / SL_ROUTE_CONFIG at T = 1 / 8 by glob arms. CG_LAYER_THREADS is consumed in prepare
(line 18045), so it needs a state override or a file.
