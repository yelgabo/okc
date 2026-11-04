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

## Sweep 2: single-T confirmation, 96 runs, both layers (pcx2.sh, arms_pcx2.{log,json}, all checks PASS)

| arm | verify 1 sl | verify 1 full | decode 16 sl | decode 16 full | decode 32 sl | decode 32 full |
|---|---|---|---|---|---|---|
| o3_8 | -0.59 (63) | +0.14 (45) | +0.06 (45) | +0.26 (38) | -0.18 (52) | -0.37 (54) |
| qkv1_16 | -0.18 (53) | +0.29 (38) | +0.06 (44) | -0.32 (59) | +0.13 (44) | +0.08 (46) |
| pool3_16 | -0.26 (55) | -0.22 (56) | +0.08 (45) | +0.03 (47) | +0.51 (38) | -0.14 (52) |
| router1_8 | -0.06 (49) | +0.00 (46) | +0.42 (32) | -0.13 (50) | +0.50 (39) | -0.03 (50) |
| qkv2_32 | -0.14 (51) | -0.08 (49) | +0.13 (43) | -0.16 (52) | +0.51 (36) | +0.32 (36) |
| o2_32 | -0.18 (54) | -0.02 (48) | +0.34 (38) | -0.35 (58) | -0.27 (53) | -0.75 (60) |
| pool2_32 | -0.16 (52) | +0.13 (39) | -0.11 (49) | -0.21 (55) | -0.05 (49) | -0.59 (60) |
| base2 | -0.19 (55) | +0.29 (34) | +0.32 (36) | +0.06 (43) | +0.18 (47) | +0.19 (44) |

qkv1_16 at decode 16 sliding did not repeat (+0.06). o3_8 at verify 1 sliding repeated (-0.59, 63/96), but only about
0.4 us past its control. o2_32 at decode 32 full -0.75 (60/96).

## Paired A/B (tools/rev_ab.sh: gate, then tools/ly_prtl_ab.sh two rounds against ly_stk3.py)

ly_rvo38.py (ly_stk3 + `PCX_CFG[("o", 8)] = (3, 0, 0, 0)` after line 10194), SHA256
2f00aeb84e8e506b5fc68524b9e71a9f658411a26756a901fa7fe6ff19387b9e. Gate: five PDL8H guards pass, chk PASS. Pooled 24
pairs: verify 1 sliding -0.12 (15/24), verify 1 full +0.61 (1/24), decode 1 sliding -0.22 (20/24, T = 1 untouched),
verify 32 sliding +1.51 (1/24, untouched). Rejected. The untouched cells show the process-level pairs count carries
systematic swings of about 1.5 us. Box dirs /root/okc/.build/prtl/rvo38{,_r2}.

ly_rvo232.py (`PCX_CFG[("o", 32)] = (2, 0, 0, 0)`), SHA256 069a1a14c15a139b9331fdf51f5326761cbf6aa429b79cd28da2ecae7ab6ae72.
Gate pass, chk 12/12 PASS. Pooled 24 pairs: decode 32 full -0.81 (21/24), decode 32 sliding -0.03 (11/24), verify 1
sliding +0.39 (2/24, untouched). Misses the bar (no sliding cell). Tables: profiles/box/rev/ab_rvo38.txt, ab_rvo232.txt.

## Sweep 3: finish and tile shapes at T = 1 / 8 (fin.sh, arms_fin.{log,json}, 48 runs, all checks PASS)

Sliding: decode 1 best f1nw4 -0.26 (29/48), control -0.19; verify 1 best t1p2 -0.08 (26/48), control +0.34. Flat.

## State at 13:30 PT

No candidate clears the bar; no ledger line written. Run log "Per-T knob sweep on ly_stk3 (box)" and three ruled-out
rows written.

## Next (for a relaunched lane)

Not yet swept: OKC_A16SM reservation at decode 1 / verify 1 (compile-time define, line 171, needs files),
CG_LAYER_THREADS for the T = 1 slice (1024, 4096) at 64 / 256 (consumed in prepare at line 18045, needs files), a
T = 1 finish smem reservation (`_v1fs_reserve` filters on the T = 8 warp count; needs a file), MoE TINY defines
(lines 6935-6941). For in-process screening, ly_arms "glob" arms only reach values read at launch.
