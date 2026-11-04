# Box revival lane, 2026-11-03

Experiments the judge quota cut short, re-run on the rented H200 (docs/box.md) against the public layer
file `muse_decode_layer/cand/g1b4_rs.py` (SHA256 01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07,
verified on the box). Every number here is box net us (median cold replay minus the fixed replay cost),
candidate against control in one session on the same inputs. No number here predicts a judge score.

Settings for every run: case seeds from `.build/judge/g1b4_pdl8d.f931abf568544018.json`, router seed
106218864389449120, per-input local_start, `--runs 60`, box_run's reference check on every arm and cell.
A load's time is the weighted sum over its two shapes (46 x sliding + 16 x full, TASK.md); the
six-load sum adds the six load times.

## Finalists

None yet.

## Tools

- `tools/rv_ab.py`: in-process paired A/B. Every arm is prepared in one process; the arms' replays
  rotate order per repetition on the same fresh inputs with a cold L2 each. It reuses box_run.py's
  loader, timer, router and case seeds unchanged. It prints each arm's per-cell median, the paired
  median difference against the first arm, load times and the six-load sum. It records each arm's
  PDL8 enable set (`state["pdl8_on"]`) and re-prepares an arm (`--pdl8-tries`) until its set equals
  the control's, which in turn is held to `--pdl8-want`.
- `tools/rv_bg.sh`: rv_ab.py in the background under the same `gpu.lock` as box_bg.sh.
- `tools/rv_knob.py`: one-knob variants of g1b4_rs.py (`rv_k_<name>.py`), each edit an exact string
  that must occur once.

## Extension cache

torch keys extension builds by name only, and the release family (g1b4_stage_release, g1b4_unp_release,
g1b4_cg*_release, g1b4_f3) and the g1b4_rs family build `muse_att_v298w1pf_90a` and
`dl_okc_bf16proj_v86y4_90a` from different sources, so a switch between families rebuilds both under the
GPU lock (prepare 50 to 85 s instead of 14 s). Runs before 15:20 PT used the default cache for both
families, mixed in one process for the mechanism runs (every arm passed its reference check); runs from
then on give release-family jobs `TORCH_EXTENSIONS_DIR=/root/.cache/te_release` and g1b4_rs derivatives
the default. Each row names the cache it used where it is not the default.

## The PDL8 gate is part of the control's variance

g1b4_rs's in-prepare PDL8 gate picks its enable set per prepare, and the same bytes come up with
different sets. Of eight prepares of these bytes this session, six chose {1: on, 8: off, 16: on, 32: on}
(the set of judge check f8a55f1043804158) and two chose {1: on, 8: on, 16: off, 32: off}.
The set moves whole cells: in-process A/A `rv_aa1` (g1b4_rs against a byte copy, the two arms on
different sets) read decode 1 -0.08 / -0.38, decode 16 +7.49 / +6.99, decode 32 +6.54 / +7.12, verify 1
-6.05 / -7.98, verify 16 and 32 within 0.7 us (paired, ctl2 - ctl, sliding / full). Decode 1 (T = 1
on in every set) does not move. So every comparison below is between arms on the same set, and the
spread table uses controls on one set only. The enable set is listed with each run.

## Control-vs-control spread

Three separate box_run.py processes of g1b4_rs (`rv_ctl_a`, `rv_ctl_b`, `rv_ctl_c`), all six loads, all
three on the set {1: on, 8: off, 16: on, 32: on}. Spread is max minus min of the three.

| cell | rv_ctl_a | rv_ctl_b | rv_ctl_c | spread |
|---|---:|---:|---:|---:|
| decode 1 sliding | 66.18 | 66.29 | 66.22 | 0.11 |
| decode 1 full | 71.95 | 72.08 | 72.03 | 0.13 |
| decode 16 sliding | 130.72 | 130.53 | 130.38 | 0.34 |
| decode 16 full | 202.90 | 203.46 | 204.46 | 1.57 |
| decode 32 sliding | 146.99 | 147.28 | 147.01 | 0.29 |
| decode 32 full | 298.16 | 298.62 | 298.69 | 0.53 |
| verify 1 sliding | 111.79 | 112.30 | 112.50 | 0.70 |
| verify 1 full | 120.11 | 120.67 | 120.80 | 0.69 |
| verify 16 sliding | 244.10 | 244.91 | 243.76 | 1.15 |
| verify 16 full | 328.77 | 327.68 | 328.16 | 1.09 |
| verify 32 sliding | 296.26 | 298.34 | 298.13 | 2.08 |
| verify 32 full | 462.00 | 462.54 | 461.63 | 0.91 |

| load (46 s + 16 f) | rv_ctl_a | rv_ctl_b | rv_ctl_c | spread | bar (2 x spread) |
|---|---:|---:|---:|---:|---:|
| decode 1 | 4195.3 | 4202.5 | 4198.8 | 7.2 | 14.4 |
| decode 16 | 9259.5 | 9259.6 | 9269.1 | 9.6 | |
| decode 32 | 11532.2 | 11552.9 | 11541.4 | 20.7 | |
| verify 1 | 7064.2 | 7096.7 | 7107.6 | 43.4 | 86.8 |
| verify 16 | 16488.7 | 16508.8 | 16463.5 | 45.3 | |
| verify 32 | 21019.8 | 21124.2 | 21100.0 | 104.4 | |
| six-load sum | 69559.7 | 69744.7 | 69680.4 | 185.0 | 370.0 |

In weighted units: the decode 1 bar is about 0.23 us per cell, verify 1 about 1.4 us per cell, the
six-load sum about 6 us per cell over all twelve. In-process nulls (the control's byte copy `rv_ctl2.py`
as a second arm, same set) are listed with each run; so far they are within these bars.

## Runs

| experiment | box log tag | arms (set) | cell numbers, box net us | decision |
|---|---|---|---|---|
| UNP mechanism: g1b4_unp_release.py 3d7f252f (unpooler slice + finish fused into the MoE tail; four mma chains, 16-byte x chunks, `UNP_LOSS_US = 3.0`) against its base g1b4_stage_release.py ba9e327c | rv_unp_mech | release has no PDL8 gate | decode 1: ctl 77.34 / 83.01, cand 78.62 / 84.42, paired +1.18 / +1.62; verify 1: ctl 112.24 / 121.58, cand 112.40 / 121.02, paired +0.80 / +0.19. In-prepare UNP_GATE ok at T = 1 and 8 (tails hot sep 74.1 / fused 75.0 us at T = 1) | Mechanism loses at decode 1 and is flat at verify 1, as the judge's third check read it (c77b95102baf4232, decode 1 80.67 / 85.35) |
| UNP port: rv_unp.py 1072c330 (tools/dl_unp_patch.py on g1b4_rs, all anchors apply) | rv_unp_port | ctl, ctl2 {1, 16, 32 on}; cand {1, 8, 16, 32 on}: mismatched | decode 1: ctl 67.44 / 73.74, ctl2 67.25 / 74.14, cand 68.10 / 74.29; paired d_ctl2 -0.27 / +0.35, d_cand +0.56 / +0.61. The verify 1 -7.26 / -8.50 came from the T = 8 PDL set, not from UNP | superseded by rv_unp_port2 |
| UNP port, matched set | rv_unp_port2 | all three {1, 16, 32 on} (cand on its second prepare) | decode 1: ctl 67.50 / 73.95, ctl2 67.30 / 73.89, cand 67.89 / 74.05; paired d_ctl2 -0.29 / -0.27, d_cand +0.10 / -0.13. verify 1: ctl 112.16 / 119.71, ctl2 112.14 / 120.16, cand 112.03 / 120.02; paired d_ctl2 +0.02 / +0.59, d_cand +0.19 / -0.13. Loads: decode 1 +19.2 (null -10.6), verify 1 -1.0 (null +6.4) | No effect inside g1b4_rs. Closed |
| cgemv (C2 12-bit GEMV at T = 1) and the cgf front (pooler + FP32 router on a side stream), on the release and on the PDL base | rv_cgf_mech | ctl g1b4_pdl8c.py 6539d8d6; g1b4_pdl8 6c94513b; g1b4_pdl8cf ff3483da; release ba9e327c; g1b4_cg_release 440397ab; g1b4_cgf_release 11ccb8e9 (one decode-1-only run, sets differ but T = 1 is on in every set) | decode 1 sliding / full: release 77.14 / 83.54, cg_release 74.06 / 79.76, cgf_release 73.34 / 79.81; pdl8 68.72 / 74.98, pdl8c 71.42 / 76.61, pdl8cf 73.73 / 78.94 | cgemv wins about 3.1 / 3.8 us on the release and loses 2.7 / 1.6 on the PDL base (pdl8c against pdl8); the front adds 2.3 / 2.3 more on the PDL base. Same signs as the judge (eba613c023414ca8). Not ported; the PDL-secondary form (cgfe) is queued |
| T = 1 knobs, one at a time on g1b4_rs: GLE_RC {1: 4} / {1: 16}; SL_CLUSTER_CONFIG[1] (4, 256) / (8, 128) / (4, 512) | rv_k1a | ctl {1, 8, 16, 32 on}; others mixed (T = 1 on in all) | decode 1 paired against ctl (sliding / full): ctl2 +0.43 / -0.78; gle4 -0.05 / -0.16; gle16 +0.29 / -0.19; cl4x256 +0.83 / -0.10; cl8x128 +0.50 / -0.58; cl4x512 +0.23 / -0.64. Loads: ctl2 -16.0, gle4 -22.4, gle16 -3.6, cl4x256 +20.2, cl8x128 +6.6, cl4x512 -0.2 | All inside this run's own null (ctl2 -16.0). gle4 repeated in rv_sk1 |
| T = 1 knobs: SL_TILE_CONFIG[1] (8, 1, True) / (4, 2, True); attention full T = 1 n_splits 264 / 198 / 132 (file: 252) | rv_k1b | as above | paired (sliding / full): ctl2 +0.11 / -0.03; tile8w +0.24 / -0.16; tile2p +0.64 / +0.06; att264 +0.03 / -0.22; att198 +0.00 / +1.58; att132 +0.50 / +3.95. Loads: ctl2 -6.0, tile8w +0.3, tile2p +14.1, att264 -13.7, att198 +2.9, att132 +66.3 | 198 and 132 splits lose (the knob is live, 252 is at or near the best); others inside the null. att264 repeated in rv_sk1 |
| T = 8 knobs on g1b4_rs: SL_TILE_CONFIG[8] (8, 4, True) / (4, 2, True) / (4, 8, True); SL_ROUTE_CONFIG[8] (8, 0, 0); SL_CLUSTER_CONFIG[8] (8, 1024) / (16, 512) | rv_k8a | ctl and every candidate {1, 16, 32 on}; ctl2 reached only {8, 32 on} (T = 1 off) after 6 prepares, so it is no null here | verify 1 paired against ctl (sliding / full): t8tile8w -0.11 / -0.03; t8tile2p -0.27 / +0.05; t8tile8p -0.38 / +0.43; t8route8w -0.24 / -0.38; t8cl8 -0.53 / -0.16; t8cl16x512 -0.08 / -0.06. Loads: -18.8, -25.9, -32.3, -11.4, -14.3, -23.3 (bar 86.8) | Every arm reads 0.2 to 0.5 us below ctl in the same direction, which points at the ctl arm rather than the knobs. None clears the bar; closed |
| MoE tiny build (T = 1) defines, one at a time: OKC_NACC 2 / 8 (file 4), OKC_WSBLK 1 / 4 (2), OKC_L2H 1 (0), OKC_WSBL 0 (2), OKC_NW 16 (8) | rv_mt1 | sets mixed, T = 1 on in all | wsbl0 and nw16 failed the file's own startup expert comparison in prepare (rel err 0.060 and 1.41) and dropped out: those builds are wrong, and the gate caught them. decode 1 paired (sliding / full): ctl2 +0.10 / -0.42; nacc2 -0.03 / +0.30; nacc8 -0.10 / +0.16; wsblk1 +0.06 / -0.42; wsblk4 -0.34 / +0.14; l2h1 -0.03 / +0.22. Loads: ctl2 +0.6, nacc2 +13.2, nacc8 -25.0, wsblk1 -14.3, wsblk4 -20.1, l2h1 -3.9 (bar 14.4) | nacc8 and wsblk4 read past the bar on medians but the paired differences are mixed in sign across the two cells; repeated with two prepares each and a combination (rv_t1rep) |
| MoE mid build (T = 8) defines: OKC_NACC 2 / 8, OKC_WSBLK 1 / 4, OKC_WSBL 0, OKC_NW 16 | rv_mm1 | ctl, ctl2, nacc2, nacc8, wsblk4 {1, 16, 32 on}; wsblk1 never reached it in 6 prepares (ended on all four on) | wsbl0 and nw16 failed the startup expert comparison (wrong builds, dropped). verify 1 paired (sliding / full): ctl2 -0.48 / +0.50; nacc2 -0.18 / +0.24; nacc8 -0.10 / -0.42; wsblk4 +0.11 / +0.10; wsblk1 -7.89 / -8.08 is its T = 8 PDL set. Loads: ctl2 -10.6, nacc2 -17.2, nacc8 -6.2, wsblk4 -28.0 (bar 86.8) | No effect, and none was possible: T = 8 runs the tiny2 build (MOE_TINY2_MAX_M = 8), so the mid build serves 9 to 16 tokens only and this sweep changed nothing at verify 1 (the arms equal the null). Re-run at decode 16 (rv_mm16); the tiny2 sweep for T = 8 is rv_m2 |
| sk2 picks at tokr 8 (T = 1 and 8): pooler -> 'pooler' / 'pooler32_r144' (file 'pooler8h_r72'); o -> 'o32h_r100' ('o8h_r88'); plus repeats of gle4 and att264 | rv_sk1 | all arms {1, 16, 32 on} (sk_pool32 and sk_o32 on their 3rd / 4th prepare) | decode 1 paired (sliding / full): ctl2 +0.22 / +0.00; pool16 +0.74 / +0.00; pool32 +0.37 / +0.11; o32 +0.94 / +0.67; gle4 +0.06 / -0.10; att264 +0.14 / +0.27. verify 1: ctl2 +0.48 / -0.21; pool16 +0.61 / -0.08; pool32 +0.14 / -0.10; o32 +0.00 / -0.29; gle4 +0.51 / -0.02; att264 -0.10 / +0.06. Loads decode 1 / verify 1: ctl2 +21.9 / +24.8, pool16 +45.7 / +11.8, pool32 +16.5 / +1.4, o32 +45.0 / +6.3, gle4 +23.9 / +23.9, att264 +35.6 / -1.8 | The file's picks hold. gle4 and att264 do not repeat their rv_k1 readings; closed |
| F3 mechanism: g1b4_f3.py 9231e381 (o projection, attention boundary and pooler/router as one cooperative launch at T <= 8, counter fix b4aa2557) against its base, the release ba9e327c | rv_f3_mech (1500 s timeout around the job because the first judge check hung at T = 8; default extension cache) | release family, no PDL8 gate | In-prepare tuner on the box: T = 1 off 68.64 / on 68.19 us, saving 0.70, wins 5 / 9 -> off; T = 8 off 121.98 / on 121.63, saving 0.48, wins 6 / 9 -> off. Timed arms (fused path off in both): decode 1 ctl 76.77 / 83.18, f3 76.22 / 82.69 (paired -0.74 / -0.54); verify 1 ctl 112.14 / 121.12, f3 111.98 / 120.11 (paired -0.26 / +0.02) | No hang. The fused launch saves 0.5 to 0.7 us on a base without PDL, and its own tuner turns it off; g1b4_rs already overlaps the three launches it replaces with PDL edges, so a hand port is not worth its risk. Closed |
| cgfe mechanism: g1b4_cgfe.py 3785f13a (the three m = 1 cgemv kernels as PDL secondaries: wait before x, x through __ldcg, trigger at the CTA's end) against its PDL base g1b4_pdl8.py 6c94513b, with pdl8c2 d07b512b and g1b4_rs as extra arms | rv_cgfe_mech | decode 1 only; sets differ, T = 1 on in all | decode 1: pdl8 68.13 / 74.82, pdl8c2 68.40 / 74.54, cgfe 66.56 / 72.90, g1b4_rs 68.16 / 73.57; paired against pdl8: c2 +0.30 / -0.43, cgfe -1.46 / -1.89, rs -3.41 / -4.69. Loads: c2 +8.2, cgfe -102.8, rs -18.5 | Mechanism wins 1.5 to 1.9 us per decode 1 cell over its own PDL base, where pdl8c2 (trigger after the weight issue) did not. Ported onto g1b4_rs: rv_cgfe_rs (tools/rv_cgfe_rs_patch.py), A/B queued (rv_cgfe_port) |
| Attention plans at T = 1 (pairs = 1): sliding override cs = 8 clusters / n_splits 16 (file: 33 splits, no cluster); full nc = 4 workspace copies (file 8) | rv_at1 | T = 1 on in all | decode 1 paired (sliding / full): ctl2 -0.11 / -0.27; a1s_cs8 +5.38 / -0.29; a1s_n16 +1.66 / -0.02; a1f_nc4 +0.50 / +0.08. Loads: ctl2 +0.5, cs8 +249.3, n16 +104.0, nc4 +30.3 | The file's plans hold; closed |
| Attention plans at T = 8: sliding n_splits 16 / cs = 8 (file 33 splits); full n_splits 264 / 198 (file 252), nc = 4 (file 8) | rv_at8 | all arms {1, 16, 32 on} except a8s_cs8 ({1, 16 on} after 6 prepares; differs at T = 32 only, which verify 1 does not run) | verify 1 paired (sliding / full): ctl2 +0.11 / +0.16; a8s_n16 +2.48 / -0.19; a8s_cs8 +5.71 / -0.61; a8f_n264 +0.42 / -0.06; a8f_n198 -0.06 / +1.42; a8f_nc4 +0.34 / +0.27. Loads: ctl2 +3.6, n16 +151.2, cs8 +282.1, n264 +20.7, n198 +39.9, nc4 +24.8 | The file's plans hold; closed |
| sk2 picks at tokr 16 / 32 (decode 16, decode 32): o 16 -> 'o32h_r100', o 32 -> 'o8h_r88', pooler 16 -> 'pooler8h_r72', pooler 32 -> 'pooler', unpooler 16 -> 'unpooler32_r144', unpooler 32 -> 'unpooler16h_r100' | rv_sk16 | all {1, 16, 32 on} | Loads decode 16 / decode 32 against ctl: ctl2 +9.7 / +31.3; o16x32 +21.8 / -13.8; o32x8 -51.9 / +8.4; p16x8 -31.5 / +2.3; p32x16 -8.8 / +65.9; u16x32 -33.6 / +33.2; u32x16 +0.6 / -27.0. Every per-cell paired difference is within 0.9 us | No knob moves the load it targets beyond the null (o32x8's -51.9 is at decode 16, which that knob does not touch); closed |
| MoE tiny2 build (the T = 8 build) defines: OKC_NACC 2 / 8 (file 4), OKC_WSBLK 2 / 4 (file 1), OKC_CPF 1 (file 0); MOE_TINY_MAX_M 8 (T = 8 on the tiny build) | rv_m2 | all {1, 16, 32 on} | tiny8 refused by the file's own native-module key check (prepare error, dropped). verify 1 paired (sliding / full): ctl2 -0.10 / -0.37; nacc2 +0.11 / -0.43; nacc8 -0.40 / +0.03; wsblk2 +0.21 / -0.42; wsblk4 -0.45 / -0.30; cpf1 -0.35 / +0.05. Loads: ctl2 -5.9, nacc2 -4.4, nacc8 -35.1, wsblk2 -22.0, wsblk4 -50.7, cpf1 -16.4 (bar 86.8) | None clears the bar. wsblk4 is the only arm below the null on both cells; kept for one combination run |
| T = 1 repeat: two prepares each of mt_nacc8, mt_wsblk4 and c1 (mt_nacc8 + mt_wsblk4 + gle4 + att264), three controls, 100 runs | rv_t1rep | T = 1 on in all | decode 1 loads against ctl: ctl2 +20.3, ctl3 +1.2; nacc8 +39.9 / -6.7; wsblk4 +19.5 / +10.3; c1 +47.1 / +42.8 | The rv_mt1 readings do not repeat; the combination is slower than the control. The T = 1 knob space around g1b4_rs is flat within the null; closed |
| MoE mid build (T = 16) defines at decode 16: OKC_NACC 2 / 8, OKC_WSBLK 1 / 4 | rv_mm16 | all {1, 16, 32 on} except mm_nacc2 (ended {1, 8 on} after 6 prepares: T = 16 PDL off, so its +7.55 is the gate) | decode 16 paired (sliding / full): ctl2 +0.06 / -0.14; nacc8 +0.37 / +0.59; wsblk1 +0.06 / +0.35; wsblk4 +0.19 / +0.51. Loads: ctl2 +6.5, nacc8 +52.4, wsblk1 +28.0, wsblk4 +17.4 | Every matched arm is slower; the file's mid build holds; closed |
| cgfe port: rv_cgfe_rs.py a65796ef (tools/rv_cgfe_rs_patch.py: g1b4_cgfe's release + cgemv prefix, g1b4_rs's PDL tail, cgfe's cgemv PDL block) | rv_cgfe_port (100 runs) | ctl, cand, cand2 {1, 16, 32 on} on their first prepare; ctl2 on its 5th | In-prepare: CG_GATE ok on all three layers (cg against sk2 4.5e-05 / 6.5e-07 / 0); CG_TUNE picked cg for the unpooler slice only (cg 7.49 / 7.52 us against sk2 7.90 / 8.00) and sk2 for qkv_gate (13.79 / 13.60 against 13.38 / 13.54) and o. decode 1: ctl 68.14 / 73.86, ctl2 67.87 / 73.68, cand 65.79 / 72.24, cand2 66.54 / 72.61; paired d_ctl2 -0.34 / -0.35, d_cand -2.00 / -1.71, d_cand2 -1.39 / -1.33. verify 1: ctl 112.62 / 121.28, cand 112.43 / 121.01, cand2 112.78 / 121.73; paired d_ctl2 +0.32 / +0.19, d_cand +0.08 / -0.27, d_cand2 +0.34 / +0.00. Loads decode 1: ctl2 -15.3, cand -134.0, cand2 -93.6 (bar 14.4); verify 1: ctl2 +28.0, cand -13.2, cand2 +14.5 | Clears the decode 1 bar on both prepares; verify 1 unchanged (cgemv runs at m = 1 only). Six-load A/B queued (rv_cgfe_six, and separate processes rv_cgfe_sa / sb) |
| cgfe port kernel table, separate process | rv_cgfe_k | {1, 16, 32 on}? (box_run) | This prepare's tuner picked cg for qkv_gate too (13.41 against 13.54): the 9.9 us sk2 qkv kernel became cgemv40<256, 2> 10.0 us and the 4.5 us unpooler slice sk2 became cgemv40<128, 2> 5.5 us, while `_sl_cluster_finish_cg` fell from 4.3 to 2.6 us (it now waits less on its PDL primary). decode 1 net 65.76 / 73.06 against the separate-process controls' 66.2 / 72.0 | The qkv_gate pick is a coin flip in the isolated tuner and the qkv cgemv may cost the full cell; the unpooler-only variant rv_k_cgfe_u (CG_LAYERS = ("unpooler",)) removes that choice. Its six-load A/B is queued (rv_cgfeu_six, rv_cgfeu_sa / sb), with the layer-subset sweep rv_cgfe_sub |
