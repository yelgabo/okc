# moe lane checkpoint (round 5, decode 16 / 32 MoE prologue)

Updated 2026-11-04 13:10 PT. Round 5 closed: no candidate clears the bar. Branch `worktree-agent-aac86ce8ff3fa9576`. Round 4's checkpoint is in git history
(commit 040d9eb4).

## Base

muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6 (MoE
device code as ly_stk2). Every A/B is paired against it.

## Candidates (tools/dl_ly_m5_patch.py mid|b4)

| file | SHA256 | change |
|---|---|---|
| muse_decode_layer/cand/ly_m5mh.py | d5fbbb083f92060f293972966135691347eec31bffb64d5359ce014a6a41b295 | MID: rb7 (round 4) plus the FAST1 warp's four topk loads issued before its count loop (-DOKC_RB7=1 -DOKC_F1HOIST=1) |
| muse_decode_layer/cand/ly_m5mh2.py | b953f364fe055e8a4fbbe9fa120d3a90cb670b596c7d865f4cc05785cc6a0e19 | MID: ly_m5mh with an empty asm register barrier after the four hoisted loads so they cannot be sunk |
| muse_decode_layer/cand/ly_m5ww.py | b05a19a2977c3f6b7a935a6e016e07384fca0302ec6174d909b89346755a9e93 | B4: dry wgs_ranges on the weights producer warp at its entry, storing nothing; the x producer's real call unchanged (-DOKC_RWARM=1) |

Stamp copies (box only): ly_m5mh_pro16.py 5ec5ab7a, ly_m5mh2_pro16.py d9ae2611, ly_m5ww_prob4.py 6162950b, ly_stk3_prob4.py 41cf47c4 (B4 base).

## Measured (box, own time in us after each CTA's entry, median of 20 inputs, decode 16 / 32 sliding cold)

MID ly_m5mh: F1 topk back 0.56 / 0.75 / 0.97 / 1.16 (still about 0.2 us apart, so the hoist did not shorten the
FAST1 count loop), F1 TMA issued 2.64, bar 4 2.89, first slot full 5.07, w0 x load back 5.33 (base 4.64 / 4.90, so
0.43 later). Gate: five PDL8H guards pass; chk rc=0.

B4 base ly_stk3: route end 2.42, xprod bar 7 2.54, xprod ranges 5.12 (wgs_ranges cold 2.58 us), consumers' WS_JOIN
(gt) 4.23, wprod after WS_JOIN 5.30, first weight issue (gt) 5.35.

MID ly_m5mh2: F1 topk back 0.57 / 0.75 / 0.97 / 1.17, F1 scan done 1.70, F1 TMA issued 2.65, bar 4 2.90: the same as
ly_m5mh, so pinning the loads changed nothing the stamps can see. Not A/B'd.

MID ly_m5mh A/B (two rounds pooled, 24 pairs, w - c): decode 16 sliding +0.21 (8/24 faster), decode 16 full +0.71
(1/24), decode 32 sliding +0.32 (9/24), decode 1 full -0.57 (21/24, thr 1.18); every cell inside. Null.

B4 ly_m5ww: xprod ranges 3.71 (base 5.12), first weight issue (gt) 4.78 (5.35), wprod after WS_JOIN 4.94 (5.30), but
gather done (gt) 4.35 (4.22), WS_JOIN (gt) 4.39 (4.23), xready released 4.71 (4.56), xprod first xready (gt) 5.78
(5.55). The weights now start 0.57 us earlier and the first x rows land 0.23 us later. Gate: five guards pass, chk rc=0.

B4 ly_m5ww A/B (pooled, 24 pairs, w - c): decode 32 sliding -0.43 (17/24), decode 32 full +0.50 (10/24), decode 16
sliding +0.08 (9/24), decode 1 full -0.68 (23/24, thr 1.82); every cell inside. Below the 0.8 us bar.

## Box

Driver tools/ly_m5_ab.sh (stamps first, then gate, chk, two paired rounds vs ly_stk3). Logs in /root/okc/.build/moe5/
(mh_run.log, ww_run.log, pro_*.log, gate_*.log) and /root/okc/.build/prtl/m5_{mh,ww}{,_r2}/.

## Next

Round 5 is closed and recorded (docs/runs-decode-layer.md "Decode 16 / 32 MoE prologue round 5 (box)", two rows in
docs/ruled-out.md, logs in muse_decode_layer/profiles/box/moe5/). With wgs_ranges warm, B4's next wait is the first
x rows (gather, xready at 5.78); at MID the FAST1 count loop's own body (0.2 us per 32 entries).
