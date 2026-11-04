# attention-d16 lane checkpoint

Updated 2026-11-04 12:36 PT (Mac clock). Branch worktree-agent-a5dc59bb03b3b9eec.

- Base: muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6
  (attention source identical to ly_a16o2). Its judge check 4a2bf53969b44e2f is the lead's.
- Base stamps (box, 20 inputs, us after qkv end, decode 16 / 32): loop end 4.69 / 7.26, reds 5.17 / 7.66, ticket
  6.72 / 8.59, attention end 7.95 / 9.57, o wait 8.51 / 10.10, o end 11.87 / 14.21.
- Ruled out at the stamp level (rows in docs/ruled-out.md, section "Decode 16 / 32 sliding attention merge (box)" in
  docs/runs-decode-layer.md): ly_aopf a7bc691a, ly_aopfl 0a58b5f6 (o stream L2 prefetch from the merge tail),
  ly_a16sp b72fc51a (8 / 4 splits), ly_a16cl e3bc81b5 and ly_a16cl2 21da0e02 (MODE 2 cluster merge).
- Round 3 (relaunch, 12:36 PT, branch worktree-agent-af551e13c3c5956fd): candidate muse_decode_layer/cand/ly_a16nc2.py
  (SHA256 dbadd5302873cea3385f497b6d24837604bc67e2118e4e1dae1f87bf456c24bb), stamp build ly_a16nc2_st.py
  (f5a5dcc44452ee22362abd0d4cc3820b0294ef641a576a9d74cd279097eb0de7, tools/dl_ly_a16_patch.py). Base stamp build on
  the box ly_stk3_st.py 95f41970. Stamp run launched 20:36:49 UTC on the box:
  `ARMS="ly_stk3_st ly_a16nc2_st" tools/ly_a16r2_st.sh` -> /root/okc/.build/a16/st_<arm>.log, driver log
  .build/a16/r3_st.log (queued behind other lanes on gpu.lock).
- Stamps (box 20:36-20:43 UTC, 20 inputs, medians, us after qkv end; logs muse_decode_layer/profiles/box/a16r3/):
  decode 16 base / nc2: reds 5.34 / 5.25, ticket 6.80 / 6.56 (reds->ticket 1.46 / 1.31), attention end 8.08 / 7.46
  (finalize 1.28 / 0.90), o wait 8.59 / 8.10, o end 11.94 / 11.41. Decode 32: o end 14.21 / 14.24.
  Literal gate (ticket -0.4) fails at -0.14, but ticket+finalize nets -0.52 us (finalize shrank instead of growing),
  above the 0.3 bar, so the A/B runs as the decisive test.
- A/B launched 20:44:08 UTC: `CAND=ly_a16nc2.py TAG=a16nc2 CTL=ly_stk3.py tools/ly_a16_ab.sh` -> box
  .build/a16/a16nc2_run.log, gate_a16nc2.log, .build/prtl/a16nc2{,_r2}/.
- Next action: read the gate (PASS, five PDL8H guards) and both A/B rounds; register only on a decode 16 sliding win
  with 20+/24 pairs and no other load losing beyond noise.
