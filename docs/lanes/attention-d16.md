# attention-d16 lane checkpoint

Updated 2026-11-04 12:36 PT (Mac clock). Branch worktree-agent-a5dc59bb03b3b9eec.

- Base: muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6
  (attention source identical to ly_a16o2). Its judge check 4a2bf53969b44e2f is the lead's.
- Base stamps (box, 20 inputs, us after qkv end, decode 16 / 32): loop end 4.69 / 7.26, reds 5.17 / 7.66, ticket
  6.72 / 8.59, attention end 7.95 / 9.57, o wait 8.51 / 10.10, o end 11.87 / 14.21.
- Ruled out at the stamp level (rows in docs/ruled-out.md, section "Decode 16 / 32 sliding attention merge (box)" in
  docs/runs-decode-layer.md): ly_aopf a7bc691a, ly_aopfl 0a58b5f6 (o stream L2 prefetch from the merge tail),
  ly_a16sp b72fc51a (8 / 4 splits), ly_a16cl e3bc81b5 and ly_a16cl2 21da0e02 (MODE 2 cluster merge).
- Built, untested: muse_decode_layer/cand/ly_a16nc2.py, SHA256
  dbadd5302873cea3385f497b6d24837604bc67e2118e4e1dae1f87bf456c24bb (two workspace copies, warmed finalize; expected
  about -0.3 to -0.4 us at decode 16, below the 0.8 us bar).
- Box paths: /root/okc/.build/a16/st_<arm>.log; logs copied to muse_decode_layer/profiles/box/a16r2/. Box was idle
  after 20:33 UTC; nothing of this lane is running.
- Next action: stamp ly_a16nc2 (`python3 tools/dl_ly_a16_patch.py --base muse_decode_layer/cand/ly_a16nc2.py --out
  muse_decode_layer/cand/ly_a16nc2_st.py`, rsync both, `ARMS="ly_a16nc2_st" tools/ly_a16r2_st.sh`); if reds -> ticket
  drops by at least 0.4 us with the finalize under +0.15, run `CAND=ly_a16nc2.py TAG=a16nc2 CTL=ly_stk3.py
  tools/ly_a16_ab.sh` (gate, chk, two paired rounds).
