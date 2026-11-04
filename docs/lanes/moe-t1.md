# moe-t1 lane checkpoint (2026-11-04)

Base: muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6.

## Candidate 1: ly_x1e (decode 1 TINY MoE: phase A x registers loaded right after the entry griddepcontrol.wait)

- File: muse_decode_layer/cand/ly_x1e.py, SHA256 c385fe76677295c38ea21337fe8466963e00676441ccf98a824fece579edf17a,
  written by tools/dl_ly_x1e_patch.py (base pinned). Gated by -DOKC_X1E=1 in MOE_TINY_DEFINES only.
- Legality: xm is written by the route kernel (`_graft_route` / `_route` / `_sl_tile_route_cg` all take xm as an
  output), the MoE's immediate stream predecessor. The load sits after the MoE's entry griddepcontrol.wait, the
  same point where the base's own prefetch.global.L2 of x already reads it.
- SASS (box, nvcc 13.0, tools/ly_micro_sass.py): TINY moe_fused has 145 registers, local 0 and stack 0 in both the
  base and the candidate (5197 -> 5296 instructions); TINY2 has 141 registers and local 0.
- Stamps (instrument tools/dl_ly_mtl3x_patch.py on pmt copies; tools/ly_box_pmt.py --raw 20, 20 inputs, decode 1;
  summary muse_decode_layer/profiles/box/x1e/st_summary_x1e.txt). Values are us from the first MoE CTA's entry,
  mean CTA, sliding / full:

  | stamp | stk3 | x1e |
  |---|---|---|
  | generation stored | 0.60 / 0.58 | 0.75 / 0.71 |
  | route() done | 2.11 / 2.09 | 2.21 / 2.21 |
  | WS_JOIN passed | 2.30 / 2.29 | 2.41 / 2.41 |
  | first job's x consumed | 2.90 / 2.89 | 2.90 / 2.91 |
  | first ring slot full (thread 0) | 3.03 / 3.01 | 3.03 / 3.02 |
  | phase A end | 12.70 / 12.50 | 13.07 / 12.46 |

  Reading: with x already in registers, the first slot still returns at 3.03 us, so the first weight unit (FAST1
  copies issued at about 1.1 us) gates the mean CTA's ramp and the x load is off the critical path. The entry loads
  delay the prologue by 0.1 to 0.15 us. The helper's slot-0 stamp (slot 3) reads 4.1 to 4.2 us, later than thread
  0's slot; this is unexplained (parity window or helper scheduling) and is not used.
- Box run: tools/ly_x1e_box.sh (stamps, gate, chk, two A/B rounds against ly_stk3). Log
  /root/okc/.build/x1e/x1e_run.log, A/B dirs /root/okc/.build/prtl/x1e and x1e_r2, table
  /root/okc/.build/prtl/x1e_table_both_rounds.txt.

## Next action

Read the A/B table when the box run ends; the stamps predict a null or a loss. Item (2), per-warp handoff stamps,
has not started. The hypothesis is that the 0.55 to 0.87 us is the helper's epilogue queue lag (it reaches the last
phase A unit's partials only after finishing the earlier units' epilogues), not warp skew.
