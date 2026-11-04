# moe-gb lane checkpoint (MoE grid barrier removal)

Updated 2026-11-04 13:22 PT. Status: ly_gb4 meets the bar; registered in the ledger at 13:17 PT.

## Base

muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6 (964,424 bytes).

## What MFIN is and why it is off

MFIN is the public code's per-expert completion counters (p.mfl + MF_CE + gen*32 + e). Phase B units release
them with red.release.gpu (wgs_flush kind 4 for whole units, MFP&2 after the designated finisher's epilogue for
split units, 4 warpgroups x 16 units = 64 per job); the final item acquires the counter of each of its token's
local experts instead of waiting at the grid barrier. The native repair forced it to 0 for WGS (ly_stk3 line 3967)
because the final's poll was bounded at 4M iterations and could fall through
(docs/steering/2026-11-01-g-native-repair-inventory.md line 78).

## Candidate 1: ly_gb4 (B4 build only, decode 32)

File muse_decode_layer/cand/ly_gb4.py, SHA256 07e145f7369c8ec96ea51fa678e84dae9769eb3b1c14f9cca398549c1aee6299.
Diff against ly_stk3 (two lines, MOE_CUDA_SRC only; the alt source copy at line 12751 is untouched):

- 3967: `#if OKC_WGS` becomes `#if OKC_WGS && !OKC_B4FIN` (MFIN stays on in the B4 build only; MID is WS and MAIN
  is WGS without B4FIN, so both keep MFIN 0 and the grid barrier).
- 6581: the B4FIN poll `for (n...; ld_acquire(cp) < target && n < 4000000; ...)` becomes an unbounded
  `while (ld_acquire(cp) < target)`.

Ordering argument: contributor epilogue stores, then bar.sync 3+wg (128 threads), then red.release.gpu by one
thread; final: ld_acquire.gpu by tid < 8, then CTA_SYNC, then __ldcg reads of sspart and ybuf. Reset: launch
parity from atomicInc on MF_EXIT (after griddepcontrol.wait, line 6307); each launch zeroes the other parity's
CE/XR/DA after route(); the MoE launch is an ordinary stream-ordered cooperative launch, so the previous launch has
finished. No deadlock: cooperative residency is checked on the host; the poll waits only on phase B releases, and
every CTA flushes its last pending release (line 6166) before its final. Final sum code is unchanged (bitwise).

## Measurements

- Gate (21:10-21:12 UTC): rc 0, all five PDL8H guards pass (T=1 sliding, T=8 full, T=8 sliding, T=16 sliding,
  T=32 full; 2000 replays per arm, ids mismatches 0). T=32 full max rel diff 5.52e-03 / 6.41e-03, equal off and on.
- chk: 12/12 check PASS. Single-run decode 32 sliding net 135.84 us (box/judge 0.903), not an A/B number.
- Paired A/B against ly_stk3, two rounds pooled, 24 pairs per cell (muse_decode_layer/profiles/box/moegb/
  gb4_table_both_rounds.txt): decode 32 sliding 142.10 -> 140.36 us (-1.75, 24/24, thr 3.62); decode 32 full 294.00 ->
  292.33 (-1.67, 24/24); decode 1 full -0.70 (24/24); decode 16 sliding / full -0.39 / -0.47 (18/24); decode 1 sliding
  -0.10; verify 1 -0.21 / -0.08; verify 16 +0.17 / -0.59; verify 32 -0.33 / +0.09. No cell beyond its threshold.
- Decision: meets the bar (decode 32 sliding >= 0.8 us, 20+/24). Ledger registration 13:17 PT; run log section
  "MoE grid barrier removal (box)"; ruled-out row for MFIN in the MAIN (RFIN) build.

## Box paths

Driver tools/ly_gb_ab.sh (PHASE=gate, PHASE=ab). Logs /root/okc/.build/moegb/ (gb4_gate.log, gb4_ab.log,
gate_gb4.log); chk and rounds /root/okc/.build/prtl/gb_gb4/ and /root/okc/.build/prtl/gb_gb4_r2/.

## Stretch note (not built): MoE as a PDL secondary of the route kernel

Hypothesis only. The MoE already waits with griddepcontrol.wait at its top (line 6307), and its launch helper has a
programmatic-serialization attribute slot (line 3603). Launching it as a PDL secondary keeps the cooperative launch,
but its 132 CTAs need one whole SM each (1 CTA per SM by __launch_bounds__(NTL, 1) and the 132-CTA cooperative grid; the B4 shared-memory size was not checked), so they cannot become resident until the
producer's CTAs leave their SMs; the gain is the launch latency plus whatever sits above the wait, which today is
nothing. Moving work above the wait is limited: which experts run depends on routing, and the MFIN parity atomicInc and
zeroing must stay after the wait (the previous launch's final must be done). Cheapest decisive test: stamp the first
wait return relative to the producer's end at decode 32 with the attribute on (tools/ly_box_moe_pro.py) before
building anything.

## Next action

Lead: decide on a judge check of ly_gb4 (registered, not dispatched). Candidate 2 (MID decode 16) not started: it
needs launch parity for ss, ctr[0..1] and doneA plus a new per-job release in ws_helper's phase B epilogue.
