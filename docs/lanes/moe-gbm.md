# moe-gbm lane checkpoint (MoE grid barrier removal, MAIN build)

Updated 2026-11-04 13:46 PT. Status: done, bar cleared, registered in the ledger at 13:42 PT.

## Base

muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e
(judge-valid 10,046.1 us, the draw base).

## Finding (code reading, confirmed by the box stress)

The ruled-out row "MFIN per-expert completion counters in the MAIN build" (docs/ruled-out.md line 835) rests on a
false premise. Under RFIN the finisher share (s0 == 0) does not return in wgs_seg: after it acquires the other shares'
stage counts and adds their fixup slots (ly_stk4.py lines 6079-6119), it falls through `#endif` into the whole-unit
epilogue (6132-6138), which sets `pd = WgsPend{4, u, gen * 32 + jE[j]}`. wgs_flush kind 4 (5897-5903) runs
`bar.sync 3+wg, 128` and then one `red.release.gpu` per warpgroup. Non-finisher shares (s0 != 0) store only fixup
partials and release the fc stage count (kind 2), and they write no y. So every phase B unit releases exactly 4 (one
per warpgroup) whether it is whole or cut, and every job releases 16 * 4 = 64, equal to the target
`64 * (tot > SK_T ? ceil(tot / FT_T) : 1)` (6636-6638). Every segment has at least one stage, so the deferred flush at
`FIRST && ci == OKC_WGSFCI` (ci in 0..7) is reached, and wgs_phase flushes the last pending release at 6166.

The only thing that kept MFIN off in MAIN was the define gate at line 3968.

## Candidate

muse_decode_layer/cand/ly_gbm.py, SHA256 c7c2bf55195b1aa40f647bd86d1a5cf2d8d2dd2e9b6fdee502b974d380c0f217.
Diff against ly_stk4 (two lines, both inside MOE_CUDA_SRC):

- 3968: `#if OKC_WGS && !OKC_B4FIN` becomes `#if OKC_WGS && !OKC_B4FIN && !OKC_RFIN` (plus a comment). Only the MAIN
  build passes -DOKC_RFIN=1 (MAIN_EXTRA_DEFINES, line 16420), so MID (WS, MFIN forced off at 3880), B4 (MFIN already
  on) and the global `_moe` path (MOE_DEFINES, no RFIN) are unchanged.
- 6640: the bounded poll `for (n...; ld_acquire(cp) < target && n < 4000000; ...)` becomes
  `while (ld_acquire(cp) < target)`.

## Box

- Driver tools/ly_gbm_box.sh (PHASE=gate: tools/ly_gb_ab.sh gate with CTL=ly_stk4.py CAND=ly_gbm.py TAG=gbm, then
  tools/ly_gbm_stress.py under flock on gpu.lock; PHASE=ab: two paired rounds).
- Stress tools/ly_gbm_stress.py: one CUDA graph per file per M, new routing per replay (uniform, all-local, hot
  experts, skewed mix with -1), checks bitwise equal output to ly_stk4, counter == target exactly for this launch's
  parity, other parity all zero, parity alternates. M = 33 (600), 128 (2000), 256 (2000), 48 / 97 / 200 / 255 (300).
- Logs /root/okc/.build/moegb/gbm_gate.log, gate_gbm.log, stress_gbm.log; chk in /root/okc/.build/prtl/gb_gbm/.
- Gate launched 21:31 UTC (13:31 PT).

## Measurements

- Gate (21:31-21:33 UTC, gate_gbm.log): rc 0; all five PDL8H guards pass (T=1 sliding, T=16 sliding, T=32 full,
  T=8 full, T=8 sliding; 2000 replays per arm, ids mismatches 0 in every arm).
- chk (.build/prtl/gb_gbm/chk.log): check PASS at all 12 cells. Single runs, not A/B numbers: verify 16 sliding net
  236.99 us, verify 32 sliding 296.80.
- Stress (21:33:52-21:34:36 UTC, stress_gbm.log): PASS. M=33 600 replays, M=128 2000, M=256 2000, M=48/97/200/255 300
  each; in every replay the output equals ly_stk4 bitwise, the counters of the launch's parity equal 64 * njobs(e)
  for all 32 local experts exactly, the other parity is all zero, and the parity alternates. Up to 4 jobs per expert
  (max_tot 256 at M=256); fat jobs 59,993 and skinny jobs 13,628 over the M=256 replays.

- Paired A/B (21:35-21:41 UTC, two rounds, 24 pairs per cell, PDL pinned; .build/moegb/gbm_table_both_rounds.txt,
  copied to muse_decode_layer/profiles/box/moegb/): verify 32 sliding -1.59 us 24/24, verify 16 sliding -1.03 20/24,
  verify 16 full -1.04 23/24, verify 32 full -1.40 22/24; decode cells within +-0.3; every cell inside its threshold
  (worst decode 1 full +0.23, thr 2.56).

## Decision

Clears the bar (verify 32 sliding >= 1.5 us faster in 20+/24 pairs, nothing outside noise). Ledger line written at
13:42 PT (box only, no job dispatched). Run log section "MoE grid barrier removal, MAIN (box)" added to
docs/runs-decode-layer.md; docs/ruled-out.md line 835 corrected.

## Next action

The lead decides whether to dispatch a judge check of ly_gbm.py (c7c2bf55). The change composes with moe-gb round 2
(MID build) since the two touch different define gates; a merged file needs its own gate and stress.
