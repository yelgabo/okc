# moe-gbm lane checkpoint (MoE grid barrier removal, MAIN build)

Updated 2026-11-04 13:33 PT.

## Base

muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e
(judge-valid 10,046.1 us, the draw base).

## Finding (code reading, to be confirmed on the box)

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

## Next action

Read the gate (five PDL8H guards, chk 12/12) and stress results; if both pass, launch PHASE=ab and tabulate 24 pairs
per cell with tools/ly_prtl_table.py.
