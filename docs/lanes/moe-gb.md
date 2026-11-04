# moe-gb lane checkpoint (round 2: MID grid barrier removal)

Updated 2026-11-04 13:37 PT.

## Base

- muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e
  (judge job 4f99c1ec20f54f0f: VALID 13/13, 10,046.1 us, 2.2516x). Every A/B is paired against it.

## Candidate

- muse_decode_layer/cand/ly_gb5.py, SHA256 6177507af66733ca4b034e2552977cc8490623fedd93d5253deac7d125108170,
  built by `python3 tools/mk_ly_gb5.py muse_decode_layer/cand/ly_stk4.py muse_decode_layer/cand/ly_gb5.py`
  (commit 8918e849). 78 insertions, 1 deletion against ly_stk4, all inside the alt source
  `_MA_ALT_SRCS['moe_helper0']` under `#if OKC_WSFIN`, plus `-DOKC_WSFIN=1` on MOE_MID_DEFINES (line 6941).
  TINY and TINY2 compile the same alt source without the define, so they are unchanged. No shared MFIN code is touched.
- Design: the MID final pass no longer waits at the grid barrier. Each phase B epilogue warp (helper and the two
  WSBL compute warps) stores, runs a named-barrier sync over the warp, then lane 0 issues one `red.release.gpu` on a
  per-job counter doneB[j] (doneA+512+mpar*88). The final pass's 8 threads poll `ld.acquire.gpu` on the counters
  of the jobs they read until each reaches RB_B=256 (unbounded, no fall-through), then CTA_SYNC, then read.
  doneB parity comes from a MID launch counter at doneA+1023; the other copy is zeroed after route. doneA and
  ctr[3] keep their base semantics through a split barrier: each CTA's helper arrives (acq_rel) on ctr[4+(gen&1)]
  after its final WS_JOIN, and the last arriver zeroes doneA[0..88), bumps ctr[3] and the MID launch counter.

## Measured (box, okc-h200)

- Gate (`.build/moegb/gb5_gate.log`, `.build/moegb/gate_gb5.log`): five PDL8H guards pass, 2000 replays per arm,
  ids mismatches 0 (T=1 s, T=16 s, T=32 f, T=8 f, T=8 s); check PASS at decode 16 and 32.
- chk (`.build/prtl/gb_gb5/chk.log`): check PASS in all 12 cells.
- A/B, two rounds pooled, 24 pairs per cell (muse_decode_layer/profiles/box/moegb/gb5_table_both_rounds.txt; box
  `.build/prtl/gb_gb5`, `.build/prtl/gb_gb5_r2`): decode 16 sliding 125.11 -> 125.22 us (+0.11, 9/24 faster, thr
  2.21); decode 16 full 198.19 -> 198.18 (-0.01, 12/24). Every cell inside noise.
- Decision: loses the bar. Not registered. Run log: docs/runs-decode-layer.md, "MoE grid barrier removal, MID (box)",
  with a ruled-out row.

## Next action

None for ly_gb5. The untested variant is a final pass ordered by job completion, so that early final items read only
early jobs; the per-token wait as built still waits for the last phase B job.
