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
- A/B: running since 21:35 UTC, `.build/moegb/gb5_ab.log`, rounds in `.build/prtl/gb_gb5` and `.build/prtl/gb_gb5_r2`.

## Next action

When the A/B finishes: `python tools/ly_prtl_table.py .build/prtl/gb_gb5 .build/prtl/gb_gb5_r2` on the box, then
decide against the bar (decode 16 sliding at least 0.8 us faster, 20+/24 pairs, nothing else losing beyond noise).
If it clears, add the ledger line and report. Either way, add the run-log section "MoE grid barrier removal, MID
(box)" in docs/runs-decode-layer.md.
