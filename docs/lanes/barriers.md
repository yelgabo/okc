# barriers lane checkpoint (synchronization-point inventory, whole layer)

Updated 2026-11-04 14:09 PT. Status: step 1 (inventory) in progress.

## Base

muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e
(judge 10,046.1 / 10,070.4 us). Worktree /Users/gretchengabo/GitHub/okc-wt-barriers, branch barriers.

## Facts read so far

- Capped polls in ly_stk4: line 6640 (`n < 4000000`, WGSFINSS final, MFIN path). In ly_stk4 the define gate at 3967
  (`#if OKC_WGS && !OKC_B4FIN` forces MFIN 0) leaves it compiled only where MFIN is on with WGSFINSS; the B4FIN poll
  (6581) is unbounded since ly_gb4. ly_gbm (ly_stk5) makes 6640 unbounded. The `4000000 + seq` at 5995 / 14686 is a
  debug tag of wgs_wait (ignored), not a cap.
- Measured decode 1 TINY MoE tail (runs-decode-layer.md 5524-5534, 5896-5904): grid barrier 1.26-1.31 us after the
  phase B end at sliding, final pass 0.39-1.29 us after the barrier.
- MID (decode 16) tail (runs-decode-layer.md 7211): B end to barrier 0.80, barrier idle 2.75, final pass 0.46; ly_gb5
  null.

## Next action

Collect the three inventory reports (MoE, attention, glue/router/PDL), build the table and ranking.
