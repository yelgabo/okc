# Lane moe-dfybf checkpoint

Updated 2026-11-04 14:11 PT. Branch moe-dfybf, worktree /Users/gretchengabo/GitHub/okc-wt-moe-dfybf.

Base: muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e.
Its prepare prints `FORK_MOE_ROUTE native repaired: cooperative+BF16, MFIN/DF/YBF/PDL off` (ly_stk4.py:17190, written by tools/dl_fork_moe.py:162).

## YBF (OKC_YBF): closed

- What it is: the WGS phase B epilogue stores ybuf as BF16 after the gamma multiply instead of FP32 (ly_stk4.py:5825-5829 force it to 0; YBF branches in the final at about 6654 and 6690). Only the WGS builds use it: B4 (decode 32) and MAIN (verify 16/32). TINY, TINY2 and MID run the WS helper and never read it.
- Why off: docs/steering/2026-11-01-g-native-repair-inventory.md:72 ("Conservative removal of an extra post-gamma quantization ... no certified restoration") and archive/docs/steering/2026-11-01-g-fork-precision-contract.md:23 ("extra quantization inside the baseline FP32 norm/finalization ... Retain off").
- Cause: a real precision constraint. The reference widens BF16 y, multiplies gamma in FP32 and normalizes in FP32 (precision contract line 21). YBF adds a BF16 round after the gamma product, so it changes numerics. Per the lane rule it is not built.
- Exact alternative: store BF16 d and rebuild the FP32 y in the final (docs/steering/2026-11-02-moe-factorized-output.md). It already lost or was null in the standalone matched gate (docs/ruled-out.md:629, MOE-FACTOR-Y-01, 13 regressions of 6 groups). In the layer's B4 final the y read is one float2 per slot and latency-bound, so halving bytes is expected under 0.2 us (hypothesis), below the 0.4 us bar.

## DF (OKC_DF, helper0df sentinel data-flag): repair plan

- "DF" here is the helper0df alias, not OKC_WGSDFIN. WGSDFIN is on (ly_stk4.py:5599) and its 2->3 change is a measured loss (docs/ruled-out.md:665).
- What it is: `_df_patch` (ly_stk4.py:15381-15453) makes phase B poll the inter rows directly (BF16 0xFFFF sentinel = not yet written) instead of waiting on the doneA counter, which is credited only after a CTA finishes its whole phase A. Only MID (decode 16) uses it, and only with PRMTSC=1.
- Why off: inventory line 79 ("Conservative exclusion of a separate unproved publication protocol, at M16 ... Safe restoration would require independent sentinel generation/publication/retirement proof"). ly_stk4.py:15455 aliases helper0df to plain helper0.
- Defects found: (a) the poll falls through after `spin > 200000` and would read sentinel rows as data, the same class as MFIN's capped poll; (b) the writer uses weak stores and the reader weak `__ldcg` loads, so the data-as-flag edge is not formally ordered under the PTX model. Retirement looks sound: the sentinel reset runs after the acq_rel grid barrier, the next launch is ordered by the kernel boundary, and `_ma_make_scratch` fills the sentinel at allocation.
- Smallest repair (hypothesis): unalias via `_df_patch(_MA_ALT_SRCS['moe_helper0'])`, remove the spin cap (cooperative residency makes an unbounded wait safe because phase A has no dependencies), and make the inter stores st.relaxed.gpu and the poll loads ld.relaxed.gpu so the flag is a same-location morally strong edge.
- Expected gain (hypothesis): 0.5-1.2 us at decode 16 out of the measured 3.02 / 3.04 us A-to-B handoff (docs/runs-decode-layer.md:1971-1990).

## Next action

Build muse_decode_layer/cand/ly_mdf.py on ly_stk4 with the repair, run the reference check, then a two-round paired A/B against ly_stk4 with tools/ly_arms.py, judging decode 16 sliding and full only.
