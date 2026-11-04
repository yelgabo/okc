# Lane moe-dfybf checkpoint

Updated 2026-11-04 14:16 PT. Branch moe-dfybf, worktree /Users/gretchengabo/GitHub/okc-wt-moe-dfybf. Lane stopped at its tool budget; a fresh lane resumes from "Next action".

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

## Candidate ly_mdf (built 14:13 PT)

muse_decode_layer/cand/ly_mdf.py, SHA256 28d4422ac37a7857da8d0f514ee13e8549f8331bf934141d084c4018250c06c8, made from ly_stk4 by
/private/tmp/claude-501/-Users-gretchengabo-GitHub-okc/aad66da8-25fc-49ca-a3a1-289176256272/scratchpad/mkmdf.py (47 diff lines). Changes:

1. `_MA_ALT_SRCS['moe_helper0df'] = _df_patch(_MA_ALT_SRCS['moe_helper0'])` instead of the alias to plain helper0.
2. Poll uncapped (`spin > 200000` fall-through removed).
3. Poll loads `ld.relaxed.gpu.global.v4.u32`; the ws_helper inter store under OKC_DF is `st.relaxed.gpu.global.L2::cache_hint.b32` (same evict-last policy as st_keep_bf16x2).
4. New defect found and fixed: the sentinel rows were inter rows [128, 256) of the one shared scratch, which every WGS launch with M > 16 (decode 32 B4, verify 32 MAIN) writes as real data. A MID launch after one of those would accept stale rows as written. The rows move to [EMAX - 128, EMAX), which no build reaches at M <= 32 (`_df_inter` fill and the `p.inter` offset both moved).
5. Status string reads `MFIN+DF on, YBF/PDL off`.

Protocol note for the lead: under OKC_DF the MID build drops the doneA credit (its `__threadfence` and atomicAdd) and the phase B `ld_acquire` wait. The replacement is a data-as-flag on the inter words themselves with gpu-scope relaxed accesses. The writer stores 32 bits and the reader loads 128 bits, so the PTX "overlap completely" clause of morally strong is not met; soundness rests on per-word single-copy atomicity of aligned 32-bit stores. This conflicts with the brief's "keep every fence and the release/acquire ordering" if read literally; the lead decides.

## Box A/B: round 1 invalid, build failure

Script muse_decode_layer/profiles/box/mdf/mdf.sh (box copy /root/okc/muse_decode_layer/profiles/box/mdf/mdf.sh), logs /root/okc/.build/mdf/arms_r1.{log,json}. tools/ly_arms.py on ly_mdf, decode 16 both layers, 48 runs, router seed 106218864389449120. Arms: base (MOE_MID_DEFINES back to @moe_helper0 with its own `ma_ext` dict, which is ly_stk4's MID kernel), df (file default), base2 (noise floor).

Round 1 finished at 14:15 PT but measured nothing: prepare printed `FORK_MOE_ROUTE Marlin: native build failed [(16, ... main.cpp -o main.o ninja: build stopped: subcommand failed.)]`, so all three arms ran the Marlin fallback (decode 16 sliding 214.9 / 215.5 / 215.2 us, full 284.4 / 283.6 / 283.7 us, all checks PASS; these are Marlin numbers, not a DF result). `FORK_MOE_SCALE ('-DOKC_PRMTSC=1',) True` confirms PRMTSC=1, so DF would be active once it builds. Round 2 was stopped; no box job of this lane is running. The GPU lock was used for round 1.

Failing build dir: /root/.cache/bx_ext/glue_moe_6f5ecd94975b9370_sm90a-bc5b91501686cbf4 (failing step is main.cpp). Likely cause (hypothesis, not yet seen): my host-side offset `p.inter += (size_t)(okc::EMAX - INTER_SENT_ROWS) * okc::HD;` is compiled in main.cpp, where INTER_SENT_ROWS or okc::HD may not be visible; ly_stk4's original line used the literal `128 * 4096`. Fix: use the literal `(size_t)(3072 - 128) * 4096` (EMAX = 3072 = _MA_EMAX) in the `_df_patch` host replacement in mkmdf.py.

## Next action

1. On the box: `cd /root/.cache/bx_ext/glue_moe_6f5ecd94975b9370_sm90a-bc5b91501686cbf4 && . /root/okc/tools/box_env.sh && ninja 2>&1 | grep -m5 error` to see the real error.
2. Fix it in /private/tmp/claude-501/-Users-gretchengabo-GitHub-okc/aad66da8-25fc-49ca-a3a1-289176256272/scratchpad/mkmdf.py (or edit ly_mdf.py directly at the `p.inter +=` replacement inside `_df_patch`), rebuild ly_mdf.py, rsync, and rerun `nohup bash muse_decode_layer/profiles/box/mdf/mdf.sh` (it takes the GPU lock itself). Confirm the log has no `native build failed` and that base and df show the native decode 16 times (about 70-75 us MoE span; layer net far below Marlin's 215 us).
3. Read results: `grep ^ARMS /root/okc/.build/mdf/arms_r{1,2}.log`; judge df minus base at decode 16 sliding and full only, against the base2 noise floor; the bar is -0.4 us at a sliding cell in both rounds.
4. Ruled-out row for YBF (text below) goes into docs/ruled-out.md; DF gets a row or a candidate line depending on step 3.

Draft YBF ruled-out row: `| 2026-11-04 OKC_YBF (WGS ybuf stored BF16 after the gamma multiply; B4 decode 32, MAIN verify 16/32) on ly_stk4 a054c8d5 | Not built: it adds a BF16 round after gamma that the reference (FP32 gamma, FP32 norm/finalize; precision contract archive/docs/steering/2026-11-01-g-fork-precision-contract.md:21,23) does not have. The exact factorized form already failed the standalone matched gate (MOE-FACTOR-Y-01, line 629). B4 final y read is one float2 per slot and latency-bound; expected gain under 0.2 us (estimate). Reopen only if the precision rules allow a post-gamma BF16 round. |`

## Box result, final (14:18 PT)

Round 2 was not stopped: it finished at 22:16 box time. Both logs contain `native build failed` twice, so every arm in both rounds ran the Marlin fallback. Neither round says anything about DF.

Round 2, decode 16, all checks PASS:
- sliding: base 215.31, df 215.39, base2 215.25 us
- full: base 283.57, df 285.04, base2 282.99 us

The next action is unchanged. Get the real compiler error from the failing build dir, fix the host `p.inter` offset (hypothesis: use the literal `(size_t)(3072 - 128) * 4096`), and rerun `mdf.sh`. Before reading ARMS lines, confirm the log has no `native build failed`.
