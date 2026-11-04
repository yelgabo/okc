# moe-t1 checkpoint (2026-11-04 12:58 PT, round 2)

Base: muse_decode_layer/cand/ly_stk3.py, SHA256 a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6 (judge 10,109.9 us).

## Item 1: decode 1 TINY x at entry - closed, null

- Candidate muse_decode_layer/cand/ly_x1e.py, SHA256 c385fe76677295c38ea21337fe8466963e00676441ccf98a824fece579edf17a (tools/dl_ly_x1e_patch.py).
- Stamps (mean CTA, sliding): x consumed 2.90 / 2.90, first ring slot 3.03 / 3.03, prologue +0.10 to 0.15 us. x is off the ramp's critical path; the first weight unit's copy gates it.
- Gate 5/5 PDL8H_GUARD pass, chk 12/12 PASS.
- A/B pooled, 24 pairs: decode 1 sliding +0.19 us (8/24), decode 1 full -0.25 (14/24), every cell inside noise. Not registered, no judge check.
- Run log: docs/runs-decode-layer.md section "2026-11-04 - Decode 1 TINY x at entry (box)"; ruled-out row added.
- Box paths: /root/okc/.build/x1e/, /root/okc/.build/prtl/x1e, /root/okc/.build/prtl/x1e_r2, /root/okc/.build/prtl/x1e_table_both_rounds.txt. Local copies in muse_decode_layer/profiles/box/x1e/.
- Instrument copies (uncommitted, regenerable with tools/dl_ly_pmt_patch.py --xld __ldca and tools/dl_ly_mtl3x_patch.py): ly_pmt_stk3.py, ly_pmt_x1e.py, ly_mtl3x_stk3.py, ly_mtl3x_x1e.py.

## Item 2: per-warp handoff stamps - not started

Question: why the eight warps' last phase A partials reach the helper 0.55-0.87 us after warp 0 finishes. Hypothesis (unmeasured): the helper's epilogue queue, not warp skew. Proposed instrument on ly_pmt_stk3: atomicMin/atomicMax over warps of the last phase A unit's mma-done and wsre-pass times into DBG slots (tid 0 after CTA_SYNC), plus the helper's stamps at the start of its q == nA-1 iteration and at its wsrf ## Round 2 item: decode 1 TINY first weight copy - in progress

- Finding from the code: in TINY (OKC_F1EARLY=1, OKC_SPLIT=0) the FAST1 warp already issues the first two weight copies before route(), right after the topk_ids loads, the lane 0 mbarrier inits and fence.mbarrier_init. It is not behind route(). The stamped issue time (1.13 us mean CTA from the first CTA's entry) is the open question.
- Instrument muse_decode_layer/cand/ly_mtf1_stk3.py, SHA256 fb43910e97a6ed3fba3b5837c98054ddde064880f059319b9bb33521da72f7e6 (tools/dl_ly_mtf1_patch.py on ly_pmt_stk3.py 2949e3c99392377ed33ae5f5eaeba7bec5c501650e1ba4f6e9db4c623c9fad00; summary tools/ly_mtf1_summary.py). Slots: 0 entry, 4 FAST1 top, 5 inits+fence done, 7 mask reduced, 3 copies issued, 1 route, 6 first slot full, 2 phase A end.
- mtf1_a (box /root/okc/.build/mtf1/mtf1_a.json, 20 inputs, decode 1), mean CTA us from the first CTA's entry, sliding / full: entry 0.07 / 0.06, FAST1 top 0.14 / 0.14, inits+fence 0.40 / 0.37, mask 0.56 / 0.53, copies issued 1.05 / 1.03, route 2.10 / 2.08, first slot 2.98 / 2.97, phase A end 11.79 / 11.78. Per-CTA segments (sliding): top to mask 0.41 (topk load), mask to issued 0.50, issued to slot full 1.93, slot to phase A end 8.81.
- The largest unexplained segment is mask to issued (0.50 us of popc, muldiv, __fns, fence.proxy.async, expect_tx and two cp.async.bulk). Instrument ly_mtf2_stk3.py, SHA256 1b9e3e3cd8d0474b286d4d25e36e73e64d552f0fc34dde429ae5517640836cad (tools/dl_ly_mtf2_patch.py, summary tools/ly_mtf2_summary.py) splits it: 1 v0/n computed, 4 past fence.proxy.async, 6 first copy issued, 3 both issued. Box run /root/okc/.build/mtf1/mtf2_a.json started 20:53 UTC.

## Next action

Read mtf2_a stamps, pick the lever from the largest FAST1 segment, build a TINY-only candidate, gate it, A/B two rounds.
