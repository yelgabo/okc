# Pinned keell0renz layer projection review and conservative repair

2026-11-01. Read-only source findings plus requested offline transform; no fork GPU dispatch.
Parent owns fork composition, NOTICE trailer, index/publication and front/glue review.

Evidence: public commit `8e30e1be994db4310794b8d04e90d18d895deb84`,
`kernels/muse_decode_layer.py` SHA256
`1ec189dc6f306cec1694946eb62d64b54daf378c6fc1ada2f0f5e501e973ff2a`.
Owned combined-candidate evidence is separate: source `dd1a4b6`, matched check
`828974e3bbcc4d9b`, clean check `47bec92021ef4bbe`, receipt commit `2276c24`.
That combined check measured ~0.34–0.36% six-load gain over our selected baseline;
it establishes no performance claim about this donor fork.

## Active routes and findings

1. **rs128 is not a full FP32 split reduction.** `RQ_USE` at16209 enables qkv128/256
   and unpooler128. WGMMA accumulators are FP32, but `rq_pk`/`stmatrix` round each
   K-split partial to BF16 before `cp.reduce.async.bulk.tensor...add`; tensor maps
   at15810 specify BF16. The 12-bit pack is checked bit-exact at16075; that does not
   make this accumulation equivalent to one final BF16 rounding. Per-KS validation
   bounds at14933 are .004–.008, not evidence of unchanged precision. Conservative
   repair disables this family, including prepare-time probes, rather than relying
   on test tolerance to authorize the rounding change.
   [Pinned rs128 implementation](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/kernels/muse_decode_layer.py#L15657).

2. **EFENCE bit9 is inactive.** The final nine `RQPROD` entries at15974 have no
   `FL & 512`. The live zeroing path retains each writer's global proxy fence,
   release GPU counter/acquire polling, global proxy fence, release CTA flag and
   issuer acquire/global proxy fence. Each shared `stmatrix` batch is fenced before
   its warpgroup barrier and TMA issue. Do not describe the dead EFENCE experiment
   at15265 as an active missing fence. `.wait_group.read 0` at15718/15736/15777
   protects source consumption; it alone is not proof that destination writes are
   visible. No intra-kernel generic consumer of that output was found; disabling
   rs128 removes this path from the fork under review.

3. **Ordinary sk2 preserves BF16 operands and FP32 accumulation/corrections.**
   Its template at8053 uses `mma.sync` FP32 accumulators; FP32 escape corrections
   have private per-warp storage. Every escape warp observes and returns every
   stage (8234–8261), then release-arrives on `cbar`; consumers acquire-wait before
   reading corrections/reusing the ring (8460). The normal BF16 epilogue rounds
   only after FP32 reduction (8558). No stage barrier is removed by the repair.

4. **PDL-to-TMA proxy crossing gets an explicit conservative fence.** sk2 already
   waits before activation TMA issue: tid0 at8170 for CL>1, all threads at8183 for
   CL1/refill producers. The template contains no global proxy fence there.
   The patch adds `fence.proxy.async.global` immediately after both waits; early
   constant-weight requests remain early. This closes an unproven cross-proxy
   handoff; it is not a measured reproducer or a claim that passing tests proves
   the original pattern safe. PTX distinguishes generic/async proxies and proxy
   causality; a dependency wait's visibility statement is not used here as a
   substitute for an explicit proxy-ordering proof.
   [PTX8.7 memory synchronization/causality](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#memory-synchronization),
   [proxy fence](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence),
   [grid dependency control](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol).

5. **Fused sk2 router global join has a retained ordering chain.** Generic FP32
   partial stores precede a consumer-CTA barrier, leader release GPU RMW, last
   leader acquire fence and another CTA barrier before readers (8426–8448).
   The RMW observation chain plus acquire fence and CTA publication support the
   generic-proxy readers; it is not an async-proxy completion argument. Repair
   leaves every operation intact. Root separately reviews fused front/glue.

6. **Lt paths retain full-K operations.** `_GraftLt` at12255 requests compute32F,
   layouts at12290 batch output features, not K. Raw algo words are opaque: the
   repair queries `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` and accepts only0/1 before
   installing any saved plan. Attribute-built QKV256 already uses splitK1 and
   matches our validated tile124/cluster2 mechanism. Rejected plans remain absent,
   using the existing dense fallback; root must preserve its BF16 boundary ABI.
   This avoids asserting that unknown opaque K-split reduction schemes preserve
   FP32 intermediates. It does not promise every plan survives the runtime guard.

## Compile flags and attribution

Main sk2 extension at9383 uses `-O3`, native H200 `sm_90a`, optional diagnostic
delay definition; no `--use_fast_math`. Its narrowed/gle siblings at13391/13972
also lack fast-math (GLE is root-disabled). `RQFLAGS` at15899 uses `sm_90a`,
`-O3 -DNDEBUG --expt-relaxed-constexpr`; the fast-math flag at14543 belongs to
the flush calibration extension, not rs128 arithmetic. These distinctions avoid
attributing unrelated boundary/diagnostic flags to the dense kernels.

Retain the layer's original header lines8–16: alygg77 commit6cae49a, chenwainuo
3d418e1, Matsvei/kernarc helper/pins, vjeux f35a41e. The package
[NOTICE](https://github.com/keell0renz/muse-open-kernel-challenge/blob/8e30e1be994db4310794b8d04e90d18d895deb84/NOTICE)
credits team keell0renz/Bohdan Agarkov, Apache upstreams and a separate Nathen MIT
portion in the standalone projection file. Preserve NOTICE/license text as root's
trailer; do not relabel a fork as an owned kernel or copy standalone MIT code
without its notice.

## Deliverable, validation and stopping criteria

`tools/dl_fork_projection.py:patch_projection(source)` truncates `_make_chen` and
the unused rs128/MFAB tail, stubs `_rq_gemm` to None, adds the two sk2 proxy fences,
and installs the no-K-split Lt guard. Exact pinned source transforms to863066B.
AST reference analysis finds only `_rq_gemm` and existing `prepare` referenced
above the removed tail; no `_rq_time`/`_CHEN` dangling reference. The actual ctypes
guard passes CPU fake-library split0/1 and rejects2/4. Source parses. No GPU claim.

Expected impact is unmeasured: rs128 wins may be lost; fences may add latency;
cutting calibration frees prepare time/payload. No numeric forecast. Root should
compose front/boundary repairs, preserve attribution, then require official full
layer correctness plus fresh matched full-six-load evidence before submission.
Stop on unresolved async/generic handoff, BF16 boundary/precision change, rejected
required route or missing validation; do not remove fences based on benchmark wins.
