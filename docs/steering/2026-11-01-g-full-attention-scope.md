# G full-context attention16/32: operation, plan and ROI screen

November1,2026. Evidence: scored layer607eb52bdaf24160, exact G source869076bytes/SHA256`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`; owned public best8914751e3b9f468b, current promoted owned pure checkd331aca81c554d70 (scored repeatc9bc5fbd85b440e2), and SGLang profile24daba2202e547d6. No GPU dispatch, candidate changes, registry/board changes or public activity.

**Go/no-go: no new controlled candidate from this screen.** The four full-context16/32 cases have substantial physical room, but comparing these receipts does not reveal a new compliant low-risk mechanism. G and owned are distinct implementations of essentially the same selected flat producer architecture. Existing named differences are already measured/closed or incorporated. This is a scoped next-job closure, not a claim that the attention memory pipeline is optimal.

## Actual operation and executed plans

`tools/dl_g_attention_scope.py <main-checkout>` extracts/executes the actual Python shape-plan functions with132SMs and the measured two-CTA resident capacity. G `_plan7`3773 and `_launch7`11234; owned `_one_plan`6431 and `_one`6483 select:

|Case|G|Owned|
|---|---|---|
|decode16/32 full|264CTAs,flat,nc1,cs0|264CTAs,mode49,flat,nc1|
|verify16/32 full|264CTAs,flat,nc1,cs0|264CTAs,mode51,flat,nc1|

Both use128 producer threads,128 decode/256 verify consumers,3 decode/4 verify FP8 K/V ring stages,64-key tiles, both KV heads in a128-byte row, full-context paging, and virtual request-boundary padding400/600. Decode uses FP16 QK `mma.sync`; verify uses FP8 QK/PV WGMMA with FP32 accumulation and the72-column PV normalization construction. No missing FA3-to-native, page-gather, head reuse, or larger-stage implementation is established by a different kernel symbol. Current G attention CUDA literal SHA256`22c116a45f1817fbaf27a1cc3656380affad49d2cb2367dd23aac83f20b7cef5`; owned current literal`3350b5a34aaf9241528a845455c0e2634b29f854b413894aa051dd9d2f547df4`; they are **not byte-identical**.

The mathematical scope is QK norm/new-KV cache writes, attention over every key, output RMS/gate and BF16 output. **Inputs/precision are not matched across these receipts:** G full cases use `dist=soft`; standalone full cases use `dist=coherent`. The latter alters the cache/query/value coherence (`reference.py:48–85,198–199`). G's `pr8` now BF16-rounds decoded attention values before output RMS. Historical and current standalone native output RMS uses unrounded FP32 values; its final output is BF16. A future comparison must put both arms on a common required BF16 boundary and retain separate literal judge-style tolerance and baseline-parity gates. Historical standalone medians cannot certify the cost of that repair.

## Four cells, whole scores and baseline evidence

|Full case|G attention profiler us|Owned891 profiler us|Current owned clean profiler us|Owned891 adjusted whole us|Official standalone floor us|SGLang baseline-copy adjusted whole us|
|---|---:|---:|---:|---:|---:|---:|
|decode16|87.5|84.7|88.0|85.046|63.177|200.511|
|decode32|170.0|160.3|169.2|162.552|126.355|385.045|
|verify16|104.1|99.6|100.0|101.555|63.392|205.166|
|verify32|186.2|178.6|184.5|183.564|126.785|390.632|

SGLang's retained profile is dominated by FA3:187.4/498.7/192.9/376.4us respectively, plus preparation/cache/combine/norm kernels. Its decode32 profiler time exceeds its adjusted whole benchmark; this illustrates why profiler rows are not timing-aligned benchmark replacements. Our native kernels already cover the operation in one launch. Neither a large FA3 gap nor a profile sum provides extra native optimization headroom.

Holding all other G cells fixed, replacing each positive G-vs891 attention **profile** difference2.8/9.7/4.5/7.6us gives **1.004595954** (+0.4596%) under the full official46/16/six-load formula. This is deliberately optimistic, cross-receipt sizing, not an exposed saving or port forecast. Current owned clean profiles instead differ from G by−0.5/+0.8/+4.1/+1.7us, demonstrating that the earlier gap is not a repeatable transfer claim.

The physics opportunity is larger: replacing all four standalone full medians with their own official analytic floors gives **1.192485572**. Per-cell six-load sensitivities are1.038606/1.035521/1.057202/1.048783. These assume an impossible complete recovery of the analytic gap, not a realizable architecture or measured bandwidth. As another scale check,10% shorter attention in all four current G full-context rows would give **1.010219684** (+1.022%). A credible broad memory-pipeline gain would therefore be material and shareable; none is identified by this screen. No utilization or actual HBM throughput is inferred from analytic bytes/floors.

## Named differences and prior evidence

- G flat decode halves its merge traffic, uses a two-chain QK and a different page-index schedule. Owned equivalents were already tested: HMRGF bce37d22a0ec4045 gave only0.04–0.07% standalone; QK2/PE16 c8e99e038d9f4575 failed the bar. The mode49 deeper ring42c4739cf6b247c4 and page-bulk bb06f8e57bd84636/f3f1e4a5249a4228 are also closed. This is not reopening the batch1 chain/finalizer work.
- Owned mode51 already incorporates descriptor carry, packed-P16-bit join and V-address hoisting, the G/public consumer-side mechanisms. XPIPE=2 lost cold verify16 in5575e7d15d3e49a8; XPIPE=1 exceeded the112-register two-CTA budget; one-CTA mode115 lost25–31% in318ae6fc3c3746ce. No new resource/schedule evidence reopens these.
- G uses per-SM split weighting1135/2048 while owned defaults to equal shares. This is not a fresh mechanism: block-index weighting7631033a0c93485d and runtime-SM weighting4ff339332fef4aff lost. No weighting sweep is proposed.
- Current mode51 consumer-arithmetic removal8ac6563343f74bcd did not improve copy-only latency versus the full kernel:110.69/197.06us versus110.88/188.38 atverify16/32. The probe has deliberately wrong outputs and does not isolate an additive recoverable phase. It supports seeking a changed producer/copy overlap mechanism, not deleting consumer work.
- Exact WST fullverify16 comparison4862f8e0dc8a4b5a was slower by10–14us. A generic replacement with that engine is not a low-risk new candidate. Its separate short sliding signal is outside this screen.

## Next action, shareability and stop rule

Do not produce or dispatch another standalone knob variant from these receipts. A future candidate should name a new copy-issue/address/lifetime mechanism, preserve physical page mapping and exact K/V coverage/precision, prove all async lifetime/fence edges, and demonstrate realistic registers/shared memory/residency before one paired test. Full producer/consumer progress records correlated to actual page/segment input metadata are a useful reopening trigger; aggregate per-pair completion spread is insufficient.

Any later standalone file must have four private states/data/cache/output graphs, independent controls, both coherent scored distribution and soft G-distribution heldouts, fresh-input and cache/math gates, affected-cell regressions, and all-six46/16 score/noise gates. Shared plan/geometry makes a successful mechanism architecturally shareable with G, but G's different native code, BF16 boundary, surrounding graph and soft inputs require its own measured integration. Standalone microseconds or scores cannot be added to G's score. No candidate file, quota request or new H200 claim follows from this closure.
