# Prepared gamma with factorized output storage

November2,2026 Pacific. MOE-PARAM-Y-01 is a prepared, target-untested complete alternative after own factorized-storage job7479372115c64702 failed its matched gate. Its lower5698.508243us point remains disclosed. No new GPU job is dispatched. This design does not require storage alone to win.

## New source evidence and coupled design

The pinned standalone baseline already prepares gamma=BF16(raw_norm_w.float()+1) once in prepare, in the [pinned baseline prepare](../../muse_moe_w4a16/baseline.py) at line102. The portable C7 interface carries raw BF16 norm weights and repeats the effective-gamma expression on the device. The completed factorized-storage screen preserves that interface and recomputes gamma while reading compact down values. That reconstruction cost belongs in the complete design.

This candidate combines lossless factorized output storage with the baseline's untimed constant-parameter preparation. Store one extra private BF16 gamma vector per owner while retaining the supplied raw norm tensor and every NVFP4 weight. The native ABI still receives14weight tensors. At candidate WGS sizes only, its norm-weight slot7 refers to prepared gamma; original helper/control calls receive the original raw slot7. No input-derived value is cached. Two controls and two candidates each own their raw and derived buffers independently.

The selected candidate epilogue and factorized reader load gamma directly. They retain d=BF16(accumulator*global_scale), reconstruct FP32(d*gamma) with the same explicit rounding, and preserve the original RMS tree, epsilon, weighted combination and final BF16 boundary. B4 exact norm partials use the same prepared gamma. No gamma folds into W2 and no post-gamma BF16 product round is introduced. Original helper source, weights and dispatch at T1/T8/T16 stay unchanged. The complete changed endpoint is compared against the exact public C7 control, without assuming additive gains from its ingredients.

The benefit is smaller intermediate traffic with fewer reconstruction expressions in the GPU call. Added parameter loads, conversion/reconstruction multiplies, larger private prepared state and altered codegen remain costs. No current measured exposed-time or speedup claim is made. The original storage-only result, when terminal, retains its own source, gates and stop. This is a new prepared-parameter contract and a coupled implementation, not a ring, tile, cache or shape retune.

## Source, precision and ownership gates

Append gamma as state weight index14 after the original14weights have been packed. Calls construct a14entry argument vector. Candidate tokens>16 select prepared slot14 into native slot7; every other argument is the original pointer, and controls/small paths retain raw slot7. Explicitly validate this mapping across all nine sizes and four owners before timing. Count all15stored tensors in the existing strict ownership inventory. Preparation synchronizes before capture, and the state retains both forms throughout every graph. Neither raw weights nor inputs are mutated.

Prove prepared gamma equals the old BF16 effective parameter over finite BF16 inputs and that both selected epilogue and reader consume this contract. Test actual Torch state/argument bindings, fallback/helper behavior, wrong-arm rejection, scratch/input restoration and failure capsules. Preserve original geometry, cooperative residency, all fences/barriers/counts, scratch offsets, source attribution and numerical tolerances. The old storage source and active waiter remain untouched.

## Finite decision

One complete private screen only after the accepted job's decision, qualified exact source publication and fresh shared-job/quota coordination. General's separate accepted7f10243c98e74746 remains its own job. No polling or duplicate posting of that ID. Use three fixed fresh normal/skewed banks,54cases/36timed rows,378observations,216captures,1152cold replays and54restores, with independent duplicate owners. Preparation stays295s and partial evidence stays failed. Keep the every-bank/allfour-cross1.01/twice-null/all-load/no-regression/no-drift gates. A successful screen needs held-out confirmation before integration or another public request.

Any source-policy, exactness, ownership, ordering/progress/lifetime, numerical, guard, source/record/coverage/budget, drift, regression or benefit failure stops this exact combined design. No larger guard, parameter/layout sweep, post-gamma rounding fallback, favorable mask or blind repeat. A source-correct build precedes speed qualification; no ingredient must win separately. Public permission remains limited to the already submitted5df257af. No GPU result or integration is claimed here.

## Final-day priority deferral

Origin/main776f8857 directs all new engineering and checks to short loads. Park this B4/MAIN design without a GPU dispatch. Paired8ec6ee0693e9c18f70ed517c5e97ddf7061a8b576e362b6855d95c3e8992319e/350452bytes, native0b864c8ed578baaad19035e323193e8a5dbe026cab0bdba16ae7b9ccb3d1f54e. Eight CPU/source/host/reader tests pass. The actual native reader CPU witness covers2088960products and8192paired loads; an injected post-gamma round fails. Both candidate CUDA12.8.1 SM90a device builds pass locally, with exact prior control builds retained. These checks establish no H200 speedup. Source generation remains reproducible through tools/moe_param_y_harness.py; qualification records are in evidence/2026-11-02-moe-param-y. No independent review was available. New next action is the assigned Tiny/Tiny2 timeline, not this deferred dispatch.
