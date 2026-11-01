# Independent owned attention wait-trace review

November1,2026. Read-only review of attention-owner commit381a09a and host parity fix91b550e. Final checked artifact `.build/today-attn/wait-trace.py` **361621bytes**, SHA256`a882bd2ad7d9ebf36a5d4f739e6c0b60930a6e733e24fed06712090bc751d9c0`. Earlier dbb306…361629-byte artifact is superseded. No GPU dispatch or public activity by reviewer.

**Clear for the parent's single authorized off-board diagnostic**, with resource/perturbation/coverage gates and interpretation limits below. This is not a performance candidate or a public-submission recommendation. Runtime Torch/CUDA host compilation, actual H200 admission and all real correctness gates remain the judge's work.

## Source and timestamp meaning

Reviewed `attn_wait_trace_patch.py`, `attn_make_wait_trace.py`, `attn_wait_trace_body.py`, emitted CUDA and the inherited graph/timing wrappers. Existing waits, proxy fences, stage arrivals, async copy operations, barriers and math remain; instrumentation is sparse at cumulative tile ordinals0/16/64/128. XPIPE0, PBULK0 and PF3 are enforced. Producer and consumer enumerate the same concatenated per-CTA segment/tile order; counters are not reset at segment boundaries. EREL releases a stage before later PV work, so the recorded release marker intentionally means **after the actual empty-barrier arrival**, not end of all consumer math.

Producer fields bracket the actual empty wait, issue/copy-publication marker and next page-refill source region. Consumer fields bracket the actual full wait, stage release and next acquisition. Presence and final producer/consumer tile counts reject missing or mismatched sampled ordinals; stale unsampled words are not interpreted as live data. Signed low32 globaltimer differences handle wrap. `copy_publish_to_ready` may legitimately be negative because the producer leader's post-arrival timestamp can follow consumer readiness; it is not an unsigned huge stall.

These are **sampled leader marker/wait intervals**, not physical HBM measurements, whole-CTA completion, additive recoverable phases or proof of a critical path. `globaltimer_ns` is volatile PTX without a memory clobber. In particular, `refetch_issue_region` must not be advertised as proving that every page-table LDG was instruction-enclosed or completed inside the two markers without a separate SASS-ordering analysis. Existing barrier helpers include compiler memory clobbers; no new weaker memory-order pattern is being justified by this probe.

## Ownership, coverage and failure gates

Both detailed timed functions receive actual nonnull private DBG storage. Four code/module placements are ctrl/ctrl2/detailA/detailB; detail buffers are private per arm. The measured graph wrapper clones tensor inputs and caches and retains the backing dict on the CUDA graph. Fresh raw collection also clones every arm's QKV and cache bank; all four pointers are distinct. Three fixed input seeds511019+k match timing and stamp collection. Each raw stamp replay is cold-flushed and synchronized before reading counters.

The four required full-context cells are B16/B32 × L1/L8.24 stamp rows and12 per-input timing rows are mandatory; math/cache/row checks are fatal and no timing follows incomplete trace collection. Extra pruned detailed instances are never invoked on unsupported shapes; ordinary judged control retains its normal native routes. Common BF16 attention-value repair precedes output RMS in every arm; judge-style FP32 reference checks remain independent of that source repair. Actual full fresh/replay math and graph-capture behavior are not proved by CPU fixtures.

The perturbation gate compares all detailed/control crosses and both duplicate nulls against **max0.5us/1%**, without a posthoc noise escape. A failed perturbation gate marks causal_usableFalse even if the unchanged judged control passes. Such a receipt cannot be cited as valid low-perturbation timing attribution.

The reviewer found that initial `_wait_resources` compared only BYTES/threads/perSM and silently allowed numRegs drift. Owner fixed this at91b550e: **all five fields must match**, both modes must have zero local bytes and at least2resident CTAs/SM. Final a882bd2… hash independently verified after this host-only fix. Local device results are mode49 128regs and mode51 80regs, zero stack/spills, with identical off/on geometry; these are compiler/resource facts, not SM utilization.

## Independent verification and limits

All **6 worker CPU tests independently rerun PASS**: source/compiled pins, preserved primitives, signed-wrap/presence negative, cross-segment ordinal model, private graph/coverage source contract, and full report transport. Four actual sm90a NVRTC12.8/PTXAS manifests were inspected through those tests; they cover device compilation, not host extension linking.

Additionally compiled and executed the **actual emitted `attn_wait_resources` C++ body** with Zig and explicit CUDA/Instance mocks. It produces2rows×7fields in the intended order and cudaFuncGetAttributes failure throws. Evidence: `muse_attention_decode/profiles/wait-review.2026-11-01/actual-export-mock.cpp` and result.txt. Actual `Instance` declares the public prepare/BYTES/NT/per_sm fields used by the exporter. The mock validates API/type/control-flow shape; it does not certify real CUDA ABI/link/runtime calls.

Stop on compile/capture/math/cache/CUDA/resource/coverage/log-budget failure; retain exact owning receipt and no performance attribution if perturbation fails. Parent owns dispatch and interpretation; reviewer launches none. Review sealed, available for the separately requested G stage probe.
