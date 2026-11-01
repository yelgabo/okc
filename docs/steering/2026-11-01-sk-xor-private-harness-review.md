# Final private SK-XOR harness review

November1,2026. Independently reviewed exact sibling attention-worktree artifact `.build/today-attn/sk-xor-private.py`,330402bytes, SHA256 `e5d82cf2f1245495e070f6953a6f0b048cf21ac786b0cf4aa97384d3fc6f7020`; implementation references4bf27c5/fdea8fa. Augmented helper source is pinned d23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586, original C2 pin5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4. No edits to the owner's harness and no GPU/public dispatch in this review.

**No remaining source/host-scope blocker found for this PRIVATE diagnostic.** This is not clearance to submit the diagnostic publicly. The separate one-arm production qualifier requires its own exact-source review.

## Actual source review beyond fixtures

The builder preserves the exact C2 source prefix, restricts its appended helper bindings, retains original kernel_fn and wraps prepare only to run the private experiment before returning original C2 state. It does not select the treatment as ordinary judged dispatch. Both global-weight provenance and normal prepared pick remain visible; all ten target anchors must match the actual native `_sk_runner` code object and intended cid/FULL selection. A fallback or different tuner winner fails instead of masquerading as the expected control.

The four computing arms are c0/c1/t0/t1 across two independently built modules: each binary supplies a control and treatment. Writable diagnostic tensors, inputs, captured outputs and capture streams are independent. Twelve cached immutable formats are shared read-only only when actual weight identity, row-balancing rule, R/NW/UPW/NT/WSK/UPS and selected row count match; token reservation and stage count do not change those byte streams. Original C2 anchor diagnostics can share within their own serial control owner, but the overlap check rejects their alias with a computing arm. No new mutable global scratch or cross-owner stream is introduced.

The private module resource ABI is checked against analytic stage headers/stride/fixed shared/escape-warps/thread count, requires zero local bytes and positive occupancy admission, equal duplicate resources, and treatment admission no worse than control. Register changes are reported rather than silently normalized. Actual attribute/admission failures remain fatal. Local prediction is not substituted for H200 values. Old main-reader fences, stage transaction accounting, metadata lifetime and BF16 fragment restoration were separately reviewed; this harness does not remove them.

`_sx_capture` joins the parent packing stream, warms the selected function, captures on an owner-private stream, joins/synchronizes and replays. Input/output storage remains referenced through synchronized final use. Every timing replay is a normal kernel graph; there are no stamps. Each round cold-writes256MiB then randomizes arm order and measures CUDA events on the replay stream, followed by synchronization. The source makes no claim that this independently measures HBM transactions or isolates one instruction's cost.

All five actual supplied weights must be present. Target cells use all four computing arms plus the real C2 anchor. Other20of30cells use the actual normal C2 kernel, one common arm with no fabricated paired/null data. Three held-out seeds each cover30cells, with90rows total. Final counts require210captured placements,840captured-output oracle checks (including420fresh replay checks and210post-timing checks), plus80nearby/onehot and80nearby/outlier checks. Fresh copies update every graph's captured input before replay, and all outputs are checked against that fresh input. Non-FULL nearby instantiations are explicit; no FULL overwrite is assumed safe for smaller M.

Reference is independent FP64 matrix multiplication rounded to BF16 for dense projections and FP32 for router. Shape/dtype/finite output, global and row norm gates, plus a stricter one-hot gate, are enforced. Although reference finite is not separately asserted, NaN/Inf ratios cannot pass the explicit `<=` assertions; unlike an inverted `if error>limit` check, NaN does not evade rejection. This review does not weaken official tolerances. All diagnostic flags must still be zero after all cases.

## Reporting, scoring and budget

Lossless SX2 transport preserves all event float32-ms bit patterns and all90rows/resources/anchors/check counts; only measurement data is compressed, never kernel source. The encoded report must be<9000characters or prepare fails. Offsets/counts/dictionary framing and exact sample coverage are decoded fail-closed. No atexit transport or deferred partial summary is used. SXPINS/SXREPORT1/SXDONE emit after complete coverage and zero diagnostic validation. Source budget is295s with repeated case/round/final guards, including after original C2 prepare and private builds. A long in-progress compile can still hit the runner deadline before the next guard; this is an actual runtime qualification risk, not a hard preemption proof or permission to accept partial coverage.

The embedded raw score model and canonical root reader remain separate from official normal-C2 score. Contemporary offsets must come from THIS eventual official receipt and apply commonly to all private arms under the preregistered30-cell model. Old or raw factors cannot be promoted as actual paired adjusted score. Full per-seed/null/regression/anchor gates remain required. Lost transport, incomplete rows or a failed terminal result cannot yield a performance decision.

## Independent checks and pending production scope

Executed the owner's9CPU source/host-facade/codec tests independently with Torch2.10CPU: all pass in1.986s; realistic90row/complete-resource transport is7854bytes. Command:

```sh
SK_XOR_SCORE_READER=ROOT/tools/proj_sk_xor_read.py /root/.cache/okc-host-validate/bin/python -m unittest discover -s tools -p test_proj_make_bench_sk_xor.py -v
```

The fixtures mock CUDA arithmetic/streams/resources and do not qualify device execution, setup time, hardware admission, capture, numerical output or latency. Those remain root-owned H200 gates. Earlier exact-helper RTX bit checks do not broaden this scope.

A later public qualifier must be independently pinned: one treatment arm only, no diagnostic/control graphs or new comparative tuning; preserve ordinary C2 behavior for the other20cells and unsupported cases, require all ten offered native cells plus independent fresh/numerical/resource/diagnostic coverage, keep the295s preparation gate and retain original output types/boundaries. Nothing in this private-harness clearance approves that not-yet-reviewed artifact or a public diagnostic submission.
