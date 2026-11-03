# Owned skinny bounded work-order progress review

2026-11-01, source-only review. Main5843874; `muse_moe_w4a16/submission.py` normalized LF SHA256 c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7, `_SKINNY_BLOCK` SHA256 e26165cb897c685feec1d2bfd1bcdeed26cf765643791d8d64b997249e91b98d. Implementation owner today_moe; no production edit or GPU dispatch by this worker.

## Mapping and ownership

The skinny block splits each expert into eight-row jobs and issues512 phaseA and256 phaseB weight-row units per job. It has no raw Stream-K fixup: helper_loop reduces all compute-warp contributions for each complete unit. A schedule transpose must be common to issue_unit (3950), compute_loop (3968) and helper_loop (4204). Ring slot/parity, partial-buffer index and handoff counts remain functions of CTA-local q, not mapped physical v. Outputs, job_q, job_n, job_e and done_a are indexed by canonical physical job and row block.

For group of J adjacent same-expert jobs, units U and fixed band32, let local t=v-group_base*U. Map physical job to group_base+(t//32)%J and row block to (t//(32J))*32+t%32. Inverse is group_base*U+(rb//32)*(32J)+(job-group_base)*32+rb%32.32 divides both512 and256; singleton groups are identity. Mapping uses only early routing histogram/job metadata, available after its __syncwarp; it must not read ent/pos before position_pass. Initial copy priming occurs before that pass.

## Publication and progress

Original helper credit (4188) is all-lane __threadfence, __syncwarp, then lane0 atomicAdd(done_a+j,n). Preserve this exact publication chain. PhaseB compute uses acquire polling and generic __ldcg of intermediate activations; it does not feed a TMA read here. No new proxy-fence elimination or substitution is justified. Existing proxy fences protecting shared weight-slot generic reads against async refill remain unchanged.

Naively retaining jcred-on-job-change is correct but adds a fence on every transposed band. Owner proposes one credit at each physical job's LAST virtual occurrence within the CTA's A interval. This is sound under the following exact count/last specification. Clip group-local x into[0,J*U], define P_i(x)=floor(x/(32J))*32+clamp(x%(32J)-32i,0,32). For the CTA interval[a0,a1), credit count=P_i(end)-P_i(start), where endpoints are group-local clipped values. When count>0, k=P_i(end)-1 and last=group_base*U+floor(k/32)*32J+32i+k%32. Publish only at that last v. Every store for this helper's credited job is then in its prior program order. Zero-count jobs must not create a credit or a wait.

The current helper flushes pending A credits BEFORE waiting for its first B partial (4230), because compute B waits for all512 A credits. Moving that flush behind the partial wait creates a direct cyclic dependency. A replacement with analytic last-occurrence credits can have no pending accumulator only after proving every positive CTA/job count publishes by its last A unit. Compute warps and helper must retain identical q/phase progression and handshake counts.

All phaseA work is independent of other CTAs' completion; helpers can publish all their A counts before waiting for B. Under the existing one-resident-CTA-per-SM grid/resources, this permits every physical done_a to reach512, breaking the B wait. Permutation bijection preserves that argument if no new cross-CTA A wait or delayed-after-B publication is introduced. The existing final grid barrier must still precede done_a reset; no counter may be cleared early.

## CPU evidence and remaining gate

`tools/test_skinny_band_progress.py`:3tests PASS. Covers bijection/inverse for512/256 units, empty/single/multiple expert runs; CTA partition credits total512; and analytic count/last against explicit enumeration for J1/2/3/4/7, grids1/2/7/132. These tests model integers, not CUDA execution or hardware fairness. They do not clear a source patch whose actual conditions differ.

Review the exact forthcoming mapper, count/last conditions and every call site before GPU. Reject missing physical512 credits, changed slot parity, premature helper waits or modified fences. Quantify additional xr register reloads when physical jobs alternate; this is a performance cost, not a correctness defect. Updated local64bank duplicate counts are routing evidence only, not traffic proof. Earlier four-bank zero samples and census conclusions must be scoped accordingly. No speedup projection is established here.

## Actual implementation review addendum

Reviewed `../okc-today-moe/tools/moe_skinny_band.py` and its generated pure artifact against the exact pins above. Owner's two CPU tests rerun PASS. All issue/compute/helper mappings agree; `position_pass` takes off as const and leaves cnt unchanged. The physical-job last-credit condition is uniform across all32 helper lanes, so the existing all-lane threadfence/warp/leader-atomic publication remains converged. Every positive CTA/job A contribution is published by its last A unit, leaving no pending credit at the first B wait. Physical inter/ybuf/ss indices and q-based ring/barrier phase are unchanged. No source blocker found. Max32jobs per expert at M256;512/256 units divide band32; intermediate integer products remain bounded.

This clearance is source-only. Owner reports local sm90a140→130registers, zero spills/stack, unchanged16barriers but408extra static instructions. Actual H200 resources remain a runtime gate. Global CTA/job incidence and hence total credit fences can increase even though each CTA/job credits once; xr reloads also increase. These are material performance costs, not correctness defects. No resource-equality or speed claim is made here. Exact generated upload hash is owned by implementation agent/root; no H200 dispatch by reviewer.

Exact reviewed worktree bytes: `tools/moe_skinny_band.py` SHA256 `a2c90b43c47b72eec9e4d553ea27b46b32b2d580678435f7aa9fe5ae0d5d280d`; generated `.build/today-moe/owned_band32.py`301803B SHA256 `a425ebdaf285456faaf1f641f074fd40622e8aaeb7aaa46ad652fce5216223eb`. Source-scope CPU test proves AST outside `_SKINNY_BLOCK` unchanged, and unchanged occurrence counts for mbar_try, named_sync, named_arrive, fence_proxy_async, __syncthreads, bf16r and mma_bf16. Manual diff confirms native math bodies, precision operations, barrier operands/counts and shared release fences remain unchanged; only coherent unit decode and credit scheduling are replaced. Operation counts alone are not a proof; the publication/progress argument above covers the changed credit placement. This is clearance of these bytes only, not a different cooperative repair or future harness.

## Cached representation review (after direct band32 loss)

Reviewed transform `tools/moe_skinny_cached_band.py` worktree SHA256 `57e7470e8a3fe8c71a9b3e49d0141235fc424bcf0aacf265db80cf53fdd86fcd`, generated302476B `owned_cached_band32.py` SHA256 `71e9a5b0b1e14e984928f740a9998e9f3901ca50760d46e70b72d5f32eba396b`. Owner's two cached CPU tests rerun PASS. This caches the SAME band32 mapping and credits, not a new band sweep or an explanation established for the prior loss.

For M<=32: routes<=256, active<=32, jobs=sumceil(count/8)<=floor((256+7*32)/8)=60. Actual G>=132 gives perCTA at mostceil30720/132=233A andceil15360/132=117B units, total350. The350-u32 array is1400B (overall struct alignment may add padding); physicalA v<30720, credit<=233, so low16/high16 packing is safe. Empty ranges never enter the fill or reads. Host launch checks actual sms>=132 and p.M<=32, not only GPU model.

Before parallel helper fill, __syncwarp publishes lane0's range stores; the second __syncwarp publishes all schedule entries before lane0 primes copies. Existing CTA __syncthreads before compute publishes the table to consumers. Metadata remains immutable. All four issue/compute/helper physical-map reads mask low16; only A helper reads high16 credit. No new asynchronous wait, changed ring q phase, altered math or removed fence is present. Existing last-occurrence publication proof applies unchanged.

Source clearance applies only to the bounded selected variant. Its guard affects every generated skinny launch, so M128/256 comparisons must use the untouched control module, not assume a cached large-shape fallback. Additional shared memory and owner's locally reported128registers/zero spills still require actual runtime resource/occupancy checks. Neither local instruction reduction nor passing CPU bounds demonstrates an H200 win. No GPU dispatch by reviewer.
