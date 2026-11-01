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
