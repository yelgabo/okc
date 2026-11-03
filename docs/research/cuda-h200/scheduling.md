# Scheduling and locality

Source sections: L2, grouped/persistent kernels, Split-K, Stream-K, clusters and pathological routing.

## Scheduling is already part of the kernel

The selected native route handles both contractions, counts and expert jobs inside its prepared-state callable. MID uses up-to-8-token jobs. WGS jobs reuse weights across up to 64 tokens and select work using their actual job representation. The engine has continuous rings and persistent work distribution. A new global queue is a replacement with acquisition, initialization and tail costs, rather than an omitted optimization.

Expert-major ordering can improve reuse while a global queue can reduce imbalance. A hybrid contiguous range plus stealing is a candidate only when the workload demonstrates both costs. Compare a complete scheduler against the current selected route. Measure both GEMMs, routing, gather, SiLU, norm/combine and synchronization. Changing a prefix or shrinking allocated capacity does not establish that fewer MMAs execute.

[The completed grouped-work census](../../steering/2026-11-02-worker-grouped-engine-work-decision.md) separates useful rows, scheduled padded rows and skipped capacity. Smaller grouped tiles can reduce padding while increasing weight requests. Larger tiles can reduce requests while increasing padded math. Do not assign a runtime saving from either count alone.

## Count requests separately from memory traffic

Repeated logical weight loads may hit L2. Record actual DRAM bytes, L2 sectors/hits and endpoint exposure before treating duplicated requests as removable HBM time. The judge starts each replay with cold L2, but reuse within a call still matters. Read [the owned locality census tool](../../../tools/moe_owned_locality_census.py) and [the previous weight-reuse model](../../../tools/moe_weight_reuse_model.py) with their stated coverage limits.

TMA multicast needs simultaneous consumers of the same tile. Verify repeated jobs, cluster pairing, occupancy, per-rank addresses, barrier phases and cluster lifetime. The October 31 skinny multicast screen, job `53c07507b9fd4f8c`, found insufficient repeated-job coverage on its inputs and stopped before implementation. It did not measure physical traffic or disprove every cluster design. [The exact scope and reopening trigger](../../ruled-out.md) remain authoritative.

## Split-K and Stream-K pay for their reduction

Use extra K partitions only when the M-by-N grid or tail leaves exposed hardware capacity. Price fixup stores/loads, atomic or barrier traffic, scheduler setup and partial precision. A static split count for every expert can hurt already-large jobs. Keep required FP32 partials until the prescribed final rounding.

Our older phase-A Stream-K experiments `098949b28fe84bd1` and `e02f60fec94f45cb` showed that the tested fixups and setup outweighed their balancing benefit. Read their rows in [ruled-out.md](../../ruled-out.md). The result is about those implementations. A lower-cost fixup or a different current loop can change the comparison, but the new trigger must be explicit before allocation.

## Route distributions and alternative engines

Retain real router distributions as the primary decision workload. Supplement with legal uniform, tiny, skewed, bimodal, empty-local and occupied-to-empty cases. Distinct top-k IDs, remote slots and normalized routing weights must match the task. The upload's 256-expert toy counts are illustrative and do not encode this EP8 shard or scored shapes.

Compare a full current alternative, including packing, precision, selected templates and epilogues, before grafting individual ideas. [The Humming assessment](../../steering/2026-11-02-worker-broad-expert-architecture.md) and [the FLUTE assessment](../../steering/2026-11-02-research-flute-engine-screen.md) record concrete format, rounding and net-cost gaps. Preserve attribution and licenses if any code is adopted. A Triton prototype can help test a distinct schedule, but a new backend, old-MMA rewrite or universal grouped engine needs a measured benefit under the same contract.
