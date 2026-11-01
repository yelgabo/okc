# Tiny Family G expert routing/weight reuse: scoped closure

November1. CPU/source audit only, no CUDA candidate or H200 trace/dispatch.
Root requested one bounded short-expert question before reset. The source shows that
weights are already coalesced by unique local expert at T1/T8; the remaining repeated
metadata has no measured exposed >=1% six-load budget. Close this scoped investigation.

## Owning source and exact route

Pinned root `.build/today/g1b4-stage-retire.py` SHA256
`295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491`,
MOE_CUDA_SRC `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`.
This is the pure stage body underlying the public #303 release; startup wrapper differs.
No source from another entrant was copied into production by this audit.

`_fork_native_moe` selects tiny for M<=4, tiny2 for M<=8, both on the main source:
NW8,TINY1,FAST1=2,WS1,WSB1,WSBL2,WSBLK2(T1)/1(T8). T8 also FINPF1.
Neither tiny define list enables SPLIT. Mid16 alone enables the split path; do not
extrapolate this closure to T16 or larger. Both selected tiny paths are outside WGS.

Source anchors within MOE_CUDA_SRC: route319,ring_issue546,skinny_ring_ws1455,
ws_issue1557,ws_helper1564,skinny_split1642,moe_fused2382/FAST1 branch2400.
The final Python local-ID bridge subtracts the actual local start and uses -1 for remote
IDs; native p.lo=0,nloc32. Global IDs are not reduced modulo32.

## Weight ownership is already one read per unique local expert

Unique top8 per token implies a local expert count c_e<=M. For T1/T8, c_e<=8=SK_T,
so each touched local expert creates exactly one skinny job, regardless of how many
of the eight tokens route to it. `ring_issue` addresses `(expert*RB_phase+rowblock)*UNIT_BYTES`.
RB_A512,RB_B256,UNIT_BYTES36864 (32768codes+4096byte scales). Contiguous per-CTA
unit ranges cover the total job-rowblock domain without overlap. Dynamic tail is off
in these tiny builds. FAST1 preissues the first two units of the same CTA range and
records `S.preissued`; helper starts at that index, rather than copying them twice.

Each tile is loaded once into shared memory for its owning CTA. The mma.sync tile
computes all <=8 routed rows from that shared weight tile. Other CTAs own other row
blocks of the expert; they do not reread this CTA's tile. Mandatory logical global
weight requests per active expert are `(512+256)*36864 = 28,311,552 bytes`.
They exactly cover gate/up and down codes+scales. Repeated per-token weight payload
available to eliminate at these loads: **zero**. This is a logical request/ownership
proof, not measured HBM bytes or a bandwidth-saturation claim. Activation register loads
can repeat across CTAs that share a job; these small X/intermediate arrays can be cached,
and this audit has no exposed activation-load attribution.

## Retained natural-route evidence and limits

`sc6_compare.05e51ddeb7014dd9.log` SC6CORE records (M,seed,localroutes,activeexperts,maxcount):

|M|Seed|Local routes|Active experts|maxcount|Required logical weights|
|---|---:|---:|---:|---:|---:|
|1|901|2|2|1|56,623,104B|
|1|902|3|3|1|84,934,656B|
|8|901|20|11|8|311,427,072B|
|8|902|15|7|8|198,180,864B|

These are the retained standalone router's actual aggregate counts, not newly measured
#303 layer inputs. Full per-expert vectors are absent; do not invent them. The tiny paths
in that SC6 diagnostic had the dedicated M8 issuer disabled, so its31–39us/T1 and68–95us/T8
timings are not exact #303 tiny-path causal measurements. Its counts demonstrate actual
route reuse with maxcount8. Earlier owned count job53c07507b9fd4f8c sampled M16/32 and
failed the broad pairable-copy screen; that is not evidence of T1/T8 weight duplication.

## Repeated metadata exists, but is already warp-parallel and partly overlapped

Every CTA builds the same local job/position table in shared memory. It loads IDs,
uses match_any/leader popcount, prefix scans32experts, and a second match_any pass
for stable positions. With E8/E64,NW8,CH32, only one/two warps scan routes, rather
than a per-slot predecessor serial scan. The other warps initialize count rows used
by the prefix. FAST1 separately computes a touched-expert bitmask and issues the
first two weight copies before the full routing table/position pass completes.
Thus routing-work reduction is not automatically exposed latency.

At an illustrative132CTA grid (actual Gactual132 in the earlier count receipt), the two
ID-load passes FAST1+route request only8,448B at T1 or67,584B at T8, heavily cacheable;
shared metadata/prefix instructions are duplicated too. This arithmetic does not bound
latency. A one-CTA global metadata design would require a new publication mechanism,
all-CTA/shared copies and possibly a global rendezvous on the critical path. No measured
edge says that replacement saves more than its communication cost. The separately profiled
`_sl_tile_route` kernel is the front router, not this in-kernel expert metadata prologue;
its timing cannot be charged as expert-metadata time.

## Six-load budget and stop

From current public607eb52bdaf24160 medians and separate kernel profiles:
D1 weighted layer/expert=78.551290/33.154839us; V1=115.990000/68.296774us.
Under frozen other-cell times and assuming equal fully exposed absolute savings at
both short loads,1% six-load score requires2.753223us off BOTH. If only D1 changes,
it needs4.552422us; V1 alone needs6.722173us. These are requirements, not projections
or a claim that all expert time is reducible. No current receipt measures that much
exposed metadata time. The owned predecessor-map loss is a different implementation;
it does not establish a universal lower bound for this G path, which already uses match_any.

No code candidate/trace is justified by the requested >=1% credible-opportunity gate.
Reopen only with an independently measured exposed routing/activation/scheduling phase
of that scale, or a concrete new duplicate payload/serial dependency that this source audit
missed. Do not rerun SC6, expert-copy multicast, fixed owner, or a blind tiny knob sweep.

## CPU proof

`tools/dl_g_next_expert_tiny_audit.py` models actual non-SPLIT contiguous ownership.
The printed aggregate fixtures match retained count totals but are explicitly synthetic,
not recovered expert vectors.2CPUtests pass:200physically valid random top8 route fixtures
at grids1/32/132/136, empty, and1..8routes on one expert. No repeated weight tile found.

`python -m unittest discover -s tools -p test_dl_g_next_expert_tiny_audit.py -v`

This proves model ownership/byte arithmetic, not hardware traffic/correctness/performance.
Parent owns report index and publishing. No GPU quota used; production unchanged.
