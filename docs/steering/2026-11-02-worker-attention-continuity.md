# Isolated attention continuity: implemented offline, target gates open

November 2, 2026. Allocation/evidence base **48111ace**; incumbent normalized-LF
ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab,
public job607eb52bdaf24160 and unchanged-source fresh cc474883117947c0.
This resumes the latest checked-in captain allocation, superseding the stale chat
handoff about the already archived RMW and SiLU tasks. No new measurement occurred.

## Outcome and exact implementation

One isolated scratch implementation survives source/CPU and the fixed secondary
compiler screen. **It has no measured speedup or target numerical qualification.**
The incumbent is unchanged. Captain owns independent review, index and any later
allocation; no target harness, GPU/Driver/challenge CLI, public/scored action or
subagent was used in this allocation.

Deterministic generator: `tools/attention_continuity.py`. Runnable generated layer
file is [candidate.py](evidence/2026-11-02-attention-continuity/candidate.py),
**872309 LF bytes**, SHA256
**5c2de030422f612e37ff186a88e8441cbad58c8db677bb9e273347e37bf90413**.
Original attention literal22c116a45f1817fbaf27a1cc3656380affad49d2cb2367dd23aac83f20b7cef5;
treatment literal4efb67147807327c18cea0554dd09fa733d0d686c8700c707d047205a528c8e0.
Both complete literals and the exact [five-change native diff](evidence/2026-11-02-attention-continuity/native.patch)
are archived. No external implementation was copied.

The existing literal remains intact in the full file. Five readable, asserted
replacements derive the second literal; no compression or obfuscation, notices
removed or duplicated 192KB literal is needed. A separate uniquely named module
is retained in the prepared state. The actual `_decode_attention7` binding selects
it only for full-context B16/B32, L1/L8, P1, Hq16/Hkv2/D64. Other shapes use the
original module. Mutable attention scratch remains explicitly owned by that state.
There is no global selection setter or modified production default.

The five native changes are CONT=1; restore the original L1 half-merge condition;
give that handoff a disjoint tail address; reserve its4608bytes for L1-flat;
and add a uniform CTA join after once-per-launch barrier initialization.
Inverse reversal restores the entire original CUDA literal byte for byte.
The merge still writes all18FP32 values per lane and performs the original
max/rescale/add expressions, ordering and global contribution count. L8 arithmetic
is untouched. F2/S2/PTR, page-index policy, stages, producer counts and numerical
switches are unchanged. There is no17-value/rescaled standalone merge substitution.

## Actual dispatch and compiler evidence

Six CPU tests pass, including execution of the generated source derivation, actual
`_plan7`/`_launch7` and final `_decode_attention7` binding for all12scored cells,
unchanged argument tuples, per-state module identity and fallback shape selection.
Wrong incumbent identity and missing native anchors reject. Inverse native
reversal checks every other CUDA expression/instruction, not just selected math
substrings. [Summary](evidence/2026-11-02-attention-continuity/summary.json) records
all12selected routes and [CPU output](evidence/2026-11-02-attention-continuity/cpu-tests.txt).

Target full cells keep264flatCTAs,1copy,nc1,cluster0,PDLoff. Decode keeps w0=1024,
noncooperative; verify keeps w0=1135,cooperative. The original launch function and
plan are not patched. The prepared selector is additive; its module can require
an additional compilation during preparation, whose target cost is unmeasured.

Fresh fixed control/treatment builds of exact `<1,true,0>` and `<8,true,0>` use
installed NVRTC13/PTXAS13 sm90a. No sweep or compiler installation. Complete ordered
target flags remain `-DOKC_QDIRECT=0 -DOKC_PK2=0 -DOKC_W0ENABLE=1 -O3
-DOKC_SM90=1 --expt-relaxed-constexpr`. NVRTC omits its unsupported NVCC-only
`-O3`/`--expt-relaxed-constexpr`; PTXAS explicitly uses-O3. This is secondary
device-prefix evidence, **not exact host-NVCC equivalence**. Virtual standard
headers, CUDA runtime include, exact FP32 INFINITY declaration and constant-only
sizeof witness are recorded in each manifest. No selected device body is removed.

| Selected instance | Control -> treatment registers | Barriers | Stack/store-spill/load-spill bytes | Threads | Dynamic shared control -> treatment |
| --- | --- | --- | --- | --- | --- |
| L1-flat |128 ->128|3 ->3|0/0/0 ->0/0/0|256|62464 ->67072B|
| L8-flat |80 ->80|16 ->16|0/0/0 ->0/0/0|384|114688 ->114688B|

All four compile/assemble calls pass. Exact adapted source, PTX, CUBIN, logs and
hashes are in the four evidence subdirectories. A constant-only layout witness
reports sizeof(Smem1)=61440 and sizeof(Smem8)=113664, unchanged between arms.
The L1tail starts at61440, occupies4608bytes and is16-byte aligned; the existing
1024-byte allocation cushion covers the runtime base alignment. Nonflat shared
allocation and layout are untouched. L8 receives no tail.

Conditional shared residency accounting includes1024reserved bytes/block: two
L1treatment blocks require136192B, two L8blocks require231424B, below233472B
per-SM shared capacity by97280B/2048B. This is arithmetic under that capacity and
reservation assumption, not an H200 occupancy/admission query. Registers likewise
do not prove runtime occupancy, particularly with warp-role redistribution.
Original launch-side occupancy/cooperative guards remain in force. A target loss
of residency could change effective coop/weighted placement and must be treated as
a failed comparison/control gate rather than silently changing launch choices.

Initial offline adapter failures were unsupported NVCC flags, NVRTC builtins
library search path, and missing virtual host declarations/namespace-extract
boundary. They were corrected without changing treatment arithmetic or geometry.
Both final controls were rebuilt with exactly the same adapters. No earlier
adapter failure is described as a kernel correctness or performance result.

## Lifetime and progress argument, with limits

| Shared region | Writer / last reader | Reuse event retained |
| --- | --- | --- |
| S.k/S.v raw ring | Producers' cp.async; consumer MMA/patch loads | Original full wait, WGMMA waits, writing proxy fence and empty arrivals; cumulative tb/phase across segments |
| Dedicated L1tail | Consumer half1 / half0 merge reader | Original consumer joins around18-float handoff; terminal consumer-only join before next segment |
| S.pidx/S.pbar | Producer TMA / all producer page readers | Existing load_pages_t followed by producer bar.sync2 per tile; every issued x<ntiles consumed before next-segment barrier reinit |
| S.kvq/S.q8s | Consumer query staging / current-segment MMA | Original waits and consumer-only next-segment join; PTR stays off, producers do not write this staging |
| S.kvst/S.kn/S.vn | Consumer new-row staging / consumer patch | Same staged copies, waits, patch/fence and last-reader retirement |
| Full/empty barriers | Thread0 initializes / both roles | New uniform CTA join publishes initialization before role split, even when fpart omits later join |
| Workspace/cache | Original per-request contributors/finalizer | Original reset, cache updates, final contributions and cooperative guard unchanged |

Empty grid-range return precedes initialization uniformly. For nonempty ranges,
the new CTA join follows mbar_fence_init and dominates role divergence. This adds
interthread publication; it does not substitute a proxy fence for a CTA join.
No existing memory-ordering fence, WGMMA wait, full/empty handshake or terminal
producer cp_wait is weakened. The already existing continuity design advances
tb only for nonempty segments; producers wait on the prior empty phase before
reusing a raw slot, consumers wait on the corresponding cumulative full phase.
The new handoff is outside every existing Smem member, so producer prefetch cannot
overwrite it while consumers merge. Consumers retain their end-of-segment join.

PGCONT remains0 and TMAPG/TMAPG8 remain1/0. Each issued page fetch has x<ntiles;
the last producer bar.sync2 follows that tile's page reads. Thus there is no issued
future segment page fetch to outlive per-segment reinitialization, including a
one-tile/partial segment. Empty segments issue none. This reasoning relies on the
unchanged producer control flow, not on matching only the number of barriers.

Sampled CPU tests cover zero/one/short/long/multisegment ordinal/phase sequences,
weighted contiguous partition coverage, and ownership of the new-row suffix.
Local negative events reject early slot reuse, missing initialization publication,
ring/merge alias and page reuse before readers finish. **Those local model
negatives are not mutated-kernel executions or an exhaustive CUDA scheduler
proof.** Target fresh-output, causal masking, BF16 numerical boundaries, replay,
empty/cache/state ownership and resource-admission checks remain NOT_RUN. The
source argument above still requires captain's independent review before any
target decision; CPU mocks do not qualify its numerics or memory ordering.

## Opportunity limits and handoff

The actual benefit is unresolved. Under a contiguous partition each request
boundary crosses at most one CTA: at most15/264 or31/264 CTAs span a boundary
for B16/B32. Most CTAs can therefore receive the scratch/resource cost without
cross-request overlap. Continuity can also alter producer/consumer tail overlap
within one segment, but no exposed-time saving is measured. Profile activity
88.6/171.4/104.9/185.2us in the four full cells is not a savings forecast. Any
whole-layer benefit is further diluted by16:46 full/sliding weighting and the
geometric mean across six loads.

This completes the finite source implementation screen with one concrete
artifact. No RMW, SiLU, pre-gamma56e0c8b or T1three-launch28248eb5 negative is
reopened. Captain can review this exact diff/state choice and choose a separately
bounded target test or stop it. No implementation extension, target harness or
GPU action follows automatically. A later comparison must retain uninstrumented
complete endpoints, fresh retained outputs and the existing every-seed1%/twice-null
whole-six/regression gates. Source/compiler success supplies no promotion claim.

All evidence bytes are pinned in [archive-manifest.json](evidence/2026-11-02-attention-continuity/archive-manifest.json).
README remains captain-owned; this report is supplied for its next index update.
