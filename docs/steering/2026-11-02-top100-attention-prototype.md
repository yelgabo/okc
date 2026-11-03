# Top100 attention implementation pass: specific STOP

2026-11-02. Allocation/evidence base `a705900cbd9432ccaada88c29ef8661718dca038`;
checkout `71eb71c1a67335d704e53d00632b4bf92ae38845`, initial branch `t3/6d331c83`
(renamed externally during the task to publication branch `t3/attention-prototype`),
remote `https://github.com/yelgabo/okc.git`. The intervening commit adds alignment/
reference-access documentation, not an attention implementation. No children,
GPU job, challenge CLI, install, default edit or public promotion. Captain retains
dispatch; T1 and compression workers were not duplicated.

## Decision and hypothesis

**STOP this pass without a CUDA candidate.** The concrete mechanism screened was
carrying the full-context K/V producer ring across request segments, so the next
request's gather can overlap the previous request's merge. This is materially
different from a tile sweep, but it is **already implemented in the complete owned
standalone comparator**, and the layer's dormant switch is not an isolated
treatment. Neither the fresh profile nor prior unmatched standalone/layer results
establish a new favorable whole-call discriminator for it.

The [preregistered screen](evidence/2026-11-02-top100-attention-prototype/preregister.md)
was written before the CPU tool. It specifies affected cells, independent controls,
numerical/order obligations, a prospective matched whole-layer gate and the STOP
condition. This is a source-screen closure, **not a measured loss for continuous
pipelines**, not proof of optimal attention, and not a request for another survey.

## Exact incumbent and actual selected paths

The complete layer file is `muse_decode_layer/cand/g1b4_stage_release.py`, normalized
LF SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
869076 bytes. Its ATT7 CUDA literal is
`22c116a45f1817fbaf27a1cc3656380affad49d2cb2367dd23aac83f20b7cef5`.
Fresh own receipt `cc474883117947c0` uses this unchanged public607 source.

[CPU screen](../../tools/attention_top100_screen.py) executes the **actual extracted
`_plan7`, `_launch7`, `_pdl`** using explicit CPU tensor/module stand-ins. It does
not import the submission or torch. All twelve scored shape categories are
exercised, including tuning overrides. The four target full cells select 264
flat CTAs, one workspace copy, no cluster; D16/D32 select `<1,true,0>`, V16/V32
select `<8,true,0>`. Decode is noncooperative and w0=1024; verify is cooperative
and w0=1135. PDL is off. Capacity is an explicit 132-SM stand-in, not a new runtime
occupancy measurement.

Actual attention build flags are `-DOKC_QDIRECT=0 -DOKC_PK2=0
-DOKC_W0ENABLE=1 -O3 -DOKC_SM90=1 --expt-relaxed-constexpr`; `_att7` sets the
Hopper architecture through `TORCH_CUDA_ARCH_LIST=9.0a`. Do not borrow C++20 or
`--fmad=false` from the separate projection builds. Active source has `OKC_CONT=0`,
`OKC_TMAPG=1`, `OKC_TMAPG8=0`, three/four decode/verify stages, 128 producer
threads and 128/256 consumers. `OKC_XP_NOFENCE=0`. These are source settings,
not freshly compiled resources.

The fresh [hotspot map reproduced offline](evidence/2026-11-02-top100-attention-prototype/hotspots.md)
records attention activities of 88.6/171.4/104.9/185.2 us at D16/D32/V16/V32 full.
They are **activities, not exposed savings**. Historical SGLang attribution remains
`7d42771c3384449d`; no new baseline profile was run. Whole-layer endpoints include
projections, experts and glue and weight sliding/full 46:16 across six loads.

## Source findings that stop the proposed toggle

1. `segment` at layer source lines1184/1389 initializes and drains each segment
   when CONT is false. The outer producer path already has cumulative tile ordinal
   plumbing and a final drain for CONT; this is not a missing design invented here.
2. At line2634, `L == 1 && !CONT` gates the half-merge; its scratch is `S.k`.
   Turning CONT on disables that merge and lets both key halves issue their own
   global reductions. Retaining the merge by simply dropping `!CONT` would let
   consumer scratch alias a ring the producer may fill for the next segment.
   Thus a one-line toggle bundles scheduling with changed reduction traffic/order;
   this is **not** a claim the dormant toggle is itself unsafe.
3. At line2926 the consumer rendezvous changes from whole CTA to consumer-only;
   page-index lifetime, full/empty phase continuity, terminal retirement and query/
   new-row staging need their own review. No fence or wait was removed in this task.
4. Complete current `muse_attention_decode/submission.py` already selects modes49/51
   for these four cells. Its `_ONE_SRC` is
   `3350b5a34aaf9241528a845455c0e2634b29f854b413894aa051dd9d2f547df4`.
   The producer loop in `attn_one.cu` lines1417-1526 carries slot/phase across
   requests and retires the ring after the outer loop. The prior complete comparison
   explicitly records this difference in `docs/runs-attention.md:2849`. Reimplementing
   it again is not a new standalone candidate. Different soft/coherent inputs and
   BF16 output boundaries still prevent claiming its historical timing as a layer gain.

## Complete alternatives and scoped objections

Reused the completed full-file/backend investigations, including preparation,
dispatch and actual flags, rather than copying isolated donor loops. The native
standalone is the directly inspected complete alternative above. The existing
SGLang FA3 and fused-preparation path remains covered by the alternate-engine
report: page-size-one native replacement already happened. No new FA3 port is justified.

The pinned keell0renz `8e30e1be...` complete WST/VP2 comparator was measured with
`ZH_SCH=0`; its default `ZH_SCH=1` approximate math must not be credited as a legal
gain. Exact WST verify16 full lost in owning `4862f8e0dc8a4b5a`. Later BF16-matched
VP2 `c79ddddb840a4fb7` improved its batch-one sliding cell but failed the frozen
whole-score/noise gate; that is outside these target cells. This pass reuses those
pinned complete-source audits and receipts; it does not claim a fresh donor download
or fresh search. November2 competitor/ruww intake supplies no new stronger complete
attention base. Public source remains measurement-only; no external code adopted.

Prior closures stay exact: XPIPE2 cold loss `5575e7d15d3e49a8`; one-CTA variant loss
`318ae6fc3c3746ce`; per-request QK2 `fe75e566c1ae4d51` has a host-budget failure
separate from its retained private comparison; wait trace `3f44f891d0ef47db` fails
perturbation admission. None is reopened by activity durations. Existing cluster
losses and the unresolved positive VP2 bundle likewise remain scoped to their
actual implementations. The [next-boundary report](../../archive/docs/steering/2026-11-01-attention-next-boundary.md)
already records the counter-access restriction; do not retry that preflight unchanged.

## Validation, next owner and remaining gates

[Machine evidence](evidence/2026-11-02-top100-attention-prototype/source-screen.json)
pins both complete sources, flags, macro values, launch arguments and seven owning
receipt IDs/hashes. CPU PASS: twelve actual layer launcher cases, four actual
standalone plan cases, source-contract witnesses, and a changed-source rejection.
Initial extractor failed on literal `dict(...)` configuration syntax; the corrected
extractor only admits keyword-only literal dict construction. That failure was a
CPU harness issue, not an attention result. No compiler run occurred; nvcc/clang++/cl
were not found on PATH, which is not an exhaustive machine capability claim.
No new registers, spills, cubin, H200 numerical result or speed is claimed.

Priority: return this attention allocation to the captain; do not spend shared
quota on the CONT toggle or duplicate existing T1/compression work. No new scored
submission or default selection follows. Parent can archive this source-only closure
in its shared ledger; this isolated worker did not edit captain-owned run logs or
`docs/ruled-out.md`.

Reopen only with **new matched segment-boundary evidence** or a source-specific
redundant dependency/transfer that survives the existing comparator and loss ledger.
An eventual continuity treatment must preserve the decode half-merge in disjoint
scratch, justify scratch/resources and page-barrier lifetimes, and isolate all other
changes. Compile exact selected instances and independently review order/progress;
then captain may allocate the preregistered fresh-output, BF16-boundary and whole-layer
comparison. No component-duration sum substitutes for its >=1%/two-null every-seed
gate. Missing exposed boundary cost and benefit exceeding integration cost are the
decision gaps, not a shortage of tile variants.

## November 2 publication addendum: independently completed profile priorities

Read the parent's new evidence commit
`80eef4dd2f4bdce0ed60bd5dc2c21ea67661c56e` directly with `git show`:
`docs/steering/2026-11-02-top100-profile-priorities.md`. It independently reproduces
the four routes above, including unequal verify w0 and PDL off. No attribution
experiment may silently equalize those splits or change the grid.

Use the actual endpoint weight: `C=(46*t_sliding+16*t_full)/62`, then
`factor=product(C_control/C_treatment)^(1/6)`. A full-cell saving therefore enters
its load at **16/62**, not at full weight. The report prioritizes the existing
longer-token expert stream over attention on broad weighted opportunity, subject
to actual exposed savings and implementation evidence. This supports returning
this nonisolated attention lane; it does not establish either stream's speed.

Any reopening starts with **uninstrumented complete attention and layer endpoints**,
same source-selected grid/split and fresh retained outputs. The failed wait trace
`3f44f891d0ef47db` remains inadmissible as causal timing evidence. No new observer,
timebox extension, candidate, dispatch or weaker acceptance rule was introduced.

## Parent acceptance, November 2

Integrated `2803aa0cdf5ea8b882e93918f6ad5ee9c92b9fb8` via merge `b092a07cdf2b1ae81229257c652e1f17e29f69da`. Parent reran `python tools/attention_top100_screen.py --output <temporary-json>`: twelve actual layer routes, four standalone plans and changed-source rejection passed. Also inspected the earlier flat-kernel source comparison in `docs/runs-attention.md` and the attention-next-boundary report. The earlier standalone implementation explicitly used separate query-region scratch for its half-merge when its producer ring remains active. This reinforces that flipping the layer toggle is not an isolated scheduling treatment; it does not disprove a deliberate integration using disjoint scratch.

Accept the bounded source-screen STOP and record it in `docs/ruled-out.md`; no fabricated run-log result is added because no target job occurred. Do not reject every possible port merely because the mechanism already exists in standalone. A distinct safe integration could be a future candidate, but would need to earn its priority against the current expert prototype and preserve the complete output/prepare/dispatch behavior. This pass supplies neither its implementation nor matched benefit, so no new attention allocation follows automatically.

The next priority remains the already allocated longer-token expert implementation and existing T1 work. Reconcile their concrete artifacts before launching another speculative attention cycle. Private layer quota remains exhausted in the latest retained15:30UTC receipt; no fresh query or scored substitution was made here. The top100 objective stays active and unachieved.
