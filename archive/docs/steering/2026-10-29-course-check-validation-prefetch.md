# Course check: keep the release, validate overlap, isolate prefetch

Evidence snapshot: pushed `main` at **b41cb8dd58fd08c7c6a02f2adac8ce078d3b698a**.
That head is dated October 30, 2026 at 02:19 UTC / October 29 at 19:19 PDT.
The head is unchanged from the preceding status answer. This is a decision update, not a new benchmark result.

Scope: repository evidence review, a targeted check of NVIDIA documentation, and documentation publication only. No GPU run, compiler run, code change, tolerance change, live quota query, or scored submission was performed. No live rank is claimed. Unpushed and in-flight work is outside this snapshot.

## Decision

**The project is broadly on course. Keep the fused/hybrid architecture and the clean 2.014x release. Adjust the next experiments toward layer-context effects and validation quality, rather than another 32-token MoE arithmetic or scheduling sweep.**

The next bounded work is: validate and integrate MoE mode 51; compare the existing MoE PDL prefetch hints against their absence while keeping PDL itself unchanged; and finish a stage-level M1 correctness diagnosis. These tracks need not block each other.

The new consideration is not another architecture family. It is that an insufficiently discriminating replay test can both block useful overlap work and reject a useful incumbent variant, while a misleading component attribution can send the next experiment to the wrong kernel.

## 1. State and evidence

| Item | Evidence | Status at this snapshot |
|---|---|---|
| Clean layer release | `3fbb3ad027e84c6b`, 2.014x, CI 1.984-2.045, 13/13 passed; source `2cee446`, file SHA256 `b9c23bda16bb9d6311f025062e415841c67b16284737172badaa190fdc506a49` | Recorded scored incumbent; M1 code/build excluded |
| Mode 51, final-pass sums of squares from 64 tokens up | `c6b0e500c3a84c9a`; held-out seeds 29-34; layer-routing means against mode 47: +0.0 / -3.9 / -3.9 us at 32 / 128 / 256 | Component gate passed; shippable build checked locally, not yet judged as a standalone release or integrated into the recorded layer |
| Whole-layer public comparison | `86ced49f092a48d8`, `f5ee8632e1f64bf2` | Public lead repeats at verify 1 and decode 32; profiled boundary attribution remains a causal hypothesis |
| Direct 32-token MoE comparison | `1fbb9fbf524f4ed7`, `209cf29bf91c4ea2` | Public leads by only 1.5-1.9 us on the tested layer-like routing; not the earlier 10-14 us inferred from a layer profile |
| Tail-split modes 52/53 | `209cf29bf91c4ea2` | Lost at every tested size/routing; not promoted |
| M1 overlap validation | `6372154fcd7a4fa9`, `3e2aa83358dd4f60` | Off and on both exceed the current replay bound on some inputs; first divergent intermediate and full ordering argument remain unresolved |

Sources: [release manifest](../../../docs/release-manifest.md), [layer attribution report](2026-10-30-layer-release-attribution.md), [layer run log](../../../docs/runs-decode-layer.md), [MoE run log](../../../docs/runs-moe.md), and [rejection register](../../../docs/ruled-out.md).

These are different evidence classes. A passing component gate is not a new layer score. A same-process kernel comparison on generated layer-like routes is not a replay of the exact routes and cache state from a complete-layer profile. A profile's exposed-time allocation is not a counterfactual measurement of removable time.

## 2. Correct the remaining 32-token attribution before optimizing it

The layer attribution report estimated 10-14 us of MoE work deficit at decode-32 sliding from a profiled replay. The subsequent MoE-owner measurements found only 1.5-1.9 us on layer-like routing. Both observations may be real in their own contexts; the difference does not prove either a slow MoE algorithm or a prefetch defect.

The useful question is now: **what changes when this MoE runs inside the actual layer?** Candidate explanations remain routing/job counts, the selected skinny/fat policy, the PDL preamble, cache state, preceding work, and profiler attribution. Do not extrapolate the generated-route result to every possible input or replace it with a hard-coded shape winner without further evidence.

The latest MoE measurements are enough to stop another generic tail-split sweep. Modes 52/53 used fixups to split the final phase-B claims; they increased cost and did not shorten the measured tail. This loss is scoped to that implementation. It is not a reason to abandon the fused architecture.

The older README summary's 10-14 us claim is historical profile attribution, not the current measured isolated-kernel gap. This report supersedes that priority assignment, while preserving the older report as evidence history.

## 3. First new performance test: remove only the existing prefetch hints

### Why this is justified

`tools/dl_moe_inline.py` shows the fat MoE PDL preamble hinting the first unit, comprising 16 stages, of every local expert: an address footprint of approximately **18.9 MB at 32 experts**. The skinny preamble hints approximately 4.7 MB. The 32-token MoE study had about 8.5 active experts on average for its layer-like routes.

Those facts motivate a test; they do not establish that all hinted bytes are transferred, or that the hints are harmful. Actual prefetch traffic and contention were not measured. See [the inliner](../../../tools/dl_moe_inline.py) and the MoE log's "Phase attribution at 32 tokens" section.

### Bounded comparison

Use the clean mode-47 release as the fixed control. Compare the existing fat preamble with a diagnostic variant that skips **only the L2 prefetch hints**. Keep the same PDL launch attribute, route kernel, mbarrier initialization, wait, fences, MoE variant and numerical operations. Do not toggle all of PDL, and do not also integrate mode 51 in this first comparison.

Hold the prepare-time policies fixed across arms for mechanism attribution. Use identical full-layer input sets, both sliding and full at decode 32, seeded randomized order, a duplicate/null arm, and enough independent seeds to reveal routing dependence. Record local expert counts, per-expert routes, job counts and the selected kernels. Where practical, reuse captured current-layer MoE inputs for an auxiliary isolated comparison rather than relying only on a similar synthetic distribution.

Measure complete-layer paired time, and the route-to-MoE chain as a diagnostic. Profile several representative replays instead of attributing a whole budget from one. Include 128/256-token regression controls. A later shape-dispatched choice may be justified, but only after evaluating held-out routes.

**Suggested stopping rule:** one diagnostic check, then a held-out confirmation only if the whole-step change is larger than the measured null and at least roughly 1 us on a targeted decode-32 shape, without an offsetting weighted-load loss. If the difference is null, close prefetch as the explanation for the large discrepancy. If hints help, keep them. If removing them consistently helps, test that isolated change as a layer candidate. A smaller hint footprint is a second experiment only after this first result provides a mechanism.

This is a regression ablation of an existing MoE preamble, not a reopening of the failed generic projection side-stream prefetch experiment. No active-expert information may be consumed before the dependency wait merely to make the hint selective.

## 4. Highest-leverage correctness work: distinguish numerical variation from stale data

### What the data supports

The M1 report records repeated-input off/on equality in its tested verify-1 full trials, while fresh-input runs exceed the current 1e-4 bound with both M1 off and on. It also records two unused MoE variants rejected by the existing replay gate without M1. This supports the hypothesis that the gate is sometimes testing reproducibility more strictly than the baseline can supply.

It does **not** identify the first source of variation or prove M1 race-free. NVIDIA's floating-point documentation explains why different accumulation order can change rounded results, but that general fact is not a diagnosis of this kernel. Likewise, observing no error above 1e-2 is not proof that every stale read would exceed 1e-2: a stale subset of a tile, one element, or a low-amplitude update can produce a smaller aggregate difference.

The potential performance issue is also real: if a useful variant is rejected during prepare, a scored run can select a slower fallback. The records show unused variants being dropped; they do not show that sk8w3s itself has been dropped or that a specific score loss was caused by this mechanism.

### Proposed diagnostic, not an approved tolerance change

Keep M1 out of the release. Preserve the current bound and all failures in the logs. Diagnose separately:

1. Retain intermediate outputs around attention, o, the attention boundary/front, MoE and the final boundary. Use fixed captured numerical inputs to isolate each boundary and locate the first divergent intermediate, comparing eager/replay and off/on. Do not assume an off/on distribution match alone is sufficient.
2. Build a small deterministic producer-to-consumer boundary test with changing input generations. Exercise the actual dependent wait and TMA input-read path, then check the bytes/values actually consumed after the transfer against that replay's expected input. Test immediately adjacent queued replays and reuse, not just synchronized individual launches. Diagnostic instrumentation can change scheduling, so keep a production-path replay/reference check alongside it.
3. Test the test: feed deliberately constructed stale observations, including a previous-generation tile or a small subset of elements, into the diagnostic comparator. It must reject them. This is an offline/diagnostic negative control, not permission to ship an intentionally unsafe synchronization path.
4. Use an independent numerical reference with the required BF16/FP8 boundaries for accuracy, and workspace/counter invariants plus the boundary test for freshness/reuse. A deterministic debug reduction can help localize atomic-order variation; it need not become the timed production algorithm.
5. Complete the written generic-write to async-read ordering argument for the PDL/TMA boundary, including compiler ordering, completion and buffer reuse. Tests do not substitute for this argument.

Only after these results should the coordinator propose a validation-policy revision, with fixed acceptance criteria, held-out inputs, false-rejection evidence on the control, and negative controls showing bug-detection sensitivity. **The existing requirement for explicit approval of a production validation-policy change remains. This review grants no such approval and changes no threshold.** Do not retry until a passing sample appears, or adopt 1e-2 solely because it exceeds the observed maximum.

Log a compact per-variant manifest at the end of prepare: checked, rejected and reason, fallback chosen, source digest and policy. Distinguish numerical/reference failure from replay disagreement, non-finite output and workspace/synchronization failure. Do not turn a failing gate into a silent acceptance.

## 5. Continue harvesting mode 51 independently

Mode 51 passed its pre-registered component gate with approximately 3.9 us lower means at both 128 and 256 tokens on layer-like routing, and no mean penalty at 32. Validate the shippable build, then port it in isolation against the clean mode-47 layer. Keep M1 absent and do not combine the prefetch ablation with this comparison.

Preserve mode 47 as a candidate/fallback; check numerical boundaries, mixed-size queued replays, scratch reuse, PDL source parity and actual selected modes. Promote on repeated component and weighted-layer evidence, not merely a higher headline reading. The expected effect is incremental, not a new architectural step. No new scored result is claimed here.

## 6. Only after M1 clears correctness: test useful overlap, not maximum overlap

The public-source analysis says its attention/projection kernels trigger dependents after their main loops, whereas the experimental M1 attention trigger is at entry. Earlier eligibility is not automatically better: the secondary may compete for registers, shared memory or memory bandwidth while the primary is still doing useful work. NVIDIA documents PDL overlap as opportunistic and warns against requiring concurrency for forward progress.

This is a scheduling hypothesis, not a diagnosed defect. Once correctness is established, test the o boundary and the front boundary separately before their combination. Keep the existing >=1 us whole-step verify-1 gate as the first performance decision. If an early-trigger arm shows substantial overlap but no step saving, one bounded later-trigger comparison is reasonable; change trigger timing alone, not ring depth and geometry simultaneously. Do not extend the scheme to the unpooler until an earlier boundary has demonstrated a net step win.

The profiled 7-8.5 us boundary budget is a lead, not a promised speedup or an amount that can be added to other overlapping savings.

## 7. Execution order and ownership

| Track | First action | Decision gate |
|---|---|---|
| Integration owner | Shippable mode-51 check, then isolated clean-layer candidate | Required correctness/stress, visible policy, paired weighted-layer evidence |
| Layer/performance owner | Current MoE prefetch hints versus no hints, otherwise identical | Full-step effect above null on held-out inputs; stop if null or harmful |
| Correctness owner | First-divergence localization plus deterministic boundary freshness test | Ordering argument, reference accuracy, negative controls; approval before changing production gate |
| Coordinator | Preserve release, query live quota, reserve confirmation capacity | No new architecture sweep without a newly measured bottleneck |

These are proposed allocations, not a live quota statement. Do not run overlapping edits in the same problem directory or replace the release with a diagnostic build. The existing submission authorization rules still apply.

Park the retested staged pair, modes 52/53, generic projection prefetch, and further unguided small-token arithmetic/barrier sweeps. Reopen only with a named mechanism that changes the prior comparison. No broad external architecture search is needed to execute this round.

## Sources and unresolved points

Repository evidence is pinned to the commit at the top; relative links will advance with main, so use that commit for reproduction.

- [Clean release manifest](../../../docs/release-manifest.md): job 3fbb3ad027e84c6b and file identity.
- [Layer report](2026-10-30-layer-release-attribution.md): M1 diagnostics, replay-gate risk and profiled boundary attribution; jobs 6372154fcd7a4fa9, 3e2aa83358dd4f60, 86ced49f092a48d8 and f5ee8632e1f64bf2.
- [MoE log](../../../docs/runs-moe.md): gate c6b0e500c3a84c9a; direct 32-token attribution 1fbb9fbf524f4ed7 and tail-split diagnostic 209cf29bf91c4ea2.
- [MoE inliner](../../../tools/dl_moe_inline.py): actual fat/skinny prefetch footprints and preserved PDL preamble.
- [Rejection register](../../../docs/ruled-out.md): failed blanket PDL, projection prefetch, staged-pair and split-tail experiments, plus M1's unresolved validation status.
- [NVIDIA CUDA 12.8.1 Floating Point and IEEE 754, Operations and Accuracy](https://docs.nvidia.com/cuda/archive/12.8.1/floating-point/index.html#operations-and-accuracy): accumulation order can change results; does not establish the source of this project's differences.
- [NVIDIA CUDA 12.8.1 Programming Guide, Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization): independent work before dependency synchronization; opportunistic overlap and progress cautions. Not an independent proof of the project's TMA proxy chain.

Unresolved: exact current-input layer versus isolated-MoE discrepancy; actual prefetch traffic/cost; first numerical divergence; M1 ordering proof and validated speedup; preparation-gate false rejection rate for useful selected variants; mode-51 layer benefit. No explanation is declared proven by this report.
