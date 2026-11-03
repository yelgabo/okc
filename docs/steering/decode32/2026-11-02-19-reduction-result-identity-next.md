# D32 reduction result and exact identity-rescale comparison

November 2, 2026 Pacific. Target remains verified D32 speedup of at least2.3x. D32-REDUCE-01 is terminal and closed. D32-IDENTITY-01 is implemented and locally qualified for one bounded private comparison, with target correctness, cost and performance still unmeasured.

## Split reduction: valid result, stop this treatment

Own private job `bd5776bdfcab42a7`, exact physical source `93a05524b1c10b44ac5bd1ea307a25fdc906db4a617737a579cfae6332fd0d49`, published `0585bd21`, passed13 tests,12 benchmarks and12 profiles. The log confirms `D32_SPLIT_COPIES (32, 1, 2048, 8, 8)` and56 native startup replays. Queue313.898s and worker231.087s are separate. [Owning receipt](evidence/D32-REDUCE-01/status-wait.raw.json), [profiles](evidence/D32-REDUCE-01/profiles/candidate.bd5776bdfcab42a7.json), [recomputed result](evidence/D32-REDUCE-01/terminal-analysis.json), [SGLang hotspot comparison](evidence/D32-REDUCE-01/hotspots.md).

D32 adjusted sliding161.438408us/full304.957678us gives C32=12.305490ms. Raw sliding165.695995us/full310.975999us gives C32=12.597632ms. Its own SGLang C32 baseline22.334400ms gives **1.814995x**. The whole-six geometric score is2.119214x,95% CI[2.087426,2.151002]. This whole-six score is separate from D32 speedup.

| Load | Adjusted weighted time, ms | Own baseline speedup |
|---|---:|---:|
| D1 |4.838919|2.234390x|
| D16 |9.964416|1.953537x|
| D32 |12.305490|1.814995x|
| V1 |7.258185|2.281518x|
| V16 |16.891973|2.157961x|
| V32 |22.171325|2.322337x|

The independent control94ef has sliding157.620500us/full317.105412us and C32=12.324230ms. The candidate's affected sliding endpoint is2.42% slower, while its unchanged full endpoint is3.83% faster. The resulting0.152% lower C32 provides no useful affected-path signal. Seeds and nodes differ, so these comparisons establish neither a causal slowdown nor a deterministic-routing correction. [Strict descriptive comparison](evidence/D32-REDUCE-01/control94ef-comparison.json).

Stop this exact eight-copy performance treatment under its registered rule. Retain the incumbent; do not sweep copy counts or claim the old routing mismatch is resolved. A distinct reduction treatment would need new evidence of an exposed bottleneck or a concrete correctness requirement.

## Identity rescaling: source, compiler and mathematical scope

Actual pickup follows [research's selected full-context mechanism](../2026-11-02-general-hotspot-search-refill.md#2-decode32-exact-identity-rescale-elimination-in-full-decode-attention). `tools/d32_identity_rescale.py` derives enabled/disabled twins from exact public607 `ba9e327c`, native `22c116a4`. The candidate tests computed alpha against exactly1 and skips only sixteen output-accumulator multiplies when every active lane in a consumer warp has identity scaling. Denominator `l_r[0] * alpha + ls`, max/exponent/probability and PV ordering remain unchanged. Producer copies, barriers, fences, final merge and FP8/BF16 boundaries remain unchanged.

The vote executes outside the active-lane branch. L1's consumer region comprises complete warps, and its tile-loop bounds are uniform within a warp. Inactive lanes vote false. The generated native entry rejects metadata outside B17..32/L1/full/page1/Hq16/Hkv2/rope0 and rejects geometry outside264 flat CTAs, one copy, noncluster, existing W0 flags and positive finite representable scale. The Python selector additionally preserves current scale and SM90/132-SM admission. Other five scored loads and sliding keep the incumbent path.

CUDA12.8.1 `sm_90a` device-only compilation succeeds for both twins with128 registers, zero stack/spills,3 barriers,61440 shared bytes,256 threads, three stages and unchanged56/200 producer/consumer register settings. SASS emits `VOTE.ANY` at0x5f10 before inactive divergence; branch0x6160→0x6270 skips exactly sixteen non-FTZ FMULs. The disabled twin's SASS SHA256 `8960896f...` equals the archived incumbent control exactly. Thus the matched instrument can use the existing incumbent module for its two control placements without compiling a second new control module. This equivalence is specific to the recorded compiler/flags, not a target-build claim. [Codegen audit](evidence/D32-IDENTITY-01/codegen-audit.json), [enabled manifest](evidence/D32-IDENTITY-01/treatment-compile/manifest.json), [disabled manifest](evidence/D32-IDENTITY-01/disabled-compile/manifest.json), [compressed exact files](evidence/D32-IDENTITY-01/archive-manifest.json).

Three CPU source/dispatch/arithmetic tests, four instrument tests, two real CPU Torch host-orchestration tests and two receipt-reader tests pass. They cover finite FP32/BF16 identity cases, neighbor/direct-override isolation, actual arm selection,36 rows/144 initial/1152 fresh/144 restored checks/96 transitions, private mutable owners, same-set order-swap rejection, partial rows on injected replay corruption, budget failures and cleanup preserving the original failure. These facades do not execute CUDA, prove CUDA graph lifetime or measure speed. [Commands and exits](evidence/D32-IDENTITY-01/cpu-validation.json). No independent reviewer checked these new changes.

## One bounded target check

Frozen private payload SHA256 `f7a5d1fb7fba3365f52285ef84f7d888e6f4b347280f257833469f85382bef9a`,920106 bytes. Standalone treatment SHA256 `3a1ad278f4b5dcb113257f42a53874121fed21cc8625318c054c92b3021d6398`; native `b599e5b21c3e53cec6cfdd752d7e37ce693ba2003ac116629781888542d0d62d`. [Frozen manifest](evidence/D32-IDENTITY-01/paired.manifest.json), [exact private source](evidence/D32-IDENTITY-01/paired.py.gz).

Hypothesis: identity skips remove exposed full-context consumer instructions at lower cost than their comparison/vote/branch, improving whole-call D32 latency. The main uncertainty is producer pacing and schedule disruption; skip frequency is unmeasured. The latest whole-layer profile has161.2us full attention and84.4us sliding MoE. This small arithmetic treatment cannot be assumed to close the full2.3x gap. CUDA Optimizer retains the separate B4 phase/tail study; this check does not overlap its mechanism.

Reuse the existing complete-layer harness, one parent preparation, four independent mutable states and two controls/two treatments. D32 sliding/full across all three fixed banks runs first, then the remaining30 mandatory rows. Preserve24 randomized cold repetitions, eight fresh replays, exact ordered-route equality, mathematical/cache/generation/guard checks and all-load regressions. `tools/d32_attn_worker_contract.py` remains the gate: every bank must have weighted D32 and full-context savings greater than twice duplicate noise, duplicate stability, and unchanged-cell preservation. No whole-six1% floor replaces this gate. Use `tools/d32_attn_worker_read.py --manifest ... --job ...` against the owning receipt.

Check parent preparation by180s, owners/module readiness by210s, work by280s and serialization by290s, within the unchanged external295s limit. The blocking compiler cannot be preempted by these checks. Emit compact source-pinned progress after preparation, owners and each row; exceptional exits emit collected rows and the original failure. Each capsule is bounded to11000 encoded characters. Partial coverage cannot pass. Host tests establish control flow only; under-budget GPU completion remains unproved.

Publish and verify exact remote bytes, perform fresh shared jobs/quota preflight, record dispatch-started, make one private POST and persist its accepted ID. Follow one own-ID waiter under ignored `.build/`; archive only after exit. Stop this exact treatment on source-policy, budget/coverage, routing, numerical, cache, regression or benefit failure. No automatic observer variant, seed/cap/tile sweep, larger guard or weakened gate. A successful matched screen needs held-out confirmation before integration. Public permissions remain unchanged.
