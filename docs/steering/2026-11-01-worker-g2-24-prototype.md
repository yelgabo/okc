# Fixed G2/24 prototype: early admission STOP, unfinished source only

November1 Pacific / November2 UTC, 2026; implementation allocation approximately06:42UTC, stopped approximately06:54UTC on captain's explicit installed-header finding. **STOP_FIXED_LAYOUT. No runnable prototype or performance candidate was produced.** This report is the bounded counterexample/unfinished-source handoff, not a source/CPU/compiler PASS or readiness request. No compiler, CUDA API, GPU, challenge CLI, judge check, public/scored/default edit, delegate or alternate design was run. The preceding outgoing seam lane stays closed and is not diagnosed or rescued here.

## Exact starting point and work actually performed

Read [the fixed design and dated attribution correction](2026-11-01-expert-pipeline-next-contender.md). The Chord favorable/unfavorable percentages are author comments without a retained controlled receipt; they were not treated as a speed forecast or a three-stage-only admission rule. Captain explicitly funded one higher-risk fixed G2/24 build rather than scalar scheduling; no several-percent gain was established.

Verified the exact public607 file muse_decode_layer/cand/g1b4_stage_release.py SHA256 **ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**, and its first actual MOE_CUDA_SRC literal **c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71**. Source inspection followed _fork_native_moe -> _ma_ext and _ma_make_scratch, rather than compiling the later dormant alternate literal. Selected H200 flags are -O3, compute_90a/sm_90a, lineinfo, existing NS/FNST/NSF, PRMTSC/SC6W/DQ7 suffixes, and B4FIN for B4. Original MAIN defines NW16. These are inspected source selections, not captured/compiler-validated prototype flags.

An initial append-only generator was written for a separate scratch variant: intended NW8/FT_T24, two compute WGs plus original helper roles, half-unit packed-weight mapping, two-chunk gather, eight-publication parent remap, larger job/flag/fixup bounds, widened range arithmetic, and strict two-CTA cooperative admission. Its intended launch bound was **__launch_bounds__(NTL,2), NTL320**, not the old 576/1 register result. Before executing any transformation or compile, captain identified the omitted reserved-shared term and explicitly stopped this fixed layout. The generator is retained as [generator-unfinished.py.txt](evidence/2026-11-01-worker-g2-24-prototype/generator-unfinished.py.txt), prominently marked **unfinished / never executed / not validated**. Its helper formulas, replacement anchors, entry assembly and protocol changes are NOT claimed complete or correct. No generated variant, native diff, full-layer harness, or emitted source exists. Production and all short/attention/projection/secondary code remain untouched.

## Decisive necessary-condition counterexample

I independently read the installed CUDA13 header and reran captain's bounded CPU witness unchanged under the existing WSL host-validation Python. [Replay output](evidence/2026-11-01-worker-g2-24-prototype/cpu-accounting-replay.json), [scope/pins/status](evidence/2026-11-01-worker-g2-24-prototype/stop-evidence.json). Exit0; source-binding assertions PASS. This witness executes only the retained layout AST function, checks exact official header expressions and does arithmetic; it does not execute NVIDIA's complete native occupancy calculator or any CUDA API.

- Existing design layout reproduction: sizeof(Smem)=23776; aligned ring start23808.
- Fixed three weight rings: 3*18432=55296 bytes. Fixed three X rings: 3*12288=36864 bytes. Dynamic total **115968 bytes/CTA**.
- Installed /root/.cache/okc-cuda130-88/nvidia/cu13/include/cuda_occupancy.h SHA256 **163b8f296ef2a33c799e2dc8c525b37166da9217bcc6a3c1d26b1c10aca5e22b**. SM90 allocation granularity is128 bytes (function at620); the calculation adds attributes.sharedSizeBytes + properties.reservedSharedMemPerBlock at1414, then dynamic bytes and rounds upward.
- Under **reserved1024/static0**, allocated bytes/CTA = roundUp(115968+1024,128) = **116992**. Two CTAs need **233984**, exceeding the campaign's233472-byte H200 SM capacity by **512 bytes**. The shared-memory-only resident block upper bound is1, not the required2. Nonzero additional static shared memory cannot cure this inequality.

NVIDIA's [Hopper tuning guide, shared-memory capacity section](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#increased-shared-memory-capacity) documents a1KB system reservation per thread block. That documentation supports the1024 assumption; **this target's reservedSharedMemPerBlock was NOT queried or measured**. The claim is conditional source/accounting contradiction under the documented reservation, not observed runtime admission failure.

The omitted reservation reverses the original paper fit. At reserved0/static0, the pair uses231936 and nominally fits. At reserved768/static0 it exactly reaches233472. At reserved1024 it does not. At least **256 bytes/CTA less dynamic memory**, with zero extra static memory, would be required to restore this particular necessary condition; authoritative evidence of a different reservation could also change it. Neither is an authorized repair. Rings, header, shape, tile, CTA count, arithmetic and fences were not shrunk or swept to rescue the design. This does **not** rule out all two-CTA architectures.

## Honest completion and resource census

| Selected prototype entry | Device compilation | Registers / spills / stack | Emitted static/dynamic shared | Native occupancy |
| --- | --- | --- | --- | --- |
| MAIN false | NOT_RUN | unavailable | unavailable | NOT_RUN |
| MAIN true | NOT_RUN | unavailable | unavailable | NOT_RUN |
| B4 false | NOT_RUN | unavailable | unavailable | NOT_RUN |
| B4 true | NOT_RUN | unavailable | unavailable | NOT_RUN |

Planned115968-byte dynamic memory and intended320,2 launch bounds are **uncompiled source/layout quantities**, not emitted-resource proof. Old96 registers do not transfer to a smaller block. No local host compiler installation or secondary-toolchain sweep occurred; host compilation/linkage also NOT_RUN.

The planned packed-address bijection, actual generated-formula contracts, legal distinct-topk fixtures, all-eight B publications, split/fixup inverses, ring last-reader/parity/replay tests, generic MMAX384/EMAX3072 bounds, progress proof, selected bool compilation and numerical qualification were **not completed** after this early decisive STOP. Replaying the layout witness does not substitute for any of them. No quota was consumed by this allocation.

## Decision, owner and next gate

End this fixed115968 layout investment now, before compiler/GPU expenditure, per captain's stop. No fallback, scalar/seam/RMW/LUT/gamma reopening, new source variant or further target dispatch is pending. Captain owns shared closure/index; this worker publishes only its report and own evidence.

There is **no justified next target performance test for the stopped layout**. A separately authorized revised architecture would first need an independently proved layout satisfying reservation+static+rounding, exact selected resource/admission evidence and the uncompleted source/CPU/progress gates. Only then could one existing legal-fixture full-six independently prepared A/A/B test, with unchanged unrelated routes, replay output and duplicate-noise controls, test the performance hypothesis; no new observer/query harness is proposed. Routing checks must distinguish identity required by the candidate from cross-preparation routing-equality assumptions, rather than copying the prior seam's unlocalized equality failure. None of that future work is allocated or implemented here.

[Archive manifest](evidence/2026-11-01-worker-g2-24-prototype/archive-manifest.json) pins exact owned canonical Git blob bytes and read-only accounting dependencies. The incomplete generator is evidence of interrupted implementation, **not a qualified candidate**.
