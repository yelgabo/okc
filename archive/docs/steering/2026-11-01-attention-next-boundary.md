# Attention: next architecture boundary, 2026-11-01

Evidence checkout `4bee6b0`; production `_ONE_SRC` SHA256 `3350b5a34aaf9241528a845455c0e2634b29f854b413894aa051dd9d2f547df4`. This is a source/instruction audit, not a new performance result. No kernel changed and no H200 check or public submission was dispatched.

## Decision

Preserve the remaining three checks. No materially different owned mechanism currently has enough causal evidence to justify one. The largest measured remaining opportunity is full-context verify16/32, but existing source and compiled lifetimes do not reveal an unnecessary matrix drain analogous to the successful MoE stage repair. The next useful attribution requires an uninstrumented hardware-counter profile in an authorized environment, rather than another timing-marker placement or ring/page knob.

## Measured opportunity and score sensitivity

The latest baseline comparison is `tools/hotspots.py` on `vp2_bf16_private.c79ddddb840a4fb7.json` against SGLang `baseline_profile/status.24daba2202e547d6.json`. Full-context decode16/32 profile times are 85.0/162.4 us against SGLang 201.1/386.7; verify16/32 are 103.0/184.9 against 206.8/392.2. The arithmetic floors (63.2/126.4 and 63.4/126.8 us) are model outputs, not measured traffic or attainable bounds. This receipt judges a private external VP2 only on verify1/sliding; its other cells are owned. Do not label the complete file an owned candidate.

The paired owned controls in that same receipt give verify16/32 full adjusted means 105.803/185.927 us. The reproducible [score sensitivity](evidence/2026-11-01-attention-next-boundary/score-budget.json) charges the actual six-load 46 sliding + 16 full mixture. A common cut confined to those two full cells needs 3.763% (3.981/6.996 us) for 1% whole score, or 14.632% (15.481/27.205 us) for the observed 2.7119/2.6043 rank50 ratio. Across all four full16/32 cells, the corresponding cuts are 1.864% and 7.413%. These are sensitivities holding every other cell fixed, not forecasts. Board numbers are the frozen comparison snapshot, not a fresh leaderboard claim.

VP2 `c79ddddb840a4fb7` passed official tests and private coverage, but its 5.71% target-cell gain produced only 0.631â€“0.906% whole-score four-cross gains and failed the frozen every-seed/noise gate. Automatic owned rewrite of that signal remains closed; foreign source remains measurement-only.

## Source and compiled lifetime proof

Actual owned `muse_attention_decode/attn_one.cu` has PF3 page-ID prefetch at `pe[PF][4]` around line1424, before the current raw-slot empty wait. A new current-tile address hoist therefore repeats existing work. The producer copies four K and four V 16-byte segments per thread; shared raw KV contains both KV heads, already providing grouped-query reuse. EREL releases the raw slot after QK retirement and raw-V loads, before probability/value work. Moving that release earlier crosses a live raw-K/V reader.

The default mode51 consumer starts around line2494. QK uses two m64n64k32 issues, then wait0 and register fence before immediate score use; V transpose overlaps this QK. P is rounded to FP8 and held in registers for the two m64n72k32 PV issues around line2680. The extra columns compute the probability denominator. PV wait0 protects output accumulator consumption and the next use of its probability operands. All writing-side proxy fences, named synchronization and full/empty handshakes remain essential. XPIPE at lines2347â€“2493 already implements the alternative QK/PV schedules.

Local CUDA12.8 sm90a NVRTC/PTXAS compilation of the exact mode8/51 instance reports 80 registers, zero stack/spills, 16 barriers, launch bounds384 threads/min2CTA and zero WGMMA serialization warnings. [Manifest](evidence/2026-11-01-attention-next-boundary/manifest.json) discloses removed Torch host headers and the declaration adapter: this proves device compilation only, not host APIs or H200 execution. [SASS census](evidence/2026-11-01-attention-next-boundary/sass-matrix-waits.txt) confirms paired QK issues followed by DEPbar0 and paired PV issues followed by DEPbar0; score registers R60â€“91, P operands R92/R96 and output R24â€“59 remain live across the relevant edges. The 80-register static report coexists with dynamic warpgroup register redistribution.

Changing PV from n72 to n64 plus scalar denominator accumulation is not a free representation win: it introduces exact FP8 conversion/reduction work and does not by itself change two-CTA occupancy. No measurement separates that cost from the observed producer pacing. It is not proposed for a check.

## Closed evidence and causal limits

`8ac6563343f74bcd` equal-resource six-arm mode51 experiment: verify16 control110.88, read/no-math110.69, zero/no-math74.24, no-copy/compute85.15 us; verify32 control188.38, read/no-math197.06, zero132.51, no-copy151.17. Removing consumer arithmetic did not improve the real-copy path repeatably. These perturbations establish sensitivity to producer/copy pacing, not additive removable time, DRAM bytes, or SM idle time. Old full/empty waits and pair spread are local pacing observations.

`3f44f891d0ef47db` passed correctness and matched every resource field, but all12 full-context sparse-wait-trace placements failed the frozen1% perturbation gate (verify16 +11.744â€“13.760 us, verify32 +21.216â€“23.136). Those marker intervals cannot justify an uninstrumented optimization; no stamp-placement sweep is warranted.

XPIPE2 `5575e7d15d3e49a8` compiled real overlap without serialization warnings but lost cold verify16/32. XPIPE1 required a register/liveness tradeoff; the one-CTA high-register test `318aeâ€¦` lost substantially. Pair-weight variants `7631033a0c93485d` and `4ff339332fef4aff` lost. QK2, WST, deeper rings, page bulk, DCARRY and VHOIST are already promoted or explicitly closed in `docs/ruled-out.md`; complete-head generic DSM `eac485ea0d554e50` also lost. No reopening is authorized by this audit. NS4 VT compaction alone still requires119392*2 >233472 bytes and cannot support two CTAs.

## Next action and stop gate

The measured gap is physical-copy/cache/issue pacing versus dependency scheduling, not yet localized. CUPTI counter availability job `04f0339fa9f24807` actually returned `CUPTI_ERROR_INSUFFICIENT_PRIVILEGES` on counter-image access after successful initialization. Torch profiler durations do not supply HBM/L2/SM counters. The next discriminating method is an organizer-provided or otherwise authorized Nsight Compute profile of this exact uninstrumented owned binary, covering full16/32 decode and verify: DRAM/L2 bytes and throughput, copy issue/long scoreboard, barrier stalls and eligible warps. It needs an environment/permission change; repeating the same preflight is not useful. No runtime file-I/O/subprocess workaround is proposed.

Reopen only with a source-specific redundant transfer or exposed dependency whose removal preserves every reader lifetime, precision boundary and handshake. Before spending a check, require local sm90a compilation, unchanged occupancy/resources or an explicit cost model, and a predicted multi-cell budget exceeding1% whole score. A future four-arm check must use independent builds/private state, three fresh seeds, all12 score cells, official FP32 and separate BF16 parity checks, fresh replays and per-seed gains exceeding two independent nulls; reject any affected-cell regression or incomplete coverage. This is a prospective acceptance rule, not an approved experiment.
