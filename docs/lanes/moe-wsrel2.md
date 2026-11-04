# Lane moe-wsrel2 checkpoint

Updated: 2026-11-04 15:55 PST (gates, SASS, four A/B rounds, decision)

## Goal

Single post-relay release at the ws_helper doneA credit (moe_helper0: TINY T=1, TINY2 T=8, MID T=16): drop the 32
per-lane pre-relay `__threadfence()` and have lane 0 fence after `__syncwarp()`, before its atomicAdds.
Acceptance: correct by the cited proof, no worse than ly_mpdl by more than 0.2 us at any sliding cell.

## Base

muse_decode_layer/cand/ly_mpdl.py sha256 a0ee638cf5342235a879054ac68da02e63d6f5e88f4834db6081e09937858a68

## Candidates (tools/dl_ly_wsrel1_patch.py; site matched once in MOE_CUDA_SRC, once in moe_helper0)

- muse_decode_layer/cand/ly_wsrel1.py (fence.acq_rel.gpu) sha256 323362a6afea3500f4c856e04361f7e5b8eb06cdbc68a65cdb88184bfd181ed5
- muse_decode_layer/cand/ly_wsrel1_sc.py (fence.sc.gpu) sha256 ef68943633ee037e16a4bab0bd4a971b3d7fce74782eec092ba937720e6d07a3
- muse_decode_layer/cand/ly_wsrel1_ab.py (A/B only; default arm = ly_mpdl helper, state wsrel_h ar1 / sc1)
  sha256 978bc1474cff880c0b91837522b5c5a7d20bdf5c3173cfe1f1ba49eb3f6b9d81

## Proof (PTX ISA, chapter "Memory Consistency Model", plus the bar.warp.sync instruction description)

Producer (one helper warp, every lane): weak `st.global.L2::cache_hint` stores of its `p.inter` rows across the
phase-A loop, then at `q == nA - 1` `__syncwarp()` (bar.warp.sync), then lane 0 runs `fence.acq_rel.gpu` (ly_wsrel1)
or `fence.sc.gpu` (ly_wsrel1_sc) and `atomicAdd(p.doneA + jj, ...)` (atom.global.add, strong relaxed, gpu scope) for
each job. Consumer: lane 0 polls `ld.acquire.gpu` on `p.doneA + j`, then `__syncwarp()`, then every lane reads
`p.inter` with `__ldcg` (ld.global.cg, generic proxy; no TMA or cp.async.bulk on this path, confirmed by moe-wsrel).

1. Lane k's stores precede its `bar.warp.sync` in program order. The instruction description of bar.warp.sync says
   it "also guarantee[s] memory ordering among threads participating in barrier", so those stores are ordered
   before every operation lane 0 performs after the barrier ("Base causality order": X precedes Z in program
   order, Z precedes Y).
2. "Release and acquire patterns": a `fence.sc` or `fence.acq_rel` followed in program order by a strong write is a
   release pattern. Lane 0's post-relay fence plus each atomicAdd is one, and its first operation (the fence) comes
   after the relay, so every lane's stores precede it.
3. "Observation order": the consumer's acquire read observes the counter through the chain of producers' atomic
   RMWs, so each producer's atom precedes the read in observation order. "Synchronizes-with": the release pattern
   synchronizes with the `ld.acquire.gpu` acquire pattern; both are gpu scope and morally strong.
4. The consumer's `bar.warp.sync` orders its lane 0 acquire before the other lanes' `__ldcg` reads (as in 1).
   "Causality order" then runs from every producer lane's store to every consumer lane's read, and the "Causality"
   axiom makes the read see the store (or a later write).

The base (ly_mpdl) has a fence in each lane but lane 0's release pattern starts before the relay, so step 2 fails
for lanes 1 to 31. The variant is strictly stronger than the base without any per-lane fence. Disclosure: the formal
"Synchronizes-with" list names bar{.cta}.sync / red / arrive; bar.warp.sync's ordering comes from its instruction
description, which the base, ly_wsrel and ly_wsrel1 all rely on in the same way. Every other fence is kept
(the PB path, the CTA credits and the MOE_CUDA_SRC sites other than ws_helper are unchanged).

## Box

tools/ly_wsrel1_box.sh, PHASE="sass gate ab", outputs in /root/okc/.build/wsrel1/ (sass.log, chk_<cand>.log,
ab_r{1,2}.log). SASS: tools/ly_wsrel1_sass.py (nvcc device build of the helper source, TINY and MID define sets).

## Results

Gate ly_wsrel1 (bx_box_run, six loads, 5 runs, judge reference g1b4_pdl8d, /root/okc/.build/wsrel1/chk_ly_wsrel1.log):
12/12 check PASS, one "G1b4 backend native" line (the same "native repaired: cooperative+BF16, MFIN/DF/YBF/PDL off"
line as moe-wsrel's accepted ly_wsrel_ar gate), 0 "native build failed", 0 Marlin. Box net us:

| load | sliding | full |
|---|---|---|
| decode 1 | 61.60 | 70.05 |
| decode 16 | 124.10 | 201.09 |
| decode 32 | 134.69 | 292.67 |
| verify 1 | 92.93 | 110.59 |
| verify 16 | 240.51 | 321.09 |
| verify 32 | 296.06 | 460.94 |

The first SASS attempt failed (C++17, no Python.h); fixed script rerun to /root/okc/.build/wsrel1/sass2.log.
The run.log "rc=" values of the first launch are date's status (the script echoed `$?` after a `$(date)`);
fixed in tools/ly_wsrel1_box.sh for later launches. Read the logs, not those rc values.

Gate ly_wsrel1_sc (/root/okc/.build/wsrel1/chk_ly_wsrel1_sc.log): 12/12 check PASS, the same native line, 0 failed,
0 Marlin. Box net us sliding / full: decode 1 63.01 / 68.54, decode 16 121.12 / 201.76, decode 32 135.78 / 292.42,
verify 1 95.81 / 112.66, verify 16 235.65 / 317.76, verify 32 294.24 / 457.89.

SASS (/root/okc/.build/wsrel1/sass2.log, .sass files in sass/): MEMBAR counts per build (two ws_helper
instantiations each), TINY and MID identical:

| build | MEMBAR.SC.GPU | MEMBAR.ALL.GPU |
|---|---|---|
| base ly_mpdl | 6 | 2 |
| ar1 ly_wsrel1 | 4 | 4 |
| sc1 ly_wsrel1_sc | 6 | 2 |

In ly_wsrel1_tiny.sass the credit fence is at 0x11d40 (base: MEMBAR.SC.GPU at 0x11d20 executed by every lane). It
follows `@P1 BRA 0x121e0` at 0x11d10 (lanes other than lane 0 branch past it) and precedes the
`@P0 ATOMG.E.ADD.STRONG.GPU` credits at 0x11e20 onward, so one MEMBAR at the credit, lane 0 only. The sc1 build has
MEMBAR.SC.GPU at the same address.

A/B (tools/ly_arms.py on ly_wsrel1_ab, arms base / ar1 / sc1, 60 runs, ab_r{1,2,3,4}.log; r3 and r4 via
PHASE=ab ROUNDS="r3 r4", run34.log). Paired median delta vs base in us, every arm check PASS in every round:

| cell | arm | r1 | r2 | r3 | r4 | mean |
|---|---|---|---|---|---|---|
| decode 1 sliding | ar1 | -0.08 | +0.29 | +0.24 | +0.30 | +0.19 |
| decode 1 sliding | sc1 | +0.03 | -0.19 | -0.22 | -0.03 | -0.10 |
| decode 1 full | ar1 | +0.86 | -0.66 | -0.42 | -0.38 | -0.15 |
| decode 1 full | sc1 | +0.51 | -0.72 | -0.22 | -0.32 | -0.19 |
| decode 16 sliding | ar1 | +0.54 | -0.05 | +0.32 | +0.40 | +0.30 |
| decode 16 sliding | sc1 | +0.26 | +0.00 | +0.38 | +0.30 | +0.24 |
| decode 16 full | ar1 | -0.21 | -0.22 | +0.26 | +0.06 | -0.03 |
| decode 16 full | sc1 | -0.06 | +0.14 | +0.34 | +0.22 | +0.16 |
| verify 1 sliding | ar1 | -0.70 | +0.72 | +0.43 | +0.40 | +0.21 |
| verify 1 sliding | sc1 | -0.43 | +0.03 | -0.03 | -0.06 | -0.12 |
| verify 1 full | ar1 | -0.03 | -0.24 | -0.13 | -0.45 | -0.21 |
| verify 1 full | sc1 | -0.05 | +0.27 | +0.02 | -0.13 | +0.02 |

## Decision

Both arms fail the bar (no worse than ly_mpdl by more than 0.2 us at any sliding cell): ar1 at all three sliding
cells, sc1 at decode 16 sliding only (+0.24 mean, 3 of 4 rounds above 0.2). Ruled-out row appended to
docs/ruled-out.md. sc1 is the form to take if the lead closes the base's proof gap for correctness at that cost.

## Next action

None in this lane. Lead decision: accept sc1 for correctness at about +0.24 us decode 16 sliding, or reopen with a
cheaper MID-build release form.
