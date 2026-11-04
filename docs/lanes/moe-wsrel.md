# Lane moe-wsrel checkpoint

Updated: 2026-11-04 15:41 PST (lane finished; decision with the lead)

## Goal

This lane is a correctness fix. The doneA credit in the MoE helper published other lanes' `p.inter` stores through a
release pattern that started before the `__syncwarp` relay (AGENTS.md publication rule). The brief says to keep the
fix unless it costs more than 0.3 us at a sliding cell.

## Base

- muse_decode_layer/cand/ly_mpdl.py, sha256 a0ee638cf5342235... (draw base)

## Sites (both source copies; helper0 = `_MA_ALT_SRCS['moe_helper0']`, used by TINY T=1, TINY2 T=8, MID T=16)

| site | MOE_CUDA_SRC lines | helper0 lines | writers and relay | active in ly_mpdl builds |
|---|---|---|---|---|
| skinny_ring (SKDYN) credit + pool dync credit | 4661-4675 | 13427-13441 | all CTA threads, CTA_SYNC | no (helper0: WS=1 calls skinny_ring_ws; MAIN/B4: WGS=1 calls wgs_phase, not run_phase) |
| skinny_ring credit | 4828-4836 | 13594-13602 | all CTA threads, CTA_SYNC | no (same reason) |
| fat_stream per-unit credit | 5248-5250 | 14014-14016 | all CTA threads, CTA_SYNC | no (TINY traps on fat units; MAIN/B4 use WGS) |
| ws_helper phase-A credit | 5468-5476 | 14244-14256 | 32 helper-warp lanes, __syncwarp | yes, helper0 copy: TINY, TINY2, MID |

MAIN and B4 (MOE_CUDA_SRC, WGS=1) credit through `wgs_flush` kind 1: `bar.sync 3+wg, 128`, then `red.release.gpu` by
thread 0 of the warpgroup. That is already the correct form.

## Fix

The leader (lane 0 or tid 0) issues a fence after the relay and before its atomics. Every existing fence is kept.

- ly_wsrel: `__threadfence()` (fence.sc.gpu, SASS MEMBAR.SC.GPU).
- ly_wsrel_ar: `fence.acq_rel.gpu`.

PTX ISA, Memory Consistency Model:
- "Release and acquire patterns": a fence.sc or fence.acq_rel followed in program order by a strong write is a
  release pattern. The pattern is per thread.
- "Base causality order" / "Causality order": X precedes Y when X precedes Z in program order and Z synchronizes
  with, or precedes in base causality order, Y. This makes the release cumulative. Lane k's store precedes its warp
  barrier. The barrier orders it before lane 0's post-relay fence (bar.warp.sync "guarantees memory ordering among
  threads participating in barrier"; for CTA_SYNC, bar.sync synchronizes with bar.sync on the same barrier). That
  fence is the first operation of the release pattern that synchronizes with the consumer's ld.acquire.
- "Causality" axiom: the consumer's later reads, which its own warp barrier orders after its acquire, must see those
  stores.

Before the fix, lane 0's fence came before the relay, so it covered only lane 0's own stores.

Consumer: lane 0 polls with `ld.acquire.gpu`, then `__syncwarp`, then `load_x_regs` reads with `__ldcg` (ld.global.cg,
generic proxy). There is no TMA or cp.async.bulk on this read, so no `fence.proxy.async` is needed.

SASS check: in the TINY build (flags TINY=1, SPLIT=1, WSBLK=2), the fixed .o has 8 MEMBAR.SC.GPU against 6 in w0,
plus 2 more CCTL.IVALL and 2 more MEMBAR.ALL.CTA. That is one inserted fence per ws_helper instantiation.

## Candidates

- tools/dl_ly_wsrel_patch.py is a text patch. It asserts each site once per copy, asserts 8 inserted fences, and
  asserts that unpatching restores ly_mpdl's helper text exactly.
- muse_decode_layer/cand/ly_wsrel.py sha256 16a2ba4b0451dc9940f2c2afbc724f0fb1f8961452ea082c0382b74cfee87b9c (fence.sc)
- muse_decode_layer/cand/ly_wsrel_ar.py sha256 e8beab882d095fa1de49a7b65262438742b5db1ad68da85a9421c488cb28956e (fence.acq_rel)
- muse_decode_layer/cand/ly_wsrel_ab.py sha256 325c8865b293f0bd2f90313c49d63bac70edacf777f0cd74f9767cb74a2ef551
  (A/B only, three arms: w0 = ly_mpdl helper, sc = default, ar)

## Results

Gates (bx_box_run, six loads, 5 runs, judge reference g1b4_pdl8d):
- ly_wsrel: 12/12 check PASS, "G1b4 backend native", 0 "native build failed", 0 Marlin
  (/root/okc/.build/wsrel/chk.log).
- ly_wsrel_ar: 12/12 check PASS, native 1, 0 "native build failed", 0 Marlin
  (/root/okc/.build/wsrel/chk_ly_wsrel_ar.log).
- ly_wsrel_ar box net in us:

  | load | sliding | full |
  |---|---|---|
  | decode 1 | 64.35 | 70.14 |
  | decode 16 | 122.82 | 201.57 |
  | decode 32 | 135.39 | 293.41 |
  | verify 1 | 94.85 | 114.05 |
  | verify 16 | 237.70 | 320.83 |
  | verify 32 | 296.51 | 458.14 |

A/B (ly_arms, 60 runs per round, paired median delta against w0 in us; every arm check PASS in every cell).
Rounds r1 and r2 ran two arms, rounds r3 and r4 ran three. Logs are /root/okc/.build/wsrel/ab_r{1,2,3,4}.log.

| cell | sc r1 | sc r2 | sc r3 | sc r4 | ar r3 | ar r4 |
|---|---|---|---|---|---|---|
| decode 1 sliding | +0.56 | +0.19 | +0.26 | +0.21 | +0.05 | +0.21 |
| decode 1 full | +0.35 | -0.08 | -0.43 | -0.32 | -0.26 | +0.16 |
| decode 16 sliding | +0.18 | +0.21 | +0.34 | +0.77 | +0.51 | +0.69 |
| decode 16 full | +0.56 | +0.30 | +0.82 | +0.13 | +0.67 | -0.10 |
| verify 1 sliding | +0.98 | +0.69 | +0.91 | +0.72 | +0.24 | +0.61 |
| verify 1 full | +0.46 | +0.83 | -0.02 | +0.64 | -0.16 | +0.74 |

Reading: both forms cost more than 0.3 us at a sliding cell.
- sc costs +0.7 to +1.0 us at verify 1 sliding in all four rounds.
- ar costs +0.5 to +0.7 us at decode 16 sliding in both rounds and +0.6 us at verify 1 sliding in r4.
- The full cells swing by about plus or minus 0.4 us, which sets the noise floor.

## Decision

Under the brief's rule, neither form is kept. Because this is a correctness fix, the lead makes the call. ly_mpdl
stays the draw base. If the lead takes the fix, use ly_wsrel_ar: it is no worse than sc and is cheaper at verify 1
sliding.

## Untested option (needs the lead; this lane may not remove fences)

Every lane keeps its pre-relay `__threadfence`, and the leader adds one more fence. One alternative is to give the
whole proof to the single post-relay leader fence and drop the 31 non-leader fences. The ly_cfar row measured the
existing per-lane fence at about 0.5 us. Under the same causality argument this form is valid, and it could cost
less than the base. That is a hypothesis.

## Next action

None in this lane. The lead decides whether to adopt ly_wsrel_ar.
