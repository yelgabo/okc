# Lane moe-wsrel checkpoint

Updated: 2026-11-04 15:33 PST

## Goal

Correctness fix: the doneA credit in the MoE helper published other lanes' `p.inter` stores through a release
pattern that started before the `__syncwarp` relay (AGENTS.md publication rule). Keep the fix unless it costs
more than 0.3 us at a sliding cell.

## Base

- muse_decode_layer/cand/ly_mpdl.py, sha256 a0ee638cf5342235... (draw base)

## Sites (both source copies; helper0 = `_MA_ALT_SRCS['moe_helper0']`, used by TINY T=1, TINY2 T=8, MID T=16)

| site | MOE_CUDA_SRC lines | helper0 lines | writers and relay | active in ly_mpdl builds |
|---|---|---|---|---|
| skinny_ring (SKDYN) credit + pool dync credit | 4661-4675 | 13427-13441 | all CTA threads, CTA_SYNC | no (helper0: WS=1 calls skinny_ring_ws; MAIN/B4: WGS=1 calls wgs_phase, not run_phase) |
| skinny_ring credit | 4828-4836 | 13594-13602 | all CTA threads, CTA_SYNC | no (same reason) |
| fat_stream per-unit credit | 5248-5250 | 14014-14016 | all CTA threads, CTA_SYNC | no (TINY traps on fat units; MAIN/B4 use WGS) |
| ws_helper phase-A credit | 5468-5476 | 14244-14256 | 32 helper-warp lanes, __syncwarp | yes, helper0 copy: TINY, TINY2, MID |

MAIN/B4 (MOE_CUDA_SRC, WGS=1) credit through `wgs_flush` kind 1: `bar.sync 3+wg, 128`, then `red.release.gpu` by
thread 0 of the warpgroup. That is already the correct form.

## Fix

The leader (lane 0 or tid 0) issues `__threadfence()` (fence.sc.gpu) after the relay and before its atomics. Every
existing fence is kept. After the fix the release pattern is relay, then fence.sc.gpu, then the strong atom.

PTX ISA, Memory Consistency Model:
- "Release and acquire patterns": a fence.sc or fence.acq_rel followed in program order by a strong write is a
  release pattern; the pattern is per thread.
- "Base causality order" / "Causality order": X precedes Y when X precedes Z in program order and Z synchronizes
  with, or precedes in base causality order, Y. This makes the release cumulative: lane k's store precedes its
  warp barrier, the barrier orders it before lane 0's post-relay fence (bar.warp.sync "guarantees memory ordering
  among threads participating in barrier"; for CTA_SYNC, bar.sync synchronizes with bar.sync on the same barrier),
  and that fence is the first operation of the release pattern that synchronizes with the consumer's ld.acquire.
- "Causality" axiom: the consumer's later reads (ordered after its acquire by its own warp barrier) must see them.

Before the fix lane 0's fence preceded the relay, so only lane 0's own stores were covered. The other lanes'
fences are fence.sc, but fence-SC order between two lanes' fences is not fixed by the relay, so it gives no proof.

Consumer: lane 0 `ld.acquire.gpu` poll, `__syncwarp`, then `load_x_regs` with `__ldcg` (ld.global.cg, generic
proxy). No TMA or cp.async.bulk on this read, so no `fence.proxy.async` is needed.

## Candidates

- tools/dl_ly_wsrel_patch.py: text patch; splits the file at the helper0 source, asserts each site once per copy,
  asserts 8 inserted fences and that unpatching restores ly_mpdl's helper text exactly.
- muse_decode_layer/cand/ly_wsrel.py sha256 16a2ba4b0451dc9940f2c2afbc724f0fb1f8961452ea082c0382b74cfee87b9c (ship)
- muse_decode_layer/cand/ly_wsrel_ab.py sha256 0822ce7bb579844056792373bb772c41b21ffc07164301b2c3aaaefa3cb4d1df
  (A/B only: arm `{"state": {"wsrel0": True}}` runs TINY/TINY2/MID on ly_mpdl's helper source)

## Box

- tools/ly_wsrel_box.sh: PHASE=gate (bx_box_run, six loads, --runs 5, out /root/okc/.build/wsrel/chk.{log,json});
  PHASE=ab (ly_arms, two rounds, decode 1 / verify 1 / decode 16, 60 runs, out /root/okc/.build/wsrel/ab_r{1,2}.{log,json}).
- Launched gate then ab at 15:31 PT, log /root/okc/.build/wsrel/run.log. Waits on gpu.lock (an attention stamps
  run held the box at launch).

## Results

Pending.

## Next action

Read chk.log for PASS and "G1b4 backend native", then the ab_r1 / ab_r2 paired deltas.
