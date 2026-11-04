# Lane pdl-moe: route -> MoE as a PDL edge (checkpoint)

Branch worktree-agent-a3c586c187190aa19. Last update 14:46 PT 2026-11-04 (handoff before restart).

## Base

muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e,
19,066 lines.

## Stamps read in the base

- Line 102 `PDL = False`, so `_pdl(T)` is always False; every `if MOE_HAS_PDL: ext.okc_set_pdl(1 if _pdl(T) else 0)`
  would set 0. Line 6957 `MOE_HAS_PDL = False`.
- Both MoE sources keep `static int g_okc_pdl` and `okc_set_pdl` (main 6808-6810, alt 15291-15293), but the launch
  ignores it: plain `cudaLaunchCooperativeKernel` (main 6905-6916, alt 15370-15373), after TORCH_CHECKs on sm_90,
  `prop.cooperativeLaunch`, grid <= SMs, grid <= 136 and grid <= resident*SMs.
- MoE kernel entry (main 6301-6308, alt 14889-14896): `griddepcontrol.wait` is the first statement after `DBG(0)`.
  MoE trigger `griddepcontrol.launch_dependents` at 6525 / 15096.
- Active layer path `_pdl8_sl_tile_after_attention` (18360-18427): route `_sl_tile_route_cg` launched with
  `launch_pdl=_pdl8_edge("route")`; MoE call at 18403-18411. Graft path MoE call at 18485-18490.
- Route kernel `_sl_tile_route_cg` (18147-18212): every CTA runs `_gdc_launch()` then `_gdc_wait()` at entry; only
  optional PRELOAD constant loads come before the trigger.

## Proofs

1. Launch support (cooperative + programmatic attributes on one launch): HOLDS, measured on the box 14:37-14:38 PT
   (tools/pdl_coop_probe.cu, box .build/pdlmoe/pdlcoop.cu, logs pdlcoop.log and probe.log; H200 sm_90, 132 SMs,
   driver 13020, runtime 13000). Secondary: 384 threads, 200 KB smem (occupancy 1/SM), grid 132, 6 grid.sync rounds
   plus a software barrier, 40 iterations per config, eager and graph, 10 s watchdog.
   - Mode 1 (Ex, coop+PDL) launched without error in every config; 0 invalid grids, 0 barrier errors, 0 data
     mismatches over 5280 CTAs per mode.
   - Grid 133 is refused with "too many blocks in cooperative launch" with and without the PDL attribute, and the
     captured node reads cooperative=1 with a prim->sec edge of type 1 (programmatic), from_port 1. So the launch
     stays cooperative. (is_valid alone is no evidence: mode 2, PDL only, also reports valid eagerly.)
   - No overlap: mode 1 overlapped_iters = 0 in every config (the secondary starts after the last primary CTA ends);
     mode 2 overlaps by 21-78 us. The driver honours co-residency by serializing.
   - Gain is graph launch latency only. Mean gap prim end -> sec start, graph: mode 1 0.38 / 0.45 / 0.50 us vs
     mode 0 0.76 / 0.95 / 0.96 us (route-like 8 CTAs / 132 CTAs / 264 CTAs); about 0.4 us per launch. Eager: no
     change (1.60 / 1.95 / 1.93 vs 1.62 / 1.95 / 1.94).
2. Residency and progress: HOLDS by argument. A secondary grid is launched only after every primary CTA has executed
   its trigger or exited, so every route CTA is already resident when the first MoE CTA can be scheduled; none is
   pending behind a MoE CTA. Route and every earlier grid wait only on their own predecessors, never on the MoE, so
   they finish and free their SMs; the MoE grid (state["nsm"] CTAs, one per SM by `__launch_bounds__(NTL,1)` and its
   shared memory) then becomes fully resident before any CTA can pass the grid barrier. A MoE CTA cannot block a route
   CTA. Measured (proof 1): with both attributes the MoE does not start before route's last CTA ends, so residency
   is trivially safe.
3. Producer publication and consumer wait: HOLDS for the minimal patch (wait stays first). No global load precedes
   `griddepcontrol.wait`: DBG writes p.dbg only with OKC_DEBUG and non-null dbg. Counters, parity and the
   generation-indexed MFIN counters of B4 are read after the wait, which returns after route has completed and
   flushed; by the chain of waits every earlier grid (previous MoE, unpooler) has completed too. Weight prefetch before
   the wait is impossible here: the first weight units depend on topk_ids, which route writes.
4. Route's trigger: at entry of every route CTA. It is needed (without it the MoE launches only at route's
   completion and the edge buys only the launch latency); it is already there.

## Candidate

- tools/dl_ly_mpdl_patch.py -> muse_decode_layer/cand/ly_mpdl.py, SHA256
  a0ee638cf5342235a879054ac68da02e63d6f5e88f4834db6081e09937858a68. Launch switch at lines 6915 and 15382,
  `MOE_HAS_PDL = True` 6967, `_mpdl_on` 18152, patched call sites 18430 (sl_tile) and 18513 (graft).
- Edge on only when PDL8 is on at T, T <= 32 and state["moe_pdl"][T]; okc_set_pdl(0) after the launch.
- Prepare guard MPDL_GUARD: per T in 1/8/16/32 and both layers, 48 fresh-routing rounds (new random stream, rotated
  local_start), replays off, off, on, on; bitwise if off-vs-off is bitwise, else PDL8H; 30 s hang watchdog.
- Box: /root/okc/muse_decode_layer/cand/ly_mpdl.py, runner tools/ly_mpdl_box.sh, outputs /root/okc/.build/pdlmoe/.

## Gate run (done, 14:44-14:45 PT): PASS 12/12, guard passes at every T

Command on the box: `cd /root/okc && PHASE=gate setsid nohup tools/ly_mpdl_box.sh > .build/pdlmoe/gate.log 2>&1 < /dev/null &`
(bx_box_run, judge .build/judge/g1b4_pdl8d.f931abf568544018.json, 6 loads, --runs 5, --pin-pdl8 1,8,16,32, PDL8 gate
not skipped). Log /root/okc/.build/pdlmoe/chk.log, JSON chk.json. prepare 74.25 s.

- Native build: "G1b4 backend native repaired: cooperative+BF16" and "G_RELEASE_STARTUP_OK native ... 56
  verified_replays"; grep for native build failed / fallback / marlin is empty.
- Reference check PASS on all 12 cells. Box net (n=5, not a timing result): decode 1 62.18 / 69.86, decode 16
  124.00 / 201.34, decode 32 135.90 / 293.02, verify 1 94.11 / 112.93, verify 16 239.33 / 318.72, verify 32
  296.00 / 458.46 us (sliding / full).
- PDL8 gate enabled T = 1, 8, 16, 32 (paired savings 7.14 / 7.82 / 7.34 / 6.34 us).
- MPDL_GUARD, 48 fresh-routing rounds per T and layer: off-vs-off is not bitwise equal (the layer is nondeterministic),
  so the PDL8H rule applied; all 8 pass, moe_pdl {1,8,16,32: True}. Each round compares off once and on twice against
  the first off replay, so on-arm counts are about 2x by construction. Off / on: T=1 sl bytes 3381 / 13987, ids 0 / 0,
  rel 3.6e-3 / 7.0e-3; T=1 full ids 0 / 0, rel 8.9e-3 / 8.9e-3; T=8 sl ids 0 / 0, rel 7.9e-3 / 9.1e-3; T=8 full ids 2 / 2;
  T=16 sl ids 0 / 1; T=16 full ids 2 / 6, rel 6.2e-3 / 7.8e-3; T=32 sl ids 1 / 6, rel 7.2e-3 / 8.7e-3; T=32 full
  ids 0 / 0, rel 9.3e-3 / 1.2e-2. No hang.

## In progress: paired A/B (launched 14:45 PT, detached, survives this session)

Command: `cd /root/okc && PHASE=ab setsid nohup tools/ly_mpdl_box.sh > .build/pdlmoe/ab.log 2>&1 < /dev/null &`
It runs tools/ly_arms.py twice (r1, r2) on ly_mpdl.py with arms off = {"moe_pdl": {}} (the ly_stk4 launch path)
and on = {"moe_pdl": {1,8,16,32: True}}, loads decode 1, verify 1, decode 16, decode 32, --runs 60,
--skip-pdl8-gate --pin-pdl8 1,8,16,32, router seed 106218864389449120, under /root/okc/.build/box/gpu.lock, each
under timeout 3000. Outputs: /root/okc/.build/pdlmoe/ab_r1.log, ab_r1.json, ab_r2.log, ab_r2.json; progress
lines in ab.log ("ab r1 rc=...", "all done"). Check: `ssh okc-h200 'pgrep -af ly_arms; cat /root/okc/.build/pdlmoe/ab.log'`.

## Next action

When ab.log says "all done": `grep ^ARMS` in ab_r1.log and ab_r2.log gives net us per cell and arm; tabulate on - off
per cell for both rounds; grep both logs for "backend native" and for native build failed / fallback / marlin.
Expected gain from proof 1 is about 0.4 us per cell (graph launch latency, no overlap). If on is not faster in both
rounds at a cell, write a ruled-out row. Judge use is the lead's decision (T = 1, 8, 16, 32 only).
