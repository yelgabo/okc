# Lane pdl-moe: route -> MoE as a PDL edge (checkpoint)

Branch worktree-agent-a3c586c187190aa19. Last update 14:46 PT 2026-11-04.

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

## Next action

Gate run launched 14:45 PT (PHASE=gate: prepare with guard, reference check 6 loads, --runs 5, chk.log). Then
PHASE=ab: two ly_arms rounds off vs on at decode 1, verify 1, decode 16, decode 32 (ab_r1.log, ab_r2.log).
