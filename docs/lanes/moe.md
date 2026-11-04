# moe lane checkpoint (round 4, decode 16 / 32 MoE prologue)

Updated 2026-11-04 12:40 PT. Branch `worktree-agent-a8c676644bda35cae`. Run log section: docs/runs-decode-layer.md
"2026-11-04 - Decode 16 / 32 MoE prologue (box)".

## Base

muse_decode_layer/cand/ly_stk2.py, SHA256 4bbf4010f412b9afe8595b1a5ede517fb8aa6e31af6efdc1e55dd34a3a335846.

## Instruments (box only, never candidates)

- tools/ly_moe_pro16.py (MID) and tools/ly_moe_prob4.py (B4): 128 stamps per CTA, new stamps on clock64 (the box's
  globaltimer ticks every 256 ns), post-barrier stamps behind a volatile shared load (a bare clock read was hoisted
  above bar.sync). Options `--probe` (first-touch vs same-page load latency) and `--r2` (route() / wgs_ranges() run a
  second time). Runner tools/ly_box_moe_pro.py (`--warm` adds an unflushed replay). Box logs .build/moe4/pro_*.log,
  copies in muse_decode_layer/profiles/box/moe4/.

## Measurements (us after each CTA's own entry, CTA mean, median of 20 inputs, sliding)

- MID route() ends 3.45; 1.26 of it is barrier 1 waiting for the FAST1 warp (four dependent topk loads 1.0, job scan
  0.45, TMA issue 0.19). route() body 1.5 us, the same when run a second time (not instruction fetch). Warp 0 scan 0.65.
  x load for the first job 0.80 (64 KB per CTA from L2). Warm (no L2 flush) replay identical: not DRAM, TLB or cold
  data. Probe: first touch 414 ns vs same 2 MiB new line 400 ns (TLB not a cost).
- B4 route() ends 2.41 (body 1.56, same when repeated); x producer's wgs_ranges 2.53 -> 5.07 (2.54 us; an immediate
  second call takes 0.90); consumers' gather and xready releases end 4.56; WS_JOIN and first weight issue about 5.2.

## Candidates

- ly_m4rb.py b12e1b47688ed569d2e8202326f475f54166788495e59debbf4b2de378b740a8 (MID, -DOKC_RB7=1: routing's first
  three barriers without the FAST1 warp). Gate pass, chk PASS 12/12. A/B vs ly_stk2, two rounds: decode 16 sliding
  +0.19 (5/24), full +0.45 (5/24); everything inside. NULL. Stamps: route end 3.45 -> 2.84, but the FAST1 warp's
  TMA issue 1.81 -> 2.59 (it now runs beside route()) and the x load 0.80 -> 1.79, first slot full 4.64 -> 5.04.
- ly_m4rw.py f66973f1aba39ae4b144fc6f56528ad2f3661dd9c6c5c750d6474fd95f25a6f8 (B4, -DOKC_RWARM=1: wgs_ranges dry run
  before routing). A/B running at the time of this checkpoint (.build/moe4/rw_run.log, .build/prtl/m4_rw*).

## Next action

Read the ly_m4rw table (`python3 tools/ly_prtl_table.py .build/prtl/m4_rw .build/prtl/m4_rw_r2` on the box) and its
stamps (.build/moe4/pro_rw.log). Unbuilt follow-up for MID: ly_m16r's hoisted FAST1 loads together with rb7, so the
weights copies go out before warps 0..6 start routing (rb7 alone delayed them by 0.8 us).
