# Lane barriers: checkpoint 5 (final; ly_fw closed as a loss)

Base: `muse_decode_layer/cand/ly_stk4.py`, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e.
Line numbers below are in ly_stk4. Inventory is from source reading (three read-only agents plus my own reads);
costs marked "measured" cite docs/runs-decode-layer.md, everything else is an estimate.

## Inventory (sync points with a "wait only for what you need" question)

| # | Kernel / build | T | Lines | Primitive | Protects | Consumer needs | Cost |
|---|---|---|---|---|---|---|---|
| S1 | MoE helper0 TINY / TINY2 / MID | 1, 8, 16 | 15112-15131 | grid barrier, all 132 CTAs spin on one counter (nanosleep 64) | ybuf + ss atomics -> final | item (token, 1024-col seg) needs all phase-B units of all its token's jobs (ss); only M x 4 CTAs have items (4 / 32 / 64) | measured T = 1: B end to barrier 1.26-1.31 us sliding, release about 0.9-1.1 us; MID barrier idle 2.75 us |
| S2 | helper0 phase B x load | 1, 8, 16 | 14097, credit 14239-14247 | lane-0 ld_acquire doneA[j] >= 512, uncapped | phase-A inter of job j | all 512 phase-A units of job j | credit is late (at q == nA-1); est. small, ranges are balanced |
| S3 | helper0 unit loop | 1, 8, 16 | 14146 | CTA_SYNC per unit | partials -> helper | whole CTA | closed (668 ly_ul_nb/na/pr) |
| S4 | MoE B4 final | 32 | 6576-6593 | tid<8 ld_acquire MF_CE >= 64 x jobs(e), uncapped | y + sspart | y needs 16 of 64 counts; sspart needs all 64 | est. <= 0.2 us (hides y load latency only) |
| S5 | MoE B4 kind-4 MF_CE release | 32 | 5897-5901, deferred 6134-6138 | red.release deferred to next segment's commit group | per-expert completion | - | est. <= 0.2 us, only when an expert's last unit is mid-range |
| S6 | MoE MAIN grid barrier | 128, 256 | 6542-6552 | grid barrier | y -> final (whole token row) | token's 8 experts | MFIN for MAIN is ly_gbm / ly_stk5 (recorded alternative); 835 corrected by the research addendum C |
| S7 | MoE B4 / MAIN phase A -> B | 32-256 | 5757, 5760 | ld_acquire xready / doneAs | rows / stage of job j | already per-job / per-stage | none |
| S8 | attention ONE mode 13 ticket | 1 sliding | 2882-2935 | last-CTA ticket | split merge | all splits of the request | measured merge ticket about 1.49 us; closed by 791 ly_acl (DSMEM cluster merge), 283 |
| S9 | attention griddepcontrol.wait before K/V | 1, 8, 16, 32 | 1515-1567 | gdc wait before cache loads | qkv | only q and the new row | closed: 785 ly_akv, 806 ly_a16kv, 759 ly_apw (K/V is off the critical path) |
| S10 | attention flat q8 W0 registration barrier | 128, 256 full | 3163-3170, 3331-3381 | grid barrier before pdl_wait | W0 pairing | same-SM peer + bad flag | est. 0.3-0.5 us at full cells worth 0.013-0.017 %/us |
| S11 | attention flat_finalize counter | 128, 256 full | 2955-2982 | counter to G | per-request finalize | about 2 requests per CTA | est. similar to S10, full cells only |
| S12 | attention per-request cooperative finalize | 8, 128, 256 | 2801-2824 | cooperative wait for all splits | merge | all splits of the request | closed for q_len 8 (299), 760 ly_sngw2 |
| S13 | finish `_sl_cluster_finish_cg` | 1, 8 | 18296-18303 | b1 loads after gdc wait | b1 outputs | only P depends on un | closed: 734 ly_finb1, finpre null |
| S14 | `_graft_row_boundary_pre_cg` | 16, 32 | 18323-18330 | Stream load after gdc wait | stream | Branch only | closed by the b1/b2 read-ahead null (research addendum D, 768) |
| S15 | cgemv un trigger | 1 | 17793 | launch_dependents at CTA end | - | - | closed: 651 (trigger after weight loads lost standalone) |
| S16 | route on whole fused pooler/router grid | 16, 32 | 18155, 18176, 18224 | gdc wait | pooled + logits | MoE needs both halves anyway | est. ~0 |
| S17 | qkv forced plain | all | 18545 | no PDL edge into qkv | previous call's output | cross-call edge | not applicable inside one layer call (unverified) |
| S18 | sk2 pre-wait weight stages | <= 32 | 8195-8200 | only P stages before wait | x | weights do not depend on x | P is tuned per role: 731, 769, 792 |

No capped poll falls through on a scored build: 6640 (`n < 4000000`) is compiled out of B4 and MAIN in ly_stk4;
the `_df_patch` sentinel spin (about 15410) is never applied. The `4000000 + seq` at 5995 / 14686 is a debug tag.
Revival triggers: 711 ly_fin1 needs MAIN MFIN, which exists only in ly_gbm / ly_stk5, not ly_stk4 (not landed on the
base). 337 (MoE barrier-to-exit tail shorter): not landed. 299 (ticket at q_len 1 batch-1 full, needs > 0.3 us repeat):
no new measurement. 750 ly_rsl_nt: not landed.

## Ranking (expected gain x value per us; full cells about one third)

| Rank | Point | Cell (value %/us) | Est. gain | Score | Status |
|---|---|---|---|---|---|
| 1 | S1 waiters only on item CTAs (ly_fw) | decode 1 sl 0.18, verify 1 sl 0.12, decode 16 sl 0.096 | 0.1-0.4 us (hypothesis) | 0.02-0.07 % | built, on the box |
| 2 | S1 option b: item CTA forms its own ss | decode 1 sl 0.18 | 0 | 0 | no lever: each route's rsqrt(ss/HD) needs y over all 4096 columns, so the item still depends on every phase-B unit of its jobs; at T = 1 that is the whole grid, and only the barrier's own latency (S1, rank 1) is removable |
| 3 | S4 B4 two-step final wait | decode 32 sl 0.084 | <= 0.2 | <= 0.017 % | not built |
| 4 | S5 B4 kind-4 release undeferred | decode 32 sl 0.084 | <= 0.2 | <= 0.017 % | not built |
| 5 | S10 W0 barrier -> per-SM pair counter | verify 16/32 full ~0.015 | 0.3-0.5 | <= 0.008 % | not built |
| 6 | S11 per-request flat_finalize | verify 16/32 full ~0.015 | 0.2-0.4 | <= 0.006 % | not built |
| 7 | S2 early doneA credit | decode 1 sl 0.18 | < 0.1 | < 0.018 % | not built |
| 8 | S16 per-half route wait | decode 16/32 | ~0 | ~0 | not built |
| 9 | S6 MAIN MFIN | verify 16/32 | measured -1.0 to -1.6 us box | - | exists as ly_stk5 (lead's call) |
| 10 | S9, S13-S15 | - | - | - | closed rows cited |

## Candidate ly_fw

- Tool: `tools/dl_ly_fw_patch.py`; file `muse_decode_layer/cand/ly_fw.py`, SHA256
  059f734b34f9a0c3b3695776f878862d10f5547c56a4d84c6859d3ca87476b17.
- Change (helper0 only): every CTA arrives with __threadfence + red.release.gpu; only CTAs with
  blockIdx.x < min(G, M x 4) poll (nanosleep only when that count > 8), the rest return; CTA 0 bumps the generation
  after its wait; the flag zeroing runs on the waiting CTAs.
- Happens-before: phase stores -> WS_JOIN / CTA_SYNC -> tid 0 fence.sc.gpu -> red.release.gpu; waiter tid 0
  ld.acquire.gpu reads G, the end of a modification order made only of these RMWs, so it synchronizes with every
  release; CTA_SYNC orders its threads after it. Progress: every one of the G CTAs executes exactly one unconditional
  arrival, the counter for this generation was zeroed by the previous launch's CTA 0 before that launch ended, the
  launch is cooperative (all CTAs co-resident), the poll is uncapped. Reset lifetime: unchanged parity scheme; the MoE
  has no PDL edge, so the next launch starts after every CTA of this one exited.
- Box: gate log /root/okc/.build/moegb/fw_gate.log, A/B prefix gb_fw, gb_fw_r2 under /root/okc/.build/prtl/.

## Gate (14:17 PT)

ly_fw gate rc=0: PDL8H guards at T = 1 / 8 / 16 / 32 with 0 ids mismatches; reference check PASS at all 12 cells
(/root/okc/.build/prtl/gb_fw/chk.log). Single-run box nets in that check (not an A/B): decode 1 sliding 64.86 us,
verify 1 sliding 95.55, decode 16 sliding 122.50.

## Native build check

Round 1 arms checked at 14:2x PT: ab_c_a.log (ly_stk4), ab_w_a.log (ly_fw) and chk.log each print
`G1b4 backend native repaired: cooperative+BF16` and `G_RELEASE_STARTUP_OK native (128, 32, 33, 256, 1, 8, 16)`;
none contains "native build failed", "fallback" or "marlin".

## How to read the result (next session)

The A/B runs on without the lane: `tools/ly_gb_ab.sh` with CTL=ly_stk4.py CAND=ly_fw.py TAG=fw PHASE=ab, two rounds
of `c_a w_a c_b w_b c_c`, started 14:17 PT. It is finished when the log ends in "all done" and
`pgrep -f ly_gb_ab.sh` is empty.

    ssh okc-h200 'tail -4 /root/okc/.build/moegb/fw_ab.log'
    ssh okc-h200 'cd /root/okc && python3 tools/ly_prtl_table.py .build/prtl/gb_fw'      # round 1
    ssh okc-h200 'cd /root/okc && python3 tools/ly_prtl_table.py .build/prtl/gb_fw_r2'   # round 2

Native build in round 2 (every arm must match the first grep and none the second):

    ssh okc-h200 'cd /root/okc/.build/prtl/gb_fw_r2 && grep -c "backend native" ab_*.log; grep -liE "native build failed|fallback|marlin" ab_*.log'

Judge only the touched cells: decode 1, verify 1 and decode 16, sliding first (helper0 TINY / TINY2 / MID). Every
other cell runs identical code, so swings there are noise (about 1.5 us on pair counts).

## A/B result (box 22:18-22:20 UTC, both rounds complete)

Round 2's native check passed: every arm log (ab_c_a, ab_w_a, ab_c_b, ab_w_b, ab_c_c) prints "backend native" once,
and none matches "native build failed", "fallback" or "marlin".
Values were checked against `tools/ly_prtl_table.py` on .build/prtl/gb_fw and gb_fw_r2. Each row gives w - c in us: round 1 mean, then its four pairs; round 2 mean, then its four pairs.

| Cell (touched) | r1 mean | r1 pairs | r2 mean | r2 pairs | thr r1 / r2 | verdict r1 / r2 |
|---|---|---|---|---|---|---|
| decode 1 sliding | +1.18 | +1.07 +1.02 +1.44 +1.23 | +0.58 | +0.62 +0.67 +0.70 +0.42 | 0.51 / 0.58 | loss / inside |
| decode 1 full | +0.89 | +1.63 +1.86 +0.46 -0.11 | +1.37 | +1.18 +1.57 +1.39 +1.46 | 1.15 / 0.90 | inside / loss |
| verify 1 sliding | +0.44 | +0.48 +0.46 +0.32 +0.45 | +0.37 | +0.58 +0.26 +0.34 +0.24 | 0.50 / 0.83 | inside |
| verify 1 full | +1.13 | +1.14 +0.18 +0.94 +1.70 | +0.61 | +0.32 -0.06 +0.43 +1.33 | 1.92 / 1.79 | inside |
| decode 16 sliding | +0.28 | +0.51 +0.08 +0.11 +0.24 | +0.25 | +0.51 +0.11 +0.10 +0.14 | 1.25 | inside |
| decode 16 full | +0.58 | +0.74 +0.56 +1.04 +0.21 | +0.85 | +0.40 +0.85 +1.81 +0.83 | 2.02 / 1.97 | inside |

## Verdict

ly_fw is closed as a loss. In both rounds, all 8 decode 1 sliding pairs, all 8 verify 1 sliding pairs and all
8 decode 16 sliding pairs were slower. The skinny grid barrier got slower when only the item CTAs waited. Possible
causes, none tested: red.release costs more than atom.acq_rel on the arrival side; the 4 item CTAs' poll without a
nanosleep competes with the arrivals at the counter's L2 slice; the generation bump and the flag zeroing now sit after
CTA 0's wait. No stamps were taken.

Ruled-out row:

| 2026-11-04 ly_fw (059f734b, `tools/dl_ly_fw_patch.py` on ly_stk4 a054c8d5): the TINY / TINY2 / MID grid barrier waited on only by the M x 4 final-item CTAs (red.release arrivals, the others exit, CTA 0 bumps the generation, the waiting CTAs zero the flags) | Box paired A/B vs ly_stk4, two rounds, native build in every arm: decode 1 sliding +1.18 / +0.58 us (8/8 pairs slower), verify 1 sliding +0.44 / +0.37 (8/8), decode 16 sliding +0.28 / +0.25 (8/8), decode 1 full +0.89 / +1.37; reference check PASS 12/12 | fewer pollers on the skinny MoE barrier counter as a decode 1 lever. At T = 1 every final item needs every phase-B unit of the token's jobs (each route's ss covers all 4096 columns), so only the barrier's own latency was in play. Untested: a hierarchical or per-SM arrival form, and stamps that split arrival skew from release latency |
