# Final-day directive: short loads only (rule of law, November 2 late Pacific)

User decision, 2026-11-02 late Pacific: the [top-100 gap reframe](2026-11-02-top100-gap-reframe.md) is
binding for the remaining day of the challenge. It supersedes the owner queues in
`2026-11-02-incumbent-hotspot-goal.md`, `2026-11-02-worker-repurpose-moe-bf16.md` and
`2026-11-02-top100-profile-priorities.md`. Every owner reads this file first on every wake. Release gates,
correctness rules, fence rules, quota rules and the no-submit-without-go-ahead rule are unchanged.

## The rule

Spend GPU checks and engineering time only on cells whose value is at least 0.2% of a board score per
microsecond saved. On every board that means **decode 1 and verify 1** (and decode 16 where it falls out
of the same change). Rank candidates by (distance to byte floor in microseconds) times (% per microsecond)
divided by (checks needed). Nothing at verify 16 or verify 32 qualifies. Decode32 stays closed.

Value per microsecond, from the public receipts (derived):

| Load | Layer | MoE | BF16 | Attention (sliding) |
|---|---:|---:|---:|---:|
| decode 1 | 0.212% | 0.62% | 0.47% | 1.76% |
| verify 1 (8 tokens) | 0.144% | 0.25% | 0.44% | 1.40% |
| decode 16 | 0.105% | 0.18% | 0.36% | 0.45% |
| decode 32 | 0.087% | 0.13% | 0.32% | - |
| verify 16 | 0.062% | 0.09% | 0.25% | - |
| verify 32 | 0.046% | 0.08% | 0.20% | - |

## Owner assignments for the day

| Owner | Task | First check | Stop |
|---|---|---|---|
| MoE worker (P0) | Tiny / Tiny2 build at T = 1 and T = 8: per-CTA `globaltimer` timeline (entry, first ring fill, each unit retire, A-to-B handoff, final pass), bounded buffer, then fix the largest exposed bubble (expected: A-to-B handoff with idle CTAs not prefetching phase B weights, or the final-pass dependency). | One private timeline check, then one candidate check. | Candidate ships to the promotion queue if T = 1 or T = 8 improves by 2 us or more with the other four loads inside noise. A fix in the Tiny build ports to the layer by file edit (same `c7c201ae` engine); do that port in the same day if a private layer check is available. |
| General Optimizer | Same lane, T = 8 and T = 16 only: the frozen row-owned build (`896b5350`) is dispatched only if the timeline shows the cross-warp reduction or ring refill on the critical path. Otherwise help with the Tiny timeline instrument. | At most one check. | No NS4 repeats, no T32+ work. |
| CUDA Optimizer (BF16) | Drop the m = 128/256 register-A brief. Target the m = 1-32 o, router and unpooler-slice cells: each is 1.7-2.5x over its byte floor and ramp-limited. Mechanism: every CTA issues its complete load set at entry (deeper first fill, more CTAs, or bulk loads in flight before any compute), no codec change. | One candidate check covering all 30 cells. | Ship if decode 1 or verify 1 improves by 1.5 us summed across those cells with no other cell losing more than noise. |
| Attention owner | decode 1 sliding (5.18 us) and verify 1 sliding (6.77 us) only. Remove the workspace round trip and ticket at batch 1: one cluster per request with a DSMEM merge, or one CTA group that owns all 2048 keys for its KV head. The `f74cbe27` cluster receipt (+21 us) describes a broken build, not the mechanism; state what differs before dispatch. | One candidate check. | Ship if both cells improve by 0.5 us or more and no other cell loses. |
| Layer owner | No megakernel today. Two allowed actions: (1) port a landed MoE Tiny win into `g1b4_stage_release.py` and run one private layer check; (2) if quota remains after that, the expert-to-unpooler fusion slice (CTAs finishing phase B stream the 8.4 MB unpooler slice and the finish). | At most two private layer checks; confirm remaining quota before each. | A layer check with a cleaner result than `607eb52bdaf24160` enters the promotion queue and is presented to the user with file, job and CI; public submission still needs the user's go-ahead. |
| Research / steering | No new briefs outside this table. Refill only with a short-load mechanism that names its cell, floor distance and expected microseconds. | - | - |

## Post-deadline backlog (recorded, not for today)

The persistent post-attention kernel for T <= 8 (section 4 of the reframe) and its two decisive tests
(section 5) are the next structural step after the deadline. The two-stage fusion microbench is the first
thing to build when work resumes.

## Reporting

Each owner appends one line per check to the relevant run log and to this file's ledger below: job, source
hash, affected cells before and after, decision. Chat-only results do not count.

## Ledger

- 2026-11-02 ~23:30 PT, user authorization: "yes, submit on confirmation" for the MoE factorized-output release (lead job 7479372115c64702, 5,698.5 us, 2.360x). The final-day lead submits the single-arm release file to the muse_moe_w4a16 board as soon as its own private check confirms within noise of that result, without a further ask. This authorization covers that one file only.
- 2026-11-02 ~23:45 PT, user authorization: "yes, run the repeats on all four boards". The lead resubmits each board's already-public best file with a changed trailing comment (the spec returns the earlier result for identical bytes and keeps the best run as the row), pacing to leave two public runs per board per day for real candidates. Base files: layer g1b4_stage_release.py (ba9e327c), MoE current_control_release.py (5df257af), BF16 C2 rebuilt by tools/proj_sk_xor_control.py (5610b8a6), attention d3eeeda repeat2 (b296c13a). Driver tools/repeat_draws.py, ledger .build/repeats/<problem>.jsonl, summarized here when done.

- MOE-TINY-TIMELINE-01 privatef4012a0531af4013, source106a05c5/observer8324f079, accepted after source05e3e875/registration6485937b. T1/T8 phase evidence pending, no before/after or gain yet. One own waiter; no other-owner polling.
