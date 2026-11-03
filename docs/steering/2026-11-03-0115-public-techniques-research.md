# Public techniques for the decode layer: GitHub, X and forum sweep, November 3 at 01:15 Pacific

Evidence snapshot: main at `31638851`. Board read 2026-11-03 09:43 UTC. Three parallel
read-only searches: GitHub code search keyed on the `# okc: problem=muse_decode_layer` header
and the top-30 handles, an X and web sweep including the Guild forum, and a freshness check of
the sources already catalogued in `docs/audit-2026-10-28.md` and the Family F and G reports.
No kernel, gate, judge job or submission changed. Query logs are in the session scratchpad
under `research-github/`, `research-web/` and `research-known/`.

## Summary

1. **The top of the board has no public code or writeup.** None of the top 30 handles has an
   OKC repository, post or thread. Five are not even searchable on GitHub. The leader
   (dolanzhao, 2.490x) and the 10-31 leader (mpeng19) have nothing public. The one top-40
   author who commented, keell0renz, wrote on 11-02 that their own current best submit is
   "judgemaxxed, showing how weak the checks are".
2. **The highest public decode-layer code is keell0renz at 2.264x** (their check; 2.2659 on the
   10-31 board). It is the file our Family G forks. Its layer file has not changed since
   11-01 07:09 UTC, the commit we measured.
3. **Our own largest measured gap is to that exact donor file.** On our account the unmodified
   donor checked 2.2488x (job `3a5f879df0d44825`, CI 2.215 to 2.283) against our retained
   fork at 2.1261x (CI 2.094 to 2.158). The intervals do not overlap. The fork's bundled
   repairs cost about 5.8% of score, which is roughly 190 board places today. The promotion
   ledger records that donor job as not approved for public submission.
4. **Rules changed 11-02:** readable source only, no compressed, encoded or obfuscated code,
   and the final review reruns top entries on nearby shapes. Our five files carry no encoded
   blobs. Prizes are paid on the serving track that opens 11-05, not on the kernel boards.
5. **Five techniques in the public record are absent from our ruled-out and runs logs.** The
   rest of what participants describe we have either adopted (12-bit codec, 6-bit scales,
   per-batch MoE builds, stream-K cost tables, fused pooler and router) or ruled out with the
   same sign as everyone else (L2 prefetch, megakernel, deeper rings, cluster split-K GEMV,
   hand WGMMA dense GEMMs at large T, router and top-k fusion).

## Official changes since October 30 (measured from llms.txt)

- 2026-11-02 08:00 PT: "submissions must be readable source code. Do not embed or load
  prebuilt binaries, and don't compress or obfuscate your code." Violating runs had to be
  withdrawn by 16:30 PT on 11-02; `okc withdraw <job>` is new and does not refund the run.
- Final review: "reruns the top entries on the scored shapes and on nearby ones. Positions
  come from the scored shapes; an entry that's much slower on the nearby shapes gets a
  closer look."
- Serving track: "The prizes are paid on the serving board; the kernel problems' boards
  carry no prizes." Opens 11-05 noon PT, closes 11-19, a diff against SGLang `a55ab7d5`.
- The official task text for `muse_decode_layer` is byte-identical to our `TASK.md`.
- Check of our files for `base64|zlib|lzma|bz2|decompress|marshal`: zero hits in all five.

## Where the public teams stand (self-reported scores, ranks at their snapshot date)

| team | layer | where stated | notes |
|---|---|---|---|
| keell0renz | 2.2659 (#38 on 10-31; #114 today at 2.2713) | repo README, LESSONS.md, issues on vjeux and alygg77 | built on alygg77 on vjeux; about 460 logged experiments; "the wins are small and stacked" |
| Sudhendra | 2.307 (#61 today) | Guild t/114 | L2 prefetch lost 0 to 18% on every shape |
| preonxtrun | 2.258 (#164) | Guild t/125 | flush-warp MoE redesign 9% slower; largest win was pinning prepare's MoE config choice |
| nikebp | 2.246 (#212) | Guild t/116, t/123 | 12-bit mma.sync-from-registers GEMM issue-bound, 3 to 5x off tensor floor |
| prateekm1007 | 2.272 | Guild t/114 reply | PDL in standalone attention gave zero delta; it pays only inside the layer chain |
| SatyamKluisz | 2.219 (#287) | Guild t/120 | per-stage T=1 table below |
| trxphxle | 2.1756 (#191 on 10-31) | repo README, BEST.json | history 1.08 to 1.70 (vjeux base) to 2.04 to 2.18 (alygg77 base) |
| vjeux | 2.1019 (#422) | repo | stopped 10-27 |
| Kushagrabainsla | 1.496 (#703) | repo EXPERIMENTS.md | measured negatives and one new mechanism |
| jacobpmiller | 1.437 | repo PROGRESS.md | mask-free q_len 8 verify probe |

Per-stage T=1 decode times on H200 posted by vkazas in Guild t/120, measured by them, not
by us: step 70.8 us, of which MoE 22.4, pooler and router 20.0, qkv 9.0, o 6.5, attention
5.0, unpooler 3.5, route 2.6, boundary 1.9. Our d1 sliding cell is 76.7 us. The same
author: "decode rank tracks the MoE row", and the per-expert fit is 8.2 us against a 6.4 us
DRAM floor with 3.9 us of dequant.

## Techniques not in our logs (candidates, all claimed by others, none measured by us)

| technique | source | their claim | our hotspot | grep of ruled-out and runs logs |
|---|---|---|---|---|
| PDL `launch_dependents` placed right after the producer's last weight issue, not at kernel entry | keell0renz LESSONS; Kushagrabainsla L11 (1.448 to 1.480); jacobpmiller (+0.8%, in noise) | +0.9% layer on the judge | every chain boundary | ruled-out line 309 and 317 tried trigger at entry (`179cecdac87b4872`, ONE_TRIG); late placement not found |
| Cold-code warming: idle producer warps execute the merge or epilogue path early with stores off, inside the timed call | Kushagrabainsla README A13; trxphxle lists it as their lead | merge 6928 to 2524 cycles; sliding verify B1/B16 12.3/21.6 to 9.9/17.2 us, bundled with other changes | attention (15.3%), MoE epilogues | zero hits for cold code, icache, instruction cache |
| Read PDL-predecessor outputs with `__ldcg` because L1 can hold the previous replay's data | Kushagrabainsla L11 | correctness: first PDL attempt failed recheck at repetition 5 | our PDL chain | `__ldcg` appears only for the MoE P2 reduction (ruled-out 375); a replay-only trap worth a source audit |
| Mask-free main loop for q_len 8 full-context verify | jacobpmiller probe | B16 207.6 to 172.8 us, B32 392.7 to 327.2 us on a check | v16 and v32 full | runs log 151 has an FP32 warp-specialized q_len 8 verify kernel; mask-free detail unverified against ours |
| Read-flush rather than write-flush of L2 when timing cold | MrNirmata notes (repo since deleted) | write flush inflates cold timings by up to 6 us | measurement only | not checked in our bench tools |

Hypothesis, not measured: the first item is the only one with a stated layer-level gain
from a team above us, and it is below our 0.8% control noise. None of the five reaches the
0.2x needed for rank 50.

## Techniques the public record and our logs agree are dead

| technique | public verdicts | our row |
|---|---|---|
| L2 prefetch of next weights during attention or from a side stream | Sudhendra 0 to 18% loss on every shape; keell0renz x0.99 to 1.00; Kushagrabainsla +10 to +25 us; trxphxle 0 | ruled-out 310, 318 |
| whole-layer megakernel | keell0renz capped by cross-SM hop cost (1.4 us idle, 6.3 us under 4.3 TB/s); preonxtrun IMA from a missing `fence.proxy.async`; ruwwww 2.075x with one | ruled-out 322, 326 |
| 4-CTA cluster split-K on the 96-CTA qkv GEMV | keell0renz x0.87 at T=1, clusters do not all fit the GPCs | ruled-out 134 to 137 |
| hand TMA/WGMMA dense GEMMs replacing cuBLASLt at large T | trxphxle 0.45 to 0.89x; nvbkdw 2 to 3x slower | ruled-out 134 to 142 |
| deeper expert weight rings, wider tiles | trxphxle -1 to -3% each; alygg77 saturates at 3 slots | ruled-out 207, 230 |
| router GEMM and top-k fusion | trxphxle -4.2%; Ranoobaba discarded | ruled-out 313 |
| fused-slice MoE | keell0renz six variants 5 to 13% slower | not separately tried; same mechanism as our slice losses |
| own FP32 residual or norm boundary that changes routing numerics | Kushagrabainsla: judge-planted experts then give the MoE more pairs, +10 to +42 us | ruled-out 308, 314, 315 |

## What this means for the last 44 hours

1. **The only measured lever larger than noise is the donor gap.** Submitting the unmodified
   keell0renz layer as the attributed G0 fork is a measured +5.8%, about rank 190, with
   non-overlapping intervals. The ledger says the user did not approve that submission, and
   the repairs were bundled for reasons recorded in the G repair inventory. Reopening it is
   the user's call. It does not reach rank 50 either.
2. **Rank 50 is not in the public record.** No public team is above 2.31x, and the teams
   above 2.4x publish nothing. Their attempt counts (20 to 93) suggest iteration, not one trick.
3. **The two cheap experiments** are late PDL trigger placement and the `__ldcg` audit. The
   first is a source-level change on a chain we already run; the second is a correctness
   audit that costs no quota.
4. **Serving track.** Prizes sit there, it opens 11-05, and nothing in this sweep covers it.

## Validation and stopping criteria

- A reopened G0 submission needs the user's explicit go, a fresh same-day check of the exact
  pinned file, and a receipt that passes all 13 cases before `okc submit`.
- Late PDL trigger: one check against a same-source control in the same session; keep only if
  the matched delta exceeds 0.8% on the six-load score.
- Stop the public search here. All platforms except GitHub and the Guild returned nothing
  new since 10-30, and the GitHub code search on the header string is exhausted at the
  repositories listed in the sub-audits.

## Gaps

- X posts behind login and the SGLang Slack channel were not readable.
- ZhimaoL (decode #29 on 10-31) deleted their branches on 11-01; no copy was found.
- The kernarc agents board needs a bearer key.
- All competitor scores above are self-reported at their snapshot dates; only the two jobs
  on our account (`3a5f879df0d44825`, `607eb52bdaf24160`) are our measurements.
