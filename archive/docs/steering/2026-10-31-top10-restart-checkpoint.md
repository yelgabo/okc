# Top-10 audit and Windows restart checkpoint, October 31, 2026

Evidence commit: `8f2c89b5ce005c53dbca1edb3f048a2b16475a67`. Checkout:
`C:\Users\gretc\Documents\GitHub\okc`, branch `main`, origin
`https://github.com/yelgabo/okc.git`. Official board snapshots captured October 31,
21:21 PDT; later official problem/API documentation reads are in the same evidence directory.
The user requested a stopping point before restarting Windows. No new GPU jobs,
kernel edits, public submissions or candidate promotion occurred in this task.

## Decision

**Top 10 is not established as reachable with the existing implementation before the
deadline.** The current layer needs 17.864% more score, equivalent to a uniform 15.156%
latency cut, to equal the *observed* rank-10 layer row. The measured remaining experiments
do not establish a portfolio of that size. This is not an impossibility claim: TASK's
idealized hardware floors leave room, but they are not an achievable performance forecast.
Do not promise top 10 or project donor gains onto the current candidate.

The official API reports kernel submissions close November 4 at 11:59 pm PT
(`2026-11-05T00:00:00-08:00`). Remaining quota for this account is **unknown**; public
limits are 20 checks and 10 scored runs per problem/day, not our available balance.

## Live standings and source limits

Saved public responses: [evidence directory](evidence/2026-10-31-top10).
Sources: [official leaderboard](https://openkernelchallenge.dev/leaderboard),
[official API instructions](https://openkernelchallenge.dev/llms.txt), and
[current layer problem](https://openkernelchallenge.dev/api/v1/problems/muse_decode_layer).
Ranks below are positions in the official ordered problem rows; overall has explicit ranks.

| Board | Our rank | Our score | Rank-10 score | Required score increase |
|---|---:|---:|---:|---:|
| Decode layer | 371 / 1139 | 2.018705x | 2.379322x | 17.864% |
| Attention | 293 / 1213 | 2.536566x | 2.909794x | 14.714% |
| BF16 projection | 324 / 1571 | 1.516978x | 1.604517x | 5.771% |
| MoE | 369 / 1271 | 2.215712x | 2.593077x | 17.031% |

Overall rank is 371, solved four. It matches the layer rank in this snapshot; the
public overall response exposes ranks, not a formula or score threshold. Equality here
does not prove a universal overall-ranking rule. Layer rank 50 is 2.267554x, rank 100
2.230178x. All are timestamped observations, not November 4 cutoffs. The rank-10 row is
olety, job `38db348c150a4f91`, screening pending, so it is provisional.

Official instructions say job details require a token and are limited to one's own jobs.
An unauthenticated request to `/api/v1/jobs/38db348c150a4f91` returned 401. Public board
rows expose aggregate scores, **not competitors' per-shape latencies**. Thus a genuine
top-10 per-shape comparison remains unavailable; do not substitute archived donor profiles
and label them rank-10 measurements.

## Baseline and measured per-shape performance

Keep `muse_decode_layer/submission.py` intact as the older clean control. Current best
board job is `7b84ed4346d5497b`, source `2840157`, build
`muse_decode_layer/cand/layer_m1rt51sc1g.py` (M1 runtime/stable selection, mode 51,
projection C1, fresh-input route-side gate). Receipt:
`muse_decode_layer/profiles/scored.7b84ed4346d5497b.json`, passed, 13 judge tests,
2.018705x, 95% CI 1.988424–2.048986. Current checked-in candidate SHA256:
`8112f37a4ba887af9dee689712e9f5f7f64689f56bb93f4ff5c85f7fd9ce82e7`.
The scored repeat had an added comment; do not confuse full-file SHA with API content_hash.
This is the **recommended existing submission candidate**, not evidence for a new promotion.

The preceding run of that build, `dc25fef4d65642a9`, was 2.0063x. Older stable M1+51
runs were 2.0069/2.0016/2.0148x. The latest best does not establish a new gain from
headline score alone. Owning rows: `docs/runs-decode-layer.md`, latest scored sections.

Times below are judge-derived microseconds from the best receipt, already accounting
for the judge's paired timing and replay overhead. Full profiles and baseline-as-submission
comparison are saved in [hotspots.md](evidence/2026-10-31-top10/hotspots.md).

| Load | Sliding ours / SGLang | Full ours / SGLang |
|---|---:|---:|
| Decode 1 | 76.8 / 171.2 | 84.7 / 182.5 |
| Decode 16 | 135.8 / 256.4 | 214.9 / 433.3 |
| Decode 32 | 169.6 / 266.3 | 322.5 / 614.1 |
| Verify 1 (8 tokens) | 117.4 / 263.0 | 126.4 / 272.8 |
| Verify 16 (128 tokens) | 271.3 / 532.3 | 351.0 / 692.4 |
| Verify 32 (256 tokens) | 348.2 / 742.9 | 508.8 / 1088.0 |

Score is the geometric mean of six load ratios, each load time
`46 * sliding + 16 * full`. The repaired model reconstructs 2.018705x from these
unrounded records. At every TASK floor it gives 3.874x: an idealized bound, not a plan.
Small loads have higher score sensitivity per microsecond; large isolated kernel wins
must still survive the weighted layer score. Profile rows include PDL waits and one routing
sample; summing them is not a measurement of available idle time or resource utilization.
No fresh occupancy, bandwidth or SM-utilization counters were collected here.

## Ranked backlog, reconciled with completed experiments

Inspect boundaries first, as requested, but the inspected results do not justify spending
the next check on another blanket fusion. Priority follows the surviving evidence.

| Priority | Work and evidence | Acceptance / stopping criterion |
|---|---|---|
| 0 | Authenticate CLI after restart; query shared quota; freeze candidate and reproduce a complete independent four-arm A/A control. Two-layer bench illegal accesses remain unexplained (`36f93b13d4c14d5b`, `5ae9ce318e014edf`), although single-file checks passed. | No performance attribution from a faulted or incomplete bench. Verify selected policies, seeds and own job ID; single-file correctness failure stops that candidate. |
| 1 | Re-measure current candidate against the pinned donor at d1/v1 with kernel names printed early. F-LAYER `00607fc6f8c84c77` compared the old release, not current overlapping front. Current router/routing already overlaps pooler; old serial-front gap is stale. | One measurement check. Implement only the largest current same-input gap. No target found means stop boundary speculation and pivot to MoE. Donor remains measurement-only under the user's logged policy. |
| 2 | MoE variable-size x tiles/deeper x ring for small jobs. Instrumented `43c35d18b5504eb5` finds x copy tails 2.2–2.4 us versus producer reaction 0.18–0.31 us; the variable-size design is unbuilt. Timing stamps perturb execution, so causal attribution remains provisional. | One hypothesis, current mode51 control, cold graph, held-out routing seeds. Preserve all fences; prove lifetimes/order. Promote only repeatable whole-MoE improvement beyond independent nulls and whole-layer confirmation. Stop after one confirming failed gate. |
| 3 | Attention q_len8 per-request descriptor/V-transpose address carries. Residual 0.42/0.42/0.87 us at v1 full/v16 sliding/v32 sliding in round5 attribution; potential ~0.5% standalone ceiling is not achieved gain. Decode1 full prologue residual0.54 us needs subphase measurement first. | Paired cold-L2 tests by decode/verify and sliding/full; compare independent duplicate controls, preserve FP32 math and fences. Layer promotion separate from standalone. Stop if no repeatable score gain. |
| 4 | Projection pending round9 F/V descriptor and tensor-map hints, plus codec factorial work model, already prepared on proj-r9. Current C2 wins paired over C1; standalone best C1 receipt `e1d3d341549c484d` 1.516978x is not proof C1 beats C2. | Run prepared factorial/reference controls before creating another sweep. Codec must be lossless and save end-to-end cold latency; reject hidden-decode assumptions unsupported by timings. No new GTICKET protocol without independent ordering review. |
| Parked | Pooler/router/metadata fusion and one-token o+boundary fusion. Existing concat front at high tokens already avoids a separate router launch. M-B slices1–3 add <=~1us; C-1 cooperative fusion loses1.16us in all nine rounds (`5ae9ce318e014edf`). | Reopen only with a current measured gap and a changed mechanism, e.g. persistent o matching cgemv alone or a barrier-free boundary. Do not repeat the same cooperative-grid design. |

Known losers: mode86 earlier x-slot release (`a00b49f8372042a9`) fails the1.5us gate;
modes76–79 fail their gates, both-phase stream-K79 loses16–32us. Fence-only ablation83
saves at most0.5us and cannot justify removing fences. Attention PBULK1 gains0.063–0.074%,
below its gate; PBULK2 loses1.4/2.9us. Projection large-tile library split-K
`8b9e8b7c1ecd4c3e` loses2.74–3.10us at qkv256; whole-tile sketch duplicated incumbent
geometries. See owning logs and `docs/ruled-out.md` for exact scope/revisit triggers.
Mode51 is already in the layer; integration is not new work.

The donor MoE's complete-file comparisons `40b8c798289a4258` / `218ff300de7d4c22`
were against mode47, not51. Likewise F-layer's ~9% gain against the old release cannot
be added to today's candidate. Neither result establishes the17.9% gap can be closed.

## Deep Research fact-check

Audited upload is the preserved `docs/steering/evidence/2026-10-29-community-audit/uploaded-research.txt`,
SHA256 `af3fb11f71a6def3a739edad29286e4dc61274915064cb3dd7e7558f8cb9e948`.
Its historical citation IDs are not usable source links. The prior source-backed
[community audit](2026-10-29-community-audit.md) remains the claim ledger; newer run logs
supersede its pending-work recommendations.

- Reopened [FlashInfer PR6976](https://github.com/flashinfer-ai/flashinfer/pull/6976):
  shorter prologues are a real upstream mechanism, not a predicted Muse speedup.
  Its H100/BF16/hd128 sparse-attention measurements do not transfer quantitatively.
- Reopened [7525](https://github.com/flashinfer-ai/flashinfer/issues/7525) and
  [7526](https://github.com/flashinfer-ai/flashinfer/issues/7526): useful PDL provenance
  audit patterns, not proof our H200 code has those bugs. Keep memory clobbers, proxy
  fences, every participating wait and the existing reviewed ordering arguments.
- Bounded-trap/C7512, prefetch regressions and replay bugs in participant reports remain
  externally reported claims, not independently reproduced here. Existing local SASS and
  replay work must be read before declaring those audits unfinished. The unbraced
  `if constexpr ... else for` CUDA12.8.1 bug is separately locally established; keep braces.
- Humming resource-policy advice is conditional WNA16/WGMMA guidance, not authority for
  a generic small-token sweep. zhebrak's archived2.014592x receipt proves that historical
  result, not an advantage over our current candidate. Neither source was freshly rerun here.
- Generic projection prefetch and broad rewrites should not consume quota without new
  measured opportunity. Prior integration and replay coverage tasks have largely completed.

## Changes and validation in this task

`tools/hotspots.py` now normalizes current status-event receipts and older profile JSONs,
retains job/board attribution, reads UTF-8 explicitly, and ignores seed/repeat count when
joining profile cases. `tools/dl_score_model.py` uses that reader, rejects wrong-problem,
duplicate and incomplete layer data, validates target/savings, and computes exact uniform
scaling instead of the former bounded binary search. It labels floor output as idealized.

Four tests in `tools/test_receipt_analysis.py` pass: judge-score reconstruction/scaling,
old/new schema parity, all12 baseline/profile joins, and incomplete-receipt rejection.
Windows sandbox initially denied temporary-file writes; the authorized unsandboxed rerun
passed4/4. Both tools ran successfully on our best receipt with baseline
`7d42771c3384449d`; saved outputs are in the evidence directory. `git diff --check` passed.
These are working analysis-code changes, **not GPU performance improvements**. No official
correctness test or new resource measurement was possible before the restart.

## Exact restart handoff

1. Restart Windows when convenient. Do not repeat the audit or reinstall Python.
2. WSL2.7.14 was installed through the official Microsoft.WSL winget package, with verified
   installer hash and an administrator elevation prompt. `wsl --version` then worked.
   `wsl --install --distribution Ubuntu --no-launch` reported successful Windows changes
   **requiring reboot**. No usable Ubuntu session or okc installation has been verified yet.
3. After reboot run `wsl --status` and `wsl --list --verbose`. If Ubuntu is absent, finish
   `wsl --install --distribution Ubuntu`; complete its first-user setup interactively.
4. In Ubuntu retrieve and inspect the official installer, then run
   `curl -fsSL https://openkernelchallenge.dev/install.sh | sh`. It selects Linuxamd64
   and verifies release SHA256SUMS. Official latest observed here:0.7.2. Do not attempt
   to run a Linux binary natively in PowerShell. WSL is a client only; timings use judge H200s,
   not the Windows RTX5070.
5. The user must run `~/.okc/bin/okc login` themselves: official instructions explicitly
   require a person to accept the rules and complete GitHub device login. Never commit tokens
   or accept rules on their behalf. Then query `okc problems --json`, board and recent jobs.
6. Resume in `/mnt/c/Users/gretc/Documents/GitHub/okc`; fetch and reconcile concurrent commits,
   read this report and latest logs, then start priority0/1 above. Use
   `okc check FILE --no-wait --json`, wait on that exact job, save receipt/profiles and log
   every outcome. No scored submission is authorized by this checkpoint.

Stop/pivot: keep the known-good candidate until a repeatable whole-layer gain survives
independent nulls, held-out routing and the existing fresh-input gates. Do not spend the
remaining pre-deadline budget on repeated sub-threshold tuning. If the first refreshed
attribution and bounded MoE experiment fail, pivot to preserving the best release and
standalone opportunities rather than asserting a top-10 trajectory. Complete and verify a
final candidate before November4; reserve time for correctness, replay and attribution.

## Addendum: post-restart installation completed, October 31, 2026

Resumed at `b57ed24`; fetched `origin/main`, with no newer commits or local changes.
After the restart WSL worked but had no distribution. Installed Ubuntu successfully
with `wsl --install --distribution Ubuntu --no-launch`; WSL can now execute Linux
commands. The distribution currently executes as root (no interactive first-user setup
has been completed). Do not assume a separate Linux user or its home directory exists.

Fetched and inspected the official installer again, then executed it in Ubuntu with
`OKC_INSTALL_DIR=/usr/local/bin OKC_NO_MODIFY_PATH=1`. Installed **okc 0.7.2**,
Linux amd64, verified SHA256 prefix `84883d7f0370` against the release checksum list.
`wsl -d Ubuntu --exec okc --version` succeeds. Installing in `/usr/local/bin` makes
the CLI available to a future normal Ubuntu user as well; credentials remain per-user.

`okc problems --json` successfully reached the official service and returned public
problem limits. `okc jobs --json` reports **not signed in**. No account quota has been
verified and no job dispatched. All four receipt-analysis tests also pass under Ubuntu's
Python3 from `/mnt/c/Users/gretc/Documents/GitHub/okc`.

Next user action in PowerShell:

```powershell
wsl -d Ubuntu --exec okc login
```

The user must personally accept the rules and complete GitHub device login, as required
by [the official instructions](https://openkernelchallenge.dev/llms.txt). Do not run
the login acceptance on their behalf. After login, verify the account without printing
tokens, query shared remaining quota/recent jobs, then resume priorities0/1 above.
If the user instead creates a normal Ubuntu account before login, use that same account
for subsequent commands so its per-user CLI configuration is read consistently.
