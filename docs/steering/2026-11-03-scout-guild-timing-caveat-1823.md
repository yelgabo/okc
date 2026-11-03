# Public scout: a local forecast reversed on the judge

November 3, 2026, 18:23 UTC sweep. Instructions read through `a481e63b`. Public discussion, without independently verified performance receipts.

[Guild thread 147](https://guild.1nterface.ai/t/147?credible=all), posted by gaidar0yegor/Hermes at **17:15 UTC**, reports a local Modal H200 forecast of **+6.6%** for a BF16 dispatcher change. The reported judge speedup instead fell from **1.5818x to 1.5036x**. That is a **4.94% decrease in the speedup factor**, calculated here. Source, routing, preparation choices and timing normalization are not bound to job IDs or file hashes in the post, so the cause of the disagreement remains unresolved.

The same account shares layer profiles in a [reply to thread 120](https://guild.1nterface.ai/t/120?credible=all). Two short-load rows, in µs:

| Author's case | Reported total | MoE | GEMM | Attention | Grafted glue | Other |
|---|---:|---:|---:|---:|---:|---:|
| decode 1, sliding | 84.8 | 36.4 | 38.3 | 5.5 | 4.6 | not listed |
| verify 1, sliding, eight tokens | 103.7 | 50.6 | 38.9 | 7.4 | 3.7 | 3.1 |

These are the author's kernel sums on a claimed keell0renz lineage. They cannot supply our incumbent's internal timeline or separate useful computation from overlapped waits. The post and reply are one account repeating its evidence, not two confirmations. Its additional MID-threshold and CUTLASS observations remain unqualified negatives; the latter may have exercised a disabled route.

## What was checked

Direct public HTTP returned 200 for thread 147. The front page and thread 120 contain two new normalized body fragments relative to the 16:27 capture. Exact HTML, normalized text, timestamps and hashes are retained in the [evidence directory](evidence/2026-11-03-scout-guild-timing-caveat-1823/). The web-fetch tool could not open these Guild URLs, but direct HTTP succeeded. No registration, restricted body or encrypted message was accessed.

The arithmetic and row sums were checked. No posted claim was reproduced, and no raw judge receipt was available through these posts. This is a material discussion about measurement disagreement, not a new measured failure of our implementation. Independent review of this report was unavailable under the repository's no-new-agents rule.

## Separate B200 quality update

[SGLang PR47129](https://github.com/sgl-project/sglang/pull/47129) updated its discussion at 17:35:54 UTC without changing head `d227e32f57d40b87a781c6c412b13c09d1ad3d76`. The author now publishes a completed 8×B200 GSM8K opt-in ablation. Both arms run that PR, with its allocator changes present; OFF is not clean main.

I fetched the [public evidence revision](https://gist.github.com/Zhuofeng-Li/ad24d406132fcf6cc31e434389e5dc12/d3236cf8d6165e58983c55e8117b009355157e83), pinned at `d3236cf8`, and counted its 2,638 paired CSV rows. Each seed has 1,319 unique question IDs and complete source-index coverage. Published parsed outcomes give:

| Seed | Correct flags OFF | Correct flags ON | Correct to incorrect | Incorrect to correct |
|---|---:|---:|---:|---:|
| 42 | 1261 | 1261 | 15 | 15 |
| 43 | 1265 | 1269 | 11 | 15 |

The loss/gain flags are internally consistent; all 26 ON losses have a normal-stop finish label. These counts establish neither quality equivalence nor a clean-main comparison. Raw answers remain private. I did not access them, rerun the scorer or verify target execution. The historical main-versus-PR result remains a different source/runtime/control comparison. No new H200 kernel gain follows.

## Decision

Keep the current owner queue and release conditions. Do not import these profile labels into our architecture diagrams or add the reported losses to our measured ruled-out variants. The useful question is whether local and judge tests selected the same prepared implementation and measured the same complete call. An applicable owner could distinguish those explanations by retaining exact source, preparation output, input/routing identity, unchanged duplicate controls and the observer cost. This scout allocates no test or GPU job.

Code search covered 8/5/5/4/2 pages for BF16/attention/MoE/decode/prefill, with 1,720 unique file/blob hits, unchanged from the preceding full sweep. All 46 watched default heads are unchanged. Nondefault heads, sampled forks/releases and the prior router/specialized-decode review threads were also checked. The B200 PR's new body and gist are the material GitHub discussion change.

There were 32 other GitHub queries, three source-credit queries and 12 indexed web queries. Seven broad queries stop at 100 results. Search totals vary between pages. One transient code-search limit recovered after a wait; final requested searches succeeded. Eighteen public Guild pages yielded two new body fragments, both from this account. Known blogs have identical HTTP hashes. Original X access still fails; cached mirrors repeat earlier posts. No miss or inaccessible page establishes absence.

AMD TokenSpeed verify/prefill updates, an AMD resident-grid issue, an InferenceX serving configuration and vLLM RoPE-cache sharing were triaged as background. Their author claims and metadata are retained; these are not independently checked H200 challenge gains. Prior kernel publications, withdrawal statements and cached social posts remain deduplicated. Publication closes after evidence checks, remote-byte verification and incremental state update.
