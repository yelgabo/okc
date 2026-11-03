# Public scout: specialized eight-token Muse Spark decode

Checked November 3, 2026 UTC, November 2 Pacific, against repository base `f29abe6c`. This sweep found a useful public implementation reference for the short verification path. It establishes no new H200 challenge result and assigns no work.

## Finding and source identity

[vLLM PR64077](https://github.com/vllm-project/vllm/pull/64077), authored by `djmmoss`, describes an NVIDIA-developed specialized Muse Spark Open decoder. Created October27, updated November3 at06:21:45UTC, it remains open and unmerged in the captured metadata. This is a newly found reference in our watch, not a claim that the implementation was first published today.

The inspected head is `31ebba9843cd6341d50434310d25b371eb2a12de` in [djmmoss/vllm](https://github.com/djmmoss/vllm/tree/31ebba9843cd6341d50434310d25b371eb2a12de). Its latest commit, November3 at05:01:33UTC, fixes a documentation build. The captured diff base is `9a519c0585b6cccb3d2363380725d974789551fd`. All42 changed files were downloaded at the full head pin and verified against their Git blob IDs and SHA256 hashes, totaling821606bytes. The [manifest](evidence/2026-11-02-scout-vllm-specialized-0724/specialized-source-manifest.json) and [source archive](evidence/2026-11-02-scout-vllm-specialized-0724/specialized-source.tar.gz) preserve the inspected bytes and their legal notices. This archive covers the PR's changed files, not its complete base checkout or dependency closure.

## What the author reports

The PR reports **1.53x single-stream throughput** on GB300 NVL72 at TP8/EP8:698.60 to1072.14tokens/s/user. The workload has one request, seven DSpark draft tokens and synthetic acceptance length6. Both arms reportedly share an allocation; the stock baseline is vLLM `1fd6337e22`. The performance setup also locally quantizes the model's two BF16 expert layers to NVFP4.

The author separately reports GSM8K agreement within one standard error and2735 locally passing tests. The returned42-file diff includes no test files, and no independently checked raw benchmark receipt accompanies this scout's source capture. Those measurements and test counts remain author claims. They are not SGLang-relative challenge scores, natural speculative-acceptance measurements, H200 results or evidence of a gain for our file.

## What the source checks establish

The [runtime guard](https://github.com/djmmoss/vllm/blob/31ebba9843cd6341d50434310d25b371eb2a12de/vllm/models/muse_spark/nvidia/specialized_decode/runtime/__init__.py) requires the SM100 device family, TP4/8/16, EP equal to TP, BF16 KV with block size16, DSpark with seven draft tokens and FULL decode graphs. It rejects pipeline parallelism and several other deployments. H200/SM90 and our FP8-KV attention contract are outside this configuration.

The [executor](https://github.com/djmmoss/vllm/blob/31ebba9843cd6341d50434310d25b371eb2a12de/vllm/models/muse_spark/nvidia/specialized_decode/runtime/decode.py) accepts one request verifying all eight tokens. Prefill, multiple requests, partial drafts and other unsupported steps use the stock forward path. This is a specialized provider loop with fused stages, rather than proof that a single persistent kernel replaces the whole layer.

The [MoE provider](https://github.com/djmmoss/vllm/blob/31ebba9843cd6341d50434310d25b371eb2a12de/vllm/models/muse_spark/nvidia/specialized_decode/runtime/providers.py) overlaps routing with the pooler on an auxiliary stream and joins before experts. On eligible NVFP4 paths, routing also prepares the experts' routing metadata. Expert activations are quantized into packed values and block scales. That differs from standalone W4A16's BF16 activation contract. The auxiliary-stream mechanism also needs a separate challenge-contract review before any transfer; this scout authorizes no such change.

The [small-M unpooler](https://github.com/djmmoss/vllm/blob/31ebba9843cd6341d50434310d25b371eb2a12de/vllm/models/muse_spark/nvidia/specialized_decode/cutedsl/gemm/unpooler.py) uses register operands and `mma.sync`, loads weights before its dependency wait, reduces cluster partials and can publish the result directly into the next boundary's Lamport mailbox. Its own runtime check requires SM100-family hardware. The source attributes the publication mechanism to an exact FlashInfer pin and preserves its notices.

The useful structural reference is the explicit producer-to-consumer handoff through GEMM epilogues and fused boundaries on the eight-token path. Whole-call benefit on H200 is unknown. Neither the reported1.53x nor the presence of a familiar MMA instruction establishes portability or a legal performance win in our benchmark.

## Scope and next action

The [final-day short-load directive](2026-11-02-final-day-directive.md) remains controlling. This report records a source reference and leaves current owners, accepted jobs and queues intact. It opens no new brief, GPU experiment or public submission. Cross-GPU collectives are outside our layer benchmark, and the broader persistent post-attention redesign remains on its existing later-work schedule.

If an owner later investigates a transfer, the first gate is the selected source's numerical, stream and ISA contract. Retain the current baseline and reject direct adoption where those contracts differ. Any eventual speed claim needs a complete H200 comparison under the owner's registered controls and finite stop. No expected microsecond saving can be assigned from this source inspection.

## Sweep coverage and other observations

All five identifiers were searched in GitHub code, repositories, commits, issues and PRs. Code pages covered BF16 1-4, attention 1-3, MoE 1-3, decode layer 1-3 and prefill1, with1028unique file/blob records and no new records relative to the preceding code capture. Totals changed across pages, so this is not a stable exhaustive snapshot. Code searches exclude our repository and two large starter collections; their watched heads were checked separately.

The32other queries include the challenge name, domain and Muse Spark kernel phrase. Commit/issue/PR coverage is newest100 per query; seven broad totals exceed100. Three source-credit searches cover `keell0renz`, `alygg77` and `vjeux/okc-2026`. The current first pages returned all their reported matches. Relevant branch/fork/release lists and five previously watched nondefault heads were checked. [Coverage](evidence/2026-11-02-scout-vllm-specialized-0724/query-coverage.json) records totals and limits.

One watched repository changed, neuro-spark-accel, from `b3e68b31` to `0d53e776`. The diff adds a Colab generator and changes its notebook to a BF16 projection wrapper. The inspected change provides no bound H200 result; its deployment wording alone establishes no useful new measured implementation. A source-credit hit in a daily GitHub catalog repeats the known alygg77 link. Both remain background observations.

A new indexed [Kempner serving recipe commit](https://github.com/KempnerInstitute/hpc-agentic-recipes/commit/cd81f355bef63610256c1e229a56134886c45f34), dated November3 at05:28UTC, publishes four-node H200 BF16 and one-node RTX NVFP4 serving configurations. Its README/engine diffs describe Marlin/FlashInfer choices for SM120. These are different deployment and precision settings, with author-reported throughput. They supply no standalone challenge kernel gain and remain a background reference.

Public Guild front page, page2 and all five problem communities were fetched. Nine comparable topics retained the same44public bodies. Topic48 returned a registered-agent notice, so its bodies were not read. Indexed X, Reddit, Hacker News, LinkedIn and participant-blog searches supplied older announcements or background accounts. The known withdrawal caveat and lxe orchestration account were deduplicated. Original X returned403; the Karmedge mirror and Kernarc remained inaccessible. No private Slack, Discord, registered-only Guild content or encrypted messages were accessed. These access limits require no user action for this sweep.

## Validation and gaps

[Evidence summary](evidence/2026-11-02-scout-vllm-specialized-0724/summary.json), [PR metadata](evidence/2026-11-02-scout-vllm-specialized-0724/specialized-pr.json), [comments and reviews](evidence/2026-11-02-scout-vllm-specialized-0724/specialized-reviews.json), compressed search captures and the source archive retain the receipts. A merge-conflict bot comment from October31 is historical; current mergeability was returned as unknown and no merge-ready claim follows.

All42 source blobs and archive members were checked. Source checks were performed in this chat without an independent reviewer. No foreign code was executed, compiled or benchmarked. No kernel, job, quota, owner, reminder or public submission changed. The alert is for a concrete public reference, with its hardware and evidence limits retained.
