# Exact pasted report: primary-source provenance

Evidence reviewed November 1, 2026. Repository evidence is pinned to coordinator commit
`6da000f`; projection experiment branch was at `08d8150`. This audit concerns the exact
[*Reaching the Top 10 in the Muse Spark Open Kernel Challenge* paste](evidence/2026-11-01-pasted-top10/reaching-top10.txt),
not the earlier community audit. No GPU job, optimization, or prefill work was performed.
The coordinator owns integration, the steering index, publication, and remote verification.

## H100 per-stage profile

The matching primary source is
[SatyamKluisz/kiro's Guild post, thread 120](https://guild.1nterface.ai/t/120).
It identifies its author and team, an alygg77-lineage decode chain, **H100 SXM**, the judge
image, cold cache **per kernel**, and a 3.35 TB/s H100 bandwidth reference. Its table matches
the paste's QKV 14.0 µs, pooler/router 27.7 µs, MoE 20.6 µs and final boundary 3.4 µs.
The post also reports approximately 139 MB per step and 58% whole-graph HBM utilization.

This is the author's reported measurement, not our H200 result. The post supplies no pinned
code revision, profile job identifier, raw Nsight counters, timeline artifact, or explicit
measurement date. A relative publication age and the agent's key-registration date do not
establish when the profiling occurred.

**Arithmetic check, not a new measurement:** the table sums to 139.1 MB and 92.1 µs.
Dividing those quantities by 3.35 TB/s gives approximately 45.1%. A 58% figure would instead
correspond to approximately 71.6 µs. Overlap or a separately measured graph duration could
explain that distinction, but neither is provided. Therefore the published table alone
does not independently reconstruct the 58% whole-graph claim; this does not prove it false.
It cannot establish recoverable time or utilization in our H200 implementation.

## Public BF16 result: 1.5186×

This belongs to [Nathen Fernandes's public repository](https://github.com/NathenFernandes/muse-bf16-proj),
not our team. Its immutable
[v18 result extract at commit `86aff75e6aca72511bf9b41b6dc190bf1e763373`](https://raw.githubusercontent.com/NathenFernandes/muse-bf16-proj/86aff75e6aca72511bf9b41b6dc190bf1e763373/results/v18.json)
identifies:

- Job `66d8b007f6c641ee`, problem `muse_bf16_proj`, H200.
- Finished `2026-10-24T09:44:46.900879+00:00`.
- Speedup `1.5185701646498175`; 95% interval `1.4957916121800703–1.5413487171195646`.
- Nine tests passed, no flags; preparation `53.798846702` seconds.

The repository's historical first-place statement concerns earlier competition standings;
it is not a current cutoff or a final-rank guarantee.

## Public BF16 result: 1.331–1.343×

The matching primary source is the
[thethings.ai projection report](https://thethings.ai/p/okc/muse-bf16-proj-report).
It reports a scored 1.331× result for `a9dfcf1c069f4f79` and a 1.343× check for
`7e998918e84d4ea8`, on RunPod H200 SXM. Its source artifacts are described with local
`~/projects/okc` paths; those paths do not identify a GitHub owner.

**The scored job's owner is independently resolved:** our archived
[official projection-board snapshot](evidence/2026-11-01-four-track/muse_bf16_proj.json)
contains the exact job `a9dfcf1c069f4f79`, team **`zzj`**, speedup
`1.3311931351047017`, and submission timestamp `1792981718.7524333`, corresponding to
`2026-10-26T02:28:38.752433+00:00`. The October 31 snapshots agree. Consequently this scored
result is external to **yelgabo**, rather than an older measurement of our repository.
The associated check's ownership and full receipt were not independently recovered.

The page's public JSON-LD metadata names the author namespace `okc`, publication
`2026-10-25T18:17:07.651481+00:00`, and modification
`2026-10-26T04:45:25.638225+00:00`. These are document dates, not benchmark execution dates.
An authenticated `okc status` request for the external scored job returned “no such job”;
that response does not invalidate the archived official board row or establish ownership.

## Use in the combined audit

1. Attribute the H100 figures to the Guild author and retain the missing-artifact caveat.
2. Attribute the two projection results separately to Nathen Fernandes and team `zzj`.
3. Do not treat public utilization, historical ranks, or either external implementation's
   tuning failures as measurements of our current candidate.
4. Use the planned current-candidate/SGLang graph timeline collection to test exposed
   dependency and materialization costs. Require correct graph capture, current prepared
   policy attribution, and per-shape coverage before interpreting gaps or proposing fusion.

No projected speedup, top-10 feasibility percentage, or new experiment is inferred here.
