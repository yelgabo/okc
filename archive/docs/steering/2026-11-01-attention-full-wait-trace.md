# Full-context owned attention: bounded causal timeline gate

Prepared against main `a63588b7d3f27ea77833b4c0f47810ea208d259d`; worker base `bd6624caab28c7bb1d54cf91ab444527f01b275d`. This is a diagnostic, not a submission or claimed optimization. No H200 job has been dispatched by this worker.

## Evidence and decision

The latest full-context scope report and `8ac6563343f74bcd` show that removing consumer math did not reliably lower mode51 whole latency: the read/no-math perturbation was approximately 110.69/197.06 us versus 110.88/188.38 us. This motivates producer attribution but does not measure physical bandwidth, idle SMs, or additive recoverable savings. Prior mode51 weighting, XPIPE, WST and mode49 page/ring/QK experiments remain closed. No new copy-order optimization is justified before separating producer slot pressure from consumer readiness.

`tools/attn_make_wait_trace.py` emits a native-only four-arm diagnostic. Controls and diagnostic arms all receive the same explicit BF16 attention-output rounding before RMS, matching the SGLang boundary; the existing FP32 judge-style reference remains separate. It is therefore a repaired control comparison, not an exact repeat of historical unrounded latency.

## Frozen scope and safeguards

Four affected cells: B16/B32, q_len1/q_len8, full context. Three identical input seeds 511019–511021 are used for correctness, stamps and timing. Two independent diagnostic builds and two independent controls have separate mutable state, caches and graph input allocations. Graph objects retain their private input backing allocations. Default XPIPE0, PBULK0 and PF3 are asserted; unsupported future source changes fail generation.

Sparse leader timestamps at tile ordinals 0/16/64/128 cover producer empty-slot wait, copy instruction issue through full-arrival issue, future page-ID issue region, consumer readiness wait, raw release and next-acquire cadence. The arrival marker is not a copy-completion timestamp. Negative publish-to-ready intervals are valid and decoded with signed low32 wrap. Metadata verifies identical producer/consumer tile totals and sample presence. All native acquires, releases, fences, barriers, cp.async instructions and attention math remain present. Sampling changes leader pacing and must pass the perturbation gate.

Before timing, runtime resource checks require zero local memory, at least two CTAs/SM and identical registers, local memory, shared memory, threads and occupancy between all arms. Each of the twelve cell/input measurements must satisfy both duplicate null and worst diagnostic-versus-control absolute perturbation <= max(0.5 us, 1% of the faster control). Otherwise stamps are descriptive only and cannot motivate an optimization. Every arm must pass math/row/cache checks; all 24 diagnostic records and 12 timing rows are mandatory. Report compression fails closed above 11,000 characters. No isolated region is treated as an additive speedup estimate.

## CPU/device compile results and remaining gates

Generated artifact: 361621 bytes, SHA256 `a882bd2ad7d9ebf36a5d4f739e6c0b60930a6e733e24fed06712090bc751d9c0`. CUDA source SHA256 `e806266e3f920e4ceb3ec3f72cb2f39ff6fc1ab374dbe61f6269bb2521d0d834`.

Local CUDA12.8 NVRTC/PTXAS sm90a device-only compilation passes for off/on MODE49 and MODE51. Both MODE49 instances use128 registers and2 barriers; both MODE51 instances use80 registers and16 barriers. All four report zero stack/spill bytes. These extracts remove Torch host headers and adapt the negative-infinity declaration; they do not validate host extension compilation, actual H200 occupancy, graph behavior or speed. Manifests/logs are under `profiles/attention-wait-trace-local/`. Six CPU tests pass (generation pin, placement, ordinal model, signed wrap, private graph/coverage assertions and realistic full-report codec).

Pending: independent source review; host/runtime compilation, runtime resource gate and bounded H200 receipt, owned by coordinator. No performance claim or public release is authorized by these CPU results. Stop after this one attribution check if timing perturbation fails or producer/consumer ordinal coverage disagrees. If transparent, inspect stable readiness versus empty-slot pressure across both detail builds and all three inputs before designing one legal copy/address/lifetime change. No old ring, QK or weighting sweep.

Reproduction (CPU generation only): `python tools/attn_make_wait_trace.py .build/today-attn/wait-trace.py --root C:/Users/gretc/Documents/GitHub/okc`.

## November 1, 05:48 PST terminal result

Owned check **3f44f891d0ef47db** completed on H200. All 16 official tests,
12 benchmark cells and 12 profiles completed; policy checks passed without flags.
The common BF16-rounded control scored **2.490692705×**, 95% CI
[2.453332314, 2.528053095]. This is a diagnostic control score, not an improvement
over the historical public implementation. The four scoped arms each completed
four cases without numerical failures; the diagnostic finished in 167 seconds.
[Raw receipt](../../../muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.receipt.json).
[Full log](../../../muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.log).
[Decoded measurements](../../../muse_attention_decode/profiles/wait-trace.3f44f891d0ef47db.decoded.json).

All four arms matched actual resource vectors. MODE49 uses 128 registers,
zero local memory, 58960 bytes shared memory, 256 threads and two CTAs/SM;
MODE51 uses 80 registers, zero local memory, 107104 bytes shared memory,
384 threads and two CTAs/SM. Despite this parity, **all 12 timing rows fail
the preregistered perturbation gate**.

| Cell | Faster control range, µs | Worst detail perturbation range, µs | Permitted perturbation range, µs |
|---|---:|---:|---:|
| B16 decode | 99.008–99.840 | 3.776–5.088 | 0.990–0.998 |
| B32 decode | 179.360–179.584 | 9.504–10.464 | 1.794–1.796 |
| B16 verify | 110.720–113.440 | 11.744–13.760 | 1.107–1.134 |
| B32 verify | 194.720–195.904 | 21.216–23.136 | 1.947–1.959 |

Retain the 24 raw detail records as descriptions of perturbed execution only.
They cannot identify the uninstrumented producer/consumer bottleneck, physical
memory traffic or recoverable score. Close this exact trace implementation;
do not spend another check on a stamp-placement sweep. Reopening requires a
different attribution mechanism with demonstrated low perturbation. Production
attention and the layer incumbent remain unchanged. No public submission occurred.
