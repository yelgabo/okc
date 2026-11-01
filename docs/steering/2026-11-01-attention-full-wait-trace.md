# Full-context owned attention: bounded causal timeline gate

Prepared against main `a63588b7d3f27ea77833b4c0f47810ea208d259d`; worker base `bd6624caab28c7bb1d54cf91ab444527f01b275d`. This is a diagnostic, not a submission or claimed optimization. No H200 job has been dispatched by this worker.

## Evidence and decision

The latest full-context scope report and `8ac6563343f74bcd` show that removing consumer math did not reliably lower mode51 whole latency: the read/no-math perturbation was approximately 110.69/197.06 us versus 110.88/188.38 us. This motivates producer attribution but does not measure physical bandwidth, idle SMs, or additive recoverable savings. Prior mode51 weighting, XPIPE, WST and mode49 page/ring/QK experiments remain closed. No new copy-order optimization is justified before separating producer slot pressure from consumer readiness.

`tools/attn_make_wait_trace.py` emits a native-only four-arm diagnostic. Controls and diagnostic arms all receive the same explicit BF16 attention-output rounding before RMS, matching the SGLang boundary; the existing FP32 judge-style reference remains separate. It is therefore a repaired control comparison, not an exact repeat of historical unrounded latency.

## Frozen scope and safeguards

Four affected cells: B16/B32, q_len1/q_len8, full context. Three identical input seeds 511019–511021 are used for correctness, stamps and timing. Two independent diagnostic builds and two independent controls have separate mutable state, caches and graph input allocations. Graph objects retain their private input backing allocations. Default XPIPE0, PBULK0 and PF3 are asserted; unsupported future source changes fail generation.

Sparse leader timestamps at tile ordinals 0/16/64/128 cover producer empty-slot wait, copy instruction issue through full-arrival issue, future page-ID issue region, consumer readiness wait, raw release and next-acquire cadence. The arrival marker is not a copy-completion timestamp. Negative publish-to-ready intervals are valid and decoded with signed low32 wrap. Metadata verifies identical producer/consumer tile totals and sample presence. All native acquires, releases, fences, barriers, cp.async instructions and attention math remain present. Sampling changes leader pacing and must pass the perturbation gate.

Before timing, runtime resource checks require zero local memory, at least two CTAs/SM and identical shared memory, threads and occupancy between all arms. Each of the twelve cell/input measurements must satisfy both duplicate null and worst diagnostic-versus-control absolute perturbation <= max(0.5 us, 1% of the faster control). Otherwise stamps are descriptive only and cannot motivate an optimization. Every arm must pass math/row/cache checks; all 24 diagnostic records and 12 timing rows are mandatory. Report compression fails closed above 11,000 characters. No isolated region is treated as an additive speedup estimate.

## CPU/device compile results and remaining gates

Generated artifact: 361629 bytes, SHA256 `dbb306301090e8209b43fac25826315eb37ef5a6425e696b63e5a8fa0b0255d4`. CUDA source SHA256 `e806266e3f920e4ceb3ec3f72cb2f39ff6fc1ab374dbe61f6269bb2521d0d834`.

Local CUDA12.8 NVRTC/PTXAS sm90a device-only compilation passes for off/on MODE49 and MODE51. Both MODE49 instances use128 registers and2 barriers; both MODE51 instances use80 registers and16 barriers. All four report zero stack/spill bytes. These extracts remove Torch host headers and adapt the negative-infinity declaration; they do not validate host extension compilation, actual H200 occupancy, graph behavior or speed. Manifests/logs are under `profiles/attention-wait-trace-local/`. Six CPU tests pass (generation pin, placement, ordinal model, signed wrap, private graph/coverage assertions and realistic full-report codec).

Pending: independent source review; host/runtime compilation, runtime resource gate and bounded H200 receipt, owned by coordinator. No performance claim or public release is authorized by these CPU results. Stop after this one attribution check if timing perturbation fails or producer/consumer ordinal coverage disagrees. If transparent, inspect stable readiness versus empty-slot pressure across both detail builds and all three inputs before designing one legal copy/address/lifetime change. No old ring, QK or weighting sweep.

Reproduction (CPU generation only): `python tools/attn_make_wait_trace.py .build/today-attn/wait-trace.py --root C:/Users/gretc/Documents/GitHub/okc`.
