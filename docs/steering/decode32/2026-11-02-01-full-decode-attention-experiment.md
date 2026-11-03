# D32-ATTN-01: one-tile score lookahead candidate

November 2, 2026. Assignment `11bfcd4c`; pickup `69cf82b3`; captain reconciliation
`28346bf5`; current readback `65b94dcc`. D32-02 at `9dea190b` now assigns this chat the full D32 cycle. This is the
preregistered attention experiment record, continued directly through private
measurement. MID a6fe2769779e4299 is terminal and parked. GPU qualification and
speed remain unresolved. No GPU or public-submission call has run at this source
checkpoint. CLI authentication and read-only job/quota access succeeded.

## Deliverable and source pins

[candidate.py](evidence/D32-ATTN-01/candidate.py) is 886106 plain LF bytes, SHA256
`5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5`.
The control is `muse_decode_layer/cand/g1b4_stage_release.py`, SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
The newer qualified calibration `94ef698e91964dfa` retained that same source
with unused metadata appended. MID is a separate candidate and is excluded here.

The original attention literal stays intact. One readable asserted replacement
derives a second literal. Its SHA256 is
`3de8f5caac947c54af133fd83b157e3d1f1ab16ebcee8d61bcd0573f8ad0b26d`.
[native.patch](evidence/D32-ATTN-01/native.patch) shows the complete CUDA change.
Attribution and original notices remain. No external implementation was added.

Regenerate and check from the repository root:

```sh
python3 tools/d32_attn_worker_candidate.py --out docs/steering/decode32/evidence/D32-ATTN-01
PYTHONPATH=tools python3 -m unittest -v d32_attn_worker_test
python3 tools/d32_attn_worker_compile.py --self-test
python3 tools/d32_attn_worker_compile.py --out docs/steering/decode32/evidence/D32-ATTN-01
```

The compiler command uses an existing cached Docker image with pulling and
network access disabled. It installs nothing and exposes no GPU. Generation
checks the complete control hash and rejects drift. The
[manifest](evidence/D32-ATTN-01/manifest.json) pins all generated source files.

## Mechanism and selection

One-tile score lookahead carries QK results across the loop edge. Current
softmax/PV updates remain chronological, and current raw storage retires only
after its final PV reader. The existing producer ring, page TMA, half merge and
finalizer remain unchanged. This differs from cross-request continuity and the
prior L8 WGMMA schedules.

The compiler retained next-K load/current-exponential interleaving but grouped
QK math separately. Longer live scores and earlier waits may outweigh that
benefit. Runtime timing must decide the result. The 20-us target is unmeasured.

Selection covers batch 17 through 32, full context, q_len 1, page 1, Hq16/Hkv2/D64,
zero RoPE theta, unit metadata cache scale and finite positive score scales.
Original supported lengths and cache-size guards remain. Preparation enables
treatment only at compute capability 9.0 with 132 SMs. Other devices and metadata
use control. The treatment module is prepared explicitly in state; timed entry
cannot compile it. `D32_ATTN_PREPARED` records observed capability/SM count,
eligibility and the prepared treatment source hash before timed execution. Direct treatment use rejects wrong metadata, geometry or PDL.
All other standard cells and MoE/projection paths retain control.

## Verification

Six focused CPU tests pass. They execute actual extracted preparation and final
bindings with CPU stand-ins, compare complete launch tuples for all 12 standard
cells, test neighboring batches and fallback/guard negatives, and check source
identity and sampled slot events. [CPU output](evidence/D32-ATTN-01/cpu-tests.txt).

| Exact selected device instance | Control | Treatment |
|---|---:|---:|
| Registers / barriers | 128 / 3 | 128 / 3 |
| Stack / spill-store / spill-load bytes | 0 / 0 / 0 | 0 / 0 / 0 |
| Shared struct / dynamic allocation bytes | 61440 / 62464 | 61440 / 62464 |

Fresh CUDA 12.8.1 builds instantiate only `attn_kernel<1,true,0>` with the original
selected flags. This is device compilation. Full Torch host build, preparation
budget, H200 admission, GPU math/cache outputs, graph replay and speed remain
NOT_RUN. The original 56-replay startup banner covers experts; it does not qualify
this attention treatment. [Compile receipts](evidence/D32-ATTN-01/compile-summary.json).

Independent source review found no required fix to numerical ordering, patch-join
uniformity, buffer retirement, progress or host guards. The
[implementation notes](evidence/D32-ATTN-01/implementation-notes.md) retain the
lifetime table, scope of that proof, exact SASS locations, compiler adapters and
the fixed receipt-reader failure. That failure occurred after successful control
compilation; it supplies no kernel correctness or speed result.

## Registered private experiment

Decode32 Worker runs one ordinary complete-layer exploration of the exact
candidate after source/live-job/capacity checks and waits only on its own ID.
Confirm H200 capability/SM count, eligible prepared treatment, actual D32 metadata,
pinned source/flags and preparation time. A control fallback cannot qualify the
candidate. Require 13 correctness tests, 12 benchmarks, 12 complete profiles and
the unchanged expert startup gates. Runtime neighbors 17/24/31/32, B33 fallback,
empty/short/partial contexts, new-cache suffixes, private-state replay and guard
buffers remain required target coverage.

Lead with raw/adjusted D32 sliding/full medians, own baseline,
`C32_us = 46 * sliding_us + 16 * full_us`, actual path/resources and all-six-load
results. Reuse `tools/hotspots.py` and `tools/dl_compare_receipts.py`. For the latter,
pass the archived control and new terminal receipt with
`--control-job 94ef698e91964dfa --candidate-job NEW_JOB_ID`. Its load endpoints
are divided by 62; multiply by 62 for weighted microseconds. Its fixed short-load/
1% descriptive fields are not this task's acceptance rule. Cross-job comparisons
remain descriptive across inputs/nodes/drivers.

A useful D32 signal earns separately registered matched held-out confirmation
with independent mutable state and cold replay. A cheap smaller gain can
contribute. A correctness/progress/admission failure blocks this implementation;
a valid loss parks this exact lookahead without a ring/tile/register sweep.
The parked exact MID implementation is excluded; any new expert combination needs a prospective complete-dispatcher test.
No production-default or public promotion follows from these local checks.

The first run decides correctness/execution and supplies an exploratory D32
signal. Control94ef is the existing unchanged-source calibration; new inputs and
nodes prevent a causal delta. A directionally lower full endpoint and weighted
C32 without a clear regression earns matched held-out confirmation. No useful
D32 signal or a math/lifetime/progress failure parks this exact implementation.
No automatic ring/tile/register rescue sweep follows. Public posting is outside
this experiment. Accepted ID and terminal evidence go in STATUS.md before
another overlapping measurement.
