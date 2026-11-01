# Stage-local sparse repair: source and local compiler gate

2026-11-01. Evidence base `43a3261`; exact owned C2 raw SHA
`5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`
(LF `7f4fd58cdd940f36c8d4d355f3de85687bc7f6ab04981842ce936db76d78adc2`).
The new prototype is OFF by default. No GPU execution, judge check or public
dispatch was performed for it. The normal C2 entry remains unchanged.

## Measured reason for changing representation

The previous global-record BEFORE-A version is closed: official
`2b537518091248f1` passed correctness and all 30 benchmarks, but scored
1.3941200708702308, CI [1.3732082698071773, 1.4150318719332842], versus
C2 `9eba96894ee1402e` at 1.5278064829557674. All ten target cells lost.
Root's receipt is `muse_bf16_proj/profiles/sk-xor-release.2b537518091248f1.receipt.json`;
the independent terminal analysis is MoE commit `8bbaa7f`. Passing admission,
math and progress gates did not establish a performance improvement.

The new mechanism changes record storage, not geometry or placement: append
the sparse u16 records to each bulk-copied weight stage, then repair from
shared memory before the same load-A/empty-release point. The global version's
header-dependent LDG.U16 instructions are present in its actual SASS. Their
exposed stall cost is **not** measured by the public profiler. No HBM-byte
saving or speedup is inferred from these instruction counts.

## Exact prototype and representation

`tools/proj_stage_xor.py` generates `.build/today-proj/stage-xor.py`, 438663 bytes,
SHA256 `b726aebd42c0749d3ff736fdb72d667d1143dce90819fc45b785580a1832bc2e`.
New CUDA value SHA256 is
`998b5efbf26f762cefdcc308b23629fa51dbbdc36569d0a466077aa44a4344da`.
C2 and the exact previous global-SX path remain independently selectable
controls inside the new module; no original dispatch is installed or tuned.
CUDA/CPP values are readable multiline literals. There is no runtime source
file access, compression, embedded cubin or external implementation.

For each stage/CTA, retain the original 12-bit payload and per-unit count.
Rebase each header's record start to the stage-local concatenation in
[warp, uu, nt] order. Keep the same u16 lane/register/half/exponent-XOR records.
`pmax` now means the maximum **actual** record count across the complete
stage/CTA population. It is never selected from a cap sweep. The per-stage
arena is `align16(2*pmax)` bytes; unused space is zero. The separate int32 table
contains only the exact original row map. Every consumed record and restored
BF16 bit pattern is unchanged. Nonfinite/admission rejection remains the
previous public packer's legal fallback; exhaustive bit transport is a broader
internal proof, not a claim that nonfinite GEMMs satisfy a numerical oracle.

`WBY = original_SX_WBY + align16(2*pmax)` and
`SST = align1024(XBY + WBY)`. Dynamic shared memory adds
`NSTAGE*(SST-original_SX_SST)`. Producer pointer stride, bulk-copy byte count,
full-barrier expected transaction bytes, ring offsets, barrier/correction
locations and allocation use these same values. The resource query exposes
the corresponding actual CUDA attributes; `sxs_expected_layout` independently
derives the required bytes and thread roles from the original configuration.

## Lifetime and ordering

The original successful full-mbarrier wait precedes all payload/header/record
reads. The single bulk copy includes the aligned arena and credits its bytes
to that full barrier. Every main-reader warp reads its complete repair records
using volatile `ld.shared.u16` with a compiler memory clobber, then loads A,
then follows the original early or late empty release. No new read survives
that reader's release. The original counted empty arrivals, producer waits,
proxy fences, full waits, CTA joins and final barriers are unchanged from
reviewed global SX. No additional escape-reader arrival is removed here.
The previous SX proof already removed correction readers/join by compiling
their entire work out; this change does not alter that decision.

PTX's [mbarrier test/wait acquire semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
and [bulk copy completion mechanism](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk)
are the existing publication pattern. This prototype preserves that pattern.
In particular, `LATE_ARRIVE = MT==2 && UPW==2` still retires after MMA;
other early iterations may still hold the slot. Safety does not establish
free overlap or imply that moving repair after A would hide its cost.

## Local evidence

Four actual Torch CPU tests in `proj_test_stage_xor.py` pass, including all
ten geometries, exact per-unit global/stage record sequence, unchanged
payload/row map, exact maximum capacity, empty stages, dense all-bit transport,
and a corrupt-count negative control. MoE independently reviewed exact
`b726aebd` in commit `023b2273c4a8eb3aa08cae48351ff95b2f342476` and passed actual stream fixtures for all ten geometries with
sparse/empty/one-512-record units and a dense all-bit population; there was
no source/lifetime blocker. That review does not qualify H200 math or speed.

CUDA 12.8 NVRTC/PTXAS sm90a compilation succeeds for all 45 unique entries
(60 full/NF requests). All entries have zero stack/spill stores/spill loads.
For every full/NF control and old global-SX entry, normalized opcode/operand
sequences equal the previous compiled artifact. HMMA counts are unchanged.
Selected full-entry results:

| Configuration | Registers C2/global/stage | Global U16 loads global→stage | Shared U16 loads global→stage |
|---|---:|---:|---:|
| QKV8 | 76/76/76 | 36→0 | 0→36 |
| QKV16 | 66/72/68 | 18→0 | 0→18 |
| QKV32 | 112/101/101 | 18→0 | 0→18 |
| O16 / unpool16 | 56/61/58 | 12→0 | 0→12 |
| O32 / unpool32 | 84/79/74 | 12→0 | 0→12 |
| Pool16 | 60/62/61 | 24→0 | 0→24 |
| Pool32 | 86/79/85 | 24→0 | 0→24 |
| Unpool8 | 58/52/54 | 14→0 | 0→14 |

These are static instruction occurrences, not dynamic transactions or
bandwidth measurements. Stage repair retains substantially more instructions
than C2 (QKV8 2400 versus 1160; QKV16 1432 versus 928); the sparse-loop
control/fragment-selection overhead could still outweigh shared access.
Per-thread registers are not uniformly lower than C2 or global SX. The
candidate has fewer reader threads than C2, but actual rounded allocation and
H200 admission must still be queried rather than inferred from register count.

`proj_stage_xor_capacity.py` generates the published natural dimensions in
order using CPU Torch seed 5151, preserving preceding random draws. It measures
an **independent CPU population**, not bit-identical owning H200 weights:

| QKV8 | QKV16 | QKV32 | O16/32 | Pool16/32 | Unpool16/32 | Unpool8 |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 7 | 7 | 6 | 9 | 6 | 5 |

All ten capacities fit existing SX stage-alignment slack, adding zero shared
allocation in this population. Arena transfers add 16 or 32 bytes per stage;
padding is several times the actual sparse-record payload in some cells,
though at most 0.131% of the corresponding weight payload here. There is no
claim that global traffic decreases. The actual supplied weights must derive
their own capacity; larger populations may increase SST or fail admission.

Evidence is under `evidence/2026-11-01-projection-stage-xor/`: exact artifact
manifest, compiler manifest/logs, compressed PTX/SASS evidence, all full/NF
resources/instruction hashes and CPU capacity data. The device-only compiler
uses declared fixed-width virtual headers; host Torch extension compilation,
CUDA runtime attributes and numerical execution are not covered.

Reproduce from the project checkout, using the sibling root's tracked C2
restorer if its frozen scratch file is absent:

```text
python ../okc/tools/proj_sk_xor_control.py .build/today-proj/control.py --root ../okc
python tools/proj_stage_xor.py .build/today-proj/control.py .build/today-proj/stage-xor.py
python tools/proj_test_stage_xor.py -v
python tools/proj_stage_xor_capacity.py .build/today-proj/stage-xor.py .build/today-proj/stage-capacity.json
python tools/proj_wait_compile.py .build/today-proj/stage-xor.cu /root/.cache/okc-stage-xor
```

The CPU and compiler commands use the corresponding isolated WSL Python
environment described in the compiler manifest; they do not launch a kernel.

## Score gate, next action and stop rule

The ten targets remain fixed. All other twenty official cells and arbitrary-M
C2 behavior stay unchanged. Using C2 `9eba...` and SGLang `bde8b4c49c4a46bd`,
a hypothetical uniform 3% cut to the **whole** ten SK operators gives a
full-30 score factor 1.011089; 5% gives 1.018707. These are score sensitivities,
not measured repair cost or predicted gains. Reaching the old C2 score from
the global-record loss is recovery, not an improvement.

The prototype clears bounded source/CPU/local-compile feasibility only.
Before any further root-authorized H200 measurement: freeze actual per-pack
capacity/layout and verify CUDA attributes, zero local memory, matching thread
roles/arrivals, and admission at least the corresponding C2 full/NF control;
retain independent FP64→BF16 normal/probe/onehot/outlier and fresh capture
checks. A same-process paired diagnostic should use independent C2 controls
and stage candidates, with global SX descriptive only. Require all ten targets,
all 30 score cells, all preregistered seeds, >=1% whole-score improvement and
twice null variation, plus existing regression guards. Stop if it only recovers
the failed variant, misses admission/math/prepare limits, or does not beat C2
repeatably. Do not tune capacity, tile, stage count, placement or selectively
drop losing targets after observing results. No public candidate is qualified
by this offline work.
