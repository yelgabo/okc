# Native gamma precompute: CLOSED_RESOURCE at the MAIN compiler gate

2026-11-01. Root-authorized fixed offline compiler screen; no GPU execution,
CUDA driver/API execution, service/official call, benchmark, tuning or production
edit. This is G integration work, separate from the earlier standalone main47
termination screen and the pre-expert Triton coefficient repair.

**Close exact full-family ONd6756bb5: MAIN spill stores increase4 to44 bytes
and spill loads32 to68 bytes in both lowered bool entries.** Registers and
stack stay unchanged. The no-resource-worsening gate fails despite fewer static
instructions. This is not a GPU correctness failure or a measured slowdown.
No short-only mask, adapter, coefficient-format or compiler-cap sweep follows.

## Frozen sources and fixed compile scope

|Source|Bytes|SHA256|
|---|---:|---|
|607e control|869076|ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab|
|new default OFF|870703|e17901d562b35f127c49110d222205fab254d078ae37e941a687774973a44c18|
|new ON|870702|d6756bb50a0362af9c9cd03d44a88c80fa593628254ffb2a931a643fccafdcad|

Owner source epoch: MoE branch d7e10e1/71a647a,
`tools/g_native_gamma_precompute.py`, source report
`2026-11-01-g-native-gamma-precompute-off.md`. Exact scratch inputs are
`.build/today-moe/gamma-precompute/{off,on}.py`; control is root tracked
`muse_decode_layer/cand/g1b4_stage_release.py` (owning607eb52bdaf24160).
Their manifests were checked before compiling.

Compiled exactly Tiny/Tiny2/Mid/B4/MAIN for all three sources, with both
`okc::moe_fused<false>` and `<true>` expressions: **15 fixed build requests,
30 entry instances**. This is not a geometry/cap/flag search. Actual prior
five-key compiler/source-factory flag recipe is reused: common NS3/NSF3/
FNST4/PRMTSC1/DQ7=1; per-key Tiny/Tiny2/Mid definitions; WGS16 and SC6W1
for B4/MAIN, B4FIN1 only for B4. ON appends only
`-DOKC_FORK_GAMMA_EFFECTIVE=1`. Default OFF uses the source's0 definition.

`tools/proj_g_native_gamma_compile.py` reuses installed NVRTC12.8.93,
PTXAS12.8.93, runtime headers12.8.90 and existing nvdisasm12.8.90. The
established driver uses compute_90a/sm_90a, C++17/O3, line info and spill
warnings; no fast-math flag. It extracts the original device prefix before
`torch/extension.h`, supplies the previously disclosed virtual fixed-width/
type-trait headers and declaration-only `wgs_cta_of` forward declaration.
It excludes Torch extension/host launch wrappers and does not link, capture,
execute native CUDA or query H200 admission. No toolchain was installed.

## Baseline to OFF: exact normalized compiler equivalence

For **all five keys**, baseline/OFF have identical normalized PTX, full
normalized SASS opcode/operand sequences, selected-entry resources and
launch bounds. PTX normalization removes only debug `.file`/`.loc`, debug
sections, comments and whitespace-only lines. SASS normalization removes
addresses/encoding bits and resolves labels to instruction indices; it
retains actual register operands, predicates and branch targets. This is
not a claim of identical debug metadata, scheduling encodings or CUBIN bytes.

The source helper/cache epoch therefore does not itself alter compiled OFF
device behavior in this selected scope. It does not certify new host packing,
fallback/format checks or inherited progress contracts.

## Actual resources: same values in both bool entries

Baseline and OFF are equal in every field below.

|Key|Regs baseline/ON|Barriers|Stack B baseline/ON|Spill store B baseline/ON|Spill load B baseline/ON|Max threads /min CTA per SM|
|---|---|---:|---|---|---|---|
|Tiny|147 /147|3|0 /0|0 /0|0 /0|288 /1|
|Tiny2|147 /147|3|0 /0|0 /0|0 /0|288 /1|
|Mid|155 /155|3|0 /0|0 /0|0 /0|288 /1|
|B4|96 /96|16|16 /16|120 /120|216 /216|576 /1|
|MAIN|96 /96|16|8 /8|**4 /44**|**32 /68**|576 /1|

All launch bounds remain identical. B4's large spills are pre-existing, not a
new gamma result. These are PTXAS stack/spill reports, not measured local-memory
transactions or CUDA `localSizeBytes`. Source shared layouts are unedited;
dynamic shared size/actual H200 occupancy and whole-grid admission were not
queried. Unchanged per-thread registers and bounds do not turn the MAIN spill
increase into a passed resource gate.

## ON instruction evidence and its limits

Existing-output census `proj_g_native_gamma_analyze.py` was run only to archive
the completed screen; no compilation/tuning continued after the MAIN failure.
Across both entries per build, the only **PTX opcode-count** changes are:

|Key|FP32 add count delta|BF16 RNE conversion delta|BF16 toFP32 conversion delta|
|---|---:|---:|---:|
|Tiny/Tiny2/Mid, each|-8|-8|-8|
|B4/MAIN, each|-212|-216|-216|

This is consistent with moving repeated effective-gamma add/round work into
preparation. The remaining BF16 RNE and BF16x2 conversions are present (for
example Tiny36 to28 scalar RNE, BF16x2 stays8; MAIN1296 to1080 scalar RNE,
BF16x2 stays8, across both entries). It is not a claim that all other virtual
register numbering, instruction scheduling or SASS allocation is identical.
The source owner's complete-expression/reversal proof establishes which source
operations changed. All explicit PTX synchronization/MMA opcode counts are
unchanged: no source fence, proxy fence, barrier, arrive/wait or MMA is removed.
Counts alone are not an independent memory-ordering proof.

Per-entry static SASS instruction totals, including padding:
Tiny 5272 to5264; Tiny2 5328 to5312; Mid 5304 to5288;
B4 26984 to26840; MAIN 24112 to23960. Fewer static instructions did **not**
prevent the MAIN allocation/spill regression. No latency or full-score inference
is supported, and the underlying compiler-liveness cause is not diagnosed here.

## Evidence and disposition

`evidence/2026-11-01-g-native-gamma-compile/` archives all15 builds: exact
input CUDA fragment, PTX, CUBIN, SASS (compressed evidence only), per-build
manifest/compiler logs, normalized hashes/counts, source pins and final blocker.
No compressed code/binary is used by an entrant or at runtime. Original local
outputs remain `/root/.cache/okc-g-native-gamma/`. The archive ledger records
uncompressed hashes and lengths.

Status **CLOSED_RESOURCE** for exact full-family d6756bb5. The default-OFF
equivalence milestone remains useful; ON is not qualified. Root holds the
factorial until a common conforming control is qualified. The separate short
RMW question remains distinct. No H200 job ID, score, CI or runtime failure
exists for this screen, and no retry/masked promotion is proposed.
