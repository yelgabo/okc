# Validation and integration

Source sections: compiler/SASS inspection, graphs, streams, sanitizer, numeric reference and promotion discipline.

## Compilation must fail visibly

This intake fixes five diagnostic wrappers: `moe_compile_check.sh`, `moe_compile_fused.sh`, `moe_compile_w4.sh`, `moe_compile_legacy.sh` and `dl_moe_compile_check.sh`, all under [tools](../../../tools/). They capture the Docker/compiler status before filtering output, keep the full log at `KEEP` or a named `.build` path, stop inner shells on failure and publish requested SASS only after a successful compile and disassembly. A filter with no matches does not turn a successful quiet compile into a failure.

[The regression suite](../../../tools/test_moe_compile_wrappers.py) exercises their shell entry points with controlled compiler, Docker and disassembler executables. It checks exact failure codes, retained compiler diagnostics, unchanged prior SASS on failure, quiet success and visible resource output. This proves wrapper behavior. It does not compile a CUDA kernel or validate GPU math.

Choose the extractor that matches the selected implementation. `moe_compile_fused.sh` assembles the standalone owned source and instantiates its variants. The older raw-literal and W4 tools describe older formats. `dl_moe_compile_check.sh` checks `_MOE1_SRC`, not the scored layer's final fork-native MoE. For the selected WGS source use [the documented CUDA 12.8 extraction gate](../../steering/2026-11-01-local-cuda128-compile.md), its explicit adapters and current exact source. Do not claim that an old fallback build validates c7c201ae.

## Inspect actual emitted instructions

Record selected entry names and source/PTX/CUBIN/SASS hashes. Hopper SASS uses `HGMMA` for WGMMA and `UBLKCP` for bulk copies; searching only for literal PTX mnemonics can miss them. Count fences, barriers, completion waits, exits and local loads/stores per actual entry and inspect the hot backedge loop. Keep a matching control build. An opcode count does not prove ordering or latency hiding.

CUDA 12.8 cross-compilation is a device-code gate. It does not validate target-toolkit host compilation, cooperative occupancy, graph capture, actual registers, numeric results or speed on H200. The existing [compiler note](../../steering/2026-11-01-local-cuda128-compile.md) explains source adapters. Preserve its disclosed boundaries and no-new-spill gate. Read [NVCC's option contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-compiler-driver-nvcc/index.html) before changing flags.

## Correctness, ordering and lifetime are separate

On an available GPU host, run focused memcheck, racecheck, initcheck and synccheck passes using the installed tool's supported features. Unsupported instrumentation or a timeout is a missing check, not a pass. Keep a PTX ordering and progress proof for async/cross-CTA protocols. See [Compute Sanitizer's tool descriptions](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Compare numeric output both with the task reference and with the same decoded quantized values and required stage rounding. A single FP32 matmul does not reproduce the two-GEMM MoE, SiLU or per-expert norm contract. Separate implementation error from intended quantization error. Test empty-local output, occupied-to-empty replay, neighboring shapes, independent prepared states and fresh input mutation. Never weaken the existing comparator to qualify a candidate.

CUDA graphs are already the timing mechanism. Keep scratch and weights alive for every captured consumer; eager completion alone does not retire graph references. Additional streams require explicit independent buffers and dependencies. PDL progress cannot rely on simultaneous residency with a persistent global-barrier kernel. Graph capture support and a source-level dependency proof need actual replay validation.

## Promotion

Follow correctness, matched complete-call improvement and end-to-end regression gates. A local win must pay packing, runtime metadata, replacement passes, observer overhead and both contractions where applicable. Preserve the other five layer loads and standalone decisions. Read [ruled-out.md](../../ruled-out.md) before allocation; record losing or stopped experiments with exact scope and a reopening trigger. Private checks and public eligibility remain separate.

Use `okc check FILE --no-wait --json` and wait only on its returned job ID. Refresh live shared quota/jobs first through the lane owner. This intake dispatches no check or public submission. D32 stays with its [existing end-to-end owner](../../steering/decode32/2026-11-02-02-end-to-end-ownership.md).
