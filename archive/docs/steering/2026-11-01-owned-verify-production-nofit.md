# Owned verify production closure: safe pruning does not fit

2026-11-01. Scope: source packaging against coordinator main `4c7233d` and its verify-pair slices; no GPU check, public action, or kernel tuning. This is a subordinate evidence report to the coordinator's verify-hybrid packaging report. The historical and private diagnostic bytes remain unchanged.

## Decision and byte budget

**NOFIT.** `tools/dl_owned_verify_production.py::build(root, batches=(16,32))` reproducibly reconstructs the frozen private arm from tracked `dl_verify_pair_slices.build(root)[0]['owned']`, then emits source-transparent closures. The input is 511911 bytes, SHA256 `384933697aadb2ebfc2abe672b216751e55fd5949e01c0c57d03caa586fe6fe3`.

| Selected batches / tokens | Raw bytes | SHA256 |
|---|---:|---|
| 16 / 128 | 389440 | 038a08e20effd39ecbc455a57b5e0587536a01d282218fdf85af05d07d5847f9 |
| 32 / 256 | 345746 | 83c5436ad5801422f51377b32b168d27889a1b3ea4dc1e02097e646179b8101f |
| both | 399155 | 6de124667ee5b56c81771b4fd5d2591dc62e1bcfbf60f18baa680438d543bf42 |

Coordinator's license-preserved G embedding is 721481 bytes and dispatcher is 3977 bytes, leaving strictly less than 274542 bytes for owned embedding under the 1MB limit. Even the smallest raw owned closure exceeds that allowance by 71204 bytes before additional embedding costs. The coordinator owns the actual combined mask matrix; these raw byte sizes are not claimed as complete wrapper sizes.

The unchanged active ONE (151256 bytes) and MoE (121690 bytes) literals alone total 272946 bytes. This explains the shallow-pruning limit, not a universal lower bound on future specialization. Rewriting compiled-out branches would require a separate source-equivalence and compiler qualification task; it is not authorized by this closure audit.

## What remains active

Attention plans are sliding128 `(8,1,3)`, sliding256 `(4,1,3)`, full128/full256 `(264,1,51)`. Native MoE remains main51/case10. QKV128 retains WD `b1n128c3s8h` cfg1; QKV256 retains recorded Lt layout0. Unpooler128/256 retain recorded Lt layouts3/1. O projection and concatenated FP32 pool/router torch.mm remain unchanged. Attention boundary128 remains SGLang;256 uses native geo4 `<1,512>`. FFN boundary remains SGLang. Lt plan values, BF16 boundaries, scratch ownership, retained modules and locks are preserved.

Removed only unreachable native projection/routing families, GEMV/SK kernels, unused boundary helpers and Python preparation/dependencies;128-only omits native boundary,256-only omits WD. CUDA values use readable raw multiline literals. Whole active ONE/MoE device literals are byte-identical; full WD literal is byte-identical. Applicable original attribution within retained literals is preserved. Full G notices remain the coordinator wrapper's responsibility.

Unsupported owned expert scales still fail closed. The parent wrapper must preflight this constraint and delegate unsupported weights to its existing G fallback; this closure does not broaden weight support or silently change fallbacks.

## Validation and reproducibility

Three source/CPU tests pass (final run 5.053s): all three closures have no unbound globals, selected source bodies and literal values match, no retained decorators are lost, recorded actual projection dispatch executes correctly, and selected plan/closure ownership is preserved. The tests reconstruct from tracked inputs, not ignored `.build` files. Default root is the test's repository; `OKC_HYBRID_ROOT` optionally selects a checkout containing the newer tracked generators.

PowerShell command used on this stale child branch:

```powershell
$env:OKC_HYBRID_ROOT='C:/Users/gretc/Documents/GitHub/okc'
python -m unittest discover -s tools -p test_dl_owned_verify_production.py -v
python tools/dl_owned_verify_sass_compare.py
```

Bounded CUDA12.8 NVRTC/PTXAS sm90a device-only original/pruned comparison:

| Selected device entry | Instructions | Registers | Stack/spills | Barriers | SASS equality |
|---|---:|---:|---|---:|---|
| WD `<1,128,3,1,false,8,BF16,0,true,0>` | 776 | 90 | 0/0 | 2 | exact normalized opcode/operands |
| Boundary `<1,512>` | 424 | 64 | 0/0 | 1 | exact normalized opcode/operands |
| Route `<float,true>` | 1664 | 80 | 0/0 | 1 | exact normalized opcode/operands |

All six raw nvdisasm texts are retained as `device.sass.gz`, together with compiler manifests/logs under `evidence/2026-11-01-owned-verify-production`. `tools/dl_owned_verify_sass_compare.py` independently replays the comparison and hashes: only offsets, whitespace, generated local labels, and the compilation-specific entry symbol are normalized. Predicates, opcodes, registers, instruction order and other operands remain. Updated normalized hashes are in `selected-sass-comparison.json`.

Compiler recipe: `python tools/dl_owned_verify_compile.py INPUT.py OUTDIR route|boundary|wd` in the isolated WSL CUDA12.8 environment. It uses the tracked `moe_decoder_compile.py` driver and `-DSK_PDL=1`. Adapters replace Torch/driver headers, supply equivalent route INFINITY/type traits and the CUDA12.8 128-byte aligned CUtensorMap declaration, and remove host wrappers. Initial adapter-only missing-INFINITY and missing-CUtensorMap failures were corrected; the route header failure log is retained. This proves device compilation and selected instruction equivalence, not CUDA host registration, tensor-map creation, capture, runtime admission, latency, or H200 correctness. Unchanged ONE/MoE literals were not newly compiled in this task.

## Stop and next action

Stop this packaging attempt: no GPU quota or public submission, no active math/scheduling change, and no automatic specialization to force fit. Preserve this generator as a reproducible safe closure and concrete byte budget. Any future deeper source specialization must first specify exact active-branch equivalence and selected compiler checks; feasibility remains unproven. There is no performance claim and no new best submission candidate.
