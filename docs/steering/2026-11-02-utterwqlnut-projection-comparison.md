# Projection comparison: utterwqlnut and our current stack

Our existing projection design covers the new source's main ideas and has the stronger archived aggregate. Keep our base. The clearest component lead is their 32-token O projection, a small unpaired timing advantage that needs confirmation before implementation work.

Evidence base `358f997624f302241e7090bc0a2a8ef32e54153a`, November 2 Pacific. [Source identities and review scope](evidence/2026-11-02-utterwqlnut-projection-comparison/sources.json), [all 30 timing cells](evidence/2026-11-02-utterwqlnut-projection-comparison/timings.json). No kernel changed, foreign code executed or GPU/judge job started.

## Which implementations this compares

The external file is [utterwqlnut submission.py at e5c79030](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/submission.py), SHA256 `28efd0826317814da738e0783072732d716126ce60c115fc30dc70504c0190ca`.

Our [tracked standalone projection source](https://github.com/yelgabo/okc/blob/358f997624f302241e7090bc0a2a8ef32e54153a/muse_bf16_proj/submission.py) has SHA256 `6a4ea57210e43e13f0d8b2c8419e0974102853be0152bc745ab4a8ef674f3d33`. It contains SK/RD/RT, wide and library candidates. Our best scored standalone receipt is C2 `9eba96894ee1402e`; C3 `40b9f54be6fe4d4e` supplies a second historical comparison. Neither receipt is presented as a fresh measurement of the current tracked bytes.

Our [scored whole-layer G1 source](https://github.com/yelgabo/okc/blob/358f997624f302241e7090bc0a2a8ef32e54153a/muse_decode_layer/cand/g1b4_stage_release.py) is separately pinned at SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. It embeds the G-line projection backend and layer-specific dispatch. The tracked `muse_decode_layer/submission.py` is another source, not this scored incumbent.

## Design comparison

| Area | External source | Our tracked standalone source |
|---|---|---|
| One-token BF16 projections | Lossless 12-bit CUDA-core GEMV for all four matrices | Compressed CUDA-core GEMV candidates, plus SK/RD and raw alternatives selected per shape |
| 8 to 32 tokens | Raw BF16 WGMMA cluster GEMM for QKV and the small unpooler slice; primarily Triton elsewhere | Lossless compressed tensor-core SK; RD register streaming for pooler/O at the small bucket. Dense SK/RD reductions stay within a CTA. |
| 128-token QKV | Weight rows as WGMMA A, tokens as B, three-way cluster K split, eight-stage ring | Existing `b1n128c3s8h` uses the same broad orientation, three-way K split and eight-stage ring. The C2/C3 profiles confirm this route ran. |
| Weight movement in that wide GEMM | Pre-swizzled packed weights loaded by contiguous bulk copies; X by TMA | Raw weights and X through TMA tensor maps. This is a concrete implementation difference within the shared design. |
| Router | FP32 Triton at up to two tokens; then BF16 hi+lo Triton with output zeroing and split-K atomics | FP32 CUDA-core candidate at one token; custom hi+lo SK/RT candidates and wide WGMMA routes. RT uses ordered cluster reduction. |
| Large dense shapes | Mostly fixed Triton rows or cuBLAS fallback | Correctness-gated wide/CUTLASS candidates and cuBLASLt plan search with stable preferences |
| Selection and preparation | Offline-tuned tables; runtime compile/numerical checks enable eligible routes | Runtime FP64/probe/diagnostic checks and cold CUDA-graph tuning, plus repeated-round Lt selection |
| Build | NVRTC through cuda-python, with CuPy fallback for eligible kernels; targets detected SM architecture | PyTorch `load_inline`, CUDA `-O3`, explicit SM90a; separate CUTLASS and cuBLASLt wrappers |

Thus the cluster/WGMMA idea is already in our stack. Their main simplification is using fixed recipes and limiting 12-bit compression to one token. We retain compression across more small-token workloads and a larger candidate portfolio. Their source is 57,776 bytes versus our standalone's 226,787 bytes. That size difference establishes neither dead code nor performance.

The scored layer has further integration costs and opportunities. Its final `dense` wrapper can select grafted Lt plans; `_rq_gemm` is explicitly disabled. Its small-token pooler path can fuse BF16 pooler work with the required FP32 router input, and the projection wrappers participate in PDL scheduling. External standalone timings do not measure those fused boundaries or establish a safe direct substitution.

## What the receipts show

The author's [74b36974863c4d1e receipt](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/profiles/submission.74b36974863c4d1e.json) reports 1.356457x, CI 1.336110 to 1.376804, with 9/9 tests. Our [C2 receipt](../../muse_bf16_proj/profiles/scored.9eba96894ee1402e.json) reports 1.527806x, CI 1.504889 to 1.550724; our [C3 receipt](../../muse_bf16_proj/profiles/scored.40b9f54be6fe4d4e.json) reports 1.516022x, CI 1.493281 to 1.538762. These are different runs, with different seeds, baselines and preparation choices. The author's JSON also lacks a source-hash binding. No fresh paired speedup is established.

Adjusted medians in microseconds, lower is faster:

| Cell | External | Our C2 | Our C3 |
|---|---:|---:|---:|
| QKV, 16 tokens | 12.122 | 9.741 | 10.159 |
| QKV, 32 tokens | 12.817 | 10.557 | 10.708 |
| Pooler, 32 tokens | 18.109 | 15.084 | 14.899 |
| QKV, 128 tokens | 14.874 | 14.664 | 15.058 |
| O, 32 tokens | 5.568 | 5.869 | 6.098 |

C2 has lower times in 27/30 cells; C3 in 23/30. The external O32 path is a raw BF16 Triton GEMM with `BLOCK_M=32`, `BLOCK_N=64`, `BLOCK_K=128`, `SPLIT_K=1`, four warps and five stages. Our corresponding receipts name compressed SK. This is the strongest small exception to examine. Near-ties at one token and 128 tokens vary with the chosen own receipt and remain unresolved.

As a bounded arithmetic check, replacing only C2's O32 time with the external number while holding every other time and baseline fixed changes the six-load standalone score by a factor of 1.000973, about 0.10%. C3 gives about 0.17%. These are unpaired substitution sensitivities, not measured or predicted gains, and they do not transfer to the whole-layer score.

## Decision and next gate

Retain our complete implementation and current worker priorities. Their packed contiguous weight stream and explicit cluster-cost experiments are useful reference details. Their aggregate does not reopen our closed cluster geometries or justify another full-base port.

The newer [worker reassignment](2026-11-02-worker-repurpose-moe-bf16.md) at origin/main `d06d53fb` gives CUDA Optimizer P1 standalone BF16 ownership and puts MoE first. This comparison is reference input for that owner, not a competing assignment. The source and timing evidence above remains pinned to the stated evidence base.

If the BF16 owner selects this small lead over stronger opportunities, the cheapest decisive comparison is one matched O32 component test against our selected SK and raw control, with fresh inputs, independent state, cold graphs and duplicate controls. Require a repeated advantage beyond control spread, then price it in the full standalone dispatcher and separately in the layer before integration. Stop if the difference disappears or its overall contribution cannot justify the work. No test is dispatched by this report.

Review covered static preparation, final wrappers, compiler flags and both own receipts' actual profile names. It did not establish full correctness, ordering/lifetime safety or current compiled resources for the external source. Independent review was unavailable under the current no-new-agent instruction. The earlier [source-discovery report](2026-11-02-scout-utterwqlnut-projection-0328.md) retains retrieval and attribution limits.
