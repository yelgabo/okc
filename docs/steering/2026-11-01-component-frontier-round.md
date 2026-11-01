# Non-prefill standings, new comparator and two MoE decisions

November 1, evidence starts at 8e3b681. The official CLI placed us at rank 377 overall and layer, 212 attention, 300 projection and 377 MoE at the start of this round. Layer and projection had no remaining checks in the UTC quota window. The owned public layer remains the recommended submission. G1b3 needs H200 qualification after the reset. [Board rows](evidence/2026-11-01-component-frontier/boards.json). [G1b3 qualification](2026-11-01-family-g-qualification.md).

## Standalone MoE

The controlled SC6 study, 05e51ddeb7014dd9, compared two lossless scale formats with identical expert math and inputs. Its byte control passed five official tests at 2.259311×. Four diagnostic arms passed 48 original, 48 fresh and 48 empty checks. Whole-score factors of 0.999807 and 1.007119 missed the duplicate-control noise gate, and 256 tokens regressed. Close the compression-only port. [Controlled test and resources](2026-11-01-moe-sc6-controlled.md).

A second H200 check, 1c6ebc76b91d4b7f, passed five official tests at 2.256420× for the repaired SC6 control. Writer issue times, CTA compute endpoints and actual token ownership sized an optimistic schedule in which each token can finish before unrelated experts. The most favorable samples gave only 0.34–0.43% of the six-load score on three seeds. Tracing changed latency by under 1%. Park fixed-owner MFIN at the measured 128/256 workloads. New MoE work needs evidence from the main computation, other loads or a genuinely different assignment. [Model and trace](2026-11-01-native-finalizer-frontier.md).

## Attention

An independent public code-content search found a complete attention submission in rohanarun/Auto-research-agent. We verified its source hash and ran the unmodified file off-board. Job cd6805e7ebce45ee passed 16 official tests at 2.639606× against our public 2.604290×. Rank 50 was 2.7093×. Decode batch 1 had the largest score-weighted opportunity. Substituting its two cross-run medians while holding the other loads fixed gives a 1.020034× sensitivity. Taking each lower median across all 12 cells gives 1.033593×. Neither represents a measured hybrid. The published study used the same speedup ratio across every case, so the per-shape decision uses our official receipts. [Twelve medians and source pin](evidence/2026-11-01-component-frontier/attention-shapes.md).

Two owned experiments target distinct loads. For sliding verify batch 1, let a cluster own a complete output head and finalize it locally. Require a proof of output/cache ownership, acceptable resources, current-input replay correctness and a repeatable positive six-load result. For decode batch 1, investigate the public file's two shorter QK accumulation chains against our four-step chain. Earlier flat-grid results do not measure the per-request kernels. Require a same-process comparison on both contexts, private inputs/caches, fresh output checks and duplicate controls. External standalone files remain measurement comparators.

## Source coverage

The user authorized registration of research handle `yelgabo_okc_research` on Kernarc. The account key is protected by Windows user-scoped encryption outside this repository. Subsequent requests were reads. Four relevant tags yielded 18 distinct discussions. We also read the cross-topic measurement index. There were no attachments in the file index on this visit.

- Post p_mvt72bdp3eca20d5 describes consumer work at expert unit ends in another implementation. Our H200 trace above determines whether the apparent wait is recoverable under the measured fixed-owner rule.
- Posts p_mvsyho369f57745b and p_mvto7tm4bb57cef9 identify two traps in related code. A preparation-time choice can change short-token router precision, and state copies can share a packed tensor. G1b3 pins its short chain and uses the FP32 router activation. Its Lt holder deep-copies the workspace and the plan copies its weight. The wrapper returning itself owns a compiled module. This is source evidence; a fresh-thread copy on H200 needs independent validation.
- Projection correction p_mvu4un8k37b169bc and reproduction p_mvr3rvawa3f630ea distinguish wins against stock cuBLAS from wins against tuned routes. G1b2 already selected eight cuBLASLt plans. The optimistic large-token dense substitution accounts for 0.42% of layer score across the twelve cells. The other published finalizer also omits required intermediate BF16 rounds. Neither warrants spending a check on that implementation. [Dense score budget](2026-11-01-layer-dense-stock-budget.md). [Finalizer audit](2026-11-01-rohan-finalizer-fusion-audit.md).

We screened public repository and gist metadata for 109 distinct handles in the archived top fifty. Two endpoints returned 404 and eleven lists reached the first-page cap of 100. This survey misses unrelated project names, other branches and alternate competition handles. The independent code-content search found the attention comparator. Gated discussion bodies and downloaded third-party kernels remain outside tracked evidence. [Public benchmark repository](https://github.com/rohanarun/Auto-research-agent/tree/018b7a102348ea3b73701e8be52d24f82d8c91ee/benchmarks).

Isolated CUDA 12.8 tools now cross-compile Hopper device code locally. They caught an undefined macro in the first frontier draft. Both final WGS variants compiled at 96 registers and zero spills. The H200 separately confirmed equal residency. [Compiler setup and limitations](2026-11-01-local-cuda128-compile.md).

No new public entry qualified from these MoE or comparator runs. The public standings remain outside the requested rank range.
