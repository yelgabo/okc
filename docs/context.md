# Context: what this repo is for and what we know so far

Written 2026-10-27 from two earlier chats (one with Claude, one with ChatGPT) about the
[Open Kernel Challenge](https://openkernelchallenge.dev), plus the task files `okc init` wrote.

- Claude chat: https://claude.ai/share/0d838d06-5c71-4ffd-ad81-b89f7869cd86
- Claude's field guide: https://claude.ai/artifact/DRUUXPVgr7v5A2HcLyZ8TA
- ChatGPT chat: https://chatgpt.com/share/6ae13c1f-5170-83e8-9529-3dd09b85caa3 (its HTML guide,
  `open_kernel_challenge_guide.html`, only opens inside ChatGPT, so it could not be read here)

## Goal

Get a submission that beats SGLang's baseline (score above 1.00x). The top of the board is not
the goal.

## The challenge

Meta released Muse Spark Open, an 816B-parameter mixture-of-experts model. The challenge asks
for faster GPU code to run it on NVIDIA H200s. There are two tracks:

- **Kernel track** (Oct 22 to Nov 4, 11:59 pm PT). There are five problems, and each one is a
  single Python file with a `kernel_fn`. Each is timed against the kernel SGLang calls for the
  same step. It has no prizes, but the top 50 on the overall kernel board get priority in the
  serving queue.
- **Serving track** (Nov 5 to Nov 19). Contestants patch SGLang itself, and runs are scored on
  a full 8xH200 node. This is the track with the prizes.

## Concepts, in the order they came up

- **Kernel.** A kernel is one function that runs on the GPU, written in CUDA (NVIDIA's C++
  dialect) or Triton (a Python-based language). Kernels compute every number the model
  produces. A faster kernel computes the same math, equal within rounding, in less time. The
  judge rejects any output that drifts past a small tolerance from an FP64 reference.
- **Where kernels run.** Tokenizing the prompt runs on the CPU. Prefill then runs every prompt
  token through all 62 layers at once. Decode follows, with one step per generated token, and
  each step is a full pass through every layer. A layer is about a dozen kernel launches, so
  one decode step is roughly 800 launches. SGLang batches users together, so one decode step
  might process 1, 16 or 32 users' tokens; these are the "decode 1/16/32" loads.
- **SGLang.** SGLang is the open-source serving engine. It batches requests, manages the KV
  cache and launches kernels, mostly calling libraries: cuBLAS, FlashAttention 3, Marlin.
  `baseline.py` in each problem is exactly SGLang's call sequence. The task is to replace a
  piece of SGLang, not all of it.
- **The floor.** Every shape's time is bounded below by the larger of two figures: the bytes it
  moves divided by 4.8 TB/s, or the FLOPs it does divided by 989 TFLOP/s. Decode shapes are
  memory-bound, so the way to win is to read the weights once at full bandwidth with few
  launches. `TASK.md` prints the floor for every shape.
- **Speculative decoding ("verify").** A draft model guesses 8 tokens and the big model checks
  them all in one step, so verify 16 processes 16 users x 8 tokens = 128 tokens.

## What is unusual about Muse

The model has 256 experts with top-8 routing. It shrinks the hidden vector from 8192 to 4096
around the experts (the pooler and unpooler). Its router is FP32. Its attention is gated, and
46 of the 62 layers use a sliding 2048-token window while 16 attend to the full context. The
expert weights are NVFP4, which the H200 has to unpack to BF16 in software.

## The five problems

| problem | what it computes | plan |
|---|---|---|
| `muse_bf16_proj` | `x @ W.T` for qkv_gate, o, pooler, unpooler, router | start here |
| `muse_attention_decode` | QK-norm, RoPE, FP8 cache write, attention, norm-gate | fuse the 5 launches on sliding layers |
| `muse_moe_w4a16` | top-8 experts with NVFP4 weights | hard; later |
| `muse_decode_layer` | the whole decode layer | port the router win |
| `muse_prefill_layer` | the layer at prefill sizes | skip; not on the overall board |

Both chats agreed on starting with `muse_bf16_proj` and on the router being its weak spot.
ChatGPT's chat added one rule worth remembering: no submission may pick fewer than 8 experts
to save time.

## Why the router first

Per-shape SGLang times come from a public contestant repo
(https://github.com/ozgureyilmaz/okc-muse-kernels), as quoted in Claude's guide:

| load | m | SGLang router | floor |
|---|---|---|---|
| decode 1 | 1 | 10.5 us | 1.8 us |
| decode 16 | 16 | 14.8 us | 1.8 us |
| decode 32 | 32 | 15.4 us | 1.9 us |
| verify 1 | 8 | 26.4 us | 1.8 us |
| verify 16 | 128 | 16.5 us | 2.2 us |
| verify 32 | 256 | 17.6 us | 2.7 us |

These numbers have not been re-measured here. The first `okc check` will show this account's
own per-shape times.

## Rules that matter for code

- `kernel_fn` is captured as a CUDA graph. It must not sync with the host (`.item()`,
  `.cpu()`, printing a tensor), and it must not branch on input values. Branching on shapes
  is fine.
- `prepare(weights)` runs once, untimed, for up to 300 s. Compiling and repacking belong there.
- Weights stay at their own precision. The one exception is the FP32 router, which may be
  split into a BF16 hi + lo pair with FP32 accumulation.
- The router tolerance is 0.01% relative L2. About one row in eight of `x` is a single power
  of two (a probe row), and its output must be exact to within 0.1%.
- Native CUDA is allowed through `torch.utils.cpp_extension.load_inline`, and the rules page
  says so explicitly.
- The daily limits are 10 scored runs and 20 check runs per problem, reset at 00:00 UTC. A run
  has 10 minutes in total, compiling included.
- `okc login` must be run by a person, because it accepts the rules.

## Setup on this Mac

- `okc` 0.7.2 is installed at `~/.okc/bin/okc`, and the installer added it to `~/.zshrc`.
- There is no NVIDIA GPU here. Kernels are checked locally in two ways: a CPU emulation of
  the work split (`tools/emulate_router.py`) and an `sm_90a` compile in Docker
  (`tools/compile_check.sh`). Real timing comes only from `okc check`.

## Leads from ChatGPT's DeepSeek review (2026-10-27)

Source: https://chatgpt.com/share/6ae15b95-3b30-83e8-942d-321b6fd22d8b. Status as checked
against our local baseline files:

- **MoE finalizer below 32 tokens** (`muse_moe_w4a16`, `muse_decode_layer`): ChatGPT read in
  the pinned SGLang source that `fused_post_expert_norm_weighted_sum` falls back to separate
  PyTorch ops below `_MIN_TOKENS_FOR_FUSED = 32`. Not confirmed here; the function is inside the
  image, not in our files.
- **Single-token MoE shortcut skipped** (`muse_moe_w4a16`): confirmed.
  `muse_moe_w4a16/baseline.py:124` takes the shortcut only when `expert_map is None`, and lines
  88-91 build an expert map for the local experts.
- **Per-expert tile choice** for skewed routing (MoE): unexplored.
- **Gate/up paired layout with fused SwiGLU** (MoE): unexplored; ChatGPT estimates 1.03-1.10x.
- **K/V reuse across 8 query heads x 8 verify tokens** (`muse_attention_decode`): unexplored.
- **Lossless BF16 compression decoded in-kernel** (`muse_bf16_proj`): same idea as our 12-bit
  exponent code (about 25% smaller on the judge's Gaussian weights); not built.
- **Router split-K / orientation**: done differently on `muse_bf16_proj` (own kernels up to 32
  tokens). The split-K 32 setting it cites may be in `muse_decode_layer`'s fused routing.

## First leaderboard submissions (2026-10-27)

| problem | job | score (95% CI) | rank |
|---|---|---|---|
| muse_bf16_proj | ee4dd1b9e36e4910 | 1.139x (1.122-1.156) | 986 of 1202 |
| muse_moe_w4a16 | ebf5711b1b4f46ee | 1.324x (1.304-1.344) | 715 of 952 |
| muse_attention_decode | fd376b11ca9b49ef | 1.291x (1.272-1.315) | 735 of 911 |
| muse_decode_layer | 2c49c79bafcd4b56 | 1.207x (1.189-1.225) | 692 of 827 |

Overall kernel board: rank 787 of 1384. Leaders: 1.637x, 2.685x, 3.032x, 2.399x.

Update 2026-10-27: muse_decode_layer resubmitted with the attention port, job d156c66614694f25,
1.269x (1.250-1.288), rank 695 of 829 (from 1.207x). Overall rank 791 of 1391.

Update 2026-10-27: muse_moe_w4a16 resubmitted with our own NVFP4 W4A16 GEMM kernels (BF16 at
SGLang's boundaries), job 49af3c0930e24ff2, 1.699x (1.673-1.724), from 1.324x. Overall rank 596 of 1394.

Update 2026-10-27: muse_decode_layer resubmitted with the W4A16 MoE port, job 9846e29b2d664091,
1.390x (1.369-1.411), from 1.269x. Overall rank 597 of 1395.

muse_attention_decode: warp-specialized q_len 8 kernel checked at 1.348x (dcea57a9e84f441a) and
1.347x (dc37819eef384acb), NOT submitted: its Q.K accumulates in f16 and the softmax runs in half2,
where FA3 accumulates in FP32. TASK.md does not specify accumulator precision, but AGENTS.md warns
against computing at lower precision than the rules allow. Pending a decision or an FP32-accumulator
variant.

Update 2026-10-28: muse_attention_decode resubmitted with the FP32-accumulator warp-specialized
verify kernel, fenced variants only (var 16), job cb42811004a9440c, 1.321x (1.302-1.341), 16/16,
from 1.291x. Sliding verify unchanged vs the f16 kernel; full verify about 8% slower (b16
191.3 vs 175.8 us, b32 357.6 vs 331.7): about 3% is the fence, the rest the FP32 math, so the
math warps are now at least partly on the critical path on full layers. Next: backlog A1.

Update 2026-10-28:
- muse_decode_layer resubmitted: verify kernel port + X1 (routing metadata on the side stream),
  job 0e28c26271c94570, 1.403x (1.382-1.424), from 1.390x (its check bd28a84740ac4b84 was 1.433x).
  X5 (no zero pad before the FFN boundary, check b216cf4adb0d4033 1.449x) is held back: its
  boundary kernel keeps FP32 where SGLang's appears to round the branch to BF16, against
  TASK.md's activation rule. Saved at muse_decode_layer/experiments/x5_unpadded_boundary.py.
- muse_attention_decode: A1 (bounded-shift softmax, tau 8, var 272) submitted as
  submission_a1.py, job 06966cba66924747, 1.315x (1.295-1.338): no gain over var 16's 1.321x.
  Full verify b16 191.3 -> 193.6 us, b32 357.6 -> 364.7. Skipping rescales (emulated 18% -> 3%
  of steps on full verify) saved nothing, so the rescale is not what the FP32 math costs. The
  entry stays var 16 (submission.py).

Update 2026-10-28 (the rewrite push; details in docs/audit-2026-10-28.md, docs/gap-analysis-2026-10-28.md and the runs logs):
- Decision: the overall board ranks by the decode-layer entry, so every component rewrite is judged by its
  effect inside the layer. Target changed from "beat 1.0x" to top 50 (2.135x on 10-28, rising).
- Orchestration (W0, merged): X5 with SGLang's two BF16 roundings (bitwise), own pre-expert norm and attention
  boundary kernels, PDL across our chain (no gain of its own inside a captured graph), a front tuner that picks
  one concatenated pooler + router GEMM above 8 tokens; checks 1.431-1.463x.
- Re-implemented from the public entry's design memos (vjeux/okc-2026, Apache-2.0; re-implemented, not copied):
  MoE as one persistent launch (job 36b5b559b0344d0f, 1.990x on its board, from 1.699x), projections on a
  no-reduction skinny kernel with 12-bit weights and escape warps (job 3843af6418a84a72, 1.349x, from 1.139x),
  attention as one launch per call (its first judge run hung on a full-mask shuffle inside a K-only branch,
  fixed in 4352d4e; standalone 2.162x as b458555d3e3e4e9a).
- Layer entry: job f23f1c8171374285, 1.643x (1.618-1.667), rank 362 (from 513), with the fused MoE at every T
  (sk8w4s at T <= 32, the previous W4 chain at 128 by the tuner, the fat wgmma path at 256) and the sk
  projections at T <= 32. Decode 1 sliding 116.8 -> 91.8 us. Still open: attention (14.4 us at decode 1 sliding
  against the public 4.9; 250 and 354 us at full batch 32 against 170 and 186), the MoE at 32-128 tokens, and
  the dense GEMMs above 32 tokens.
- Layer entry: job cc7a07e2964a4d67, 1.761x (1.734-1.787), with the one-launch attention on every tp8 shape
  (full verify at batch 16 / 32 on the non-producer flat mode 19). Down on all 12 shapes; full batch 16 / 32
  -41 / -84 us at decode and -83 / -124 at verify. Attention at decode 1 sliding 6.1 us (public 4.9), full
  batch 32 181.9 / 236.2 us (public 170 / 186).

## 2026-10-28 23:25 UTC: unfenced MoE runs withdrawn

An independent review found that the MoE kernel writes x tiles with generic stores that another
CTA reads through `cp.async.bulk` (async proxy) with no generic-to-async proxy fence, which the
PTX memory model asks for. Fixed in 1099d9b (standalone and the layer copy); the fenced runs
scored 1.979x (1b20c5b390f044b2) and 1.985x (8078730bfa434074). The user chose to withdraw the
unfenced runs that ranked above them so the board shows only code whose ordering we can argue:
84de5bbafa344b7a (2.009x), 0400cfbdfa2845fe (2.007x), c524612c78454486 (2.002x),
36b5b559b0344d0f (1.990x). The layer entry a41ace0b94ba4369 (1.782x) already carries the fence.
