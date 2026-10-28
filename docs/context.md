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
