# Review: mode 37 phase A publish ordering and SiLU reciprocal, 2026-10-29

Independent, read-only review of `muse_moe_w4a16/submission.py` mode 37 ("main-lean-silu-bar"),
the build behind scored run 6307d0c91c8d4ada (commit 96f2c43). I did not run okc and did not
edit code. PTX citations are from the PTX ISA 9.4 HTML at
docs.nvidia.com/cuda/parallel-thread-execution, fetched 2026-10-29; quotes are verbatim.

## Verdict

**SOUND.** Dropping the writers' per-thread `fence.acq_rel.gpu` keeps the phase A publish
correct under the PTX memory model. The removed fence was not a link in the release chain and
was never a proxy fence, so the cross-proxy part of the argument is unchanged from mode 28. No
fence needs restoring.

One note that applies equally to mode 28 and to every generic-write, TMA-read pattern: the 9.4
text of proxy-preserved base causality order (8.9.5) names only "aliases" with "an alias proxy
fence" as the cross-proxy case. The generic-to-async case rests on 9.7.10.28.2 and on 9.7.15.4's
sentence that proxy fences compose with other synchronization. Mode 37 keeps a proxy fence on
both ends of the path, so it satisfies every reading of that rule that mode 28 satisfied. This is
a gap in the model's wording, not a defect in mode 37.

`rcp.approx.ftz.f32` for SiLU is not a precision concern under the task rules: SGLang's own
`silu_and_mul` is built with `--use_fast_math` and computes the same expression in FP32 with an
approximate divide, from BF16 gate and up, writing BF16.

## What the code does (96f2c43, unchanged at HEAD for these paths)

Writer side, `publish_a<MODE>` (submission.py:968 at HEAD), called from `epilogue_a` with
`PUB = !defer_a(MODE)`, which is true for 37:

```
each of 128 threads of the warpgroup:
    st.global (bf16 SwiGLU outputs into xb)      // generic, weak
    fence.proxy.async.global
    [fence.acq_rel.gpu]                          // mode 28 only; skipped when bar_release_a(37)
    bar.sync kBarWg + wg, 128
thread tid_wg == 0:
    red.release.gpu.global.add.s32 [done_as + j*16 + st], 1
```

`bar_release_a` is read only in `publish_a` (submission.py:971). The fixup path keeps
`fence_ar(MODE)` fences and is untouched by the change. Defer variants (32 to 35) are not mode 37.

Reader side, the x producer thread (submission.py:1871 and 1909):

```
spin: ld.acquire.gpu.global.s32 [done_as + j*16 + st] until >= 8   // 2 units x 4 warpgroups
fence.proxy.async.global
mbarrier.arrive.expect_tx [xfull]
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes  (reads xb)
```

## Question 1: does bar.sync plus one thread's gpu release publish every writer's stores?

Yes. The chain for a store W by writer thread T1, released by thread T0, observed by the x
producer T2 in another CTA:

1. W precedes T1's `bar.sync` in program order.
2. 8.9.4: "A bar{.cta}.sync or bar{.cta}.red or bar{.cta}.arrive operation synchronizes with a
   bar{.cta}.sync or bar{.cta}.red operation executed on the same barrier." So T1's `bar.sync`
   synchronizes with T0's. Scope does not enter this rule.
3. T0's `bar.sync` precedes its `red.release` in program order.
4. 8.8: a release pattern can be "A release operation on M", so `red.release.gpu` alone is a
   release pattern. 8.9.4: "A release pattern X synchronizes with an acquire pattern Y, if a write
   operation in X precedes a read operation in Y in observation order, and the first operation in
   X and the last operation in Y are morally strong." `red.release.gpu` and `ld.acquire.gpu` are
   both strong, gpu scope (includes the other thread), generic proxy, same 4-byte word: morally
   strong per 8.7. 8.9.2 lets observation order run through "an optional sequence of atomic
   read-modify-write operations", so the acquire that reads 8 synchronizes with each of the eight
   `red.release` adds, not only the last.
5. 8.9.5, base causality order: "X precedes Z in program order and Z precedes Y in base causality
   order", and "X precedes Z in base causality order and Z precedes Y in base causality order".
   Chaining 1 to 4 gives W before T2's acquire, and by program order before T2's later
   instructions.

The question of cumulativity through the CTA barrier is answered directly by two footnotes. 8.8
footnote 1: "For both release and acquire patterns, this effect is further extended to operations
in other threads through the transitive nature of causality order." 8.9.5 footnote 1: "The
transitivity of base causality order accounts for the 'cumulativity' of synchronizing
operations."

The sentence in 8.8 that limits a release pattern ("only affects operations occurring in program
order before the first instruction in that pattern") does not break this. In mode 37 the pattern
is the single `red.release`, and T0's `bar.sync` is before it in program order; the other
threads' stores reach it through the `bar.sync` synchronizes-with edge, which is exactly the
transitivity the footnotes describe. That limit is what made the older "fence before bar.sync,
relaxed atomic after" form insufficient (runs-moe.md, "Proxy ordering of the x tiles", step 3),
and it is why the release must be on the atomic after the barrier, which mode 37 has.

The 9.7.15.1 bar description agrees operationally: the barrier "guarantees that when the barrier
completes, prior memory accesses requested by this thread are performed relative to all threads
participating in the barrier".

What the removed `fence.acq_rel.gpu` did: before the barrier, followed by no strong write of its
own thread to `done_as`, it formed no release pattern for any writer except T0 (where the
`red.release` already is one). Its acquire half has nothing to order: it follows no strong read of
another CTA's data. It was redundant under the model.

## Question 2: is the proxy fence placement still sufficient?

Yes, and the change does not touch it. The read that must see W is the async-proxy read of
`cp.async.bulk`, a different proxy from W's generic store. The rules involved:

- 9.7.10.28.2: "Accessing the same memory location across multiple proxies needs a cross-proxy
  fence. For the async proxy, fence.proxy.async should be used to synchronize memory between
  generic proxy and the async proxy."
- 9.7.15.4: "Bi-directional proxy fences ... take effect within a single thread ... Their
  cross-proxy synchronizing effect composes with other forms of synchronization according to the
  rules of the Memory Consistency Model."
- 8.9.1.1: `cp.async.bulk` operations "are ordered after prior instructions in the same thread".

Path in mode 37: W, then T1's `fence.proxy.async.global` (program order), then the `bar.sync`
edge, T0's `red.release`, the acquire in T2, T2's `fence.proxy.async.global` (program order),
then the bulk copy, ordered after it by 8.9.1.1. A proxy fence sits in the writing thread
directly after W, and another in the issuing thread directly before the async operation. Whether
one reads the composition rule as "a proxy fence by the writer, composed with later
synchronization" or as "a proxy fence by the issuer after it has synchronized", the path has it.
The removed `fence.acq_rel.gpu` is a generic memory fence, not a proxy fence, so under no reading
was it a link in the cross-proxy ordering. The writer-side proxy fence followed by a CTA barrier,
with one other thread issuing the async operation, is the same shape NVIDIA uses for
generic-smem-write-then-TMA-store (fence.proxy.async.shared::cta per thread, then barrier, then
one thread issues); I cite that from memory as precedent, not as proof.

`.global` on the fences matches the state space of both W and the bulk copy's source.

Wording gap, shared with mode 28: the 8.9.5 definition of proxy-preserved base causality order in
9.4 lists three cases, the third being "X and Y are aliases and there is an alias proxy fence
along the base causality path from X to Y". The generic-to-async case is not spelled out there.
runs-moe.md ("Proxy ordering of the x tiles", paragraph "What the model requires") paraphrases
that bullet as "aliases with a proxy fence on the path", which generalizes the text; it would be
more exact to cite the bullet as written and rest the async case on 9.7.10.28.2 and 9.7.15.4 as
above.

## Things checked and found unchanged from mode 28

- Count of 8: two units times four warpgroups, all `red.release.gpu` adds, combined through the
  RMW chain (8.9.2). Unchanged.
- Completion and slot reuse (mbarrier `xfull`, implicit generic-async proxy fence on bulk
  completion per 9.7.10.28.2, `xempty` with `fence.proxy.async.shared::cta`): unchanged.
- xb is written once per launch; cross-launch reuse rests on stream order or
  `griddepcontrol.wait`. Unchanged.
- `spin_ge` gives up after a timeout and proceeds with a flag set in `p.ctr`. That is a failure
  path that predates mode 37 and is the same for 28; spin flags were 0 in the logged checks.
- Compiler ordering: the stores are plain C++ stores and every following asm (`fence.proxy`,
  `bar.sync`, `red.release`) has a `"memory"` clobber, so the compiler cannot sink the stores past
  them.

## SiLU reciprocal (`rcp.approx.ftz.f32`, modes 36 to 40)

Ours (submission.py `epilogue_a`): gate and up are FP32 accumulators scaled and rounded to bf16
(`bf16r`), then `h = g * rcp.approx.ftz(1 + __expf(-g)) * u` in FP32, stored as bf16.

SGLang's baseline (`baseline.py` calls `sglang.kernels.ops.activation.activation.silu_and_mul`
with `expert_ids`). In sgl-project/sglang at main 686cf98 (fetched 2026-10-29; the judge's pinned
version may differ, but the call signature matches):

- `python/sglang/kernels/ops/activation/activation.py`: `silu_and_mul` goes to
  `_run_activation_filtered_inplace`, which uses `activation_module(input.dtype)` with the default
  `fast_math=True`, so the JIT build gets `get_activation_cuda_cflags()`.
- `python/sglang/kernels/jit/utils/arch.py`: `get_activation_cuda_cflags()` returns
  `["--use_fast_math"]` on CUDA.
- `python/sglang/kernels/jit/csrc/elementwise/activation.cuh`: `return x_f32 / (1.0f +
  expf(-x_f32));` on gate and up converted from bf16 to FP32, product in FP32, cast to bf16.

Under `--use_fast_math` (which implies `-ftz=true -prec-div=false`) that division is an
approximate FP32 divide and `expf` is the fast intrinsic. PTX 9.7.3.8 gives `div.approx.f32` as
`a * (1/b)` with at most 2 ulp error, and `div.full.f32` at most 2 ulp; PTX 9.7.3.13 gives
`rcp.approx.f32` at most 1 ulp. I did not compile SGLang's kernel to see which divide nvcc
emits; either way both sides are FP32 with an error of a few FP32 ulp (about 2^-22 relative),
far below the bf16 rounding (2^-9) that both apply to the output.

FTZ: `1 + exp(-g) >= 1`, so the reciprocal's input is never denormal. Its output flushes to zero
only when `1 + exp(-g) > 2^126`, i.e. g below about -87, where the true SiLU is about 1e-36 in
magnitude; SGLang's fast-math build flushes there too.

Rules check: "BF16 wherever SGLang's kernel keeps BF16". SGLang keeps gate and up in bf16 and the
SwiGLU output in bf16, with FP32 arithmetic in between; mode 37 does the same. No activation is
held at lower precision than SGLang's, so the reciprocal is not a precision concern. The logged
numbers agree: self-tests 3.87e-3 to 3.90e-3 against FP32 for 37, and outputs of 36 versus 28
differ by 7e-5 to 1.5e-4 (runs-moe.md rows for 0b684d5db0834dbc and 987d024620e44eb7).
