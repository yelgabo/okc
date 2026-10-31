# M-C: fewer, fused glue kernels at one token (design, 2026-10-31; not built)

Diagram: [plan-2026-10-31-m-c-t1-fusion.html](plan-2026-10-31-m-c-t1-fusion.html).

## Why (measured), and the hypothesis it tests

M-B (docs/plan-2026-10-31-m-b-pdl-chain.md) found that PDL on our own kernels saves at most
0.8 us at 8 tokens and nothing at 16 / 32, while the donor's layer (alygg77/okc-muse-spark-kernels
@ 6cae49a, read in the scratchpad only; every line of M-C would be our own code) overlaps 7-10 us
of kernel time at d1s / v1s / d16s / d32s (F-LAYER-0, checks 20abe6291eb04ed6 and
00607fc6f8c84c77). The hypothesis from M-B: the donor's lead at small token counts comes from
fewer, fused glue kernels, not from PDL itself. M-C tests it where the kernel count differs most,
one token.

Decode 1 sliding, per-kernel medians from the F-LAYER-0 ABK lines (check 00607fc6f8c84c77, us;
the donor's kernel roles are inferred from their order and size, not from its source):

| stage | ours | donor |
|---|---|---|
| qkv projection | cgemv40 qkv 10.8 | 11.3 |
| attention | attn_one 5.0 | 5.9 |
| o projection + attention boundary (residual, two norms, FP32 router input) | cgemv o 5.5 + boundary 3.9 = 9.4 | one kernel 7.0 |
| MoE front: pooler, pre-expert norm, router, routing | cgemv pooler 17.0 + pre_norm 1.9 + router 5.1 + routing 4.3 = 28.3 | 20.4 + 3.0 = 23.4 |
| MoE | sk8w3s 24.9 | 25.3 |
| unpooler + FFN boundary | 3.5 + 3.9 = 7.4 | 4.9 + 2.3 = 7.2 |
| step span | 85.9 (11 kernels) | 74.0 (8 kernels) |

The two fused stages account for about 7 us of the 12 us gap (2.4 at o + boundary, 4.9 at the
front), so they are the test.

## Design

### C-1: the o projection's epilogue forms boundary 1

Today `cgemv<64,1>` writes the BF16 branch (8192 values) and `attn_boundary_kernel` (a cluster of
8 CTAs) reads it with the stream, adds the residual in FP32, takes both sums of squares, and
writes the FP32 stream, the BF16 next input and the FP32 router copy.

Fused: each o CTA (2 output rows per CTA in cgemv) also loads its slice of the input stream and
of the gate / norm weights before its wait, adds the residual for its rows, and publishes its
rows' partial sum of squares. The full-row norm needs every CTA's partial, so the kernel ends
with a last-arriver join: each CTA adds its partial with `red.relaxed.gpu.global.add.f32`, then
`__threadfence()` and `__syncthreads()`, then thread 0 `atomicInc`s a counter; the CTA that
reads count - 1 computes the scale and a second pass over the 8192 values writes the normalized
outputs. This is the projection agent's `WV_GRED` join pattern with its PTX argument (the
fence.sc total order, docs/proj-port-guide.md "Global split-K join"), which needs no
co-residency, unlike a grid barrier over all CTAs. The counter and the accumulator are reset by
the last CTA, so graph replays need no host reset. The second pass costs one more read of 8192
FP32 values by one CTA (about 1 us on one SM); a cheaper form has every CTA spin on a
generation flag after its arrival, which needs co-residency and is rejected for that reason
(coordinator method: progress must never depend on co-residency).

### C-2: the one-token front in one launch

The FP32 front at one token runs the pooler as `cgemv<512,1>`, then `pre_norm`, the FP32 router
and `routing_kernel`. Fused: one kernel whose CTAs compute the pooler rows (as cgemv) and the
router logits (the router's 256 rows as extra CTAs, FP32 input), with the pre-expert norm of the
pooled row and the top-8 routing in the last-arriving CTA (the same join as C-1), which then
writes the MoE's metadata. The MoE launches as its PDL secondary, as it does behind route today.

### Ordering, including the donor's E2 pattern

The donor's `_gle_route` reads the o kernel's outputs before its `griddepcontrol.wait` while its
primary is the pooler (finding E2 of the donor audit). That is a two-back read whose safety rests
on every kernel between them waiting before it triggers. M-C does not copy it:

- C-1 reads before its wait only weights, the gate / norm parameters and the step's input
  stream (harness input); the branch it adds is its own output.
- C-2 reads the boundary's outputs (the FP32 router copy and the BF16 next input) only after its
  wait; its primary is C-1, so the wait alone makes them visible.
- Neither kernel triggers its dependents before its own wait, so the chain property that the
  FFN boundary's pre-wait read relies on (docs/plan-2026-10-31-m-b-pdl-chain.md, trigger-order
  rule) holds. The last-arriver join's ordering is the projection agent's argument cited above.

### Validation

Bitwise equality is not expected (the join changes the summation order of the norms). Gates:
FP32 references of the boundary outputs and the router logits within the existing self-test
bounds, routing ids equal to the separate path's on 100 judge-like inputs (or within the
route-variant check's rule), the fresh-input replay gate on the fused front + MoE graph, and a
clean counter after 2000 replays. SASS: fence, membar and atomic counts listed per kernel.

## Cost, budget and stopping rule

Two new kernels and their host wrappers. A first slice (C-1 alone, behind a switch timed by the
stable tuner against the two-kernel path) is about a day of work. Decision: an in-process paired
saving of at least 1.5 us at d1 sliding (9 of 9 rounds) for C-1, then a four-arm bench with
independent prepares. Stop if C-1 saves under 1 us: then the hypothesis that fusion explains the
donor's lead is rejected at this stage too, and the remaining gap is per-kernel speed.

## Gaps

- The donor's kernel roles at d1 are inferred from order and duration; its exact fusion
  boundaries were not read for this plan.
- Whether the pooler (512 rows per CTA group in cgemv) and the router (FP32, 256 rows) can share
  one launch without slowing the pooler's memory stream is unmeasured.
