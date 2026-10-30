# M1: PTX ordering of the TMA reads around griddepcontrol.wait

Owed by the [attribution report](2026-10-30-layer-release-attribution.md) (section 3, gaps) and
the M1 row of `docs/ruled-out.md`. Code: the sk kernel in `_SKP_SRC` of the M1 file
(`git show 6377b5a:muse_decode_layer/submission.py`, `sk_kernel`, the `#if SK_PDL` block), as
patched by `tools/dl_gate_patch.py --m1`. Written 2026-10-30 by the layer agent. This is an
argument from the PTX ISA text, not a test result; no independent review has read it yet.

Citations name PTX ISA sections by title. The numbers follow the PTX ISA 8.x HTML that the
repository already cites (griddepcontrol 9.7.15.14); section numbers move between ISA versions,
so check the title if a number does not match.

## The two boundaries

M1 launches two single-CTA sk projection kernels as PDL secondaries (launch attribute
programmatic stream serialization, bit 16 of the config):

| secondary S | primary P | x (S's input, P's output) | P triggers |
|---|---|---|---|
| o projection (`sk_fwd`) | attention `attn_one_kernel`, q_len 1 and batch-1 q_len 8 instances (built with `ONE_TRIG`) | attention output, written by P with generic `st.global` | `griddepcontrol.launch_dependents` at entry |
| fused pooler + router (`skf_fwd`) | the attention boundary kernel | `ffn_input`, written by P with generic stores | at entry (the blanket PDL change) |

S may start once every CTA of P has triggered or exited. Both rows use the same kernel template,
so one argument covers both.

## 1. What S reads before griddepcontrol.wait, and who wrote it

Before the wait, only thread 0 touches global memory, and only to read:

| access | instruction | writer | writer live in the PDL window? |
|---|---|---|---|
| the x tensor map | `prefetch.tensormap` on a `__grid_constant__` kernel parameter | the host, at launch | no; kernel parameters are fixed when S is launched |
| weight units of the first min(NSTAGE, stages) stages | `cp.async.bulk` global to shared (`bulk_g2s`, `issue_w`) | prepare (`_skp` repack), finished and host-synchronized before any captured step | no |
| fused router pieces (`rws`, skf only) | `cp.async.bulk` | prepare | no |
| escape table (`ptab`, compressed configs) | `cp.async.bulk` into `tab` | prepare | no |

Everything else before the wait is S's own shared memory: `mbarrier.init`,
`fence.mbarrier_init`, `mbarrier.expect_tx`, the zero row. No thread of S writes global memory
before its own `griddepcontrol.wait`: every thread passes `__syncthreads()` and then executes
the wait, and the stores of `out`, `diag`, `rpart`, `rcount` and `rout` all come later. So S
cannot overwrite anything P (or an earlier kernel) still reads, and no pre-wait read has a
writer that can still be running.

Provenance is therefore the same as in a plain launch: the pre-wait data was written by work
that completed, and was synchronized with the host, before the graph replay began. The CUDA
programming model makes completed work's writes visible to later work; the async-proxy reads
of prepare-time data are the same reads every sk launch in the release already makes.

## 2. The x reads after the wait

Thread 0 issues x for the first ring of stages (`issue_x`: `cp.async.bulk.tensor`, 3-D, global to
shared, completing on the stage's full mbarrier) after its `griddepcontrol.wait`. The producer
warp issues the later stages, also after that thread's wait (it is the same block, past the
same `__syncthreads()` and wait).

`griddepcontrol` (PTX ISA 9.7.15.14): `griddepcontrol.wait` "causes the executing thread to wait
until all prerequisite grids in flight have completed and all the memory operations from the
prerequisite grids are performed and made visible to the current grid." So when thread 0's wait
returns, P has completed and P's stores of x are performed and visible to S.

The open point from the steering reports was the proxy. Memory consistency model, "Proxies"
(PTX ISA 8.x, section 8.6, title as there): accesses to one location through different proxies
need a proxy fence between them for ordering. `cp.async.bulk.tensor` (Data movement:
"Tensor copy", the bulk tensor instructions) reads global memory through the async proxy; the
ISA asks for `fence.proxy.async` to order generic-proxy and async-proxy accesses to the same
memory (membar / fence, "fence.proxy.async"). P's stores are generic; S's reads are async.

Two readings:

1. The wait gives a kernel-boundary guarantee. "Made visible to the current grid" names no
   proxy, and a plain kernel boundary has the same shape: every sk launch in the release reads
   its predecessor's generic stores through TMA with no proxy fence, as CUTLASS's sm90 kernels
   do after a stream boundary and after `cudaGridDependencySynchronize()`. Under this reading
   M1's order is correct as written in 6377b5a. This is precedent and a reading of the ISA
   sentence, not a derivation from the formal model.
2. The formal model's proxy rule applies across the wait. Then the causality chain is P's
   generic stores, then the wait (completion and visibility to thread 0 in the generic proxy),
   then thread 0's async-proxy read, and it lacks the proxy fence the "Proxies" section asks for.

The candidate does not rest on reading 1. `tools/dl_gate_patch.py --m1` adds
`fence.proxy.async.global` in thread 0 right after `griddepcontrol.wait` and before the first
`issue_x`. The chain becomes: P's generic stores of x, then thread 0's wait returns (P complete,
stores performed and visible), then thread 0's `fence.proxy.async.global` (orders thread 0's
generic-proxy view of global memory before its later async-proxy accesses), then thread 0's
`cp.async.bulk.tensor` reads of x. The producer lane (warp NW, lane 0) issues the x copies of
stages npre onward after its own `griddepcontrol.wait`; it gets its own
`fence.proxy.async.global` after the wait as well, so every thread that reads x through the
async proxy has the proxy fence in its own program order, without relying on causality through
thread 0. This is the pattern the MoE kernel uses for its x tiles (each issuing thread fences
again after its acquire; `docs/moe-port-guide.md`, "What the layer must not change").

SASS (CUDA 12.8.1, sm_90a, `tools/dl_moe_sass_counts.sh FILE skp1`): all 23 single-CTA sk
instances gain exactly two `FENCE.G` (0 to 2, thread 0 and the producer lane); FENCE.S,
MEMBAR, SYNCS, BAR, ERRBAR, UBLKCP, ACQBULK, ATOMG, REDG, LDL and STL are unchanged in all 39
kernels; instruction totals rise by 4 to 9. The fences are compiled only where `PDL` is true, so
the SK_PDL build pays them with M1 off too, and the release (SK_PDL 0) does not compile them.

## 3. The full barrier cannot complete on the weights alone

For each of the first npre stages, `issue_w` sets `mbarrier.expect_tx` to `stage_tx` = x bytes +
weight bytes (for a router piece, x bytes + piece bytes) before the wait, and posts only the
weight copy. The x copy posted after the wait completes the remaining transaction count on the
same barrier (mbarrier, "complete-tx" semantics, PTX ISA 9.7.13.15). Consumers wait on the full
barrier's phase, so no consumer reads a stage whose x has not landed.

## 4. Progress

S launches only after every CTA of P has triggered, that is, after all of P's CTAs are resident
(attention mode 11 is cooperative and needs that). S's CTAs then wait in hardware at
`griddepcontrol.wait`; they spin on nothing P does, and P never waits on S. So the early launch
cannot block P (the coordinator's rule: progress must not depend on co-residency).

## 5. What remains unproven

- Reading 1 versus reading 2 is not settled by the ISA text. The fence makes the question moot
  for the candidate at the cost measured in the M1 timing (both M1 arms carry it).
- The other PDL secondaries of the layer (pre-norm, boundary kernels, router, route_align, W4
  GEMMs) read their predecessors' outputs with generic loads, not TMA, after their waits; they
  were not re-audited here (docs/runs-decode-layer.md, PDL provenance table).
- The fresh-input gate (docs/runs-decode-layer.md, "Validation policy change") is the test
  evidence. Passing it does not prove the ordering; this argument and the fence do the work.
