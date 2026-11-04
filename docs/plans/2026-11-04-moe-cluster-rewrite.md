# MoE cluster rewrite: one expert slice per thread-block cluster

Status: plan, 2026-11-04 00:10 PT. Owner: lead session. Implementers: Opus 5.5 agents, one per task, each in its own
worktree. Deadline context: kernel submissions close 2026-11-04 23:59 PT. Phase 0 is worth running whatever the clock
says; phases 3 and 4 land on the board only if every gate passes on the first try.

## 1. Why

The fused MoE (`okc::moe_fused`, five builds TINY / TINY2 / MID / B4 / MAIN inside the layer file, helper0 in the
standalone `muse_moe_w4a16`) is the largest kernel on every critical path: 23 us of 66 at decode 1, 58 to 65 at verify 1,
69 at decode 16, 77 to 80 at decode 32 (docs/runs-decode-layer.md, fusion audit table). Inside segments it streams
at or near the byte floor. Everything above the floor is synchronization that the current design ties to the stream:

| piece (decode 32, B4, box timeline) | us | mechanism |
|---|---:|---|
| segment switch | 4.9 to 6.3 per CTA | the previous expert's deferred flush (`wgs_flush`): bar.sync plus `MEMBAR.ALL.GPU` on the consumer warpgroups, which hold the weight slot for about 1.5 us |
| phase B waits | 3.6 | the down projection's x producer waits on phase A results published through global memory by other CTAs |
| end spread plus final pass | 5.8 to 6.9 plus 2.0 | static unit assignment; the final pass waits behind a grid-wide dependency |
| ramp | 4.3 | routing 1.75, x gather 3.4, stage-range computation about 2 us after routing |
| total above the 55.8 us floor | 22.8 | |

The T = 8 build (TINY2, verify 1) shows the same shape: ramp 3.7, phase A excess 5, A-to-B handoff 2.8, tail 3.8, of a
17.5 us gap at 7.5 experts (docs/runs-moe.md, "T = 8 MoE floor and gap"). Six incremental attempts on these pieces
(fci6, dfa, wpre, HALF, BAL, NW16) moved the stall or lost. The design below removes the mechanisms instead.

## 2. Design

### 2.1 Facts the design rests on

- Operator: per token, 8 of 256 experts, 32 local. `h = x @ w13[e]^T` (8192 outputs: 4096 gate, 4096 up), SwiGLU,
  `y = (silu(gate) * up) @ w2[e]^T` (4096 outputs), `out[t] += g * rmsnorm(y * (1 + post_norm_w))`. H = 4096,
  intermediate 4096. NVFP4: E2M1 codes two per byte, one e4m3 scale per 16 values, one FP32 global scale per expert.
  Tolerance: relative L2 at most 1.5% against FP32, any row at most 5x its bound. Activations BF16 where SGLang keeps
  BF16 (muse_moe_w4a16/TASK.md).
- One expert = 28.31 MB = 768 units of 36,864 B (16 rows of codes at 2048 B plus 16 x 256 B of scales): 512 units of
  w13, 256 of w2. Floor 6.37 us per distinct local expert at 4.45 TB/s (box `sum` rate; TASK.md uses 4.8 TB/s).
- Shapes: decode T = 1, 16, 32; verify T = 8, 128, 256. Distinct local experts: median 7 to 9 at T >= 8 under the
  judge's skewed routing, 2 to 3 at T = 1. H200: 132 SMs, 228 KB shared memory per SM, 8 GPCs; portable cluster size 8,
  non-portable 16 (one 16-cluster per GPC, so at most 8 of them). `tools/fu_occ.cu` measured 15 co-resident clusters of
  8 and 30 of 4 for a 416-thread one-CTA-per-SM kernel.
- Intermediate per expert: `[T, 4096]` BF16 = 8 KB per token: 64 KB at T = 8, 256 KB at T = 32. It does not fit one
  SM's shared memory at T = 32; spread over a 16-CTA cluster it is 16 KB per CTA.

### 2.2 Work decomposition

An expert's work is split into S slices. Slice s of expert e owns intermediate columns `[s*4096/S, (s+1)*4096/S)`:
the matching gate rows and up rows of w13 (so every SwiGLU pair lives in one slice, as the current packing already
arranges), and the matching K-range of w2. One cluster of C CTAs executes one slice; inside the slice, CTA c owns
`1/C` of the slice's gate/up rows for phase A, and streams `1/C` of the slice's w2 K-range for phase B. Phase A needs
no reduction (output columns are owned). Phase B is split-K inside the slice: each CTA produces a partial
`y_partial[T, 4096]` over its K-range; the cluster reduces the C partials through distributed shared memory in
column chunks (phase B exchange, decision D1 below). Slices of the same expert are independent until the per-token
combine.

Per-token combine: `out[t] += g * rmsnorm(y)` needs the full `y[t]` row of one expert, so the S slices of an expert
must sum their `y` before the norm. With S = 1 (one cluster per expert) that sum is cluster-local. With S > 1 it is
one global exchange per expert (FP32 partials written once, a per-expert arrival counter, the last-arriving slice
does the norm and the weighted add). That is one release-acquire pair per expert instead of one per segment per CTA.

### 2.3 Four rules the new kernel follows

1. Consumer warpgroups never execute a GPU-scope fence or a grid-scope wait. They arrive on an mbarrier when a tile
   is ready and continue. A dedicated epilogue warp waits on that mbarrier and does every store, fence and release.
2. Every exchange inside a slice is cluster-local: DSMEM plus `barrier.cluster.arrive.release` /
   `wait.acquire`, or an mbarrier in the receiving CTA's shared memory.
3. Work is claimed, not assigned. Inside a slice, stages are claimed from a shared-memory counter in the cluster's
   leader CTA (DSMEM atomic); across slices, leftover slices are claimed from a global counter by clusters that finish
   early (decision D2).
4. The schedule (which slices exist, which cluster starts where) is computed before the kernel's first weight load:
   by the route kernel in the layer, by a tiny prologue kernel under PDL in the standalone.

### 2.4 Open decisions, each settled by a measurement in phase 0

- D1, phase B exchange: (a) split-K with a DSMEM reduction of C partials (C x T x 4096 x 4 B of DSMEM traffic per
  slice: 2 MB at T = 32, C = 16, against 9.4 MB of w2 bytes per expert), (b) all-gather of the BF16 intermediate
  over DSMEM so each CTA streams full-K w2 rows (C x T x 4096 x 2 B: 1 MB at T = 32, C = 16, but the activation
  operand comes from remote shared memory inside the unit loop), (c) the intermediate written to global once and
  read back through L2 after a cluster barrier (today's data path with the dependency made cluster-local). Pick by
  measured phase B time against its byte floor at T = 8 and T = 32.
- D2, cluster size and slicing: C = 16, S = 1 (at most 8 clusters: idle SMs when distinct experts < 8, a second wave
  when > 8) against C = 8, S = 2 (16 clusters, 128 SMs, one global combine per expert) against C = 8 with S chosen per
  input so that S x distinct = 16 and leftover slices claimed dynamically. Pick by the phase 0 streaming skeleton's
  span against floor at distinct = 3, 7, 9, 12.
- D3, ramp: whether the schedule prologue runs as a separate PDL-overlapped kernel or inside cluster 0 with the other
  clusters spinning on a flag. Pick by measured time from kernel entry to first weight issue.

### 2.5 Expected effect (hypothesis, not a measurement)

Decode 32: segment switches 5 to 6 us go to zero (a cluster never switches experts with S = 1; with S > 1 a cluster
switches at most once); phase B waits 3.6 us fall to one cluster barrier per slice (about 0.3 us); end spread and
final pass 6 to 9 us fall to one stage of claiming jitter plus one last-arriver combine (about 1.5 us); ramp 4.3 us
falls to about 1.5 us. Floor 55.8 us stays. If every piece lands, decode 32 MoE goes from 78.6 to about 60 us, which
is 18 us on the decode 32 critical path, about 0.6% of the layer score from that cell alone, with proportional gains at
decode 1, verify 1 and decode 16 (ramp and tail are the larger shares there). The main uncertainty is D2: whether 7 to
9 experts can keep 128 SMs streaming without idle GPCs. The MoE board values one microsecond at decode 1 and verify 1
at about 0.9% of the MoE score, so the standalone board is where the kernel proves itself first.

## 3. Interfaces (fixed now so tasks run in parallel)

All new code lives under `muse_moe_w4a16/cl/`. Nothing in this plan edits `muse_moe_w4a16/submission.py`,
`muse_decode_layer/cand/*.py` or any existing tool until phase 4.

### 3.1 Schedule buffer (`cl_schedule.h`, written by task 0.4, read by every kernel task)

```
struct ClSchedule {            // one per call, int32 fields, device memory, written by the prologue
  int n_slices;                // S x distinct local experts, at most CL_MAX_SLICES (64)
  int slice_expert[64];        // local expert id of slice i
  int slice_col0[64];          // first intermediate column of slice i (multiple of 256)
  int slice_ncol[64];          // columns in slice i (multiple of 256)
  int slice_tok0[64];          // first entry of this expert's token list in tok_list
  int slice_ntok[64];          // tokens routed to this expert (same for all its slices)
  int tok_list[32 * 8];        // token ids grouped by expert, in ascending token order within an expert
  float tok_gate[32 * 8];      // routing weight g for the matching tok_list entry
  int expert_nslices[32];      // S for each local expert (0 if unrouted)
  int next_slice;              // dynamic claim counter, initialised to the number of clusters launched
};
```

Byte layout is fixed by this struct; the prologue (Triton or CUDA) and the kernel both include the header. A Python
mirror `cl_schedule.py` builds the same struct on the host for tests.

### 3.2 Kernel entry (`cl_kernel.cu`, task 1.1 onward)

```
template <int T_MAX, int C, int NWARPS>
__global__ void __cluster_dims__(C, 1, 1) cl_moe_kernel(ClParams p);

struct ClParams {
  const __nv_bfloat16* x;      // [T, 4096]
  const uint8_t* w13; const uint8_t* w13_sf; const float* w13_gs;   // as in Weights, repacked by prepare()
  const uint8_t* w2;  const uint8_t* w2_sf;  const float* w2_gs;
  const __nv_bfloat16* post_norm_w;   // [4096]
  __nv_bfloat16* out;          // [T, 4096], zero on entry (the caller zeroes it or the kernel's cluster 0 does)
  float* y_partial;            // [n_slices_max, T, 4096] FP32 scratch for S > 1 (D2)
  int* expert_arrive;          // [32] arrival counters for the per-expert combine, zero on entry
  ClSchedule* sched;
  int T;
};
```

Weight repacking in `prepare()` is allowed by the rules (lossless). Task 0.6 decides whether the current unit layout
(16 rows, codes then scales) is kept; the default is to keep it so phase 0 can start from the existing packer.

### 3.3 Bench harness (`cl_bench.py`, task 0.5)

`python muse_moe_w4a16/cl/cl_bench.py --kernel <module> --T 8 --distinct 7 --inputs 20 --runs 30 --control <file>`
loads weights through the candidate module's `prepare`, builds inputs with `reference.generate_input` at a fixed
seed and the judge's routing shape, checks every output against `reference.py` at the TASK.md bounds, times graph
replays after a 1 GiB L2 flush, and prints candidate and control medians, the byte floor for the input's distinct
expert count, and the paired delta. It takes `--lock /root/okc/.build/box/gpu.lock` and holds the lock only around
GPU work, as `tools/bx_box_run.py` does.

## 4. Tasks

Each task: owner is one agent in its own worktree; it edits only the files listed; it commits and pushes its branch;
its report carries the numbers, file SHAs and a one-line verdict against the done criterion. Estimates are agent
wall-clock on one shared H200 under the GPU lock. "Gate" means the lead decides whether the dependent tasks start.

### Phase 0: measurements and scaffolding (no dependencies between tasks; all start at once)

| id | task | owns | done when | est |
|---|---|---|---|---|
| 0.1 | Weight stream skeleton: a cluster kernel (C = 8 and C = 16 instantiations) that streams one expert slice's w13 and w2 units through a cp.async.bulk / TMA ring of depth 3 to 4 into shared memory and discards them (one XOR per unit so the loads are not dead), with stage claiming from the leader CTA's shared counter. Launch k clusters each on its own expert. Measure span against the byte floor for k = 3, 7, 9, 12 and for C = 8 / 16. Record per-SM achieved bandwidth and whether clusters of 16 co-schedule on this GPU. | `muse_moe_w4a16/cl/cl_stream.cu`, `cl_stream_bench.py`, `docs/runs-moe.md` section "CL-0.1 stream skeleton" | a table of span / floor for every (k, C); the GPC co-residency count for C = 16 and C = 8 measured with `cudaOccupancyMaxActiveClusters` and by timing | 4 h |
| 0.2 | Exchange micro-benchmarks for D1: (a) DSMEM split-K reduction of C FP32 partials `[T, 4096]`, (b) DSMEM all-gather of a BF16 `[T, 4096]` intermediate into every CTA in 16 KB chunks as a consumer would read it, (c) global write plus cluster barrier plus L2 read-back. T = 8 and 32, C = 8 and 16. Also: cluster barrier round-trip latency, and the latency of an mbarrier handoff from 8 consumer warps to one epilogue warp. | `muse_moe_w4a16/cl/micro/*.cu`, `micro/run.py`, `docs/runs-moe.md` section "CL-0.2 exchange" | a table in microseconds per exchange and a recommendation for D1 with the arithmetic | 4 h |
| 0.3 | Schedule prologue: `cl_schedule.h`, `cl_schedule.py` (host mirror), and a CUDA prologue kernel (one CTA) that reads `topk_ids [T, 8]`, `topk_weights`, local range, and writes `ClSchedule` for a given (C, S policy). Unit tests against the Python mirror on random routings including the five harness tests (bs 4, 33, 257, 128 skewed, 1). Time the prologue alone and under PDL behind a dummy producer. | `muse_moe_w4a16/cl/cl_schedule.h`, `cl_schedule.py`, `cl_prologue.cu`, `cl/tests/test_schedule.py` | tests pass; prologue time at T = 32 in microseconds reported | 3 h |
| 0.4 | Bench harness `cl_bench.py` per section 3.3, validated by running it on the current public MoE file (`muse_moe_w4a16/cand/current_control_release.py`, 5df257af) as both candidate and control: deltas must be zero within spread, and its T = 8 median at 7 distinct experts must agree with docs/runs-moe.md "T = 8 MoE floor and gap" within 5%. | `muse_moe_w4a16/cl/cl_bench.py`, `cl/README.md` | the self-check table in the README | 3 h |
| 0.5 | Math extraction: lift the current unit loop's device code (NVFP4 code unpack via PRMT, e4m3 scale application, the `mma.sync` tile shapes, SwiGLU with `__expf` and the ftz exp fix, BF16 rounding points, the FP32 sum of squares and final norm) into a header `cl_math.cuh` with no dependence on the helper0 ring, plus a one-CTA test kernel that computes one unit's output for T tokens and compares bitwise against the current kernel's per-unit output extracted from a debug build. Document every BF16 rounding point the 1.5% bound and the layer's checks rely on. | `muse_moe_w4a16/cl/cl_math.cuh`, `cl/tests/test_unit_math.cu`, `cl/tests/run_unit_math.py`, `docs/runs-moe.md` section "CL-0.5 math" | bitwise match on 1,000 random units at T = 1, 8, 32 | 5 h |
| 0.6 | Epilogue-warp pattern: a standalone kernel where 8 consumer warps produce tiles into a double-buffered shared memory slot and arrive on an mbarrier, and one epilogue warp waits, stores to global, and issues the release; measure consumer stall per tile against the same kernel with the consumers storing and fencing themselves. Covers rule 1 of 2.3. | `muse_moe_w4a16/cl/micro/epilogue_warp.cu`, `micro/epilogue_bench.py`, `docs/runs-moe.md` section "CL-0.6 epilogue warp" | consumer stall per tile for both forms, in nanoseconds | 3 h |

Gate G0 (lead): 0.1 shows a span within 1.10x of floor at k = 7 and k = 9 for at least one C; 0.2 gives a D1 choice
whose exchange cost is under 15% of the slice's w2 stream time at T = 32; 0.5 matches bitwise. If 0.1 fails at k = 9
for C = 16 but passes for C = 8, D2 is C = 8 with S = 2. If 0.1 fails for both, stop: the HBM queue cannot be kept full
by few clusters and the design is wrong for this part.

### Phase 1: phase A kernel (depends on G0; 1.1 and 1.2 in parallel)

| id | task | depends on | owns | done when | est |
|---|---|---|---|---|---|
| 1.1 | Phase A compute: on the 0.1 skeleton, consumers compute `h` for the CTA's gate/up rows over all T tokens of the slice's expert using `cl_math.cuh`, apply SwiGLU per pair, and hand the BF16 intermediate tile to the epilogue warp (0.6 pattern), which writes it to the CTA's own shared-memory intermediate slab (D1 a or b) or to global (D1 c). x staging: the slice's T x 4096 BF16 x rows (256 KB at T = 32) are read per unit from L2 as today, or from a cluster-wide DSMEM copy if 0.2 shows it cheaper. Verify `h` against `reference.py` intermediate (add an `intermediate` return to a local copy of the reference, never to the judge's file). | 0.1, 0.5, 0.6 | `cl_kernel.cu` (phase A section), `cl/tests/test_phase_a.py` | intermediate matches reference within 2^-7 relative per element on 20 inputs at T = 1, 8, 32; span of the w13 part within 1.15x of its floor at distinct = 7 | 8 h |
| 1.2 | Phase B compute, standalone input: consumers stream the CTA's w2 K-range and compute the partial `y` from a precomputed intermediate placed where D1 says (shared, DSMEM, or global), then the slice reduces per D1 and produces `y[T, 4096]` for the slice, with per-token FP32 sum of squares over the slice's columns. Test from a host-provided intermediate. | 0.1, 0.2, 0.5 | `cl_kernel.cu` (phase B section; a different file region from 1.1, merged by the lead), `cl/tests/test_phase_b.py` | y matches reference within 2^-7 relative per element; span of the w2 part within 1.15x of its floor at distinct = 7 | 8 h |

Gate G1: both tasks within 1.15x of their floors at T = 8 and T = 32 and numerically correct.

### Phase 2: whole kernel (depends on G1)

| id | task | depends on | owns | done when | est |
|---|---|---|---|---|---|
| 2.1 | Join phase A and B inside one launch: the cluster barrier between phases, the per-expert combine (S = 1 cluster-local; S > 1 through `y_partial` and `expert_arrive`, last-arriver does the norm and `out[t] += g * rmsnorm(...)`), `out` zeroing, dynamic slice claiming from `sched->next_slice`, T-specialised instantiations (T_MAX 8 and 32). | 1.1, 1.2, 0.3 | `cl_kernel.cu`, `cl_launch.py` (host side: prologue launch, kernel launch, graph capture rules) | passes `reference.py` checks on the five harness tests and the six scored shapes at T <= 32 through `cl_bench.py`; T = 128 / 256 dispatch to the current kernel unchanged | 10 h |
| 2.2 | Timeline instrument for the new kernel (globaltimer stamps outside the unit loop, SASS and register check against the plain build, per the B4 timeline method), run at T = 1, 8, 16, 32, decomposed into ramp, phase A, barrier, phase B, combine, spread. | 2.1 (can be prepared against 1.1 and 1.2 in parallel, applied after 2.1) | `cl/cl_tl.py`, `cl/cl_tl_parse.py`, `docs/runs-moe.md` section "CL-2.2 timeline" | the table for all four T | 4 h |
| 2.3 | Tuning sweep on the box, paired A/B through `cl_bench.py` against the current public MoE: ring depth, C, S policy, consumer warp count, x staging mode. One knob per run, control interleaved. | 2.1, 2.2 | `cl/sweep.sh`, `docs/runs-moe.md` section "CL-2.3 sweep" | a table of medians and deltas; the best configuration named with SHAs | 6 h |

Gate G2: the new kernel beats the current public MoE kernel on the box at T = 8 and T = 32 by more than
max(2 x control spread, 1%), and at T = 1 by more than 0.5 us or ties within spread, with every check passing.

### Phase 3: standalone release (depends on G2)

| id | task | depends on | owns | done when | est |
|---|---|---|---|---|---|
| 3.1 | Standalone submission file `muse_moe_w4a16/cand/cl_release.py`: `prepare()` repacks weights and compiles the extension (within 300 s, measured), `kernel_fn` dispatches T <= 32 to the cluster kernel and T >= 128 to the current kernel's code path copied verbatim, no host syncs, graph-safe. Run the clean-release audit (`tools/release_audit.py --tracked-only`). Lead dispatches one private check on the MoE board (`okc check muse_moe_w4a16 ... --no-wait`, registered in the ledger). | 2.3 | `muse_moe_w4a16/cand/cl_release.py`, `cl/build.py` | check VALID 5/5 tests, speedup above the public best's 2.323x | 5 h plus judge queue |

Gate G3: the check is VALID and the six-load MoE speedup beats the public best. The lead submits under the standing
rule for the MoE board (user go-ahead required: this is a new file family, not the PDL layer family covered by the
standing authorization).

### Phase 4: layer port (depends on G3)

| id | task | depends on | owns | done when | est |
|---|---|---|---|---|---|
| 4.1 | Layer build: add the cluster kernel as a sixth MoE build in the layer file via a patch tool `tools/dl_ly_cl_patch.py --base ly_fabq.py` (or the current public file), selected for T <= 32 only where the box shows it faster per T; the route kernel writes `ClSchedule` directly (rule 4) so the prologue kernel is not launched in the layer; PDL edge route -> MoE turned on for the cluster launch (today's cooperative launch forbids it) with the pdl8h guard protocol (ids mismatches at most 2x off plus 16, output within max(2e-2, off drift)); boundaries 2 and the unpooler unchanged. Box paired A/B on all six loads against the base. | 3.1 | `tools/dl_ly_cl_patch.py`, `muse_decode_layer/cand/ly_cl.py` | box: every cell within spread or better, decode 32 and verify 1 better by more than the threshold; G_RELEASE_STARTUP_OK; every reference check PASS | 8 h |
| 4.2 | Layer check and submit by the lead under the keep rule (VALID 13/13, every T on, guard and CG_GATE clean, six-load score below the current row, no changed cell worse than the base by more than 3%). | 4.1 | ledger | board row | judge queue |

## 5. Dependency graph

```
0.1 stream skeleton ─┐
0.2 exchange micro  ─┤
0.3 schedule        ─┼─ G0 ─┬─ 1.1 phase A ─┐
0.4 bench harness   ─┤      └─ 1.2 phase B ─┴─ G1 ─ 2.1 join ─ 2.2 timeline ─ 2.3 sweep ─ G2 ─ 3.1 standalone ─ G3 ─ 4.1 layer ─ 4.2 submit
0.5 math header     ─┤
0.6 epilogue warp   ─┘
```

Six agents in phase 0, two in phase 1, one plus one (2.2 prepared early) in phase 2, one each after. The GPU is the
shared resource: phase 0 tasks 0.1, 0.2, 0.6 and 0.4 all need it, so under one box they serialize on the lock and
phase 0 takes about 8 wall-clock hours instead of 5. A second H200 for the rewrite (same template as okc-h200,
$5.29 per hour) halves that and keeps the layer lanes' box free; the lead rents it only on the user's yes.

## 6. Agent rules for this plan

- Agent type `moe` for kernel tasks, `efficiency` for 0.4; the label starts with the task id (`moe 0.1 stream
  skeleton`). Model Opus 5.5 unless the user names another.
- One worktree per task; a task edits only the files in its "owns" column; the lead merges with the standard recipe
  and resolves any `.cu` or `.py` conflict by hand (docs/*.md only may be union-stripped).
- Box use through the GPU lock (`/root/okc/.build/box/gpu.lock`, held only around GPU work). Never kill another
  lane's process; check `nvidia-smi --query-compute-apps` first.
- No judge job from any task. Phase 3 and 4 checks and submits are the lead's, registered in the ledger before dispatch
  and waited on asynchronously.
- No fence, barrier or `griddepcontrol.wait` is removed from existing kernels; the new kernel keeps every
  release-acquire pair on a data path across CTAs (rule 1 moves them off the consumer warps, it does not drop them).
  Writer-side `fence.proxy.async` before any bulk-copy read of data written by generic stores.
- BF16 bitwise at every point where SGLang keeps BF16; the 0.5 rounding-point document is the reference.
- No routing slack of any kind: the set of experts that run is exactly the top-8 routing's local set.
- Every measurement in the report states control and candidate medians, spread, run count, and the floor; every
  hypothesis is labelled as one.
- Reports in `docs/runs-moe.md` under the task's section name; ruled-out variants get a row in `docs/ruled-out.md`.

## 7. Risks and the measurement that retires each

| risk | consequence | retired by |
|---|---|---|
| few clusters cannot keep HBM saturated (per-GPC fabric limits, cluster co-scheduling) | the whole design loses to 132 independent CTAs | 0.1 at k = 7 and 9 |
| DSMEM exchange at T = 32 costs as much as the global path | phase B keeps a global dependency, S = 1 only | 0.2 |
| C = 16 clusters do not co-schedule 8 at a time with this smem footprint | D2 forced to C = 8, S = 2 | 0.1 occupancy row |
| the per-expert combine with S > 1 reintroduces a grid-scope wait on the critical path | tail returns at 1 to 2 us | 2.2 timeline; fall back to S = 1 at T >= 16 where distinct is near 8 |
| prepare() exceeds 300 s with a new extension plus the old one | standalone rejected | 3.1 measures prepare time on the box first |
| cluster launch inside the layer's CUDA graph with PDL attribute misbehaves | layer port blocked, standalone still stands | 4.1 guard protocol and the pdl8h stress |
| schedule prologue adds a launch at T = 1 | ramp gain lost at decode 1 | D3: fold into the route kernel in the layer (rule 4), measure in 0.3 |

## 8. What the lead does while tasks run

Merge each branch as it lands, run the gate arithmetic, open the next phase's agents, keep the layer lanes (ly_fabq,
repeat draws) going on the quota, and record every decision in docs/steering/2026-11-02-final-day-directive.md.
