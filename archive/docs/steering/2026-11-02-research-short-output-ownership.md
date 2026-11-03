# Short output ownership: T1-only phase-boundary reserve

November 2, 2026. Independent source screen of status-and-next-research sections6/7. **Decision: concrete T1-only three-launch design is source-feasible, but no exposed saving or implementation allocation is established. Do not displace B1 or consume the reserved check.** Current tiny already owns output rows. The new mechanism can only be replacing persistent cooperative phase scheduling with ordinary kernel boundaries and an explicit norm-partial consumer; keep T8 incumbent.

## Pins and selected implementation

Public **607eb52bdaf24160**, muse_decode_layer/cand/g1b4_stage_release.py, source commit **0ce0b23f53422a1883825ed0771a54992fa87d1c**, SHA256 **ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab**. Fresh unchanged check **cc474883117947c0** has identical server content **5cf950e1819c8c2bf0edf9bfdc0ccf05b191759c0a8263a45cd962f4e934ff5e**,13tests/12benchmarks/12profiles PASS and56startup replay checks. Read fresh incumbent report/receipt/hotspots, status sections6/7, retained tiny metadata audit **cd010c83**, relevant ruled-out rows and actual source. Exact anchors/arithmetic are in [screen.json](../../../docs/steering/evidence/2026-11-02-research-short-output-ownership/screen.json). No worker WIP inspected.

T1 selects prepared-state **tiny**, T8 **tiny2**, both **@moe_helper0**, not main WGS. Defines: NW8/TINY1/FAST1=2/NACC4/WS1/WSB1/WSBL2/FINPF1, WSBLK2 for tiny versus WSBLK1/CPF0 for tiny2. Loader O3/sm90a/NS3/FNST4/NSF3 plus actual prepared PRMTSC1 scale suffix. Neither enables SPLIT/WGS. Fresh log selects repaired cooperative+BF16, MFIN/DF/YBF/PDL off (source6858–6862,6893–6918,15934–15935,16084–16106,16824–16859). “Legacy” here means selected non-WGS tiny computation, not an assumed historical fallback. The separate Marlin fallback16750–16786 already has alignment, two GEMMs, activation and per-expert norm/combine; it is NOT this fresh selected path or proof the proposed specialized schedule was measured.

## Existing ownership and precision

Tiny CTAs own disjoint gate/up or down output-row units with full K reductions. A weight tile serves all tokens of one unique expert. Activation fragments are already register-resident; BF16 intermediate and FP32 y already exist. No missing preprocessing or duplicate-per-route weight read may be credited. PhaseB waits for its expert's complete A via doneA acquire, with producer fences/counters; helper join and cooperative grid barrier publish complete sums before final output (13885–14063,14879–15015). Removing these within the existing kernel is NOT proposed.

Actual call16682–16699: BF16 x[T,4096], INT32 local ids[T,8], FP32 route weights[T,8], BF16 output[T,4096]. Skip nonlocal ids, do not modulo into a local expert. Preserve slot order. Full-K FP32 gate/up accumulators are scaled and individually BF16-rounded; SiLU times up is BF16-stored. Down scaled accumulator is BF16-rounded, then multiplied by incumbent BF16-rounded (1+normw) in FP32; retain FP32 y and squared norm. Only AFTER complete per-expert norm, accumulate slots in FP32: weight * rsqrt(sum(y*y)/4096+1e-8) * y; final BF16. Reference.py427–443 requires each expert normalized after gamma, NOT RMSNorm of weighted sum. Incumbent rounding approximates reference; preserving it is not mathematical bit identity. A new norm reduction tree still needs numerical qualification.

This does not repeat parked serial-front: front/router/unpool are unchanged. Nor does it repeat G224's larger-token chunk schedule. Novelty is the short expert phase boundary, not the already-present output ownership.

## ONE fixed challenger including normalization

**T1 only**, current immutable packed weights and runtime routing. Eight fixed slot lanes skip invalid local experts; no compaction/conversion launch. Zero local pairs returns zero.

1. Gate/up: output-tile CTA per valid slot and8intermediate coordinates, fullK4096, incumbent BF16 boundaries.512tiles/pair.
2. Down: output-tile CTA per valid slot and16output coordinates, fullK4096. FP32 y[slot,4096] plus one FP32 squared-norm partial/tile;256tiles/pair. No cross-CTA spin or norm atomics.
3. Norm+combine: **one256-thread CTA/token**. Each thread reads one of256partials per valid slot; block reductions produce up to8inverse RMS values in shared memory. CTA barrier, then column-parallel slot-ordered FP32 weighted output and BF16 store. There is no hidden fourth norm launch or assumption that one output tile knows the complete norm. Slot-sequential processing avoids requiring all y simultaneously in registers. Actual resource feasibility remains uncompiled/unmeasured.

Three ordinary same-stream launches replace one cooperative launch: **two extra boundaries**, global A completion before any B (loses current per-expert overlap), and a one-CTA final consumer (less final parallelism). CUDA13 guide §6.2.8.5 documents stream ordering: [primary](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#streams), HTTP200 read this task; document hash in evidence. This supports proposed phase publication only, not unwritten-kernel or graph-integration correctness. No harness/allocator/PDL modification.

Logical matmul work **100,663,296 FLOPs/pair**, excluding decode/activation/norm. Packing payload **28,311,552B/unique expert** =768*36,864, not a bandwidth forecast. Per pair: BF16 inter store8,192B; FP32 y store/read16,384B each; partial store/read1,024B each. Inter/y are already present: partial materialization versus current atomics is changed traffic, not all-new or all-saved traffic. T1 top8 uniqueness gives P=E: no extra logical weight read intended. Actual cache/transactions are unknown.

## Routed pairs and strongest objection

Retained owning standalone **sc6_compare.05e51ddeb7014dd9.log**, seeds901/902: T1(P,E,maxcount)=(2,2,1)/(3,3,1); T8=(20,11,8)/(15,7,8). These are NOT fresh cc474 layer populations. Full expert-count vectors and fresh layer histogram were absent from examined artifacts. Do not invent uniform8*T local routes or assign these seed counts to current layer. T1 allows0–8local pairs. AtT8 pair-owned work can increase logical weight work by P/E (~1.82/2.14 on retained examples), whereas incumbent reuses each unique expert. No T8 expansion without concrete reuse-preserving mapping.

**Strongest counterargument:** tiny already performs the proposed output-row math/weight reuse. Removed coordination is unmeasured; two extra launches, lost A/B overlap and single-CTA finalization could outweigh it completely. No trace isolates a removable scheduler/norm tail. Perfect resource usage would not establish a win. One-CTA finalization is an explicit bounded compromise, not free norm fusion.

## One future discriminator, not a check request

If later authorized: ONE fixed T1 three-launch complete-expert callable versus selected tiny, same prepared weights and actual runtime routes, all other layer paths unchanged. Include all three launches/scratch/dependencies and BF16 output; cover zero/varied local counts and validate complete per-expert norm plus weighted output after graph replay. **Does removing persistent coordination repay ALL replacement costs on real T1 routes?** No win ends this exact schedule, not a tile sweep/T8 rescue. Expert-call timing does not prove whole-layer gain; a survivor still needs matched whole-call confirmation. No implementation/target execution is authorized here.

Fresh46:16 adjusted endpoints: D1 **78.764531us**, V1 **112.681662us**. Fresh expert activities D1sliding/full29.1/22.6us; V1 50.6/85.7us are attribution, **NOT exposed saving**. Fixed-baseline six-load factor for T1-only NET endpoint saving d: (D1/(D1-d))^(1/6). Illustrative1% needs **4.564780us net**, after added costs. Equal saving on BOTHshortloads needs **2.725547us each**, but that scenario does NOT apply to this T1-only reserve. No available launch-cost measurement or exposed coordination budget establishes either opportunity. Drift and unknown fresh routing prevent a forecast;1% is not a physical floor or top100 guarantee.

B1 remains the allocated worker task. Source-feasible reserve only, not a shippable candidate or readiness approval. No implementation/compiler/GPU/driver/challengeCLI/submission/quota/subagent/newthread, competitor recensus or worker interruption. Only this report/evidence are owned; README/sharedlogs untouched.

