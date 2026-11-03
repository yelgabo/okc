# Alias exposure: missing layer observations; exact-boundary staging removes fragment serialization

November1 local / November2UTC2026. Source/CPU-only follow-up to80f3aec, evidence checkout7076754. No GPU, compiler, host link, production edits, quota, table work or subagents. Captain owns index. Replacement worker owns implementation/dispatch; retired worker was not contacted. RMW-only promotion is closed per captain's valid performance-gatefalse update; no#4 proposed. Publishing only this new report/evidence prefix.

## Decision

**Archived evidence cannot establish the requested six actual layer route frequencies or a useful final-fragment-to-ci4 timing bound.** Exact selected-source standalone route populations ARE recoverable, but substituting them for layer router outputs would be false attribution. Do not implement the original8192B streaming protocol from these observations.

**A different representation avoids serializing NTT-1 nonlinear epilogues with no added shared allocation:** compute performs the ALREADY REQUIRED FP32 scale and BF16_RNE boundary, snapshots all gate/up BF16 operands into one reserved weight slot, and leaves exp/division/up-multiply/finalBF16 to the weight helper. Maximum32768B fits the existing36864B slot. This is not lower precision or the universal-SiLU table. It removes the earlier inter-fragment helper ACKs, but does not remove producer backpressure or the ci4 publication deadline. It is a source-level candidate, not a compiled/progress-qualified implementation or speedup.

**Chosen next discriminator:** only a separately authorized, low-distortion selected-native handoff-window observation could earn implementation budget. Required record and stopping rule are below. Hold implementation/GPU now; no new measurement authorized by this report.

## Owning evidence and exact missing observations

Selected Python `muse_decode_layer/cand/g1b4_stage_release.py` SHA256ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab; actual AST-bound MOE_CUDA_SRC c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71. Not the later alternate literal. MAIN/B4 selection, both bools/defines and layout remain as80f3aec: D32=M32/B4; V16=M128/MAIN; V32=M256/MAIN; sliding/full. WGSFCI4, WGSGS2, per-stage activation wait0,512 empty arrivals, selected MFIN0. CPU helper asserts both source hashes.

Inventory: tracked paths plus `rg --files --hidden --no-ignore` across the checkout, including.build; excluded.git, node_modules and.venv contents. Checked trace/timeline/endpoint/c193/frontier/stage paths and route/input-dump candidate names. This is a bounded local-archive finding, not a claim that another machine or unretained runner state has no data. Initial Windows wildcard-path searches errored; corrected searches used directories plus glob filters/inventory, not error-as-empty reasoning.

|Requested actual layer cell|c193 expert activity us|Nonsplit NTT1 / NTT>1 / terminal counts|Final fragment to ci4|
|---|---:|---|---|
|D32 sliding|82.6|not retained|not retained|
|D32 full|75.1|not retained|not retained|
|V16 sliding|136.7|not retained|not retained|
|V16 full|183.5|not retained|not retained|
|V32 sliding|198.3|not retained|not retained|
|V32 full|204.3|not retained|not retained|

Owning jobc1931f0c9e834485 profile JSON contains case metadata and kernel report strings, not jC/jS/job-order/wgr or per-unit clocks. Seeds, total tokens, expert kernel name and duration cannot recover actual routed expert multiplicities. No CPU regeneration of GPU layer router outputs was attempted. Expert activity is not exposed epilogue time.

Historical all-layer timeline01183c2992eb4153 retains24 kernel-level traces but its actual routing metadata is explicitly INVALID (read captured output before replay). Even its intended local-slot totals/histograms/unique count would not specify per-expert jobs or CTA ranges. No internal wgs_seg/ci4 markers exist there. Short-path trace body explicitly restricts M to1/8, not D32/V16/V32. Edge trace-manifest is a compiler-only glue-cluster artifact with gpu_execution=false, not a MoE route trace.

Standalone stage jobf147d06a64504c1e uses the EXACT selected CUDA hash and preserves route-count vectors for six M128/256 natural inputs. Its ready/retire timing failed every distortion gate; retain counts, reject timing attribution. Representative M256/76117 capsule stores phase ranges for CTAs0/66/131 in four arms. The new CPU model reproduces all12 archived A-range endpoint pairs, original integer cost-prefix partition and trailing-partial-first order. Grid132 is the one prepared state shared by all six inputs, not an assumed layer grid.

|Standalone M / seed|Whole NTT1|Whole NTT>1|Whole terminal|Whole nonterminal|Split segments|
|---|---:|---:|---:|---:|---:|
|128 /76111|523|115|132|506|196|
|128 /76117|497|128|132|493|222|
|128 /76123|412|196|132|476|192|
|256 /76111|336|321|132|525|222|
|256 /76117|366|361|132|595|210|
|256 /76123|423|323|132|614|236|

These are CPU-reconstructed phase-A segment counts over all132 CTAs, not directly measured counters. Whole means16stages in one segment; split counts segment shares, not unique units. Terminal means the last executed A segment per CTA, after reorder, which immediately hits A's final flush and has no next-A compute window. NTT and terminal columns overlap; they are not additive categories. These observations demonstrate why neither NTT1 rarity nor terminal rarity may be assumed. They do NOT establish any of the six requested layer frequencies; D32/B4 was explicitly excluded from this standalone check.

Other archived frontier1c6ebc76b91d4b7f stamps precede phase-B epilogue writers and record compute/grid-join/token-finalizer endpoints. Device source hashes differ from the selected stage-retirement source; B4 is explicitly excluded. Its finalizer tail is not a phase-A accumulator-end to next-first-stage-ci4 interval. Stage-probe phase entry/end/retire marks likewise do not bracket those endpoints, and detailed timing failed distortion. A whole-kernel/whole-phase containment inequality would be vacuous for this decision, not a useful exposed-window bound. No microsecond window is inferred from commit-group counts or kernel totals.

## Lower-cost alternative: snapshot at the existing BF16 boundary

The selected `wgs_epi_tile<false>` takes `gt=fork_bf16(d[h]*gsc)` and `up=fork_bf16(d[2+h]*gsc)`, then executes unchanged `SILU_DIV(gt,1.f+__expf(-gt))*up`, `probe_q`, and final BF16 store. Selected `fork_bf16(v)` is exactly `__bfloat162float(__float2bfloat16_rn(v))`. Instead of transporting raw FP32 d, transport the bit pattern produced by THAT conversion, not a newly approximated SiLU output.

Compute side: after existing accumulator retirement, same gsc evaluation and FP32 products, same valid-token predicate and __float2bfloat16_rn for the four components. Store BF16 bits once, without expanding them locally. Helper side: __bfloat162float each loaded operand, then unchanged nonlinear/up/final store arithmetic. Do not rescale, reround, move nonlinear work outside the timed path, change fast-math flags, drop fences, or change split accumulation. Keeping the original conversion result bits preserves signed-zero/extreme/nonfinite conversion behavior structurally; actual compiler lowering/FTZ/reassociation and executable equivalence remain unvalidated. This report does not claim an all-values CUDA execution test.

Byte offset inside the reserved final weight slot: `8*(nt*512+tid)+2*component`, nt0..NTT-1,tid0..511,component0..3. Each fragment4096B, maximum NTT8=>32768B; D32 NTT<=4=>16384B. Slot starts22272/59136/96000, so maximum occupied intervals[22272,55040),[59136,91904),[96000,128768), leaving4096B per36864B slot at NTT8. CPU checks every halfword address for NTT1..8. Padded inactive-token positions remain unread; maintain the original tk<cnt predicate.

This alternative intentionally uses more than the original8192B mailbox, but no more ALLOCATED shared memory: it occupies up to32768B of the same existing slot. If the8192B cap itself remains mandatory, this construction is ineligible for NTT>2. Four independent exact-BF16 operands per512threads already require4096B perfragment; merely packing them cannot fit NTT8 into8192B. A full FP32 snapshot needs65536B and does not fit one36864B slot.

At NTT8, original FP32 streaming moves65536B each way (131072B logical shared traffic) and needs15 ready/ACK rendezvous. Exact-BF16 snapshot moves32768B each way (65536B traffic) and one ready rendezvous; completion acquisition remains. General formulas: original16384*NTT total shared bytes and2*NTT-1 ready/ACKs; new8192*NTT bytes and1READY. These are logical byte/handshake counts, NOT latency or bandwidth savings. Compute still serially writes its NTT snapshots, but does not wait for any NTT-1 helper nonlinear epilogues before entering next compute.

Retain the prior reserved-slot protocol: withhold final-slot512 BW_E arrivals, keep all32 weight-helper lanes alive, service mailbox before waiting for q+3 overwrite, preserve all original per-reader fences/wait0, and ready-barrier544 only after the complete snapshot. Helper iterates16 virtual compute warps perfragment using four BF16 operands in registers, completes all NTT nonlinear epilogues, performs the required per-lane fences and replacement512 empty arrivals plus completion notification. Original compute WgsPend waits completion at next FIRST/ci4 before original doneA release; terminal A flush drains immediately. Reuse unused bars[12] and separate completion parity as80f3aec; no new mbarrier allocation. Split/empty/PB=true/B4FIN paths remain untouched. Next compute cannot overwrite this mailbox before previous completion; the existing ci4 wait is earlier than its next epilogue.

No helper work depends on next-unit input or another CTA, and producer must service before its own reserved-slot wait: same absence-of-new-local-cycle argument as80f3aec, not a fresh proof of inherited whole-grid progress. X producer remains responsible for xA/xB and doneA acquire; it is NOT repurposed. Weight producer cannot issue further TMA while its warp services all16*NTT virtual epilogues. At M<=64 its existing n-2 landing guard also remains. This can stall next operands despite physical fit. One helper warp replaces16 compute warps' nonlinear work, so throughput/exposure can easily lose. Nothing here establishes that ci4 hides it.

## One next discriminator and stop

Keep alias implementation HOLD. If captain later allocates new measurement, choose ONE fixed selected-native handoff-exposure observation, not a streaming implementation or another broad stage-readiness observer: for D32/V16/V32 sliding/full, retain post-replay job counts/order and per-CTA A ranges, plus paired same-thread endpoints at the original final accumulator drain (before epilogue), end of original epilogue, and entry to next FIRST/ci4 flush; mark terminal/split explicitly. Count unit categories by CPU reconstruction validated against stored ranges. All CTA/warpgroup variation matters; one thread/CTA sample cannot establish a global maximum or critical path. Require unchanged-source duplicate controls and a predeclared observer-distortion gate before using clocks; the rejected f147 readiness observer is not reopened.

This single discriminator asks whether there is enough nonterminal nonsplit nonlinear work AND actual native interval to consider hiding an all-NTT helper service. It would not yet time that service or establish a gain. If metadata cannot be captured with acceptable perturbation, or measured interval/budget is negligible, close this ownership allocation rather than invent timings or proceed to a kernel port. Existing six-load cutoff gap remains approximately6.1%; profile subtraction from the prior scope report is scenario sizing only, not an available saving. No observed figure in this report supports closing that gap with ownership.

## Validation / handoff

`python docs/steering/evidence/2026-11-01-alias-exposure/analyze.py` passes: exact source hashes,12 representative range-pair matches, six standalone CPU populations, all exact-BF16 mailbox addresses NTT1..8. Output analysis.json pins owning route receipt/decoded data. No GPU/compile/table work, no source candidate, no numerical execution validation. No worker child result or SiLU compilation is claimed. Captain indexes; replacement worker remains sole implementation/dispatch owner. All prior gamma/alias holds stand.
