# G-local split overlapped short front: reviewable candidate

November1. No dispatch/public action by this worker. Root owns actual release and receipts.

## Why this is worth a bounded measurement

Current #303 stage607eb52bdaf24160 has fused BF16pooler/FP32router FRT17.9us atD1,
18.7/18.6us atV1 sliding/full. Isolated external projection3c6a2baf7b4b4bf3 poolerM1/M8
13.393/13.945us, and ownedC3 repeat40b9f54be6fe4d4e13.55/13.28us, show that precision-
preserving poolers of this size can finish sooner in isolation. They are different kernels,
inputs and contention conditions: those values are **counterfactuals, not G split measurements**.
Assuming similar pooler duration and a fully hidden router gives roughly4.35–4.51usD1 and
4.73–5.39usV1 potential exposed saving. On current weighted layer78.55129/115.99us and
other four loads frozen, that gives about1.69–1.76% six-load score. No router duration is
separately added as saving. Required equal saving for1% is2.753us on BOTH short loads.

Competing bandwidth can erase this: BF16 pooler weights67,108,864B (before compression),
FP32router8,388,608B, plus small activations/metadata. Existing packed pooler reduces its
physical stream; both kernels still compete for HBM/L2/SMs. No counter/bandwidth claim.
The pending G serial trace will measure arithmetic separately but cannot by itself prove overlap.

Important negative evidence: cb71 log selects T1fp32 splitoverlap but T8skf-r FUSED.
Its T8 splitfp32 loses9.76us wholefront versus fused in the stable tuner; T1 split beats
fused by~2.02us wholefront, which alone does not supply1% layer score. The valid owned
01183c2992eb4153 T1 trace has router/topk finish11.8/11.2us before pooler normalization;
that establishes no exposed router tail there, not a G pooler win. Owned T1 pooler spans
16.5/16.4us in these instrumented traces, slower than isolated projection results.
Thus this is a bounded forward hypothesis with meaningful possible coverage and real
counterevidence, not a projected gain or a reason to force another public attempt.

Source-path correction: current GLE_T={} and `_pdl(T)=False`. The deferred `_gle_after_attention`
2*T-row uv path is dead. Actual path `_sl_tile_after_attention` materializes T-row BF16h and
FP32hr, then `_pool_router_fp32` calls `_gle_frt`. Split fallback also uses actual Trows.
An initial chat assessment followed the dead uv path; this report corrects that mistake.

## Exact artifact and policy

`tools/dl_g_next_expert_overlap.py` pins full incumbent public source
`muse_decode_layer/cand/g1b4_stage_release.py` SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` (869076B).
It appends helpers/prepare wrapper only; full existing byte prefix, attribution, device
literals, original startup validator, attention/expert math/fences remain unchanged.

Generated `.build/today-moe/g-next-overlap.py` **878527B**, SHA256
`5b5ec131f1d9d427d04817bf1489903a9fabfa4a35c21eda05a15e6c5ae9a7d4`.
BothT1/T8 use a fixed split-overlap policy only when their original ppick/strides/router
association and existing scratch-free `sk2` pooler choice are eligible. No per-input timing
selection. All other loads and legitimate unsupported paths call the incumbent.
The original prepare/tuners run with `_g_overlap_active` absent, so tuning cannot silently
run the new front before a side stream exists. It is attached/activated afterward.
Qualification requires BOTH T1/T8 eligible, with printed actual ppick+resolved choice.
The fact FRT used a packedSK2 internally does not prove the separate bfp.choice isSK2;
if it is another path, this hypothesis fails qualification instead of changing plans.

## Stream/lifetime and mutable state

Short identifiable `_g_overlap_join(torch,h,hr,side,pooler,router)` is reused unchanged for
local stand-in callback testing. Standard PyTorch fork/join: side waits current parent,
FP32router runs on side, BF16pooler on parent, parent waits side **in finally**, including
callback-error paths. Inputs are not read before the producer dependency. No side work escapes.
Eager hr records side stream, side-produced logits records parent. Captured work forks and
rejoins within the same graph; input/output graph lifetimes remain owned by that capture.
No custom events, global stream cache or data-derived persistent activations exist.

Each prepared state has a stream holder created on its device. Deepcopy makes a DISTINCT
side stream, never returns self or enlists a prepare stream; copying during capture fails.
Concurrent capture on the SAME owner is not supported: the incumbent already shares mutable
MoE/attention scratch. Different copied owners have independent sides. No claim extends to
arbitrary concurrent calls sharing an incumbent state.

Selected bfpSK2 uses newly allocated output and readonly weight/patch streams; its dummy
buffer is debug-only atdg0. The skinny path's `ws_shared`/cnt and GEMM shared workspaces
are deliberately ineligible. Private startup copies bfpdict and clonesdummy per owner,
then explicitly checks unique writable storage across all graph inputs/debug/outputs.
`_bfp_gemm` calls `_bfp_pdl` with original `_pdl=False`, setting globalSK2PDL0; code guards
against unexpected nonzero policy. Its normal extension setter writes the same zero.
No new flag mutation is used to overlap; no PDL or cross-stream publication shortcut.

## Private validation and remaining gates

Incumbent native-MoE startup test remains first. Added front qualification:

- Fixedseed940771,4fresh normalized FP32 input banks atT1/T8, BF16h fromFP32hr.
- Independent FP32matmul pooler/refBF16 output and FP32router reference, TF32 disabled
  during oracle construction and restored in a nestedfinally even on CUDA failure.
- Existing route kernel on reference/candidate outputs: global/local IDs exact, weights
  relative<=1e-4, expert input global<=.008,row<=.075; pool<=.005/.01; router<=1e-4.
- Two distinct owners/parentcaptures per size, four distinct sidestreams; each eager result
  and all32 fresh graph replays observed. No zero-check/skipped-owner pass is possible.
- Final concurrent replay checks all pool/logit/ID/localID/weight/pre-expert outputs.
- Scratch/input/output storage uniqueness asserted for the selectedSK2 path; unsupported
  runtime inputs preserve incumbent fallback while expected qualification drift is fatal.
- Final280s process preparation guard (cannot interrupt prior native compilation).

Three CPU tests pass: actual helper call/fork/join/exception/capture-record order, actual
holder deepcopy and bootstrap delegation, exact prefix/pin and coverage guards.
`python -m unittest discover -s tools -p test_dl_g_next_expert_overlap.py -v`

Root independently provisioned CUDA PyTorch2.10/cu128 on RTX5070, compiled the exact FP32
router forsm120 (maxabsolute error5.37e-7 versusFP64), and exercised unchanged helper with
substituted same-shape BF16pooler. Different owners/captures and20fresh concurrent replays
passed. This is coordinator-reported local API/math evidence, not native H200 pooler,
full startup, layer correctness, performance or residency qualification. Final5b5 helper
and native Triton routing rerun was pending when this checkpoint was written.

Peer review is requested against exact final bytes. Root must retain all official13cases,
12medians and native eligibility, intended front coverage manifest, original validator and
new startup checks. No startup/timing-driven path selection. Stop on mismatch, alias,
capture error, unsafe ordering, elapsed preparation guard, all-fallback drift, or no useful
whole-six-load result. One single-arm public score cannot causally prove the overlap win;
use exact per-load results and noise limitations. Do not repeat the old losingT8 split blindly.
Parent owns report index/push integration and actual public source/receipt.

## Coordinator local-fixture completion

Root reports final5b5ec1 exact wrapper/holder/router/Tritonroute/full startup fixture
PASS on RTX5070 with32qualified fresh replays, both expected natural-route/owner manifests,
and two concurrent captures plus20fresh overlap replays. Only the BFPsk2pooler arithmetic
was represented by torchBF16linear. This closes local stream/API fixture execution;
native H200 packed pooler correctness/performance and the official layer remain unqualified.
Root archives owning local evidence; no H200 claim or dispatch by this worker.

## Official split-overlap measurement

Root's job942ebdec400d4e68 passed13official cases plus32front/56expert private replaychecks,
score2.0959003. WeightedD1 regressed5.73%,V1improved2.10%; V1-only substitution contributes
about0.3552% six-load score. The two-load candidate does not beat2.12608268 #303 production.
Close these exact bytes for public retries; no V1-only adapter was authorized. This result
supersedes the earlier conditional1.69–1.76% opportunity scenario. Root owns full receipts.
