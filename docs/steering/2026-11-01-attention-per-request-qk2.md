# Attention decode1: public comparator audit and owned two-chain experiment

## Qualification correction: common BF16 boundary, November1 subsequent revision

Root identified that the historical owned finalizer and comparator both normalize
the decoded attention value in FP32, whereas the official attention/layer baselines
round it to BF16 before output RMSNorm/gate. The initial638149-byte artifact976c...
below is **superseded and must not be dispatched**. Its resource evidence is only
the initial unrepaired baseline study, not the final experiment.

Both treatment and matched controls now use a common exact repair in
`resource_source`: in `finalize_rows` and `finalize_det`, after `v_scale/lsum`,
each component is `BF16_RN(acc*inv)` promoted to FP32 **before** forming the RMS sum,
normalizing and gating. This applies identically to all four modules/all native
routes; the QK flag still changes only L1MODE11/13. The judged path is this newly
BF16-repaired QK-off control, **not historical891 bytes or historical score parity**.
No old score is attributed to the repaired common baseline.

The actual `_ob_reference` helper now applies the same BF16 activation boundary
before `_ob_rms` in both BF16-query and E4M3-query references. A fifth real Torch
CPU test executes that exact extracted oracle on an independent small attention
fixture for both query modes; removing the BF16 cast is a negative control with
different results. Five CPU tests pass. Each fresh replay additionally requires
bitwise-equal complete K/V caches across the independent arms. The attention owner
independently re-reviewed the common boundary/oracle repair and found no blocker.

Final generated `.build/attn_pr_qk2.py`:638640bytes, SHA256
**7ce6bbe7761916e4917c8304d765987652b3459804b2b37f7b8a441f85e68049**.
Refreshed actual NVRTC/PTXAS compilation passes: MODE13off/on72/72registers,
MODE11off/on128/124; all stack/spills0,32HMMA each. SASS counts including NOPs
are1648/1648 and2448/2464 respectively. Every sampled finalizer now contains the
BF16 conversions. Unaffected L1MODE1 and L8MODE3 off/on instructions remain
identical under the common repair (2048 and2328instructions respectively).
The final manifests/logs/test evidence are in
`muse_attention_decode/profiles/pr_qk2-bf16.local.2026-11-01/`.
No GPU job has been launched. The rest of this report records initial source
discovery and experiment design; this correction governs its precision and artifact.

## Initial source discovery and registration

November1. Frozen owned `8914751e3b9f468b`; external `cd6805e7ebce45ee`.
No public action or H200 check yet. New owned variant is **OFF by default**.

## Source-backed opportunity

The exact external source SHA256394f2517ca8eaf69b7d422832edfd59451a9f73103981d13b1059070abb17b4b
passed16 official tests at2.639606424x. Its decode1 sliding/full adjusted medians
are4.673/10.794us versus the frozen owned5.183/12.390us. These independent receipts
give a decode1-weighted, sixth-root sensitivity1.020034, not a paired gain or forecast.
Evidence: root `muse_attention_decode/profiles/rohan-normalized.cd6805e7ebce45ee.*`,
`scored.8914751e3b9f468b.json` and steering component-frontier/attention-shapes.md.

Active external profiles are **attn_kernel<1,false,13> sliding**, and
**attn_kernel<1,false,11> full**. `_plan`4991, `_TUNED`4969, `_launch`5078 and
launch dispatch4664 confirm32 splits/nc1/last-CTA and252 splits/nc8/cooperative
software finalization respectively. Owned `_one_plan`6419 selects the same grids,
copies and corresponding modes13/11. Both use128-thread per-request CTAs, three
cache stages, page-size1 speculative length with device verification, one tile per
sliding split, and full-context multi-tile ranges. There is no new252-vs264 plan win.

The header's FP8 QK description does **not** describe active q1: external2564–2695
uses FP16 QK and PV MMAs with FP32 accumulation, normalized/RoPE query packed FP16,
cached E4M3 rows converted exactly to FP16, FP16 probability fragments, FP32 online
softmax/merge, per-head64-dimensional RMSNorm eps1e-5 and sigmoid gate, final BF16
output. New K/V are normalized/rotated/quantized and written by the range containing
their position; incoming positions are patched locally into issued stages. Owned
4682–4760/4887–5010/3587–3697 has the same precision class/cache contract. Neither
comparison introduces FP8 q1 query rounding or removes an owned BF16 boundary.

Relevant existing differences already investigated: key-half merge is owned HMRG1;
FIN8 groups workspace loads; both exist. The external full-context launch is
ordinary after an occupancy guard, while owned retains guaranteed cooperative
launch. `ce92d32e26024bd2` sized dropping that attribute at.38us and did not justify
the safety loss. Keep the owned attribute. The same receipt closed the shared
batch1 prologue as the main gap; last-CTA mode9 remains under its placement noise.

One distinct source mechanism remains: external2432–2446/2564–2623 uses an unpadded
query quad layout and **two independent two-MMA QK accumulator chains**, then adds
them. Owned currently has a four-MMA dependency chain. Its existing QK2 branch can
test the chain alone with the current padded fragments, without copying external
packing/code. Existing `ONE_F1` QK2 loss `c8e99e038d9f4575` applies only to **FLAT**
q1 modes17/49; the constexpr explicitly excludes per-request modes11/13. This new
scope changes the one-tile/short-range pipeline and is not a repeat of that closure.
No claim is made that its dependency chain is already measured on the critical path.

## Exact owned change and mathematical scope

`tools/attn_pr_qk2.py` pins frozen root attention_repeat2.py: raw CRLF SHA256
b296c13ad2745336e97488e297fcde751a5fc10c818c92b6f2250656369efe75;
LF SHA2566e29818bad3607a99b85c36d1a415f31e5dead5be8a95713653d837363188ef0.
It adds ONE_PR_QK2 default0 and enables the **already owned** QK2 branch only when
L1 and MODE11/13. Four K16 groups0,1,2,3 become even0/2 and odd1/3 chains, followed
by one FP32 add. Same dot products, FP16 operands and FP32 accumulator type; finite
FP32 association changes and must pass the official oracle. Existing fragment
loads, conversions, masks, query layout, cache writers, reduction frame, HMRG,
all fences/barriers and all other routes remain unchanged. The reverse textual
transform and normalized whole-Python AST equality enforce this scope offline.
No head-cluster L8/MODE3 edit overlaps it.

## Actual local compilation and independent review

Used the attention owner's `attn_local_cuda128_compile.py` adaptation of the
isolated NVRTC/PTXAS12.8.93 gate, with genuine NVIDIA headers, sm90a and exact owned
device source. Host Instance/API tail is excluded; header/NEG_INF declaration
adapters are fully recorded in manifests. This is not a nvcc13/H200 runtime proof.
Installed pinned official nvdisasm12.8.90 from NVIDIA redistrib archive SHA256
8ac63f4079c47707ec958af3a97ceeb18194e16827a2766058f5872f6c3412ae.

|Actual entry|QK2off/on registers|Stack/spill stores/loads|Static instruction count incl NOP off/on|HMMA count|
|---|---:|---|---:|---:|
|L1 MODE13|72/72|all0|1632/1632|32/32|
|L1 MODE11|128/124|all0|2432/2448|32/32|
|L1 MODE1 unaffected|117/117|all0|2032/2032|32/32|
|L8 MODE3 unaffected|114/114|all0|2312/2312|0/0 (WGMMA)|

Affected entries retain128-thread/four-minimum-CTA launch bounds and1/2 barriers.
Actual disassembly shows the independent QK accumulation chains, not just a flag
in the source. Unaffected MODE1 and L8MODE3 instruction sequences are **identical**
between off/on; source scope covers the other modes, and the runtime resource gate
checks all of them. No source-level removal of arithmetic is inferred from lower
register count. Archive: `muse_attention_decode/profiles/pr_qk2.local.2026-11-01/`.

Four actual CPU tests pass: exact source/ABI scope, private tensor cloning/full
input copy, actual six-load/noise/three-seed acceptance gate, and math/nonfinite
fail-closed controls. Independent attention owner reviewed the exact device/host
flow and found no blocker.128 replays below are repeated-input stress, not128
distinct fresh-input or every-replay math checks.

## Concrete one-check diagnostic awaiting root review

Generator emits `.build/attn_pr_qk2.py`,638149bytes, SHA256
976c64c808f40f819e78c7cf6e6998ced1cca4d69144adf1789d47aa18723369.
The judged route remains control0. One embedded owned Python body creates three
additional module namespaces with different extension names: independent control1
and candidates2/3. This is four actual compilations/states, not shared compiled
duplicates. No external comparator code is embedded. Default-off release is unchanged.

- Before timing: actual CUDA resources for every native instance; no local memory
  at affected modes; static/dynamic shared memory, threads and blocks/SM equal.
  Unaffected instance resource vectors must exactly match. All12 scored native
  paths get eager official-style math checks on each of four arms, TF32 disabled.
- Private QKV, caches, metadata tensors, outputs, workspaces and CUDA graphs per
  arm. Input copies cover **all** tensor fields and assert static metadata/shape.
  Both d1 contexts use three heldout seeds713301/302/303. Fresh captured-pointer
  replays are explicitly checked, then128 repeated replays and a final oracle check.
  A deliberately stale output under a changed gate must fail the same math check.
-80 randomized cold rounds per seed/context after256MiB L2 flush, single replay
  CUDA-event timing with no invented overhead subtraction. Both control/candidate
  duplicates define noise. Each seed's46/16 load mix must make all four cross-arm
  sixth-root score factors exceed1.002 and twice duplicate log noise; each context
  has a .1us/1% regression gate. All three seeds must win. The other five scored
  loads stay fixed in this sizing; full public promotion needs the actual pure-file
  official correctness/timing gate afterward.
-280s total guard includes import/compilation, checked between phases; it cannot
  interrupt a compiler call. All math/build/resource failures are fatal, and the
  core results print before the final decision. No catch continues into timing.

Stop on any failed gate or no gain beyond duplicates. Do not proceed to query quad
permutation/extra caps/barrier tuning automatically. A future combined head-cluster
candidate has distinct shape dispatch but shared compilation/workspaces; its score
must be measured, not inferred by adding this d1 counterfactual to a head-cluster gain.
Root review is required before the authorized single off-board check; no submit.
