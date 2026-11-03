# Verify16/32 paired slices: first-draft policy review

2026-11-01; root checkpoint `af3b550`, reviewer base `08e38cb`. Read-only review
of root's uncommitted `tools/dl_verify_pair_slices.py` and its three CPU tests,
plus their previous single-cell builders and exact source/receipt policies.
Root is implementing corrections; this report does not clear a later revision.
No root files edited, GPU dispatched or performance claim made.

**First draft is not policy-faithful at owned128.** Unchanged native literals
and Python binding coverage do not prove an unchanged chosen pipeline. Two
specific selectors differ from the owning receipt and need correction or an
explicitly different fixed-policy comparison label.

## Exact source owners and blockers

G source is public `607eb52bdaf24160`,869076 B, rawSHA
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Owned input is `muse_decode_layer/cand/layer_lt_boundary.py`, source851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a,
owning public `cb71acec29d141c6`; the slice applies the common BF16 attention
output repair. This repair is an intentional difference from historical cb71,
not proof that every other policy may also change silently.

1. **Owned QKV128:** `_slice_proj` directly falls back to `_gemm`; original
   `_proj`20665 can select `state['skp_wide']` for65..128. Actual cb71 verify16
   sliding/full profiles contain `wd_kernel<1,128,3,1,false...>` at15.0/14.8us.
   This is QKV128, not a Torch-mm fallback. The other dense rows are concat-PR
  26.4/25.8us, pinned unpool21.2/21.0us, O7.9/7.5us. Exact existing constructor:
   `_skp_wd_runner(_skp_ext(), 'b1n128c3s8h', state['dense']['qkv_gate'], diag)`
   with private CUDA int32[4] diagnostic storage. Retain exact selected WD
   source/helpers/registration and prepare/warm it, or limit comparison to256.
2. **Owned attention-to-FFN boundary128:** first draft inherited hard-coded
   `_abnd(...,'c1x512')` from256. cb71 owning log `SUMMARY stable abnd T=128`
   (line85) selects **sglang**; T256(line86) selects c1x512. Original `_abnd`
  19576 already implements geo`sglang` by calling
   `state['attn_boundary'](stream,branch,branch_dtype=torch.bfloat16,router_dtype=None)`
   for these token counts. CUSTOM already constructs `post` (unweighted
   post-attention Muse RMSNorm), `ar` residual and `pre` pre-FFN norm. Prepare
   `FusedPostNormResidualPreNorm(post,ar,pre)` as `attn_boundary`, then select
   sglang128/c1x512256. Merely retaining the custom CUDA literal is insufficient.

## Other selected policy audit

|Stage|Owned128/256 finding|G128/256 finding|
|---|---|---|
|QKV/O/unpool|QKV256 and unpool128/256 are exact three DLPLT pins; unpool128 layout3 verified. O128/256 use original `_gemm`; no O-wide path. QKV128 blocker above.|Eight original Lt routes retained, includingFP32 PR and custom QKV256 initializer; fail if any pin is absent or split-K rejected.|
|Attention|Owning summary v16/v32 full264/1/mode51; sliding8/1/3 and4/1/3. `_one_setup` actual residency/cooperative-probe owner must satisfy pinned assertions.|Same ATT7 literal/launcher/flags as607. Its actual128 profiles show sliding `attn_kernel<8,false,3>` and full `<8,true,0>`.|
|Boundary|SGlang128 blocker; customc1x512256 matches. Final FFN boundary remains original SGLang module, BF16 branch + FP32 residual.|Actual `_graft_boundary`12306 uses NW32 at128, NW16 at256. First draft correctly fixes both pre/post launch arguments. Double BF16 branch rounding and FP32 stream preserved.|
|PR/router|cb71 summary front128/256 selectsconcat-r. Torchmm FP32 concat output plus `_route_fwd(...,lo_off256,pooled)` matches `_moe_front_route`19780.|Same packed concatFP32 + `_graft_route`, no separate small-route mode at128/256. Local IDs remain-1/0..31; raw pre-expert BF16 gamma unchanged.|
|Experts|cb71 summary and profiles selectMAIN51 for both; ordinal10 dispatch preserved. Effective post-expert gamma stays BF16 then float ABI.|607 log confirms native repaired/cooperative/BF16, MFIN/DF/YBF/PDLoff. Both counts choose main, not tiny/mid/B4. Per-state PRMT/SC6 eligibility and DQ7 application retained; raw BF16 gamma packed atma_w[7].|

Native G128/256 flags are common mainNW16/FAST1=0/EPI4=0/WGS=1,
SC6W=1 plus PRMT source suffix, with DQ7 appended by `_ma_ext`. `_fork_native_moe`
16084 uses this same route at128/256. The slice intentionally requires native
SC6 and fails unsupported weights; it is not an implementation of every
production fallback. Original native host cooperative/resource guards remain.

G preparation compiles ATT7 but `_workspace7`/cluster buffers and input-shaped
kernel paths must be warmed on actual private owners before capture. The full
four-cell harness must warm/check every cell; compiling a module is not warming
its persistent workspace. Neither source pins nor three CPU tests establish
H200 prepare duration, captured allocation safety, numerical parity or timings.

## Required next validation

Root was notified immediately of both blockers. After repair: assert exact
128/256 selectors against owning source/log rows, exercise actual emitted
prepare/kernel through CPU facades (including SGlang module constructor/call
and selectedWD runner), retain all native source pins and guard all four
geometry/cache/scale domains. Then independent final package/owner/transport/
deadline review is needed before any reset-time GPU check. No global same-
process or historical incumbent performance equivalence can be claimed from
these source-only slices.

## 2026-11-01 corrected package addendum

**Source/host review cleared for the scoped diagnostic.** This supersedes
the two first-draft selector blockers, not the remaining H200 qualifications.
Final package is968389 bytes, SHA256
`2bad52ceca9ef72d73baa539dc9db0c776a51b6009a4e4a4f8a8d9b725761c0f`.
Reviewed root `tools/dl_verify_pair_slices.py`, `dl_verify_pair_body.py`,
`dl_verify_pair.py`, both test modules and the unchanged underlying single-
cell slice/owner/reference helpers. Root's source was uncommitted during
review; this report pins the generated artifact, not an invented root commit.
Reviewer checkpoint `bc07b23`. No root files edited or GPU job dispatched.

The owned128 QKV now uses `_skp_wd_runner(_skp_load(),"b1n128c3s8h",...)`
with private `verify_diag`. Three additional SKP native literals are exact
copies from source851c. `_slice_proj` selects that runner at128, QKV256 Lt,
and ordinary O GEMM at128/256. Owned Lt plans are exactly unpool128 layout3
`[66,18,1,0,0,2,35,0,5]`, unpool256 layout1
`[66,20,1,0,0,4,35,0,3]`, QKV256 layout0
`[66,124,1,0,0,0,35,0,2]`; all have independent weight/workspace closures.
The128 SG attention boundary now constructs
`FusedPostNormResidualPreNorm(post,ar,pre)` and calls it through original
`_abnd(...,"sglang")` with BF16 branch and no router output. At256 it retains
`c1x512`. Both retain original effective norm/gate weights, FP32 concatenated
pool/router output, original main51 MoE with BF16-effective expert gamma,
and original FFN boundary. Common BF16 attention-output repair remains an
explicit difference from historical cb71.

G's four selected pipelines retain all original selected native literals,
eight Lt plans, ATT7 call, SC6 eligibility/packing/flags and original native
cooperative-launch guard. The row-boundary constructor/launch uses NW32
at128 and NW16 at256 as in607. Unsupported shapes/scales/context capacity
fail closed rather than silently selecting an unmodeled fallback. This
does not certify support for every valid general challenge input.

Two separately prepared owned states have separate resident tensors,
SGLang module buffers and Lt workspaces; two G owners use the audited clone
with private scratch/workspaces and retained immutable weights/code only.
The alias walker includes lazy `torch.nn.Module` buffers and Lt closure
defaults; tests reject both alias types. Each capture owns/retains its
private input bank and output. Prewarm uses an ordered side stream before
capture. Original and changed-input replays each require all four natural
global ID outputs to be exactly equal; no fixed external IDs bypass routing.
Four cells/two seeds require96 numerical checks: four owners times original,
fresh and post-timing checks. Repeated fresh replays use one changed bank,
not four independently drawn fresh banks. WD diagnostics, original G guard
storage and cloned guards are checked. The final narrow addition checks
both owned MoE `ctr[8]` spin flags after synchronized post math, outside
capture; it changes no timed entry or native literal.

Seven actual TorchCPU tests reran successfully in7.374 seconds, including
four-cell actual entry launch-policy facades, selected128 projection and
boundary selectors, three Lt plan/closure contracts, native literal/global
binding equivalence, lazy module/workspace alias negatives and report
transport. Independent in-memory regeneration produced exactly the final
size/hash above and included the spin check. Legacy single-cell builder
default output remains850769 bytes/SHA305d8e2804d8d10c3fdda902d659c25177ce7ad31d7575dda803c318830e8dc7.
These tests do not execute the complete four-owner CUDA preparation.

The report retains all eight records/four-arm24-sample vectors, mandatory
finite positive timing checks and96-count assertion. It labels the46/16
mixture comparison as only two of six loads. Per-cell sixth-root leverage
uses owning G607 cell medians and a conditional transfer of private event
ratios; it is not a measured full-six-load hybrid score. No other-cell
speedups, cross-receipt equality or public hybrid size compliance are
claimed. SGLang remains the normal judged path of this private artifact.

Remaining gates: actual H200 host/device compilation and resource admission,
runtime memory/capture behavior,96 oracle checks/natural IDs, lossless log
transport with real timings and completion within the295-second local guard
and300-second external setup limit. The source package's sub1MB fit does
not establish a future public hybrid's byte fit. No H200 readiness or
performance decision is made by this offline clearance.
