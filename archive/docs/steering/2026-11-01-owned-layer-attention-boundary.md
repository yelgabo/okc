# Owned layer attention value-boundary repair

November1,2026. Parent evidence1f6dcf7. No GPU quota or public action. Known-good
historical source remains untouched. Root owns integration, index and publication.

## Exact source and measured path reachability

The historical owned incumbent `cb71acec29d141c6` is
`muse_decode_layer/cand/layer_lt_boundary.py`, SHA256
`851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a`.
Root's receipt/run-log records PASS13,2.043438469x,95%CI2.012786892–2.074090046.
This audit makes no new performance or correctness measurement of the repaired file.

The owning receipt's SUMMARY records `_attn_one` on all12 scored cells:

| load | sliding mode | full mode |
|---|---:|---:|
|decode1|13|11|
|decode16|1|49|
|decode32|1|49|
|verify1|15|11|
|verify16|3|51|
|verify32|3|51|

Those modes use `_ONE_SRC::finalize_rows`. Its scaled value remainedFP32 before
squaring for output RMS and applying sigmoid gate. The adjacent `finalize_det`
contains the same choice, but is explicitly diagnostic under `ONE_DETM`, whose
default is0 and which `_one_load` does not enable. The generator covers both
definitions; no claim is made that the diagnostic ran in the incumbent.

All attention port flags are enabled in the source, but preparation self-tests
independently decide their state flags. `_attn_path` first chooses `_attn_one`
when eligible; failure/ineligibility falls through to `_attn_fused` for qualifying
decode shapes, `_attn_ws` for qualifying verify shapes, then `_attn_prep`, then
SGLang. Unsupported head dimensions go directly to SGLang. Therefore repairing
only the observed ONE finalizer would leave reachable fallbacks inconsistent.

`_ATTN_SRC::merge_row` serves fused cluster/noncluster output combination and
keeps acc.x/yFP32 before output RMS. `_WS_SRC::ws_combine_kernel` does likewise.
Both are covered. The WS host selects `launch_var<7,3,0,16>`: its score path is
FP32, rather than the separate dormant FP16-accumulation variant in that source.

The remaining production fallbacks already implement the value boundary:

- `_attn_prep` converts FA3's output toBF16 if needed before `_ext().attn_gate`;
  `_PREP_SRC::gate_kernel` loads BF16 values before squaring and RMS/gating.
- `_attn_sglang` explicitly calls `.to(torch.bfloat16)` before `o_norm_gate`.
- `_fa3` can return a tensor or tuple; both callers perform their conversion after
  selecting the tensor. No alternate unrounded return path was found there.
- `_attn_ref32` intentionally retains FP32 for a mathematical diagnostic. It is
  not a timed fallback and is not changed by this repair.

## Minimal separate generator

`tools/dl_owned_attention_boundary.py::patch_owned_attention(source)->str`
requires the full incumbent hash and exact hashes of `_ATTN_SRC`, `_WS_SRC` and
`_ONE_SRC`. It replaces four narrowly identified pre-RMS statements:

1. ONE ordinary eight-value loop;
2. ONE dormant diagnostic eight-value loop;
3. fused attention's x/y output pair;
4. WS combination's x/y output pair.

Each scalar computes BF16 RNE of `acc*inv`, then converts it back toFP32 for
the existing squared-sum, RMS and gate math. No extra cast is placed on Q/K,
scores or probabilities. No fence/barrier, cache store, scheduler, dispatch,
projection, MoE or tolerance changes. Every other Python AST node and device
literal remains identical. Changed source or a second application fails closed.

Offline output `.build/today-moe/layer_lt_boundary_bf16_attention.py`:927823B,
SHA256 `cb0d8b73fadb1c4dc6b38c729c2ed72dbabff726934702ed2e3b4dfd5abaecd4`.
This is an unqualified diagnostic repair, not a replacement public candidate.
The original incumbent's bytes/hash are preserved.

Three source/dispatch tests pass: reverse the exact four substitutions and recover
the original device strings; enforce full-source pin and non-device AST identity;
execute the actual `_attn_path` selection and every fallback; verify existing
BF16 fallback code and unchanged FP32 diagnostic. Two real Torch2.10CPU tests pass:
RNE bit conversion and observable pre-RMS placement, including negative controls
for omitting the round or rounding values after unrounded statistics; and execution
of the actual Python FA3/prep/SGLang fallback wrappers with CPU stubs to verify
BF16 reaches the gate. These do not execute native CUDA, FA3 or SGLang kernels.

Logs are under `muse_moe_w4a16/profiles/owned_attention_boundary.*.2026-11-01.log`;
the component owner retained them there rather than editing root's profile tree.

## Precision policy and separate unresolved choices

The [official precision restriction](https://openkernelchallenge.dev/rules) preserves
BF16 activations where SGLang keeps BF16, retains the FP8 KV cache, and permits
queries no coarser than FP8 E4M3. It is not a requirement for every reduction to
be bit-identical to SGLang. Nor does it enumerate every fused-register rounding
boundary or explicitly grant arbitrary FP32 substitutions across BF16 boundaries.
This audit does not issue a new eligibility ruling on the historical submission.

The repository's baseline-parity release gate resolves this particular ambiguity
conservatively: the official layer baseline at312 explicitly materializes BF16
before output RMS/gating, so preserve that operation boundary in the fused path.
Passing numerical tolerances or keeping extra register precision does not itself
establish that parity. Do not label this repair a speed optimization or infer
that earlier official passes proved intermediate-boundary identity.

Previously documented differences requiring separate attention remain unchanged:
native key normalization/RoPE flows FP32 directly to FP8, whereas the SGLang path
has a BF16 step first. The historical `_attn_self_test` explicitly records resulting
one-step FP8 differences and the `003409c1282043d6` cache mismatch; the repaired
output boundary does not settle that issue. ONE decode queries use FP16 with FP32
score accumulation, while verify queries round to E4M3; those different query
paths and scale placement are outside this repair. No claim of a newly measured
query/cache failure follows. The inactive WS FP16-accumulation variant must not
be mistaken for the selected VAR16 path. The FP32 mathematical diagnostic also
does not reproduce every BF16 boundary and should be labeled accordingly.

Next gate is a separately authorized H200 qualification of the exact repaired
file, including all active and fallback output-finalizer paths, fresh replay and
official correctness. Preserve the incumbent until then. If native setup fails,
verify the actually selected fallback instead of claiming the native repair ran.
No quota expenditure, public action or broader precision rewrite is authorized
by this no-GPU task.
