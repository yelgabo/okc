# Owned VERIFY32 sliding slice: source checkpoint, 2026-11-01

Scope: Family G fair cross-architecture diagnostic only. No GPU, dispatch, public action,
or kernel tuning. This checkpoint is deliberately **not ready for H200**: parent requested
priority switch to the G1b4 native-MoE qualification review before prepare conformance work.

## Pins and measured context

Input tracked root `muse_decode_layer/cand/layer_lt_boundary.py` SHA256
`851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a`.
Public historical job `cb71acec29d141c6`: PASS13, 2.043438469,
CI 2.012786892–2.074090046. Its receipt preparation was 188.332763011 seconds.
G1b2 separately prepared in 119.270750898 seconds; arithmetic sum 307.60 seconds
is not a preparation forecast for specialized sources.

The builder first applies our pinned BF16 pre-output-RMS repair. This derivative
was CPU/source checked, never H200 qualified. It is not identical historical board bytes.

Selected scope: VERIFY32 sliding, B32, q_len8, T256, window2048, page1,
context-capacity80000. Weight46 on the layer board; a single-load diagnostic is not an
aggregate score. Historical selections: Lt QKV L0/unpool L1, ONE `(4,1,3)`,
attention boundary `c1x512`, front `concat-r`, main MoE51. All twelve attention
cases originally selected ONE; other shapes/routes are intentionally rejected here.

## New semantic slicing result

`tools/dl_owned_slice.py` emits **504,940 bytes**, SHA256
`43ee0f46af76349f4e48ae7bb9c5c77b07564858ee6ac096fda4bf73e0b724a1`.
This leaves 495,060 bytes under a decimal 1MB cap for the other arm and pairing harness.
It establishes an upper bound for this source-only owned arm, not a verified combined artifact.
It is distinct from whole-file formatting: modules, builders, autotuners and branches for
other paths are removed. Retained device literals remain byte-identical to the repaired input.

Active extensions and registrations:

- Main: `_CUDA_SRC`, `_BND_SRC`; `route_fwd`, `attn_boundary`.
- ONE: original `_ONE_SRC` and host source; original three registrations and resource planning.
- MoE: original `_MOE1_SRC`, host ABI; dispatch only ordinal10 to `launch_main<51>`.
- Lt: original `_DLPLT_CPP`, original API list, frozen T256 QKV/unpool configurations.
- O projection and pointer-alignment fallback use the original SGLang BF16 dispatch helper.
- Final FFN boundary uses the original SGLang module/weight semantics.

Prepare constructs only required norms/residuals, pool/router concatenation, rotary inverse
frequencies, original ONE resources/scratch, original skinny+fat expert packs/scratch, and
two pinned Lt plans. It has no opaque dispatch tuner. Attention planning must reproduce
`(4,1,3)` or fail. No unsupported-scale fallback exists. Unused template definitions remain
in retained CUDA literals; pruning them is not needed for this initial owned byte budget.

## Verification and gaps

Three CPU tests pass: exact source pin and retained-literal identity; actual emitted
registration-builder execution with inert `load_inline`; actual emitted kernel Python
execution verifies stage order and rejects wrong load. Reproduce:

`python -m unittest discover -s tools -p test_dl_owned_slice.py -v`

These are source/inert-API tests. They do not prove extension linking, actual prepare,
resource admission, graph capture, correctness, timings, or a valid cross-architecture pair.
Prepare still needs a real CPU facade test covering effective norms/state/packing arguments;
then independent source review and local compile/link feasibility. Exact page-table capacity
and original O-projection selected-path eligibility should be rechecked against live inputs.
Shared full CUDA literals can contain unused registrations/definitions with external dependencies;
local compile/link is required before claiming preparation integrity. No measured gains claimed.

Stop if the combined artifact exceeds cap, any retained computation/ordering differs outside
explicit common BF16 parity repair, prepare cannot reproduce pinned dispatch, or actual
correctness/resource/capture gates fail. Do not use a single-load timing as a six-load score.
Parent owns report-index integration/publishing; this checkpoint is committed locally only.
