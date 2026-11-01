# OFF native effective-gamma representation prototype

November1,2026. Root separately authorized this narrow source/CPU prototype
after publication of exact ladder `5958b28`. No GPU, official/service call,
compilation, new broad harness or production action. Campaign remains paused.
README is intentionally left to root. Existing RMW and common inherited
progress repairs remain separate source epochs, not bundled here.

## Source and coefficient contract

`tools/g_native_gamma_precompute.py::build(source, enabled=False)` requires the
exact607 layer LF SHA
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
main literal `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71`
and helper0 `b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716`.
The pure source has no timing selector, reference output substitution or
comparison code. Generated second line warns SOURCE/CPU ONLY, NOT QUALIFIED.

|artifact|bytes|SHA256|
|---|---:|---|
|default OFF|870703|e17901d562b35f127c49110d222205fab254d078ae37e941a687774973a44c18|
|explicit enabled prototype|870702|d6756bb50a0362af9c9cd03d44a88c80fa593628254ffb2a931a643fccafdcad|

Scratch output directory: `.build/today-moe/gamma-precompute`. Full manifests,
including all old/new native consumer expressions and literal hashes, are
tracked under `evidence/2026-11-01-g-native-gamma-precompute`.

The immutable effective coefficient remains exactly
**BF16_RNE(float32(rawBF16)+1)**. The enabled host helper allocates that BF16
tensor once and passes it in native `ma_w[7]`; it never writes the supplied
raw coefficient. Size remains4096BF16 values/8192bytes and the native pointer
ABI remains `const __nv_bfloat16*`. Disabled preparation returns the historical
`raw.contiguous()` value. Enabled preparation explicitly rejects non-BF16,
wrong-size or nonfinite parameters outside this proof domain.

Native consumer changes are complete, not short-only numeric partial edits:
all16balanced `fork_bf16(1+read)` expressions in **each** main/helper literal
become `fork_gamma(read)`. Under fixed compile flag
`OKC_FORK_GAMMA_EFFECTIVE=1`, fork_gamma returns the already-effective converted
BF16 coefficient; flag0 executes the original add/RNE expression. The flag
accepts only0/1. Actual selector appends it consistently to all five native
module keys, including the helper0df alias, and rejects a state-format mismatch.
The state records `ma_gamma_effective`; module cache keys include the format
define. Early compile-before-pack is consistent because both use the same
fixed source-epoch boolean, not a per-input or timing decision.

This handles direct BF16 reads, cached `__ldg` BF16 reads and unsigned16-bit
reads converted through `__uint_as_float(bits<<16)`. The actual selected short
functions contain four such expressions; the other literal sites remain
consistent for all other selected and compiled paths. Changing host slot7
alone, or only those four sites, would be an invalid format mismatch.

All GEMM1 gate/up, SwiGLU and GEMM2 BF16 boundaries remain. Global scales,
gamma multiplication, RMS/sum ordering and output conversion remain in place.
No threadfence, proxy fence, atomic, barrier, launch/admission, PDL/MFIN/DF/YBF,
routing, stage retirement or empty guard is edited. The whole-source AST test
proves everything outside the two native literals, native packer/selector and
new helper/constant is identical, including both fallback implementations.

Marlin keeps its own effective-gamma preparation from the original raw weights;
owned fallback keeps its separate effective-FP32 ABI. Their code is unchanged.
If a native resource fallback occurs after native packing, its existing adapter
uses its private bank/gamma, not native slot7. No globally supplied gamma field
is rewritten, and no raw tensor alias is modified.

## CPU proof and actual selection tests

Three actual TorchCPU tests pass in the existing WSL host-validate environment,
0.806s on final bytes. The first uses the **generated host helper**, not a
second implementation: enumerate all65536BF16 bit patterns, cover every65280
finite input, and compare its prepared coefficient bits against independent
binary32-add plus integer BF16 ties-to-even rounding. Nonfinite patterns are
excluded explicitly; bad dtype/length/nonfinite inputs fail closed. Tests also
verify no raw mutation, separate enabled storage and unchanged OFF aliasing.
This is host scalar equivalence, not an executed CUDA helper or kernel proof.

The second reverses all32native expression edits and the helper insertion to
recover exact prior literal bytes. It restores only the two allowed host
functions/removes the new definitions and asserts full original AST equality.
It also executes the actual packer's slot7 expression structurally: that slot
calls the derived coefficient helper. The third executes the **actual modified
native selector** on all1/8/16/32/128/256 token sizes under both formats, with
five independently recorded module keys, unchanged source suffixes/SC6/B4
selection, cache reuse and rejected mismatched state-format metadata. CUDA
extension construction is an explicit stub in this CPU test.

Reproduce:

`OKC_REPO_ROOT=ROOT python -m unittest discover -s tools -p test_g_native_gamma_precompute.py -v`

The initial default-root invocation on the stale worktree failed source lookup
before tests; the explicit root override resolves tracked source dependencies.
An initial selector fixture attempted literal-evaluation of the derived MID
tuple and was corrected to use the actual audited tuple interpreter. Final
three tests pass; no failed native/GPU result exists or is concealed.

## Interpretation and next gate

This is an exact alternative **representation** of a required BF16 effective
parameter. It removes both the newly introduced runtime round and the FP32 add
already present in G0. Therefore any later runtime comparison would not isolate
only the added RNE cost. Existing prefetch amortizes some gamma expressions;
neither four source expressions nor8192bytes is an exposed-time estimate.
No speedup, register reduction, instruction savings or G0-gap recovery is
claimed without selected code generation and conforming matched measurement.

Stop now at source/CPU as authorized. Before any GPU eligibility, independently
review the format/lifetime proof and actual consumers; compile selected
Tiny/Tiny2/Mid/B4/main entries to verify rounding/read representation, preserved
math/fences and resource/admission effects. Retain source pins and compare to
the same common repaired baseline. The inherited ATT7 progress issue applies
to this full-model source too; root's common repair must be applied to both
arms in a later epoch. No current artifact is public or GPU qualified.
Any future root-authorized comparison needs full-six/private four-cross and
three-bank>=1%/>2xnull, context regression, finite/fresh/cache/routing/generation/
capture/resource/budget gates. No broad harness, new knobs or dispatch is
authorized by this report.
