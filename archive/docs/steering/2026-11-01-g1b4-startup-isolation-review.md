# G1b4 native-MoE startup isolation review, 2026-11-01

Read-only architecture review of root `.build/today/g1b4-stage-retire.py`, exact SHA256
`295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491`.
No source changes, CUDA execution, or performance claims. Startup helper was not yet
present at review time; this is an isolation recommendation, not clearance of that helper.

`dict(state)` alone aliases every original value. Replace `ma_scratch` with
`[_guarded(t) for t in _ma_make_scratch(dev)]` and `ma_ext` with `{}`. The three
scratch buffers contain (1) counter/doneA/ss-copy/MFIN integers, (2) FP32 partials/fixup,
(3) BF16 intermediates and WGS buffers. `_ma_make_scratch` returns fresh allocations and
preserves the original `_df_inter` sentinel initialization at rows128:256. `_guarded`
allocates private backing plus canary and enrolls it in process-global `_GUARDS` and
`_GUARDED_IDS`. These registries intentionally gain entries; they do not alias existing
scratch storage. Do not clear the registries, zero original counters, or reuse the
original scratch list in an isolated test. Private scratch should stay live until tests finish.

Sharing `ma_w` is safe under the reviewed ABI's read-only kernel use. Its global scales
and raw norm parameter may alias original input weights; no test may mutate them.
`_fork_prepare_marlin(weights,{})` builds a private control: `_marlin_bank` clones all six
packed/scale/global-scale tensors before the loader transforms them, and allocates independent
Marlin workspace. Its adapter retains the private control dictionary. Do not shallow-copy an
already-bound `_fork_marlin_adapter` from production state. Gamma is exactly
BF16(1+raw FP32), as in the official layer baseline. Native weight slot7 is raw BF16;
its repaired device computes the matching effective gamma. Do not pass effective gamma as raw.

`_fork_native_moe` writes only the supplied `ma_ext` cache; five keys select tiny<=4,
tiny2<=8, mid<=16, b4 through32, main above32. Module binaries may be shared by compiler
cache, but `MOE_HAS_PDL=False`; avoid changing binary-global state through setters. The
private wrapper/cache is separate from the production state. Startup tensors and references
must be newly allocated, rather than using real incoming layer inputs or cache tensors.

Recommended bounded matrix: sizes1,8,16,32,33,128,256; each with local, skew-one-local,
and all-remote routes. Construct real global IDs with nonzero start128, then use exact
subtract/filter local-ID ABI (-1 or0..31). Each token still has8 distinct global experts.
Use two fresh occupied banks per pattern, fixed captured pointers, copies before each replay,
and in-graph output poison. Check every observed replay against its private Marlin reference:
finite, global relative error<=.008 and maximum row relative error<=.075 (established prior
component gates, not a claim these are the challenge's general precision rule). All-remote
output must be exact BF16zero; poison every replay so stale zero cannot pass. Include canary
checks after synchronization, and fail CUDA/launch/resource errors fatally. This is bounded
startup conformance, not exhaustive epoch stress or independent FP32 math verification.

Next: review exact helper pin and CPU facade tests before release composition. No unreviewed
startup helper is qualified by this architecture review. Parent owns integration/publishing.
