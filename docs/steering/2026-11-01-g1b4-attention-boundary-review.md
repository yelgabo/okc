# G1b4 attention boundary: independent source/CPU review

November1,2026. Read-only audit of root's `tools/dl_build_g1_qualification.py`
and `tools/test_dl_fork_attention_boundary.py`. No source edits, CUDA execution,
MoE experiments or public submissions by this reviewer. Root owns publication.

No blocking source defect found. The official layer baseline at
`muse_decode_layer/baseline.py:312` casts the attention result toBF16 before
calling `o_norm_gate`. The revised native `pr8` performs BF16 RNE and converts
back toFP32 before squared-sum reduction, RMS normalization and sigmoid gate.
That restores the explicit value-vector boundary; it does not change Q/K or
probability precision.

All five native ATT7 output-RMS finalizers flow through this hook: four scalar
uses in the four-value path, eight in the eight-value path, and three loop-body
uses, for15 call sites plus one definition. Every call operates on the scaled
attention value (`*inv`) before its sum-of-squares. The three other native rsqrt
uses are in prelude/QK contexts and remain unchanged. No hook call is applied
to Q, K, scores or probabilities.

The builder exact-pins ATT7 literal
`0754d5ffd18e00614787c6bbb8a78c59f503fb9ccc2691bd10420fb65f7729d7`, requires
one unique identity-hook body and16 total references, then replaces just that
body with `__bfloat162float(__float2bfloat16_rn(x))`. The test reverses that one
replacement and checks byte identity of the device literal against G1b3. Thus no
fence/barrier, scheduling, Q/K/P conversion or other GPU arithmetic change hides
inside this ATT7 delta. Existing empty-route/Marlin changes are common to G1b3/4.

Independently ran both actual Torch2.10CPU tests: PASS. The numeric fixture proves
the intermediate round can change final BF16 output and verifies the finite-value
RNE bit construction against Torch's BF16 round. These are source/boundary and
host numerical checks, not H200 correctness/performance qualification or a complete
proof of native attention arithmetic against FA3. The usual official layer gate
is still required. No further MoE work follows from this attention review.
