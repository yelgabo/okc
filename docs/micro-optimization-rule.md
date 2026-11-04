# Micro-optimization rule (user, 2026-11-03)

Look past the specific variable names in the CUDA source and optimize for these four structural
hardware patterns to save microseconds. Analyze every hot kernel for the patterns first, write the
findings down (kernel, pattern, line, what the SASS shows), then change one thing at a time with a
SASS diff and a paired box A/B.

1. **Global memory coalescing.** Trace how `threadIdx.x` propagates into pointer arithmetic. Flag
   any access of the form `BasePointer + (ExpressionA * Stride + ExpressionB)` where `threadIdx.x`
   changes ExpressionA (a stride of more than one element between adjacent lanes) while ExpressionB
   is constant: that shatters coalescing. Fix by restructuring the thread mapping or staging the
   data through `__shared__` memory.
2. **Multi-store vectorization.** Find loops or consecutive lines that issue several narrow
   back-to-back stores (16-bit half or bfloat16) to adjacent offsets. Collapse them into one vector
   store: two 16-bit elements into a `half2` / `bfloat162`, up to 128-bit `float4` / `uint4`.
3. **Math precision truncation.** Find calculations whose inputs originate as 16-bit types, pass
   through 32-bit operators or transcendentals (`__expf`, `__sinf`, standard division) and are cast
   straight back to 16-bit storage. Where the architecture allows, demote execution to native
   16-bit or SIMD2 intrinsics (`hexp`, `hexp2`, bf16x2 FMA) to remove the upcast and truncation.
   The BF16 bitwise-equality requirement where SGLang keeps BF16 activations still binds.
4. **Speculative masking.** Find boundary-check `if` statements that enclose a short block of math
   and a store. Clamp the index unconditionally to a safe fallback
   (`Index = Condition ? Target : SafeFallback`), run the math unconditionally, and use a predicated
   or masked store instead of an execution branch.
