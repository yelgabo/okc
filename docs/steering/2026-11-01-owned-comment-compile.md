# Owned native comment-only compiler equivalence — 2026-11-01

CPU-only local NVRTC12.8/PTXAS sm90a qualification for owned verify16/32 backend. Original399155B SHA256 `6de124667ee5b56c81771b4fd5d2591dc62e1bcfbf60f18baa680438d543bf42`; revised comment-preserving326005B `38e5f274a2289d4cecff8e200f6cf3ebfa825054b867ecb3fdeb921aa98245b1`. These are the final owner-frozen `prod-size/frozen/owned-15-{original,comment}.py` pair after named-credit retention. Both selected masks use the same native values; this does not qualify dispatcher/runtime behavior.

| Entry | Instructions | Registers | Barriers | Stack/spill bytes | Matrix instructions |
|---|---:|---:|---:|---:|---:|
| ONE L8 mode3 |2328|114|16|0/0|4 QGMMA|
| ONE L8 mode51 |6344|80|16|0/0|12 QGMMA|
| MoE mode51 |17528|96|16|0/0|192 HGMMA|

All three original/comment normalized instruction streams match exactly, including predicates, opcodes, operands, branch targets and barrier/fence instructions. Only compiler-generated anonymous namespace hash/identifier is normalized, including its RET symbol operand. Address comments and ELF/debug metadata are excluded. No nonempty-stream loophole: comparison asserts each stream exists and is nonempty. Zero LDL/STL instructions; PTXAS reports zero stack/spill loads/stores. PTX launch bounds and shared declarations match between arms. Dynamic shared-memory admission and host launch APIs are not measured by this device-only compilation.

Flags: actual source O3/sm90a, ONE additionally `-DONE_TRIG=1` because `PORT_SK_PDL=True`; no fast-math added. Established MoE NVRTC driver reused with narrow adapters: select native `_ONE_SRC` or `_MOE1_SRC`; stop before host Instance/launch_main; remove host-only early moe1_launch template for MoE; remove Torch/ATen/algorithm/vector headers; provide CUDA runtime/fixed-width declarations; replace unsupported ONE infinity constant declaration by equivalent device bit constant. Both native inner and anonymous namespaces closed. Every device function and synchronization instruction is retained. This is not nvcc host/API validation or GPU execution, and gives no H200 performance claim.

Evidence: [resource vectors](evidence/2026-11-01-owned-comment-compile/resources.json), [instruction hashes](evidence/2026-11-01-owned-comment-compile/comparison.json), and [full compiler outputs](evidence/2026-11-01-owned-comment-compile/compiler-outputs.zip). Archive contains all four extracted CUs, NVRTC logs, PTX, PTXAS logs, CUBINs, complete disassemblies and manifests; it is repository evidence, never runtime submission content. Original/comment first compiler-generated symbol IDs differ as expected. Resources JSON additionally records exact shared declarations; instruction equality is stronger than matrix/fence count equality.

Reproduction: from WSL invoke `tools/attention_owned_comment_compile.py SOURCE OUT {one,moe} --root ROOT` for both files, then `python tools/attention_owned_comment_compare.py` from the attention worktree using `.build/today-attn/comment-compile`. Cache `/root/.cache/okc-cuda128` contains the established CUDA12.8 toolchain. No installed GPU, official job or public action was used.

Decision: selected owned device code equivalence is cleared at these pins. Final full-wrapper notice/source audit, G-family compiler review, native dispatch/ownership/startup qualification and separately authorized official check remain required. Stop/recompile if any source pin, active macro or device transform changes; comments/byte-fit cannot establish speedup.
