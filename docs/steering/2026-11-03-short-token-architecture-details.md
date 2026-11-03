# Short-token incumbent architecture details

November 3, 2026. [Open all five diagrams](../.archify/architecture-short-token-details-20261103-015503/index.html). The [earlier overview](2026-11-03-archify-incumbent-architecture.md) remains available.

The diagrams expand the retained public607 decoder layer, rather than a pending standalone MoE or layer candidate. The integrated source is `muse_decode_layer/cand/g1b4_stage_release.py`, pinned at `8c09d60554e72db65e595c9c2a5eb35a743d87b6`, SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. The newer main commits through `a4e80e63` leave that source unchanged. The [source evidence manifest](../.archify/architecture-short-token-details-20261103-015503/source-evidence.json) records the selected helper hash and archived profile rows.

## What the views show

| View | Mechanism |
|---|---|
| [Tiny / Tiny2 hardware](../.archify/architecture-short-token-details-20261103-015503/tiny-hardware.html) | 288-thread CTA, eight compute warps, physical helper warp 0, three 36 KiB weight slots, register-held activations, four partial buffers, retained compute barrier and compute-owned tail epilogues. |
| [Expert A / B handoff](../.archify/architecture-short-token-details-20261103-015503/tiny-phases.html) | Gate/up, BF16 intermediate, per-job doneA readiness, down, y/square sums, post-B grid barrier, full-expert RMS and ordered route sum. |
| [Attention internals](../.archify/architecture-short-token-details-20261103-015503/attention-detail.html) | QK norm and optional RoPE, new FP8 cache rows, paged loads, split-KV arithmetic, online softmax, PV, merge, BF16 boundary, per-head RMS and sigmoid gate. |
| [Projection internals](../.archify/architecture-short-token-details-20261103-015503/projection-detail.html) | BF16 storage compression, exact escape corrections, TMA stage ring, BF16 MMA, CTA-local K reduction and output. Names the selected QKV, O and sliced-unpool configurations. |
| [After attention](../.archify/architecture-short-token-details-20261103-015503/post-attention-detail.html) | Seven actual launches, FP32 router arithmetic, tile routing, Tiny/Tiny2, sliced unpooling, residual carry and the full-feature norm boundary. |

T means batch times query length. Decode1 has B1/L1/T1 and selects Tiny. Verify1 has B1/L8/T8 and selects Tiny2. A workload with B8/L1 also has T8 but uses the L1 attention arithmetic. The close-ups explicitly distinguish these dispatch inputs.

Both Tiny paths select `@moe_helper0`, with eight compute warps and one helper. They use BF16 `mma.sync`, rather than the B4/MAIN WGMMA topology. Tiny retains two final B epilogues on compute warps; Tiny2 retains one. The selected WSB implementation retains its per-unit 256-thread compute barrier and compute-thread ring refill. The helper handles startup copies and normal epilogues. Source comments describing other branches were checked against the selected operations.

The retained QKV T8 narrow override reads escape-correction activations directly from global memory, separate from the ring. Its escape workers remain distinct from the tensor-core compute warps. The source uses the fused `gle_pr` pooler/router with FP32 router FMAs. `gle_o`, the larger O/boundary fusion, is disabled because `GLE_T` is empty. PDL is disabled. The diagrams show these selected conditions explicitly.

## Evidence and limits

The archived public607 log confirms the native backend. Its decode1 sliding/full and verify1 sliding/full profiles each report nine kernel calls: QKV, fused attention and the seven post-attention kernels. The manifest retains those four rows with their cases and original report text. Their rounded times and truncated kernel names corroborate launch structure; they do not isolate internal wait or arithmetic costs. No new timing result or speedup is claimed.

Each native diagram passed Archify validation, atomic delivery, strict provenance and real-browser checks. The final candidate and HTML hashes match their receipts. The final five views were rendered and visually inspected. All navigation buttons and their full-diagram links were exercised. The Tiny compute node's source inspector exposed the expected pinned GitHub ranges through the node index. No independent review was performed under the project prohibition on new agents.

The custom gray styling reuses the prior checked-in CSS. The local build recipe requires Archify 3.2.0, copies its package into an isolated ignored build directory and applies CSS before rendering. It preserves the installed package and runs finalizers sequentially. Native HTML, editable candidates, CSS, rebuild recipe, screenshots and final receipts are checked in together. The navigation page uses sibling files and works without the local review server.

This documentation task changes no kernel source, worker assignment, quota or GPU job. Current experiment priorities remain with their owners. Internal phase timing remains a separate profiling gap. The task ends after commit, push and remote-byte verification.
