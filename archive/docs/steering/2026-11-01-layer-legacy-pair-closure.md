# Legacy G-versus-owned layer pairing: offline closure, November 1, 2026

Evidence: root `26e85addbd608e79296e8824480af9055c3c3376`; owned-boundary audit `3d89e6a`; this branch starts at `581d2e9`. No H200 job, scored run, runtime file I/O, or entrant-source compression was performed. The result is a packaging/interface closure, not performance evidence.

## Frozen inputs and result

The requested comparison cannot run through the unmodified legacy four-arm harness. No H200-ready candidate was produced.

| Input | Bytes | SHA256 |
|---|---:|---|
| root `.build/today/g1b4-stage-retire.py` | 861924 | `295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491` |
| MoE worktree `.build/today-moe/layer_lt_boundary_bf16_attention.py` | 927823 | `cb0d8b73fadb1c4dc6b38c729c2ed72dbabff726934702ed2e3b4dfd5abaecd4` |
| root `tools/dl_make_bench_pub.py` | source pin | `b286a256c8589ca4daf379c71ceb5e2cc94bdee3a738554c93bfcb1d95959722` |

`tools/dl_pub_pair_offline.py` reproduces these CPU gates, importing only the generator, parsing both entrant files, and executing the actual legacy import helper against inert source namespaces. It never imports either entrant or calls CUDA.

1. The owned source passes the legacy `inc_patch` requirement of exactly five load-inline names. The G source fails `pub_patch`: its historical threading/build sequence is no longer present.
2. More importantly, choosing G as the candidate and owned as `--other=inc` still fails the executed `_ab_import` helper with `KeyError: 'PORT_ATTN_ONE'`. G has none of the legacy `PORT_*` flags, `_bg_start/_bg_ext/_bg_seconds/_BG_BUILDS`, `_one_load/_moe1_load/_skp_load/_w4_ext/_ext`, or `_FAILED` globals. This is an architectural interface mismatch, not a missing alias alone. The helper's broad exception handler would otherwise hide it and permit a judged candidate with no valid paired experiment.
3. Additional legacy BODY globals are absent from G: `RMS_EPS`, `POST_NORM_EPS`, `FP8`, `ROUTE_EPS`, `ROUTER_MULTIPLIER`, `_E2M1`, `UNPOOL_SLICE`. In particular `_ab_rms(x, eps=RMS_EPS)` needs the constant when BODY is executed. Supplying the import flag alone cannot make this input work.
4. Uncompressed sources plus the existing BODY require **1,829,940 bytes**, before namespace/parameter scaffolding. A tested Python-AST formatting alternative, which preserves every embedded string's value but removes Python comments/doc formatting, still requires **1,795,393 bytes**. That alternative would also require restoring notices. It is **795,394 bytes over** the largest allowed 999999-byte file even before wrapper overhead. There are **zero identical string values longer than 256 characters** shared between these exact ASTs; simple identical-CUDA-literal interning saves zero bytes. This does not rule out a substantial, separately audited semantic refactor.

Reproduce from this worktree:

```powershell
python -X utf8 tools/dl_pub_pair_offline.py C:/Users/gretc/Documents/GitHub/okc/.build/today/g1b4-stage-retire.py C:/Users/gretc/Documents/GitHub/okc-today-moe/.build/today-moe/layer_lt_boundary_bf16_attention.py --legacy C:/Users/gretc/Documents/GitHub/okc/tools/dl_make_bench_pub.py
```

## Policy and preparation gates

The coordinator documents a strict file size below 1,000,000 bytes, 300-second prepare, no runtime source reads/importing reference.py/processes, and forbids making a comparator bench a public entry. The legacy generator actually embeds an LZMA/base85 representation of the other entrant and decompresses/executes it at import (`_ab_module`). This historical packaging is not approval for a new compressed entrant under the current explicit instruction to keep code reviewable and avoid obfuscating it. I did not use it to hide the excess bytes or claim it complies. The official [rules page](https://openkernelchallenge.dev/rules) returned no readable text through the web extractor in this task; no additional official compression prohibition is invented from that retrieval failure. Existing attribution/license/NOTICE must remain in any future package.

The legacy `dup=own` mode performs **four complete prepares** with independently tuned states; `dup=shared` performs **two** and shares each side's state/module between its duplicates. The import-build registry assumes the owned module structure and does not cover G's five state-dependent native expert build keys. Moving those builds to arbitrary import-time work or disabling fallback/gates would change preparation semantics and is not a validated repair.

Historical G1b2 `736a5cbc977f4ad4` took **119.270751 seconds** alone, while the prior G1b `ffe303a12a954304` failed at **353.520482 seconds**. These are different preparations, not estimates for the current stage file. Current four-prepare or two-prepare runtime and device-memory cost remain **unmeasured**; multiplying a historical time does not prove either fits or fails. No CPU-generated file is certified for H200 preparation or capture.

## Bounded alternative and stopping criteria

The tested source-preserving alternative above fails size. A transparent external H200 runner without the one-file cap could host these two intact modules: at least **1,829,940 bytes** with the old BODY, plus independent namespaces and the heterogeneous build/state adapter. It would still need corrected import/preparation interfaces and baseline-parity versus judge-style references; merely removing the size limit is insufficient. Such a runner is not an authorized/available OKC input established by this task.

For a future single-file comparison, the concrete engineering gate is an audited factoring/pruning change eliminating **at least 795,394 bytes plus wrapper/notices**, with both dispatch policies and all active CUDA literals proved intact. If that cannot be met transparently, use a separately provisioned authorized runner rather than disguised source. Start with two independent per-file states plus private data/cache/outputs/graphs, disclose shared-state duplicate limitations, and only extend to four independently prepared states after a measured budget. No gain or null threshold is relaxed.

Any eventual six-load comparison must cover all 12 sliding/full shapes with the official 46/16 weights, both heldout input/routing banks and all four cross ratios beyond duplicate nulls, full affected-path math/cache checks, fresh-input replays, selected-policy/resource manifests, and a fail-closed deadline/log summary. The existing BODY uses a restated FP32 reference and broad exception swallowing; its apparent successful judged run is not proof of paired completeness or BF16 boundary fidelity. Retain separate baseline-parity boundary gates and judge-tolerance gates. Missing arms/shapes, swallowed build errors, excess bytes, changed precision/order, or preparation overrun close the attempt without a promotion claim.

**Decision:** stop this scoped offline task. The stage-versus-G1b4 scheduling result does not supply the missing fair matched comparison to the repaired owned incumbent. No additional investigation, check, public release, or performance forecast follows from this report.
