# D32-MOE-01: B4 already selects the parallel prefix scan

November 2, 2026 Pacific. Picked up after the user asked for B4 prefix work while the GPU queue clears. Source checkpoint `5ab861e3`, inspected remote `ce1d2f77`. This supersedes the selected-source premise in [D32-12](2026-11-02-12-h200-460-next-leads.md).

The pinned B4/T32 build already runs the all-lane prefix scan and four-lane range lookup. The serial code cited by D32-12 belongs to the helper literal used by Tiny, Tiny2 and MID. Creating the prescribed B4 treatment would repeat existing work or change the active cost model. Close this exact serial-to-parallel proposal as already implemented. There is no new candidate or GPU allocation.

## Helper execution correction, November 2

[General Optimizer's selected-flag preprocessing](../2026-11-02-general-short-prefix-result.md), published in `3a961a9d`, shows that Tiny/Tiny2/MID compile out the helper's serial WGS functions under `OKC_WGS=0`. The helper entries below identify raw literal contents, not an executed serial scan. B4's parallel-scan conclusion remains unchanged. Future execution claims require preprocessing with the captured flags.

## Source and factory evidence

[Reproducible factory audit](../../../tools/d32_b4_prefix_audit.py) executes the final `_fork_native_moe`, `_main_defines` and `_ma_ext` definitions with a CPU stand-in for `load_inline`. It uses the selected private state with PRMTSC and SC6 enabled. It records the actual compiler inputs and token selections in [selected-source.json](evidence/D32-MOE-01/selected-source.json).

| Input or build | SHA256 | Selection |
|---|---|---|
| Complete public607 control | `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab` | `muse_decode_layer/cand/g1b4_stage_release.py` |
| B4 and MAIN native | `c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71` | `MOE_CUDA_SRC`, all-lane scan |
| Tiny, Tiny2, MID native | `b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716` | `_MA_ALT_SRCS['moe_helper0']`, serial scan |

The actual factory chooses B4 for tested token counts 17, 24 and 32. Its flags include `OKC_WGS=1`, `OKC_B4FIN=1`, `OKC_SC6W=1`, `OKC_PRMTSC=1` and `OKC_DQ7=1`, without a helper source selector. Counts 33, 128, 256 and 384 choose MAIN with the same native literal. Counts 1, 8 and 16 exercise the three helper builds.

In the selected native literal, the x-producer calls `wgs_ranges` with all 32 lanes after barrier 7. The function scans 32-job chunks with `__shfl_up_sync`, carries lane 31 totals between chunks, writes terminal totals in lane 0 and computes the four range boundaries in lanes 0 through 3. It retains both warp synchronizations.

Its A/B costs are `((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5` and `((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4`. The per-unit terms are part of the active cost model. The different serial formula in D32-12 is from the helper literal.

## Validation and limits

Ran:

```sh
python3 tools/d32_b4_prefix_audit.py --out docs/steering/decode32/evidence/D32-MOE-01/selected-source.json
python3 tools/d32_b4_prefix_audit_test.py -v
```

All three CPU tests pass. They capture actual factory selection and reject injected B4 helper substitution and a serial producer call. [Complete test output](evidence/D32-MOE-01/cpu-tests.txt) is retained. The helper-substitution negative case modifies the last factory definition, because the file contains older definitions too.

These checks establish selected source and flags. They do not establish compiled execution, numerical correctness or performance. No native code changed, so no new compilation, serial/parallel numerical comparison or GPU timing was required to decide this source-identity question. The external note's reported 0.9% remains unverified and cannot be credited again to this control.

## Decision and next action

Preserve the frozen attention diagnostic `9e5e5d64`, published in `77023d44`. The latest shared job list still shows another owner's check running; no new D32 check was sent. Continue the one registered attention routing diagnostic when the shared queue clears, under its existing preflight and source gates.

Any renewed B4 prefix task needs a mechanism that differs from this selected implementation. Examples include reducing repeated range work or changing where metadata is produced, but neither is an implemented or measured candidate here. Research owns the next design under D32-04. Reopen this specific proposal only if a different selected B4 source is shown to use a serial scan.
