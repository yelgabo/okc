# Common/RMW qualifier: focused host-resource peer review

2026-11-01. Source/CPU/real CUDA12.8 host-header review only; no GPU execution,
driver/official/service API invocation or candidate dispatch. Root owns index
and integration. The attention peer owns the full12-cell/fallback/owner protocol;
this report does not independently certify that whole orchestration.

**Focused scope clear after one host-getter qualification fix.**
`g_common_release.py` initially passed function pointers directly to the new
native getter's `cudaFuncSetAttribute` and `cudaFuncGetAttributes`. Ordinary C++
compilation against the actual CUDA12.8 headers rejected those conversions.
The author added explicit `(const void*)kernel` casts, consistent with the
ATT7 observer. This was a header-qualification issue, not a demonstrated NVCC
device compilation failure. Corrected exact getter passes the independent
header and mock gates below. No further blocker was found in this scope.

## Reviewed source epoch

Exact607e base `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.
Known prepared-source inputs remain common `4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280`
and fixed short-RMW `2884adb6d0d6947391c81cd67a0c727a04e7d41694d7c780cf8cee80dfa9dbcd`.
The author builder pins both; this is not a new expert math experiment.

Post-cast regeneration produced these reviewed scratch outputs:

|Arm|Bytes|SHA256|
|---|---:|---|
|Common|913645|`bded49570e1192dd5d9b6a3f19c702e7ac9e0004c1bda3be59b41eb9eb534873`|
|Fixed short-RMW|914133|`a12eb0d5744551c067403762b72cf272ef672c213c32822c0dc028149fed1043`|

Earlier1d1e01bf/d84733db scratch pins precede that fix. Owner tool hashes and
both complete manifests are archived alongside this report. Any later source
change needs a pin relationship/review; these are not immutable final release
claims. Normal `kernel_fn` AST is asserted unchanged by the author generator.

## Actual native resources and fixed policy

`MOE_RESOURCE` appends a prepare-only exported getter to both actual native
CUDA literals, with its declaration/registration added coherently. It selects
`okc::moe_fused<false>` and `<true>` explicitly and queries actual
`okc_smem_bytes()`/`okc::NTL`. Both variants receive checked dynamic-SMEM setup,
checked attributes and checked occupancy. Two10-field rows record registers,
local bytes, static/dynamic shared bytes, threads, max threads, resident blocks,
SM count, cooperative capability and block capacity. This is the G resource
schema: existing G B4/main local memory is recorded, **not** rejected by an
unrelated owned-MoE zero-local gate. The Python gate checks both rows and the
existing selected cooperative launcher separately enforces its whole-grid
admission on every launch. Getter success alone is not a launch/progress proof.

The resource/pipeline witnesses read the module's actual captured compile
flags. Only `tiny` and `tiny2` may carry `OKC_SHORT_FINAL_RELAXED=1`, and only in
the fixed RMW arm. Mid/B4/main must not carry it. The new runtime witness rejects
the closed gamma flag and a stage-disable flag; it does not introduce a timing
selector. Source pins plus the earlier exact per-key flag/compiler proofs are
stronger than treating this short runtime blacklist as a complete flag census.

The retained short-counter proof and compiler evidence are unchanged:
`2026-11-01-g-ordering-restoration-proof.md` (`988d618`/`a5cc063`) and
`2026-11-01-g-short-final-rmw-compile.md` (`9f7d072` plus `f46f79c`). All explicit
writer fences, joins, reader acquire, CTA publication, cache load policy,
BF16 rounds and cooperative progress requirements remain. This review does
not infer safety from tests or assert a speedup from the relaxed RMW.

## ATT7 witness and disabled normal path

The audit is thread-local and disabled by default. The observer returns before
any CUDA query or vector write when disabled. New launch-site branches then
leave the old cooperative, extended or ordinary launch arguments intact. The
author's reverse-source assertion restores the original translation unit.
The normal entry does not enable the witness; prepare restores the Python
build/attention/MoE hooks and forcibly ends the audit on success or exception.

When enabled, each observation occurs after the actual resolved launch reports
success: checked cooperative result, checked extended result, or checked
ordinary-launch error status. It records the resolved kernel family, metadata,
grid, threads, dynamic shared memory, actual attributes/occupancy and cluster
capacity. Waiting finalizers require the reviewed cooperative path and capacity;
cluster routes query the actual cluster configuration. PDL is rejected. Python
success counters advance only after native return, and require a matching host
record, so a silently caught attention fallback cannot masquerade as native
coverage. These checks witness host launch submission, not asynchronous kernel
completion; the preparation synchronization/math protocol has its own scope.

## Executed gates and limits

`tools/proj_g_common_host_review.py` executed:

- Exact corrected native getter against installed official CUDA12.8 runtime/
  NVCC headers: compile-only PASS. Only kernel declarations, launch constants
  and `TORCH_CHECK` are stand-ins; no getter statement is replaced.
- Exact native getter CPU mocks: two10-field rows including deliberately
  nonzero local bytes8; all5 injected query/setup API failures reject.
- Exact ATT7 AUDIT plus actual patched launch branches against those same
  headers: compile-only PASS. The author emitter supplies minimal Params/Cfg
  and converts only the unchanged chevron launch into a host API declaration
  call for ordinary-C++ parsing. New launch/query calls remain exact.
- ATT7 CPU mocks: audit-disabled zero CUDA calls/no record, successful
  cooperative/cluster rows, and7 rejection cases PASS.

Commands, return codes, exact C++ sources and owner hashes are under
`evidence/2026-11-01-g-common-host-resource-review/`. These gates do not link
Torch extensions, compile full device modules, run CUDA, qualify H200 resource
values or prove full preparation stays within295s. CPU mock success is not
actual hardware admission/capture success. Whole-orchestration facade and
independent owner/protocol review remain separate prerequisites.

Stop this bounded peer review here. No additional variant, kernel change,
timing harness or GPU action is requested. A final release decision still needs
the exact final artifact, all other peer gates and the root-owned runtime plan.
