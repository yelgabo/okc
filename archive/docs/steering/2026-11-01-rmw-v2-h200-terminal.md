# Repaired RMW arm: H200 qualification and descriptive result

November 1, 2026 Pacific / November 2 UTC. Evidence checkout main
7b8ce81; implementation and pre-dispatch registration
**a0b6af2ec00dfed54f37fd81de00c14098748711**, accepted-job handoff3ab9fa8.
This report supersedes the queued/result-null snapshots for own
**private check c1931f0c9e834485**. No new check or public submission is made here.

## Decision

The repaired single arm passes complete native startup and official evaluation
on NVIDIA H200. The preparation blocker is resolved for this exact source/run:
prepare completed in109.269811068s, within both295s qualifier and300s judge limits.
The selected RMW treatment was exercised. This is the first qualified GPU
measurement of the repaired arm, not a promotion or an isolated causal RMW result.

Official score **2.1212410493206884x**,95%CI
**[2.089422433580878,2.1530596650604985]**. Historical607 is2.1260826764231093x;
the new point observation is0.227725% lower. Both short-load weighted latencies
are lower, while decode32 and verify16 are higher. All12 official case seeds
differ from607, and baselines/observer overhead also differ. This comparison
cannot attribute gains, regressions, or recovered historical gap to the RMW.
Retain the existing board incumbent and matched promotion gates.

Next priority is the captain's actual remaining G short-path cost/dependency
budget and a preregistered matched OFF/ON attribution decision. This successful
setup does not justify another setup rewrite or a blind repeated single-arm run.
No second check is dispatched from this receipt. Initial tranche usage remains
1/4; the live terminal ledger shows9 private checks/5 scored slots remaining and
no live jobs. Public measurements still require direct user authorization.

## Exact source, receipt and execution ownership

- ON file .build/rmw-v2/short-rmw.py:918978B, SHA256
  978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a.
- Reviewed, regenerated OFF file .build/rmw-v2/common.py:918490B, SHA256
  32196b8bc1c954e9b7f2c6667e6b3e0aeed14da0fed6fe53bc153c9d563e5047.
  OFF has not been independently GPU-qualified by this job.
- Own receipt: board muse_decode_layer,mode check,id c1931f0c9e834485,
  status done,result status ok,passed true,returncode0,clean flags/checks/error.
- Server content identity ce36aa2c06c6003cea3613d1e40d94f986d29cd38fc6352ccb7363b82aab30bd
  and matching payload_ref are unchanged from acceptance; this server hash is
  distinct from the physical source SHA256.
- Submitted00:10:29.155UTC; worker started00:45:20.293UTC; finished00:49:11.578UTC
  on November2. Queue time is separate from preparation/runtime.
- H200 node n-ddrylp36a7j3gkvnb7dtiy2xnm,132SM,capability9.0,
  device UUID f8e84ee7-4a58-3e7e-8d36-8a4ba4394c62. Reported startup physical
  identity remains unconfirmed; no cross-run machine parity is asserted.
  Driver595.91.07,Torch2.14.1+cu130,NVCC13.0.88; actual driver/runtime versions
  13020/13000. Full evaluator image/code/file hashes are in the owning receipt.

The exact raw CLI JSON stdout is preserved without parse/reserialize, including
64-bit seeds. The owning log is the original CLI output path; the strict reader
runs there with --mode check and validates path/job/source/terminal identity.
Portable copies of its log and all normalized profile JSON/text are retained
under muse_decode_layer/profiles/g-common-rmw-v2.c1931f0c9e834485.*.
Source association is the preregistered local upload/source proof, not an assertion
that the server content identity equals file SHA256.

## Startup, resources and preparation phases

Strict reader exits0 with complete_native,treatment_exercised=true,paired=false,
release_decision=false,causal_RMW_effect=null. Native startup covers12 cells,
72 full-model checks,12 captures,36 individually checked fresh replays,60 empty
checks,12 empty captures and48 occupied restores. The final deadline and official
runner also pass:13 correctness tests,12 benchmark records,12 profile records.
Both complete coverage and actual native selection are retained; compiler success
or expert-only replay success is not being substituted for those gates.

Five actual expert resource records each contain both bool entries. Both entries
have the same vector for each key in this run:

| Expert key | Registers | Local bytes | Static shared B | Dynamic shared B | Threads | Max threads | Resident CTA/SM | Grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 145 | 0 | 0 | 214784 | 288 | 288 | 1 | 132 |
| tiny2 | 145 | 0 | 0 | 214784 | 288 | 288 | 1 | 132 |
| mid | 152 | 0 | 0 | 214784 | 288 | 288 | 1 | 132 |
| b4 | 96 | 8 | 0 | 231168 | 576 | 576 | 1 | 132 |
| main | 96 | 0 | 0 | 231168 | 576 | 576 | 1 | 132 |

Source-asserted compile policy enables the returning relaxed final RMW only for
tiny/tiny2; all12 actual attention dispatch/resource records and expert keys
passed the reader's selected-path/admission contract. This is not an independent
printed cubin/flag-vector audit or an SM/HBM utilization measurement. Ordering
and progress obligations retain the reviewed source proof; passing replay tests
alone are not a new memory-ordering proof. No barriers/fences were removed.

| Host phase group | Seconds |
| --- | ---: |
| Normal production prepare | 106.901449 |
| All remaining private/host phases through cleanup end | 2.368223 |
| BF16 + FP8 front-reference phases, included above | 1.539021 |
| Back-reference miss phase, included above | 0.301577 |
| Cleanup begin to cleanup end, included above | 0.000111838 |
| Summed phase elapsed through cleanup end | 109.269672 |
| Official setup record | 109.269811 |

Reference memo:90 hits/54 misses,peak4 entries. Phase count385=331 fixed+54
miss events, with full count coverage and finite timing sum accepted by the reader.
These are host wall durations around existing execution/waits, not isolated GPU
phase times. The unchanged validation and295s guard now finish with ample margin
in this run. Old3fd failed at the guard on another source/machine; the pair is not
a matched setup benchmark, so do not attribute the whole difference to memo reuse
or inference scope. No further generic preparation optimization is indicated.

## All six weighted loads and aggregate log changes

For each load C=(46*sliding+16*full)/62,using official adjusted medians in us.
Positive latency change means slower.607 is an archived unmatched observation,
not a contemporaneous common-OFF control.

| Load | New sliding us | New full us | New C us | Historical607 C us | Latency change |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode 1 | 75.560 | 83.929 | 77.720 | 78.549 | -1.055% |
| decode 16 | 137.613 | 216.569 | 157.989 | 158.241 | -0.159% |
| decode 32 | 156.440 | 315.228 | 197.418 | 191.740 | +2.961% |
| verify 1 | 109.166 | 123.306 | 112.815 | 115.991 | -2.738% |
| verify 16 | 250.416 | 339.250 | 273.341 | 269.347 | +1.483% |
| verify 32 | 314.084 | 488.369 | 359.061 | 359.297 | -0.066% |

The latency-only mean log ratio mean(log(C607/Cnew)) is
**-0.0005463675050261879**,geometric factor0.9994537817265194
(**-0.0546218%**). The official score log ratio is
**-0.0022798493885410652**,factor0.9977227474942008(**-0.2277253%**).
The distinction reflects changed per-load official baseline timings as well as
candidate timings; it is not extra candidate loss. Full12 raw/adjusted candidate
and baseline medians, per-cell seeds, observer overhead and six weighted raw/
adjusted costs are in the strict observation JSON.

Short loads are mixed across contexts: decode1 sliding-1.548%/full+0.241%;
verify1 sliding-5.428%/full+4.854%. Decode32 full adjusted+5.823% is raw+1.383%;
its archived observer deduction changed from14.437us to1.412us. These facts make
an unpaired component attribution especially unreliable. No measured result
establishes the registered every-heldout/all-four-crosses >1% and twice-null
gain threshold or per-context regression guard. Do not promote this file based
on a selected short load or interpret these regressions as a causal family loss.

## Hotspot evidence and next decision

The existing tools/hotspots.py was run on this exact profile with the archived
SGLang baseline-as-submission7d42771c3384449d. The resulting hotspot map is
archived here; it is not a new best, so docs/hotspots.md is not replaced.
In the submission profiles, short expert kernels report22.8/22.3us for decode1
sliding/full and50.0/65.1us for verify1. These four measurements identify current
expert activity for the captain's next dependency screen. They are profiler
activity totals, not exposed RMW time or additive achievable latency savings.
The same screen must account for required compute, bytes and producer/consumer
dependencies and the full six-load score contribution before another local tuning
round. No gamma/PDL/sparse or old97935e7d variant is reopened.

The remaining three private checks in the initial cap are a ceiling, not an
instruction to spend them. A new attribution registration must hold unrelated
source/dispatch choices fixed and use independent fresh OFF/ON preparations,
duplicate/null states, matched inputs, every heldout seed, all four crosses,
all six loads, actual resources and regression gates. A plain OFF single-arm
score would be descriptive too. This receipt resolves setup/GPU qualification;
the next decision is whether the remaining RMW cost budget warrants that matched
experiment versus a materially different conforming design. Captain owns that
research priority; this worker retains sole implementation/dispatch ownership.

## Evidence

- [Exact terminal receipt](evidence/2026-11-01-qualifier-inference-repair/c1931f0c9e834485.terminal.json)
- [Owning startup log](evidence/2026-11-01-qualifier-inference-repair/c1931f0c9e834485.log)
- [Strict reader result and full12 comparison](evidence/2026-11-01-qualifier-inference-repair/c1931f0c9e834485.observation.json)
- [Derived six-load/phase summary](evidence/2026-11-01-qualifier-inference-repair/c1931f0c9e834485.measured-summary.json)
- [Hotspot map with archived SGLang source](evidence/2026-11-01-qualifier-inference-repair/c1931f0c9e834485.hotspots.md)
- [Implementation, complete CPU tests and registration](2026-11-01-qualifier-inference-repair.md)
- [Raw normalized profile JSON](../../../muse_decode_layer/profiles/g-common-rmw-v2.c1931f0c9e834485.json)

The evidence directory also preserves original CLI transport, terminal quota/jobs,
source manifests, accepted source/job association and artifact byte hashes.

