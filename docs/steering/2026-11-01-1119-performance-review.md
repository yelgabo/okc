# Architecture performance review, November 1 at 11:19 Vancouver time

Snapshot `fa8173a` covers 53 remote commits since `af3b550`. The private VP2 comparator
saved 0.616 µs at eight-token sliding attention. Its six-load factors fell below the
registered 1% threshold. The owned global sparse-XOR projection ran slower on all ten
target cells than archived C2. The stage-local version has a dispatch receipt and awaits
its own H200 result. Family G retains the 2.1261× layer entry.

**E. The private VP2 comparator saved 0.616 µs (5.71%) at sliding verify batch 1, eight tokens. It missed the six-load acceptance gate.** Job `c79ddddb840a4fb7` used source `e86b0d5`, frozen at `23fb760`, with the external VP2 bundle from `8e30e1b`. Four independently prepared arms shared the same BF16 output boundary. Three held-out inputs per cell and 32 rotated cold rounds produced 36 timing rows. All 640 private math and cache checks passed. The private judged path passed 16 official tests and scored 2.594320× (95% CI 2.555405–2.633234). The interval applies to the official standalone score. [Official receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.receipt.json)

| Held-out seed | Owned mode 8/15 µs | VP2 µs | Delta µs | Delta % |
|---|---:|---:|---:|---:|
| 824731 | 10.856 | 10.136 | -0.720 | -6.63% |
| 824737 | 10.744 | 10.144 | -0.600 | -5.58% |
| 824743 | 10.752 | 10.224 | -0.528 | -4.91% |

The four contemporaneously calibrated six-load comparisons ranged from 1.006309× to 1.009060×. The duplicate log-null was 0.002371. Every seed failed the combined gate, which required all four comparisons to clear both 1% and twice the duplicate noise. VP2 combines 128-key tiles, dedicated ownership and an asynchronous mailbox. The evidence isolates the bundle only. The earlier owned 64-key complete-head cluster loss stands. This round closed the automatic rewrite and promotion route. A new owned design would need independent evidence. [Terminal decision](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-attention-vp2-bf16-terminal.md)

At 128 and 256 tokens all four arms ran owned code. The following raw averages across the three seeds and two placements per arm record coverage. Differences measure variation among those unchanged-code placements.

| Tokens | Context | Control µs | Comparator arm µs | Difference µs (%) |
|---|---|---:|---:|---:|
| 128 | sliding | 15.771 | 15.717 | -0.053 (-0.34%) |
| 128 | full | 109.637 | 109.413 | -0.224 (-0.20%) |
| 256 | sliding | 19.869 | 19.888 | +0.019 (+0.09%) |
| 256 | full | 190.675 | 190.971 | +0.296 (+0.16%) |

[Paired timing rows](https://github.com/yelgabo/okc/blob/fa8173a/muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.analysis.json)

**MoE.** This interval adds a CPU audit of low-token weight representations. It parks full BF16 expansion, scale expansion and generic activation caching pending evidence that decoding or activation handling limits the skinny kernel. The audit provides no H200 candidate timing, so A through D and the 128/256 A-vs-B decision keep their prior measured status. [MoE audit](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-owned-lowtoken-representation.md)

### Projection: global sparse XOR lost on all ten target cells. Stage-local XOR awaits its result.

Job `2b537518091248f1` tested the owned global-record E7-XOR repair before activation loading (source `c138969`, registration `1bd55da`, SHA256 `685ffab5d1614343d3c46d505967ae6817e3784736297018c78984354333f765`). This H200 public run passed nine official tests with 30 timed cells and 30 profiles. All ten overrides executed with zero fallbacks. Startup passed 160 math checks, ten captures and 30 fresh-input checks in 123.33 seconds. All 40 resource vectors passed the local-memory, layout and CTA-admission gates. The candidate scored 1.394120071× (95% CI 1.373208270–1.415031872) against archived C2 job `9eba96894ee1402e` at 1.527806483× (1.504889386–1.550723580), an 8.75% lower point score. Each receipt contains one scored submission. Different inputs, machines, driver versions and calibration preclude a paired estimate or a mechanism-specific confidence interval. [Owning receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_bf16_proj/profiles/sk-xor-release.2b537518091248f1.receipt.json), [terminal analysis](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-sk-xor-release-terminal.md).

| Projection / tokens | Adjusted change vs C2 | Raw event change vs C2 |
|---|---:|---:|
| qkv_gate / 8 | +3.562 µs (+36.46%) | +4.176 µs (+30.53%) |
| unpooler / 8 | +0.234 µs (+5.16%) | +0.624 µs (+7.37%) |
| qkv_gate / 16 | +4.246 µs (+43.59%) | +4.128 µs (+29.66%) |
| o / 16 | +1.544 µs (+28.77%) | +1.536 µs (+16.16%) |
| pooler / 16 | +3.876 µs (+28.08%) | +3.648 µs (+20.00%) |
| unpooler / 16 | +1.893 µs (+13.96%) | +1.456 µs (+8.15%) |
| qkv_gate / 32 | +5.533 µs (+52.41%) | +5.184 µs (+35.14%) |
| o / 32 | +1.457 µs (+24.83%) | +1.488 µs (+15.00%) |
| pooler / 32 | +4.853 µs (+32.17%) | +5.008 µs (+26.37%) |
| unpooler / 32 | +1.211 µs (+8.23%) | +1.488 µs (+8.08%) |


At 128 and 256 tokens all five operators used existing C2 paths. The table gives across-receipt changes in the sum of five medians as coverage context. Attribution is limited to the ten target rows above. [All 30 cell medians, timing offsets and seeds](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/evidence/2026-11-01-sk-xor-release-terminal/rows30.csv).

| Tokens | Adjusted C2 → candidate | Adjusted change | Raw C2 → candidate | Raw change |
|---:|---:|---:|---:|---:|
| 128 | 67.096 → 68.591 µs | +1.496 µs (+2.23%) | 87.168 → 89.568 µs | +2.400 µs (+2.75%) |
| 256 | 82.416 → 84.070 µs | +1.654 µs (+2.01%) | 103.408 → 103.920 µs | +0.512 µs (+0.50%) |

The recorded decision closes the global-record schedule and its layer port. The review corrects an earlier dependency claim. Global record loading can delay stage-credit publication and back up the producer, even with valid lifetime ordering. The public timing leaves the cause of the slowdown unresolved. C2 remains the projection entry. [Corrected explanation and decision](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-sk-xor-release-terminal.md).

Public job `dfcfdd01d6f04525` tests a distinct stage-local representation (source and preregistration `a08f501`, 450,781 bytes, SHA256 `de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229`). It derives capacity from supplied weights. It retains the same ten targets and 100 nearby keys with C2 as comparison, subject to unchanged C2 resource admission, 160 math checks, ten captures and 30 fresh replays before enabling overrides. Local CPU and source checks passed. H200 correctness, coverage, timings and scored outcome are pending at `fa8173a`. The 128/256 cells use the original C2 paths. [Dispatch receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_bf16_proj/profiles/stage-xor-release.2026-11-01/dispatch.json), [preregistration](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-projection-stage-xor-release.md).

Two offline projection audits recorded no new GPU timing. The local compilation found terminal waits needed for partials and ring reuse on the selected C2 WGMMA paths. The mailbox audit found that the selected SK/RD reducers are CTA-local and RT already uses async completion. Both score sensitivity scenarios are hypothetical. No layer or A–E performance attribution follows. [Lifetime audit](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-projection-wgmma-lifetime-audit.md), [mailbox audit](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-projection-mailbox-scope.md).

### Family G precision and layer packaging

The latest retained layer receipt is still `607eb52bdaf24160`, Family G stage retirement, 2.126082676× (95% CI 2.094191436–2.157973917). The board snapshot at 17:27 UTC on November 1 records layer rank 316. This review interval adds no layer H200 timing, paired result or submission. [Board snapshot](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/evidence/2026-11-01-projection-sk-xor-root/live-muse_decode_layer.json), [retained receipt](https://github.com/yelgabo/okc/blob/fa8173a/muse_decode_layer/profiles/scored.607eb52bdaf24160.json).

The source audit at `bf75af6` narrows the earlier precision claim. The rules preserve BF16 activation boundaries where SGLang uses BF16, including the expert GEMM outputs before the next operator. The pinned JIT short router receives the unrounded FP32 normalized vector while the pooler receives BF16. The audit found no categorical rule requiring every internal split partial to be FP32. Keeping rs128 BF16 partial reductions, YBF and opaque Lt split-K paths disabled is a conservative implementation choice. MFIN can leave its poll before readiness, an independent reason to keep that path disabled. The audit measured no per-repair speedup and did not qualify a new release. [Precision contract and source identities](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-g-fork-precision-contract.md).

At 128 and 256 tokens, the private verify comparison now covers sliding and full contexts in one 968,389-byte package. A corrected owned-128 selector restores the recorded projection and boundary choices. Seven CPU tests and independent source review support that correction. H200 linking, capture, matching natural expert IDs, 96 reference checks and the 295-second local preparation guard remain open. These four cells cover two of six scored loads. The six-load G control-versus-stage comparison and short-path trace retain priority. [Private comparison review and runtime gates](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-verify-four-cell-slice-review.md).

The corresponding **production hybrid fails the size gate** for every nonzero override mask in the tested representation. The default is byte-identical to the 869,076-byte incumbent. One verify32 cell requires 1,072,104 bytes, both verify16 cells require 1,115,832, and all four require 1,125,548, against a strict limit below 1,000,000. The 16-mask matrix reproduced and 19 host tests passed. Local device-only builds preserve the selected WD, boundary and router instructions. This closes the current packaging approach without a runtime or latency result. [Production matrix and stop rule](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-verify-hybrid-packaging.md).

The separate sparse-XOR applicability map identifies 24 potential projection cells on four short loads. G uses SK2 and distinct packed weights. Direct transfer would require a separately qualified consumer and repack. The 128/256 Lt paths are outside this map. Its 830,130-byte figure is a source inventory with allowances, not a generated layer payload. No layer gain follows from the standalone projection experiments. [Layer applicability map](https://github.com/yelgabo/okc/blob/fa8173a/docs/steering/2026-11-01-g-sk-xor-applicability.md).

The next evidence checkpoint is the stage-local projection receipt. For layer decisions, retain the registered matched G comparison and fixed-routing short-path trace ahead of the four-cell diagnostic.
