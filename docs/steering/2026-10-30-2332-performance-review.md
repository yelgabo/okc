# Performance review, 2026-10-30 23:32 PDT

Evidence through `8f2c89b`, reviewed 2026-10-30 23:32 PDT. PBULK now passes GPU checks;
variant 1 has a small gain below its promotion bar, while variant 2 loses. Mode 86 fails
its performance rule. M-C C-1 loses its local whole-step test before the bench faults.
The recorded layer entry is now 2.018705x, within prior uncertainty, with a changed
route-side validation rule and report-only sensitivity findings. No isolated family gain
or new A-versus-B decision follows from that scored result.

### Layer: one-token fusion loses its local test; gate update passes and is scored

M-C C-1 is now tested. `layer_mc1s`, SHA256 `2998259e7109cfaadfd21c33db79266ba73297e0dbbaf577088aff02f7d1c38f`, built at `b5ebf08`, fuses the one-token o projection and attention boundary into one cooperative kernel with two grid barriers. Job `5ae9ce318e014edf` compares it with `layer_m1rt51sc1` (`9373b976`), using independent prepares and bench-only fixed M1 policy with 250 gate replays. Its completed candidate prepare passes the cooperative probe, self-test and C-1 fresh-input gate before timing. The stable pick records a **+1.16 us loss against a 66.0 us control, +1.76%**, for the weighted one-token step (46 sliding + 16 full over 62), with C-1 faster in **0 of 9 rounds**. No paired CI is recorded. This is one prepare's local whole-step comparison, not a completed four-arm result. The predeclared stop rule closes this implementation; C-2 remains unbuilt. M-C C-1 is a layer fusion name, not architecture family C. [Raw log](../../muse_decode_layer/profiles/bench16_mc1.5ae9ce318e014edf.log), [source and method](../runs-decode-layer.md), [M-C plan](../plan-2026-10-31-m-c-t1-fusion.md).

The bench later exits with an illegal memory access and has no judged score, completed cross ratios or 128/256-token results. The faulting kernel is absent from the tail. This resembles the unresolved two-layer-process fault in `36f93b13d4c14d5b`; shared module or device state is a hypothesis, not an established cause. The new follow-up treats bench-process faults separately from candidate correctness until attribution is found, and stops on a fault in a single-file check. It does not erase either fault or the local C-1 timing loss. Separately, the reduced 250-replay M1 gate failed at 1x8 sliding and switched M1 off in candidate/inc2 prepares. The bench patch now requires at least 500 replays. This validation-sampling failure is distinct from the memory fault and C-1's own gate. [Run record](../runs-decode-layer.md), [open fault follow-up and scoped rejection](../ruled-out.md).

The prior projection-C1 layer comparison `b70cfa48c0264d20` remains inconclusive at 1.0007, but its fallback explanation changes. The old route-side 1e-4 self-test rejected cand2 at 1.21e-4. That fallback **does affect the timed one-token FP32 front**: cand2 d1 sliding was 84.8 us against cand's 83.1 us, a cross-arm +1.7 us (+2.05%) observation, not an isolated gate-change gain. The skf-r path exemption at other sizes did not exempt d1. [Follow-up and gate decision](../runs-decode-layer.md).

The run record documents a user decision to replace that route-side fixed threshold with the existing fresh-input control-envelope method, retaining the order-1 threshold of relative L2 >= 1e-2. This is an explicitly changed validation rule, not proof that the old failure was harmless. Attention and norm checks retain their own bounds. `layer_m1rt51sc1g` at `2840157`, SHA256 `a00a0efa0a761d990d7298a64340751e6a3979369aa8247a1f3d958f975c7480`, changes only that self-test and one summary line from the projection-C1 layer port. Single-file check `2edb120747cb4611` passes **13/13**, no fault, **2.0045898445x**, 95% CI **1.9745209968-2.0346586921**. All six route-side gates pass, including exact zero candidate/control differences at 128/256, route gates total 14/14 and the manifest has no failures. At one token, candidate/control relative error is 6e-5/6e-5, element error 1e-3/1e-3, and 0/200 calls exceed 1e-4 on either side. Its route-side negative controls reject injected errors. [Check receipt](../../muse_decode_layer/profiles/layer_m1rt51sc1g.2edb120747cb4611.json), [raw check log](../../muse_decode_layer/profiles/layer_m1rt51sc1g.2edb120747cb4611.log), [recorded decision](../runs-decode-layer.md#validation-policy-change-fresh-input-replay-gate-user-decision-2026-10-30).

The coordinator then scored the same candidate twice; the second submission adds only a repeat comment. These are cross-run observations, not a paired comparison with the old release:

| Job | Score | 95% CI | Correctness and limitations |
|---|---:|---|---|
| `dc25fef4d65642a9` | 2.0063008666x | 1.9762063536-2.0363953796 | 13/13 judge tests; route gates 14/14. Report-only element-injection sensitivity misses in MoE main-lean-silu-bar and main-lean-grp-finss64 at 256. |
| `7b84ed4346d5497b` | 2.0187050168x | 1.9884244415-2.0489855920 | 13/13 judge tests; route gates 14/14. Report-only element-injection sensitivity misses in route side at 16 and 256. |

Both raw manifests record two sensitivity findings, with no reference, replay, nonfinite or workspace failures. Thus the clean check's negative-control outcome must not be generalized to the scored repeats. Both repeats select M1 o+front at 32, whereas the single-file check selects o. [First scored receipt](../../muse_decode_layer/profiles/scored.dc25fef4d65642a9.json), [second scored receipt](../../muse_decode_layer/profiles/scored.7b84ed4346d5497b.json).

The recorded board entry moves from `0f58d5d8223f45af` (2.0148147x) to `7b84ed4346d5497b` (2.0187050x), within the recorded uncertainty. This establishes no isolated A, projection C1 or gate-update gain. At 128/256, both scored runs cover sliding/full workloads with 20 timings per shape, but there is **no new same-process candidate-versus-prior-layer comparison**. Existing A-versus-B evidence at those sizes is unchanged. [Scored-run record](../runs-decode-layer.md#scored-layer_m1rt51sc1g-two-runs-2026-10-31).

The one-token bottleneck explanation also needs restraint. The donor's earlier front comparison used the serial release, not the current overlapping incumbent. Its 28.3 us summed front versus donor 23.4 us does not establish today's target. The current trace overlaps router/routing with pooler: its main-stream pooler plus pre-expert norm is 18.7 us, but that trace and donor trace use different inputs. The plan now calls for a same-input incumbent-versus-donor d1/v1 comparison with kernel names before choosing another kernel target. No such new comparison is recorded at this snapshot. Persistent-o memory parallelism and barrier overhead remain explanations to test, not measured causes of C-1's loss. [Reframe and proposed measurement](../plan-2026-10-31-m-c-t1-fusion.md#reframe-the-one-token-front-per-kernel-2026-10-31-after-three-weak-structural-results).

### A: earlier x-slot release passes correctness but misses its performance gate

Job `a00b49f8372042a9`, source `30d6b96`, tests mode 86 against 51 in the same process on seeds 181-186, layer, standalone and skewed routing. Cold-L2 timing uses 20 replays per arm and six seed means. This is an A ring-release variant, not a new owner-completes C or ready-job D design. Negative latency deltas mean faster. Percent is paired mean delta divided by the displayed control mean; rounding limits precision. No paired confidence interval is recorded.

| Tokens | Routing | Mode 51 us | 86 minus 51 us | Delta % | Seeds faster / 6 |
|---|---|---:|---:|---:|---:|
| 32 | layer | 87.0 | +0.2 | +0.23% | 0 |
| 32 | std | 132.3 | -0.8 | -0.60% | 3 |
| 32 | skewed | 69.1 | -0.1 | -0.14% | 3 |
| 128 | layer | 162.1 | -0.3 | -0.19% | 3 |
| 128 | std | 194.0 | +0.0 | +0.00% | 4 |
| 128 | skewed | 102.2 | -0.1 | -0.10% | 2 |
| 256 | layer | 216.4 | -0.6 | -0.28% | 5 |
| 256 | std | 244.3 | -0.1 | -0.04% | 3 |
| 256 | skewed | 160.8 | -1.4 | -0.87% | 6 |

Outputs equal 51 at 128 and 256 tokens on every seed. Maximum 32-token difference is 3.4e-5, below the 1e-4 bound. FP32 self-test reports 3.88e-3 for both modes and spin flags are zero. The receipt passes at 2.19501x, which is separate from the candidate's adoption gate. Mode 86 fails the required 256-token saving of 1.5 us on both layer and standalone, and the 128-token saving of 1.0 us on both. It is not kept. The 256-token skewed saving is measured, but does not satisfy those gates. [Raw log](../../muse_moe_w4a16/profiles/bench_m86.a00b49f8372042a9.log), [receipt](../../muse_moe_w4a16/profiles/bench_m86.a00b49f8372042a9.json), [rule and result](../runs-moe.md#earlier-x-slot-release-mode-86-2026-10-31).

The new diagnostic text reinterprets existing mode-56 job `7063f86352344efa`, rather than adding a GPU run. Its stamped refill delay is 0.18-0.31 us per stage, weights-copy latency 2.7-3.2 us, x-copy latency 2.2-2.4 us, and release-to-need lead 3.4-4.1 us. Instrumentation itself lengthens this lead. A donor counter build spilled and was abandoned before a check. Readiness waits on `done_as` and `xready`, and x/weights HBM competition, remain unseparated hypotheses. Per-job x-slot sizing is unbuilt. The next step is to separate the x producer's readiness and empty-slot waits before building it. This adds no A-vs-B or whole-layer win. [Diagnostic analysis](../runs-moe.md#where-the-consumers-wait-comes-from-refill-delay-or-copy-latency-2026-10-31).

### E: PBULK now passes GPU checks, with a small gain for variant 1 and a loss for variant 2

The fixed producer exit has now run successfully. Retain the earlier hangs in history, but replace the current pending-correctness label. These are mode-49 producer-side page-index pipeline variants, not whole-layer results. Bench p and r use three held-out inputs and 200 rounds; bench s uses two inputs and 120 rounds. All three receipts pass 16/16 judged tests on the board path. Real arms report ABCHECKSUM 11 passes and zero failures in p/r, 19 passes and zero failures in s; diagnostic probes are excluded from those correctness claims. No paired confidence interval is recorded. [p log](../../muse_attention_decode/profiles/bench_p.bb06f8e57bd84636.log), [r log](../../muse_attention_decode/profiles/bench_r.f3f1e4a5249a4228.log), [s log](../../muse_attention_decode/profiles/bench_s.1c861d3944f149a0.log).

The following cold pooled deltas subtract displayed pooled arm times, so they can differ from the log's separately rounded delta by 0.01 us. d16/d32 are q_len 1 with batch 16/32, not 128/256 tokens.

| Job @ source | Candidate | Workload | Control us | Candidate us | Delta us | Delta % |
|---|---|---|---:|---:|---:|---:|
| `bb06f8e57bd84636 @ 0671cc7` | PBULK=1 | d16 full | 100.45 | 100.38 | -0.07 | -0.07% |
| `bb06f8e57bd84636 @ 0671cc7` | PBULK=1 | d32 full | 179.42 | 178.91 | -0.51 | -0.28% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=1 | d16 full | 100.42 | 100.26 | -0.16 | -0.16% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=1 | d32 full | 179.58 | 178.98 | -0.60 | -0.33% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=2 | d16 full | 100.42 | 101.79 | +1.37 | +1.36% |
| `f3f1e4a5249a4228 @ 8bdf607` | PBULK=2 | d32 full | 179.58 | 182.50 | +2.92 | +1.63% |
| `1c861d3944f149a0 @ a191c34` | PBULK=1 | d16 full | 98.05 | 97.95 | -0.10 | -0.10% |
| `1c861d3944f149a0 @ a191c34` | PBULK=1 | d32 full | 173.70 | 172.90 | -0.80 | -0.46% |

In p, matched-null deltas are +0.03/+0.06 us and independent-null +0.06/-0.10 us at d16/d32. Variant 1's own-state twin saves 0.13/0.51 us. Its gated score gain is +0.063% against each null, below 0.10%. d32 wins all three inputs; d16 has mixed signs. In r, matched null is +0.03/+0.03 and independent null +0.00/-0.13 us. Variant 1 repeats a +0.074% score gain, again below the bar, with d32 faster on all three inputs and d16 mixed or tied. This is a measured small component gain, off, with no layer integration test. [Gate evaluations](../runs-attention.md#attention-round-5-10-31-ledger).

Variant 2 replaces the producer barrier with per-window free mbarriers. It loses on all three inputs. Its own-state twin is +1.34/+2.98 us, and variant 2 plus HMRGF is +0.22/+0.70 us in the printed raw deltas. Variant 2 and its HMRGF combination fail the score gate, with reported G -0.45%/-0.09%; variant 2 also breaches the d32 1% regression cap. Producer-only timings are 37.57/63.49 us for control and 53.73/95.39 us for variant 2. Variant 1's earlier producer-only probe is also slower, 47.33/82.75 against 37.60/63.42. These probes omit normal work and are diagnostic, not candidate speedups. The interpretation that lockstep or slow-reader waits limit refill remains mechanism analysis. Reopen variant 1 as a rider for a promoted mode-49 change, or with a producer retaining its gain without per-tile lockstep; variant 2 needs a refill that avoids waiting on the slowest previous reader. [Results and scope](../runs-attention.md), [ruled-out entries](../ruled-out.md).

The s run covers all 12 shapes. PBULK=1's full-run ABSCORE 0.9974 lies within independent-null 0.9968 variation; changes on untouched shapes prevent attributing that overall score to PBULK. Its q_len-8 rows provide explicit 128/256-token coverage but are unchanged by mode-49 PBULK: v16 sliding/full controls 15.68/112.13 us and pb1 15.65/112.99; v32 sliding/full controls 19.90/190.37 and pb1 19.84/190.21. These are context observations, not PBULK wins/losses at 128/256 tokens.

The same s run is an attribution diagnostic, with no adoption gate. Consecutive phase-exit differences put our q_len-8 tile loops at 6.21/4.35/8.13 us against public 5.79/3.93/7.26 for v1 full, v16 sliding and v32 sliding, gaps +0.42/+0.42/+0.87 us. The d1 full prologue is 10.27 versus 9.73, +0.54 us. Descriptor/V-transpose address carries in the PRQ loop are untested, and matching sub-phase exits for the d1 prologue are pending. Do not sum these gaps into a realized gain. No candidate beat the board entry, and no PBULK layer test was recorded. [Raw phase exits](../../muse_attention_decode/profiles/bench_s.1c861d3944f149a0.log), [round-5 next steps](../runs-attention.md#attention-round-5-10-31-ledger).

Receipts for these three runs are 2.49640x with CI 2.45895-2.53384, 2.51884x with CI 2.48105-2.56956, and 2.58437x with CI 2.54561-2.63520. These cross-run judged board-path scores are not PBULK deltas. [p receipt](../../muse_attention_decode/profiles/bench_p.bb06f8e57bd84636.json), [r receipt](../../muse_attention_decode/profiles/bench_r.f3f1e4a5249a4228.json), [s receipt](../../muse_attention_decode/profiles/bench_s.1c861d3944f149a0.json).

### Projection preparation and protocol review

Projection context, reviewed through `8f2c89b`: no new projection measurement or profile was committed after `9bf7f05`. The two untracked projection receipts (`db5be7d500704692`, `64ed6e6b5e204dc0`) are historical October 29 scored runs already recorded in [runs.md](../runs.md), not new evidence. The C1/C2 paired results and C1 board score therefore remain unchanged. The GTICKET ordering review is now recorded as sound, with fixes in `e6510e4`: 64-bit tickets and published generations prevent the 32-bit wrap misalignment; timeout adds nothing and sets diagnostic bit 128; a generation/owner-pointer mismatch adds nothing and sets bit 256; `CLK <= 16` bounds the state arrays; `_self_test` rejects any nonzero diagnostic. The [port guide](../proj-port-guide.md#in-kernel-tile-clear-for-the-split-router-wv_gticket-review-pending-2026-10-31) records the independent review and emitted zero-store/release/acquire instructions. Its heading still says “review pending”, but the later review paragraph supersedes that status. This is a completed protocol review and code correction, not a new correctness or performance run of the corrected implementation.

Pending benchmarks: `62bf3e9` prepares rk-c to test tensor-map prefetch `F` (`WV_PFTM`), uniform descriptors `V` (`WV_UDESC`), their combination and duplicate controls on production routers at 128/256, plus memset-zeroed `R+M+O` with/without `F+V`. `d0a274b` adds rk-d to [the router builder](../../tools/proj_make_bench_rg.py): corrected GTICKET `H` as a separate arm beside `R+M+O`, with/without `F+V`, at 128 (`r64c16s8`) and 256 (`r64c8s8`), including duplicate controls. The [round 9 preparation record](../runs.md#2026-10-31-utc-round-9-prep-no-quota-branch-proj-r9) also prepares [codec factorial probes](../../tools/proj_make_bench_cx.py) for cgemv at one token (qkv_gate/o/pooler/slice) and rd at pooler/o eight tokens. These extend timing models, not implemented codecs. Production SASS was recorded as 92/92 instruction-identical to C2 after the preparation changes; rk-d was recorded as compiled without spills. No benchmark job ID, new timing, repeat count or uncertainty exists for these pending variants in this snapshot. The old 256-token diagnostic win and 128-token losses remain scoped to the earlier implementation and jobs.

## Whole-layer outcome and attribution

Recorded scored layer is `7b84ed4346d5497b`, 2.0187050168x, CI 1.9884244415-2.0489855920,
from M1 + mode 51 + projection C1 with the revised route-side gate. Both scored repeats
pass 13/13 tests and retain report-only negative-control sensitivity misses. The score
change from 2.0148147x remains within uncertainty. No new paired comparison establishes
a gain from A, projection C1 or the gate change. The prior C1 layer result is inconclusive;
its fallback affected the timed one-token path. M-C C-1 loses locally and its later bench
fault remains unresolved. [Layer run record](../runs-decode-layer.md).

## Decision and next evidence needed

Keep the A-versus-B conclusion at 128/256: tested B remains slower and needs its phase-B
or phase-A-tail trigger before retesting. C ownership remains favored over modes 72/74/79.
D's complete ready-only queue is untested. Mode 86 passed correctness but missed its timing
rule. Separate the x producer's readiness waits from copy latency before a deeper ring.

PBULK's hang fix now has GPU validation. PBULK1 stays off below the promotion threshold;
PBULK2 loses. Preserve PRQ and measure the pick-free bundle in the layer. Per-request tile
loop and d1 prologue gaps are diagnostic targets, not forecast additive savings.

M-C C-1's local loss closes that implementation; C-2 stays unbuilt. After weak PDL and fusion
experiments, the next decisive one-token comparison uses identical inputs and kernel names
for the current overlapping incumbent and donor. Earlier donor/release traces do not locate
today's bottleneck. Find the bench-process fault separately before using failed benches
as candidate correctness evidence. Keep changed validation rules and sensitivity misses visible.

F stays measurement-only. Owned projection C1/C2 component wins are unchanged. GTICKET's
review and fixes are recorded, but the corrected implementation needs its own benchmark.
Prepared rk-c/rk-d and codec probes are pending, not gains. Never add family or component scores.
[MoE evidence](../runs-moe.md), [attention evidence](../runs-attention.md),
[projection evidence](../runs.md), [M-C reframe](../plan-2026-10-31-m-c-t1-fusion.md).

