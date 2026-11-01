# Exact pasted report: attention and layer boundary mapping

November 1, 2026. Read-only audit; no GPU jobs or kernel changes. Evidence pin:
main `6da000f35cbc3bc666659cf6e0ff0877b9f60eda`. Exact report is
`docs/steering/evidence/2026-11-01-pasted-top10/reaching-top10.txt`, normalized
SHA256 `f47bc1242e1a18e58afe82dc8d72bde6247f269e8771fa9781aedab029fffff0`.
This is distinct from the earlier community audit. Root integrates this report
and the steering index; line anchors below refer to the evidence pin.

## Seven attention recommendations versus owned code

| Exact-paste recommendation (lines71–78) | Existing implementation and evidence anchors |
|---|---|
| Separate q_len1 and q_len8 kernels/configs | `muse_attention_decode/submission.py:_one_plan`6431–6475 selects L-specific builds; `attn_one.cu`2031ff uses L1 f16 mma.sync and2270ff L8 E4M3 wgmma. Already implemented. |
| Separate sliding and full paths | `_one_plan`6437–6466 selects full batch>1 flat producer modes49/51, versus sliding per-request/one-tile geometry and distinct cooperative choices; batch1 SPEC path is separate. Already implemented, although templates share source. |
| Reuse each KV tile across eight query heads per KV head | L1 `attn_one.cu`2031–2044 maps query head hk*8+g; L8 2270–2278 assigns one warpgroup per KV head and rows8*j+head. Both KV heads reside in each CTA. Already implemented. |
| Fuse Q/K normalization and RoPE into preparation | `attn_one.cu:q_piece`1838–1887 normalizes and rotates queries into shared fragments; own-KV preparation1900–1934 normalizes K and applies RoPE conditionally. No global normalized-Q/K intermediate on this path. Already implemented. |
| Write KV directly in FP8 cache format | `attn_one.cu`1921–1930 converts to E4M3 and writes the owned new-token cache location, also staging new rows for attention. Already implemented. |
| Reuse KV across eight verify positions | L8 rows are eight positions times eight heads per KV head (2270ff); qd0/qd1 and QK/PV wgmma operate on that shared64-row query tile. Already implemented; the report does not identify an additional missing reuse axis. |
| Warp-specialized producer/consumer where resources permit | `attn_one.cu`1241ff producer warpgroup and2270ff consumers; host resident-limit selection in `_one_plan`. Actual mode51 diagnostic8ac6563343f74bcd:107104 dynamic shared bytes/CTA, two resident CTAs/SM across all six arms. Already implemented, not proof of perfect overlap. |

Output RMSNorm and sigmoid gating are also fused into `finalize_rows`
(`attn_one.cu`755–866); `sigmoid`622–624. This closes the attention-internal
postprocessing boundary. Standalone production now has ONE_LATE_TID2, a
restricted full L1 scheduling change; its paired causal gain was approximately
0.24%, not the independent-score movement. Source and math gates:
a9b416a0c7324ecc,80415fe736bb4890; clean pure checkd331aca81c554d70.
Failed strict studies3588375a529c4fe8 and26c8a3ea28554454 remain failed.

## Explicit layer candidate and surviving boundaries

Verified file `muse_decode_layer/cand/layer_attn31s.py`, exact SHA256
`585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b`.
Its `_one_plan`18687ff contains the same L1/L8, sliding/full and producer
specialization. Inlined query preparation is5575ff; finalize_rows4488ff.
These mechanisms therefore exist in the explicit layer candidate as well.

Actual use is conditional: `_attn_path`18985–18997 requires supported heads,
state enable flags and `_one_wins`; then tries fused/ws/prep paths, finally
SGLang. Do not claim every layer load uses attn_one, or that a standalone
improvement transfers without measuring its selected layer path.

`kernel_fn`20993ff explicitly calls QKV projection20997, passes its output to
attention21004, then calls O projection21005. Thus attention-internal fusion
does NOT establish QKV-projection epilogue fusion or attention-to-O fusion.
The proposed QKV epilogue remains an unmeasured hypothesis: the independent
projection output still supplies query/key normalization, RoPE, new-KV cache
writes and gate logits. Legal ownership, global reductions, consuming layout,
cache ordering, register lifetime and residency require proof before coding.
No speedup or saved-byte percentage follows merely from identifying a tensor.

Do not cite M-C C-1 as a QKV experiment. Its owned implementation fused the
one-token O projection with the attention boundary, using two grid barriers;
`docs/runs-decode-layer.md`253–254 and ruled-out row244 identify check
5ae9ce318e014edf. Candidate prepare measured it1.16us slower in all9 stable
whole-step rounds and kept it off; the later multi-prepare process faulted.
That closes this O/boundary implementation, not an unbuilt QKV epilogue.
The later donor-mechanism label M-C also refers to O/boundary/front deferred
normalization (`runs-decode-layer.md`609), not QKV. Preserve these scopes.

## Measurements constrain the remaining opportunity

New mode51 attribution8ac6563343f74bcd, source4f99802, receipt/log committed
6ba862b on the attention worker: official unchanged control16/16,2.575636x
(CI2.537002–2.614271). Cold graph medians include replay overhead, microseconds:

|shape|control|control2|read producer/no consumer math|zero producer/no consumer math|no cache copies/math retained|
|---|---:|---:|---:|---:|---:|
|v16full|110.88|110.82|110.69|74.24|85.15|
|v32full|188.38|188.10|197.06|132.51|151.17|

All six arms met107104B/two-CTA resource gate before timing. The deliberately
wrong-output probes are never submission candidates. Removing consumer math
alone is mixed on v16 inputs and slower on both v32 inputs; producer/cache-copy
pacing is the measured direction, but perturbations are NOT additive recoverable
gains. Stamped pair spread12.1/22.7us is completion spread, not SM idle time.

Read the ruled-out history before selecting a mechanism. Mode51 weighting
already lost7631033a0c93485d and4ff339332fef4aff; counter arrival failed to
identify the favored CTA. Mode49 page-refill/QK experimentsc8e99e038d9f4575,
bb06/f3f, producer attribution4b60264a276c4805 and ring42c4739cf6b247c4
are L1 studies, not new L8 proof. VT-only NS4 proposal was closed offline:
119392B/CTA times2 exceeds233472B. No repeat sweep is authorized by this audit.

## Estimates, transfer limits and next gate

The paste explicitly calls5–11% attention,2–6% layer and8–18% whole-layer
ranges engineering estimates. All seven structural recommendations largely
exist already, so those ranges cannot be applied anew to this implementation.
H10058% whole-graph/87% B32-attention utilization and FA3 H1001.5–2x/75% TC
figures are different hardware, workloads and kernels; they neither measure
our H200 bottleneck nor establish achievable improvement. Opaque citation IDs
in the paste have not been independently source-verified in this subaudit.

At the saved board snapshot,2.60429 to2.9098 requires11.73% score increase,
or10.50% uniform latency reduction, not11.73% latency reduction. These are
snapshot arithmetic, not a current/final cutoff or shape-specific target.
Actual scoring weights and selected layer paths must govern acceptance.

Next action is the coordinator's full boundary/traffic mapping and actual
current-layer timeline, before another kernel hypothesis. Any new design must
preserve math/cache semantics and fences, prove ownership/layout/lifetimes,
pass resource and official correctness gates, then beat independent paired
nulls on affected shapes with aggregate score accounting. Stop a proposal
that merely restates existing specialization, relies on an imported H100
percentage, repeats a closed experiment without new evidence, or lacks a
measured materialization cost. No GPU check was dispatched for this report.
