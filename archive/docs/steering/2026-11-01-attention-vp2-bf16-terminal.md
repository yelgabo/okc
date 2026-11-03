# Private BF16-output VP2 comparator — terminal scoped closure

2026-11-01. Evidence source e86b0d5, freeze23fb760; pending registration6121dba.
Exact uploaded artifact866845554b3d694c40165fabbdfde6739f6fa22b1efb5abb6b5831f32e368f24,
412952B. Root-owned private attention check **c79ddddb840a4fb7**. No public submission,
foreign code adoption or owned implementation promotion. This report supersedes the pending
state of [the source/requalification report](2026-11-01-attention-vp2-bf16-requalification.md).

## Actual receipt and correctness

Independently read root's FULL raw status receipt
`muse_attention_decode/profiles/vp2_bf16_private.c79ddddb840a4fb7.receipt.json` and ran the
frozen reader, not a normalized profile or reconstructed result. ID/board/check provenance
verified. PASS16,12benchmarks/profiles,checks.ok,no flags. Actual PRIVATE comparator
normal entry2.5943196585146002x,95%CI[2.5554048636368814,2.633234453392319]. This official
standalone comparator score is not a paired improvement, layer result, public entry or
owned rewrite score. Different-run score movements against older control receipts are
not credited to VP2.

All640 private checks passed: eager16,normal144,fresh336,post144;36raw rows, all12cells x
3heldouts. Required20VP2/284owned Python warm/capture calls pass; these are not graph
replay counts. Actual plan IDs/private owners/resource gates and independent inputs/cache/
graph lifetimes pass the frozen source. Preparation102.040889424s. Normal private judge
uses actual VP2 only at the declared eligible shape; other TP8 native-owned and nonTP8
stock fallback. BF16 attention output boundary is matched in both arms. VP2 raw-normalized
Q/K/P math still differs; this does not establish every internal baseline rounding boundary.
FP32 judge and separate BF16-output-oracle/cache checks retain their original thresholds.

Actual H200 target resources (not local-compiler predictions):

|Target backend|Registers|Local bytes|Dynamic shared bytes|Threads|Block admission/SM|Active16CTA clusters|
|---|---:|---:|---:|---:|---:|---:|
|Owned8/15|82|0|90704|256|2|not applicable|
|VP2|97|0|81984|256|2|14|

This is capacity/admission, not achieved utilization. Owned local83 prediction differs
from measured82; actual numbers above take precedence. No fence/barrier changes were
needed. Initial/empty/partial partition and graph replay gates passed the source assertions.

## Frozen full-score decision — FAIL

All four contemporary six-load factors are
1.008694551457367/1.009060016147697/1.006308726565348/1.006673326836658.
Largest is below1.01; contemporary duplicate null.002370867735752. All three per-seed
>=1%/2xnull gates fail. Contemporary regression false; raw four-cross-positive-every-seed
condition TRUE and raw regression false. **Registered decision FALSE.**

Raw factors1.006134117851295/1.006085285745614/1.004303066594281/1.004254323357555,
rawnull.001823205880695. Old3f44-offset descriptive factors
1.008349459537297/1.008643310637703/1.006036833060715/1.006330010220327 also fail.
No choice of a preferred cross, seed, raw/adjusted column or old calibration rescues the
predefined gate. The reader derived12 contemporaneous raw-minus-adjusted offsets from
THIS official job and applied each equally to the private arms. These recalculated factors
are paired sensitivities, not official paired scores. All raw times are preserved in the
canonical log and committed decoded record.

|Held-out seed|Target controls mean raw us|Target VP2s mean raw us|Contemporary four-cross min–max|Seed null|Frozen gate|
|---|---:|---:|---:|---:|---|
|824731|10.856000|10.136000|1.009034424–1.011312940|0.001305507|FAIL|
|824737|10.744000|10.144000|1.005692964–1.011608322|0.005827716|FAIL|
|824743|10.752000|10.224000|1.002882006–1.005613639|0.002637996|FAIL|

Target pooled controls10.784000158us versusVP210.168000124us, difference.616000034us,
about5.71% lower raw latency. All three target banks improve (.720/.600/.528us mean).
That is a real PRIVATE bundled-cell advantage under the matched output boundary; it does
not satisfy the whole-score acceptance rule. The source-phase wait budget remains
unmeasured. No physical HBM/copy/DSM-barrier latency conclusion follows.

## Comparator discrepancy and scoped engineering decision

Retain both sides of the discrepancy: owned complete-head64-key/genericDSM screen
eac485ea0d554e50 lost11.10→14.98/15.01us, while the complete VP2 N128/dedicated-owner/
asynchronous-mailbox bundle now wins the target cell under BF16-output-matched controls.
These results do not close all cluster ownership designs. They also do not prove async
stores alone, N128 alone, query/P precision, shared footprint or barrier placement caused
either result. Earlier WST full16 failure remains closed, and positive VP2 signal cannot
be assigned to D1, full context or other batch sizes where it was not used.

Close AUTOMATIC owned rewrite, retry, public/adoption action and parameter sweeps from
this exact requalification. Existing owned public submission and Family G release bytes
remain unchanged. Foreign standalone code remains private measurement-only. An eventual
reopening needs a DISTINCT owned mechanism or independently validated portfolio with
score-covered headroom beyond duplicate noise and the predefined gate; this is not a
blanket permanent ban on attention storage/ownership work. In particular, no retuning of
N128/cluster size/hints/rounding or posthoc single-cell promotion is authorized by this loss.

## Reproducibility and handoff

Reader: `tools/moe_attn_vp2_read.py` on the FULL raw receipt above. Independently recomputed
all raw per-seed46/16 products without the frozen score helper; results agree<1e-12.
Decoded full source manifest, private raw rows, all resources/plans, raw/old/current gates
and contemporary offsets are in
`evidence/2026-11-01-attention-vp2-bf16-private/c79ddddb840a4fb7.decoded.json`.
Root owns canonical receipt/normalized profile/text/log/analysis files; no duplicate profile
copies added by this branch. Source/ordering review9bd60da and eight-test real-TorchCPU
host review preceded dispatch; H200 now supplies the separately reported runtime evidence.
No additional job or timeout retry was launched by this owner.
