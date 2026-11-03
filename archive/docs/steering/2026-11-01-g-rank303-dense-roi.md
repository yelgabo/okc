# G #303 dense-projection ROI closure, November 1, 2026

Evidence: root `5dc5d8d`; scored G-stage **607eb52bdaf24160**, PASS13, **2.126082676×**, CI[2.094191436,2.157973917],127.759508s prepare; external G0 **3a5f879df0d44825**, PASS13,2.248790532×. Exact submitted G source `muse_decode_layer/cand/g1b4_stage_release.py`:869076bytes, SHA256`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. No source changes, H200 job, standalone/C3 work, or quota request.

**Decision: close this scoped next-job proposal.** No credible new BF16-preserving, multi-load projection mechanism is established by these receipts. The G0 difference is not a throughput recipe. Replacing positive low-token QKV/O/unpool row gaps plus positive large-token dense-group gaps gives only **+0.54474%** optimistic whole-score arithmetic. This is a comparator-gap sensitivity, not a hardware ceiling, a measured exposed saving, or a limit on all future dense architectures.

## All twelve cells

Per-load latency is `(46*sliding +16*full)/62`; score is the geometric mean of six SGLang/entrant load ratios. `tools/dl_g_dense_roi.py <main-checkout>` validates the two owning job IDs, exact submitted file SHA, shape alignment and complete twelve-cell coverage, and reconstructs the scored result to better than1e-9.

|Load|Context|G whole us|G dense rows us|G0 dense rows us|
|---|---|---:|---:|---:|
|decode1|sliding|76.748|38.8|39.8|
|decode1|full|83.727|38.9|39.5|
|decode16|sliding|138.661|50.5|50.8|
|decode16|full|214.534|48.4|51.9|
|decode32|sliding|154.820|54.0|58.3|
|decode32|full|297.883|53.1|61.9|
|verify1|sliding|115.432|38.5|40.4|
|verify1|full|117.598|38.5|38.6|
|verify16|sliding|247.131|70.8|64.4|
|verify16|full|333.218|70.3|62.5|
|verify32|sliding|316.390|86.9|85.9|
|verify32|full|482.657|86.7|84.6|

Weighted G load times:78.549045/158.241010/191.739736/115.990823/269.347029/359.297216us. Dense rows include pooler/router and some fused work; they are retained profiler samples, not exclusive critical-path intervals. Row sums may overlap and differ from adjusted official medians. Separate weights/routing/GPU calibration confound receipt subtraction. No bandwidth, utilization or causal speedup is inferred.

## Low tokens: no new weight-traffic mechanism

The active configs are explicit in submitted source9490–9521, mapped to template arguments by `bfp__sk2_src`9559–9590. QKV uses qkv8/qkv16/qkv32; O uses o8h/o32h; unpool uses slice8 (only1024 output rows throughT8), unpooler16h/unpooler32. The fused pooler/router is a distinct operation. These routes already use lossless compressed weights (`bfp_COMPRESS`8944), cooperative/clustered staged GEMV and FP32 accumulation/corrections with required final BF16 output. This is not a raw-BF16 stream missing the existing codec. O and unpool consume different activations after attention/experts; merely combining their weight buffers cannot eliminate either matrix read.

The script maps the low-token profiler templates to actual configs and counts **only positive** individual QKV/O/unpool differences. Every low-token **total dense group** is already faster than G0. G0's small individual advantages yield:

|Substitution into current G, other cells fixed|Full six-load factor|
|---|---:|
|Positive low-token QKV gaps|1.000270167|
|Positive low-token O gaps|1.000115254|
|Positive low-token unpool gaps|1.000213376|
|All three low-token groups jointly|1.000599052|
|Positive large-token dense-group gaps|1.004845477|
|Low individual plus large group gaps|1.005447432|

This closes a G0-gap-driven low-token weight-stream job. It does **not** prove there is no slack: deleting all low-token QKV/O/unpool rows would produce an impossible1.16406 factor, and a hypothetical10% reduction in every such row gives1.01376. Neither scenario supplies a schedule, legal traffic elimination, or measured critical-path saving. There are no counters here establishing a newly dominant decoder/escape/HBM phase.

Relevant prior losses remain closed absent a changed mechanism: QKV8 register-direct1759290da58047ab/bcabe48caa884e67; split-K QKV16/32 geometry01703396ea1a409e; QKV128 depth4542b2b04cb344ec and push/transpose4edaa20c98904404; codec/escape/ring families in `docs/ruled-out.md`. This is evidence against repeating named designs, not proof that their different source geometries cover every G-specific change.

## Higher loads: selected Lt already present

Startup receipt reports all **eight** Lt plans: QKV/O/pr/un128/256. Submitted source12185–12283 defines full-K BF16 operands with compute32F, nonsplitK0/1 acceptance, BF16 QKV/O/un outputs and FP32 concatenated pr output. QKV256 is algo66/tile124/stage35/custom0/cluster2/splitK1, the already validated pin; no missing port is identified. `_rq_gemm`16741–16742 returnsNone. G0's rq128/256 instead rounds split partials toBF16 before reduction; it cannot be reinstated as an unchanged full-FP32-reduction alternative. The retained G0 rq128 row aggregates **two calls**; it does not isolate QKV versus unpool.

The measured legal owned layer comparison from `47bec92021ef4bbe` offers only a **QKV+O128 group** sizing check: current28.5/28.6us versus owned14.8+7.8 /14.5+7.6. Substituting that entire group gives **1.003796558** (+0.37966%), affects onlyverify16, and does not attribute an individual QKV win. Owned un128 Lt was slower (21.2/21.7 versus current16.9/16.8);256 uses the same QKV pin with no material stock multi-op advantage. The prior audited wide-Lt and raw-tile/residency/transpose/depth results provide no newly supported shippable replacement.

**Updated fit condition:** this G release has130923 bytes of room under999999, so the earlier near-1MB payload objection alone is obsolete. An owned QKV device literal of41179bytes could fit in principle before host/packing/tests, but this alone would add another prepare/build path for an unisolated <0.4% one-load envelope. Current127.76s prepare is measured; incremental compile and conformance cost is not. Neither spare bytes nor spare time is performance evidence.

## Next action and stop/reopen rule

Keep the scored G projection routes unchanged; prioritize the independently owned expert/overlap questions. Do not spend a scored measurement on a broad Lt/codec/cap/depth sweep, restored rq precision, or a merged-source experiment with no isolated gate.

Reopen only for a specific G-local precision-preserving mechanism with actual component attribution, a meaningful six-load budget beyond nulls, and a source/prepare budget. Require full-K FP64/one-hot checks where projections change, exact BF16 boundary ABI, real-routing/fresh/empty capture tests, all affected contexts and load weights, actual backend/resources, and fatal CUDA/math/budget handling. A single-arm official measurement may be authorized separately but must not be relabeled as paired causal proof. No quota request follows from this audit.
