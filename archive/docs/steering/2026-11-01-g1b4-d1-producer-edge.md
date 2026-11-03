# G1b4 D1 O projection to first boundary: trace preparation

2026-11-01. Source main `1a78018`, G1b4 pure SHA1607293eee3403109df5d37b6e2d1ee34b528c3578afa6c2c8668440afde1658 (862176bytes). This follows independent steering755f456. No H200 layer check or public submit dispatched. G1b4 still requires its separate qualification first.

## Source edge and safe independent work

Pure source lines15869–15885 (`_sl_tile_after_attention`), or15667–15683 (`_sl_cluster_after_attention`), call `dense(...,"o")` producing BF16 P with S=1, followed by `_sl_cluster_module().b1`. Latest `_after_attention` at16718 cannot select GLE when `_pdl(T)` is false. Active cluster/tile selection is checked dynamically; fallback fails the probe. D1 b1 dispatch is `glue_cluster_boundary<4,32,false,false,true,false>` (host15617–15629).

At15497 the consumer first executes griddepcontrol.wait. Stream,alpha,beta,alpha2,pre vector loads at15513–15518 and rank/index setup do not depend on P. Stream data is already available before O, coefficient/pre arrays are prepare-owned. A cluster.sync after this prefix establishes DSM destinations. A second unchanged wait/launch_dependents precedes first P-dependent `glb(a.branch...)` at15553. No first-P read can move ahead of producer completion. D1 does not use PRELOADED_BASE or SLICED paths. Both norms and all publication/retirement cluster barriers remain necessary.

G1b4 sets PDL=False at93; `_pdl`97 is therefore false. `glue_launch`15611 uses programmaticStreamSerializationAllowed=0. The current ordinary same-stream graph requires O to complete before b1 starts. Consequently there is no current overlapping O/b1 wait to remove by moving independent loads above the first wait. Host allocation, module lookup, and argument construction occur during capture and do not constitute steady graph replay savings. Enabling PDL or a second stream would change synchronization and requires a separate producer/consumer proof; this diagnostic does neither.

## Existing bound and next edge

Owning G1b2 receipt736a5cbc977f4ad4 profiles b1 at3.2us on BOTH D1 contexts; whole cells76.92/83.62us, weighted D1=(46*76.92+16*83.62)/62=78.649us. Even deleting the ENTIRE3.2us boundary gives only1.006947six-load score factor (~0.695%). Independent setup alone is smaller, and G1b4 has not been measured. This is an optimistic duration-based sensitivity, not a measured removable critical-path cost; do not fund a performance implementation based on summing kernel durations. G1b2→G0 D1 endpoint difference10.45us is not explained by b1 alone.

The same D1 profile has a distinct~18.6/18.1us GLE pooler/router producer and23.4/22.3us native MoE activity. Source `_pool_router_fp32`16638 and `_sl_tile_route`15901 feed the native expert adapter. That edge has larger measured activity, but its exposed setup/wait is unmeasured and routing differs across endpoints. Scope the next trace to pool/router→route→expert only after native G1b4 qualification and actual path/ownership mapping. No claimed gain or schedule change there.

## Distinct off-board diagnostic

`tools/dl_make_g1b4_edge.py` emits a check-only881689byte artifact SHA9395dfe464acc44b46a82224adc5d715fe05b43a27a7de7e2a0b22b887768e53. It reuses existing dl_make_layer_timeline's input helper extraction and dl_timeline_codec's lossless interval transport. Two separately prepared states have equal selected policy, separate input/cache tensors, graph/output ownership, attention_scratch and ma_scratch. Read-only weights are shared. The traced b1 receives an explicit per-state8x5timestamp tensor. No global timestamp pointer or cross-state cache. Native attention/dense-O/cluster-or-tile/b1 counts must each match expected path; no GLE/fallback credit. Probe flags are active only during private calls; prepare and the judged path execute the original b1.

Both contexts, seeds911031/911032/911033,15random-order cold samples per arm and3isolated CUDA profiler replays per arm. Graph outputs must match pairwise exactly; mismatch or distinct policy blocks coverage. PDL must remain false, all stamps nonzero/monotonic, all12rows complete within280s or fail closed. There is no external profiler process, runtime filesystem I/O, or data/ref import. Output is one bounded log blob. H200 compilation/host APIs/equality/runtime costs remain unverified.

Five per-CTA tags are entry beforewait, afterentrywait, issued-independent-prefix end beforecluster.sync, aftercluster.sync beforesecondwait, aftersecondwait immediately before dependent branch loop. Only thread0 records globaltimer. Tag intervals are PERTURBED elapsed source regions, including instrumentation; load issues may remain outstanding, so prefix duration is not completed-load latency and afterwait tag is not exact LSU issue time. Do not call these SM stall/counter measurements. Original statements, fence count and barriers compare byte-exact after stripping only new stamps. No alternative scheduling is embedded.

Uninstrumented versus stamped whole-graph medians/spans expose instrumentation criticality. Raw profiler intervals provide same-stream O end→b1 start gap, b1 duration and remaining graph tail; no summed duration is called a saving. Setup/wait observations are usable quantitatively only if paired graph perturbation stays within registered max(.2us,1%) AND repeated traces agree within that perturbation; otherwise report instrumentation-distorted. A scheduling proposal must derive six-load sensitivity from measured exposed interval with all other rows held fixed, exceed duplicate/null variability and preferably1%whole-score. Historical entire-b1 ceiling already fails1%, so recommend keeping this trace optional/informational rather than spending a quota solely on this edge.

## CPU/device validation and commands

Four CPU tests pass: original kernel statement/math/sync inverse, unique stamp bounds, overlapping timeline-not-sum fixture, private scratch pointers and inactive probe guards. Actual CUDA12.8 NVRTC/PTXAS sm90a with the module's --use_fast_math flag compiles original and stamped selected kernel: BOTH64registers,0stack/spills,1CTAbarrier,384staticSMEM. Cluster barriers are separate instructions. Logs/adapter disclosure are in muse_decode_layer/profiles/g1b4-edge-local. Device-only compilation removes TVM host headers and uses exact metadata adapter; it does not compile the host TVM wrapper, validate API behavior, or run an H200.

From the attention worktree (root supplied explicitly):

```powershell
python tools/dl_make_g1b4_edge.py ../okc/.build/today/g1b4_native_experts.py .build/today-attn/g1b4-edge.py --root ../okc
wsl -d Ubuntu -u root --exec bash -lc 'cd /mnt/c/Users/gretc/Documents/GitHub/okc-today-attn && /root/.cache/okc-host-validate/bin/python tools/test_dl_g1b4_edge.py ../okc/.build/today/g1b4_native_experts.py'
wsl -d Ubuntu -u root --exec bash -lc 'cd /mnt/c/Users/gretc/Documents/GitHub/okc-today-attn && /root/.cache/okc-host-validate/bin/python tools/dl_edge_local_compile.py .build/today-attn/g1b4-edge.cu .build/today-attn/g1b4-edge-compile'
```

Future coordinator-only dispatch, AFTER G1b4 qualifies and if this small ceiling still merits quota: `okc check .build/today-attn/g1b4-edge.py --no-wait --json`, then wait on that exact ID. This command has NOT been executed. No performance claim, promotion or best-submission change follows from these offline checks.
