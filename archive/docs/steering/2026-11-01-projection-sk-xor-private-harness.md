# Private SK10 sparse-XOR harness: source/CPU checkpoint

2026-11-01. This is an off-board-only projection diagnostic scaffold, not a measured win, GPU qualification, public candidate, or permission to spend another problem's quota. Root owns future dispatch. No H200 or public action occurred here.

## Source pins and normal entry

C2 owner `9eba96894ee1402e` comes from `3044ee7` plus its authorized repeat comment. Raw216576B SHA256 `5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`; normalizedLF212002B `7f4fd58cdd940f36c8d4d355f3de85687bc7f6ab04981842ce936db76d78adc2`. Root's `tools/proj_sk_xor_control.py` reconstructs that durable pin from Git at generation time, never inside the uploaded artifact.

Reviewed implementation owner supplies the separate helper source `d23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586` (310434B); device source `8cdbb4a9…` is owned by today_proj and its independent lane/packing proof by today_moe. This report clears only the harness scope below, not their device proof. The harness generator enforces both exact input pins and the normalized C2 prefix, restricts suffix bindings to the reviewed XOR constants/helpers, and appends one prepare wrapper. Original C2 `kernel_fn` and all its normal implementations remain unchanged. The wrapper calls normal prepare once, performs private diagnostics, returns the exact same state and never changes picks or weights.

Current generated artifact `.build/today-attn/sk-xor-private.py`:329061B SHA256 `7f51d9bc8052f0ac333326baefa5b72cc8bffcc9be8760501a117bdbb42a6e13`. [Manifest](../../../docs/steering/evidence/2026-11-01-projection-sk-xor-private/harness-manifest.json) pins body, codec and root's score-model source. Future generator/input changes supersede this physical pin and require regeneration/review.

## ABI and private ownership

`xor_build(tag)` builds two independent modules, each containing both variants and nearby-M entries. Four placements `c0,c1,t0,t1` use separate diagnostic tensors, captured inputs, outputs, graphs and streams; the two placements of a variant share only inventoried immutable packed tensors. `xor_runner(module,name,weight,variant,nearby=False,packed=...)` returns its callable plus resources/pack/private-counter metadata. All five supplied weight matrices are used; no replacement benchmark weights are generated.

The ten cells are qkv8/16/32, O16/32, pool16/32, unpool8/16/32. Before comparison, the actual prepared C2 closure must be the original `_sk_runner.run`; cid/full/TOKR selection must resolve to the expected SK name. A tuner fallback or other selected route fails qualification. A fifth captured/timed `anchor` is that actual C2 callable, not a reconstruction. Its watchdog tensors form one legitimate shared anchor owner and must be disjoint from all four private owners. Every original/private/nearby diagnostic tensor must remain zero.

Full backing-storage intervals reject overlapping mutable views, not just identical first-element pointers. Each graph warms on its private stream, explicitly forks/joins current stream, captures and replays once before reading its first output. Fresh replays copy identical values into independent captured inputs. C2 Lt workspace belongs only to the one unchanged common arm and is never concurrently replayed by these serial measurements. No cloned C2 prepare or shared mutable Lt plan is disguised as an independent null.

## Frozen coverage and decision

Three held-out activation seeds903151/903157/903163; M1/8/16/32/128/256; five layers;90 rows,24 cold-L2 randomized rounds each. Ten target cells have five actual vectors; the other20 cells have one actual normal-C2 common vector charged identically to all comparisons. Common rows are not independent nulls or paired evidence.

The harness performs210 captured placements,840 full-output FP64-reference checks (including420 fresh replays and210 post-timing checks),80 one-hot checks at M and M−1, and80 outlier checks at those sizes. Outliers scale1% of FP32-generated activation channels30x before BF16 conversion. Reference matmul is FP64; projection output rounds to BF16 and router to FP32 before comparison, matching the official reference boundary. Global/row/probe bounds are the official .01/.05/.001 for BF16 and .0001/.0005/.001 for router. This does not require bitwise identity to the old FP32-delta accumulation.

Root's canonical `score_model` is AST-renamed and embedded visibly, rather than reimplemented. It requires all90 rows and24 values, reconstructs six full five-operator load sums with the uniform62 weights, and gates both treatments against both rebuilt controls and actual C2 anchor: every seed factor>=1.01 and log-gain>2×global control-null log; affected regressions and anchor drift must stay within max(.5us,1% comparator). Raw event factors are primary. Future same-receipt official30 raw-minus-adjusted offsets may supply only a separate positive-domain sensitivity, never an official paired candidate score or stale-C2 promotion basis.

Actual CUDA getter vectors are `[registers,local_bytes,static_shared,dynamic_shared,threads,max_threads,blocks_per_SM]`. Duplicate builds must match. An independent Python translation of the actual capped SkCfg requires exact static/dynamic/threads/max-thread geometry; SX can use expected additional header padding (including the slice), but must retain zero local memory and resident blocks>=control. Register×thread products are reported, not mislabelled rounded physical allocation. [Eighty actual-Cfg crosschecks](../../../docs/steering/evidence/2026-11-01-projection-sk-xor-private/host-layout-crossproof.json) agree with the owner's independently emitted layout helper; these are CPU formula checks, not observed H200 resource vectors.

## CPU validation and practical gaps

`tools/test_proj_make_bench_sk_xor.py`:7/7 pass under `/root/.cache/okc-host-validate/bin/python` (Torch2.10 CPU). Tests execute the complete host scope with small CPU tensors and mocked CUDA scheduling, assert all counts, unchanged production picks/weights, bad-anchor and watchdog rejection, view-alias rejection, corrupt-output rejection, budget failure, original score-model equivalence, and capture fork/join/first replay. They do not prove H200 graph execution, host linking, device progress, performance or admission.

The lossless measurement codec stores CUDA float32 milliseconds in a binary pool only after every sample proves exact roundtrip to its original Python microsecond float. It reconstructs all vectors before score math; no rounding or timer-quanta assumption. A90-row realistic independent-jitter fixture with rich resource metadata encodes11987 ASCII bytes and roundtrips exactly; negative offset/count/overlap/non-lossless fixtures fail. Production encoding fails above13000 bytes without truncation. Full source remains readable; only measurement data is compressed.

Remaining transport gap: the13KB payload limit plus visible pins/footer does not by itself prove retention in a16KB terminal log when the normal official suite appends profiler messages. Root must resolve or accept a proven smaller actual bound before dispatch; no speculative atexit or timed-path printing hook was added. Source/CPU clearance alone is not dispatch readiness.

The normal C2 owning receipt logged prepare106.4s (SK build23.4s, CUT build49.3s, Lt build18.3s). This prior measurement is not the new scaffold's runtime. New total cost is that normal preparation plus two SK-only builds,20 packed formats, supplemental native calls,210 graph captures,5040 cold timed replays, FP64 references and output checks. Its actual time/memory cost remains unmeasured. The295s guard checks before/after preparation work, every cell and every timed round and at final completion; incomplete coverage fails closed. Existing framework300s timeout still applies to an individual long compile.

## Reproduce and stop

After integrating root's restore/reader and the owner's implementation:

```powershell
python tools/proj_sk_xor_control.py .build/today/c2.py
python tools/proj_sparse_xor.py .build/today/c2.py .build/today/sparse-xor.py
python tools/proj_make_bench_sk_xor.py .build/today/c2.py .build/today/sparse-xor.py .build/today/sk-xor-private.py
python tools/test_proj_make_bench_sk_xor.py -v
```

Tests in the merged tree resolve the reader beside themselves; the isolated worker used `SK_XOR_SCORE_READER` to point explicitly at root's current reader. Stop on source drift, incomplete counters/coverage, watchdog activity, resource-admission regression, unsupported actual C2 pick, transport overflow or295s deadline. A performance loss is recorded as a complete losing experiment, not converted into an exception that hides its raw vectors. No public release is eligible from this diagnostic; a genuine paired win still requires the exact pure candidate and official qualification.


## Later November1 addendum: tail-safe codec and exact immutable sharing

This addendum supersedes the earlier physical artifact/codec pins and the stated13KB transport limit. Final current artifact is330402B SHA256 `e5d82cf2f1245495e070f6953a6f0b048cf21ac786b0cf4aa97384d3fc6f7020`; implementation input stays exactly `d23aa17be4cf9e7c5869803b3b4603d8e72395953687adc0287fb1c75738d586`. Updated manifest is linked above. No new GPU data exists.

Root measured4890 characters of official profiler output after prepare in the owning C2 receipt. The final measurement format is fixed `SX2:`/`SXE2` framing: full JSON metadata plus a dictionary of exact IEEE float32-ms words and unsigned1/2-byte indices, compressed by stdlib LZMA2 (1MiB dictionary, lc3/lp0/pb0), then base85. Every sample still proves exact microsecond roundtrip; no values or rows are rounded/dropped. Word IDs use one byte iff the dictionary has<=256 entries, otherwise two; this is a deterministic lossless format, not selection between benchmark variants. Fixed byte-plane/XOR representations were CPU-tested and compressed worse, so they were not selected. Source text remains unencoded.

The encoder now rejects payloads>=9000 characters. Rich90-row independent-jitter fixture is7854 characters; tests reconstruct exact original vectors and metadata, cover two-byte IDs, invalid ID/dictionary/version/bounds/count/duplicate coverage, and preserve the full report under a16000-character tail with4890 appended profiler characters,1000 pin characters and<500 footer characters. This resolves the identified13KB retention defect for the recorded append budget. A future longer official suffix or actual payload overflow still fails the diagnostic; no atexit, stdout interception or altered timed path was introduced.

The new local immutable-pack cache has12 formats (six geometries×two variants), rather than20 repeated scans. Its key requires identical actual weight object, variant, output row extent, qkv balancing policy and exact R/NW/UPW/NT/WSK/UPS. Only qkv16/32, O16/32, pool16/32 and unpool16/32 pair within their own weight bank. `stage_stream` and header byte order use those fields; TOKR/NSTAGE alter the ring reservation, not immutable bytes. Reused pack dictionaries receive only a shallow name view for the reviewed runner guard. All mutable diagnostics, inputs, outputs, graphs and captures remain private; all90 rows and1000 oracle checks remain. The native metadata crossproof now also records all12 equivalence classes. No process-global packing cache is added.

Nine CPU tests pass after these changes. Final expected counter report adds `immutable_pack_formats=12` alongside210 placements/840 capture+replay+post checks/80 one-hot/80 outlier checks. H200 runtime, host linking, selected actual resources, device progress and performance remain unqualified; root must review before any future projection check. The artifact judges original C2 and cannot itself become a promoted sparse-XOR candidate.
