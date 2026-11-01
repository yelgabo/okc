# Performance review, 2026-11-01 01:16 PDT

Reviewed 2026-11-01 01:16 PDT against fetched `origin/main` at `6e078b5`.
The checkout remains `d6337e1`; 148 remote commits were inspected without merging or
changing kernels. New source links pin the remote evidence, which is absent from the local
checkout. Parsed 93 added remote profile JSON files; 18 local untracked receipts remain
already logged historical evidence.

A's deeper ring and skinny wait policy fail their gates. B's masked grouped library
implementation loses at 128/256. E's earlier HMRG/PRQ layer port has one paired win and an
unresolved repeat; the newer late-tid2 scheduling change is promoted standalone with its
layer port deferred. Projection C3 router gains are component results, separate from family C.

### Layer: attention port wins once; repeat leaves gain unresolved

The attention-only port `layer_attn31s.py` (SHA256 `585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b`, campaign recorded at `aea73a1`) replaces only `_ONE_SRC` in `layer_m1rt51sc1g` with owned HMRG1/PRQ31 code, without the withdrawn workspace pick. Candidate check `1c1fcf1d49924050` passes 13 tests, 2.048009646x, 95% CI 2.017289501-2.078729790. Its raw log retains route-side T1 and front skf-r T1 replay fallbacks; this score alone does not establish a gain. [Check receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31.1c1fcf1d49924050.json).

First same-process four-arm job `c0402e4c92bc4b20` uses two held-out inputs, eight rounds, 16 observations per arm, independent preparations, input seeds 610001+ and order 601. All 24 checks per arm pass. Pooled control/candidate is **1.0042**, four crosses 1.0036-1.0048, nulls 0.9995/1.0007: the registered rule selects candidate faster. The complete prepared candidate differs in M1 selection at 32 tokens (inc o+front, other three arms o), so this is not an isolated attention-instruction measurement. Per-load pooled ratios d1/d16/d32/v1/v16/v32 are 1.0102/1.0022/1.0074/1.0020/1.0026/1.0008. [Raw first comparison](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31_ab.c0402e4c92bc4b20.log).

Independent repeat `9414182118414889`, same source and method with seeds 620001+ and order 701, passes all 24 arm checks but fails the timing decision: **1.0015**, crosses 1.0006-1.0024, larger absolute log-null 0.0011. All four M1 policies are empty after fail-closed gates. The original candidate's 16x1 sliding gate reports relative error 0.14 and four order-1 events. This is a different prepared regime, and does not replicate the first gain. No paired confidence intervals are recorded. [Repeat receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31_ab2.9414182118414889.json), [raw repeat](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/attn31_ab2.9414182118414889.log).

At 128/256 tokens, representative cand-inc paired deltas are below. Percentages divide those deltas by the corresponding inc median, not a difference of marginal medians. Other crosses and nulls remain in the linked logs; these cells do not override the aggregate decisions.

| Comparison | Tokens / context | Inc median us | Paired delta us | Elapsed change |
|---|---|---:|---:|---:|
| c0402 first |128 / sliding|278.9|-0.0|0.00%|
| c0402 first |128 / full|362.8|-1.2|-0.33%|
| c0402 first |256 / sliding|355.0|-0.8|-0.23%|
| c0402 first |256 / full|499.4|+0.3|+0.06%|
| 9414 repeat |128 / sliding|278.5|-1.4|-0.50%|
| 9414 repeat |128 / full|356.5|-1.8|-0.50%|
| 9414 repeat |256 / sliding|361.0|-0.7|-0.19%|
| 9414 repeat |256 / full|538.0|+1.5|+0.28%|

The source was scored after the first paired win. Recorded board job `8f2780d7382047e6` passes 13 tests at **2.022870479x**, 95% CI **1.992527422-2.053213536**. Raw diagnostics retain element-injection sensitivity misses in MoE main-m2-ar2-ub T256 and route side T128, plus the unselected skf-r T1 replay failure. The scored M1 policy is {8:o+front,16:o,32:o}. Frozen old-control repeats `36e57280ff1f4c63` and `43026195f8c84ffb` pass at 2.014780x (1.984558-2.045002) and 2.009960x (1.979811-2.040110); these cross-run scores establish no isolated improvement. [Scored port](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/scored.8f2780d7382047e6.json), [control 1](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/scored.36e57280ff1f4c63.json), [control 2](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/scored.43026195f8c84ffb.json).

Matched-M1 diagnostic `273ff292536e4429` passes official tests but loses its early observation lines, so supplies no gate explanation. Fixed-tail `03bb478f8b0b4629` passes 13 tests; matched off/off, on/off and on/on gates at 16x1 sliding and 1x8 sliding all pass with zero order-1 events. The 0.14 event is neither reproduced nor explained. Its judged M1-forced-off control score, 2.035941422x (2.005402301-2.066480543), is diagnostic and must not replace the board score. Thresholds and CUDA are unchanged. [First diagnostic](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/m1matched.273ff292536e4429.json), [fixed-tail diagnostic](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/m1matched.03bb478f8b0b4629.json), [qualification](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md).

### Donor comparison recovered; timeline retained with invalid routing metadata

Historical job `0778baa78efa4aad` is recovered evidence, not a new dispatch. Candidate `layer_m1rt51sc1g` a00a0efa (sent b7dd9976), donor 8f17613b at pinned 6cae49a, d1/v1 only, 24 observations per arm. Donor pooled advantage is 1.0768, crosses 1.0693-1.0843, nulls 1.0132/1.0009. Candidate copy 2 rejects M1 while copy 1 enables it, so v1 is qualified; d1 uses no M1 and donor advantage survives both candidate copies. Paired donor-minus-candidate deltas: d1s -5.6/-5.9 us, d1f -8.4/-8.0, v1s -6.4/-6.0, v1f -12.3/-11.7. Official candidate check passes 13 tests, 2.013476x (1.983274-2.043678). No 128/256 coverage. This fulfills the previous d1/v1 measurement request; donor remains measurement-only. Profile sums that include PDL waits cannot identify exposed savings. [Recovered receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/recovered.0778baa78efa4aad.json), [raw comparison](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/recovered.0778baa78efa4aad.log).

All-12-shape timeline `41788b39fb824179`, source 097c38f, collects 24 rows but fails its 11,000-byte output guard at 12,192 bytes; no usable timeline or official correctness result survives. Corrected `01183c2992eb4153`, source d327abd, payload SHA256 `dae91104f666fbfc323145dda56dfad39593e1631c12624f39b056b9bc26c66d`, passes 13 tests and retains all 24 owned/SGLang traces. It uses one input per shape, five cold-event and three isolated-profile replays per arm, with explicit 0.1 us trace coordinates and raw-resolution span/union. Control score 2.017452507x (1.987190720-2.047714295) is not an optimization result. [Failed receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/timeline.41788b39fb824179.json), [retained receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/timeline.01183c2992eb4153.json), [decoded timeline](https://github.com/yelgabo/okc/blob/6e078b5/docs/steering/evidence/2026-11-01-fusion-audit/layer-timeline.md).

| Timeline workload | Owned cold median us | SGLang cold median us | Owned uncovered us |
|---|---:|---:|---:|
|128 / sliding|292.416|572.608|2.947|
|128 / full|415.968|825.568|2.944|
|256 / sliding|343.264|691.680|2.817|
|256 / full|543.936|1205.600|3.263|

These are diagnostic workload observations, not gains over the prior candidate. Across all shapes, uncovered time is 1.056-3.263 us; CUDA activity includes waits and neither union nor uncovered time measures SM utilization. At T1, top-k finishes at 34.8/42.4 us before pooler/norm at 46.6/53.6, so these samples show no exposed router tail. **Actual routing counts are invalid** because they were read from captured output before its first replay. Reference group/local_start, timed graphs and official tests are unaffected; the decoder marks routing_metadata_valid=false. The host helper now replays and synchronizes before inspection and passes five CPU checks, including a negative capture-without-replay case. No new GPU result validates corrected metadata. [Evidence and qualification](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md), [decoded data](https://github.com/yelgabo/okc/blob/6e078b5/docs/steering/evidence/2026-11-01-fusion-audit/layer-timeline.json).

CUPTI probe `04f0339fa9f24807`, source beedbca, initializes and obtains a 9,184-byte image size, but reading the availability image returns **CUPTI_ERROR_INSUFFICIENT_PRIVILEGES**. Official SGLang baseline check passes at 0.997335405x (0.982375373-1.012295436). This is counter-access failure and baseline timing, not traffic, utilization or kernel performance evidence. [Counter receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_decode_layer/profiles/counter.04f0339fa9f24807.json).

Current conclusion: E's port is layer-tested and scored, with one paired win and one unresolved repeat under changed M1 gate outcomes. Keep that mixed status visible. No new isolated A/B/C/D gain follows from these layer jobs, and the full ready-job scheduler remains untested. The next evidence gap is the unexplained gate event and reliable traffic attribution, not another claim from summed overlapping kernel durations.


### Remote evidence review through 6e078b5

Reviewed committed run ledgers and retrieved raw logs/receipts with git show; checkout unchanged. Negative timing deltas mean faster. Paired microbenchmark confidence intervals are absent unless stated. Percentages below divide the displayed paired delta by the displayed control median or mean; they are approximate, not official score changes.

### A: new ring and skinny wait-policy gates close without promotion

`e06255199f6d4de1`, baseline d3eeeda, source SHA256 65439aaa…7622e3a, is a producer-only diagnostic on mode51, with layer/std seeds211/212, 128/256 tokens. Official5/5 pass, xp equals51. Controls162.5/214.5us at128 layer/std and235.2/236.2 at256; instrumentation costs1.7/2.0/1.0/1.3us. Empty-region sums exceed readiness-region sums but overlap across CTAs and include overhead, so they cannot become recoverable latency. The old small-job-minority premise is invalidated: at256 seed211, <=4-n-tile jobs supply78.3% of layer stages and88.9% of std stages;128std is100%. [xprod.e06255199f6d4de1.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/xprod.e06255199f6d4de1.log)

Mode87 tests four x slots plus fewer weight slots against51, job `39804be1a59c429c`, dispatch41bf882, generated SHA2565e8eaa8f…156de31. Four seeds221-224 per routing,20 cold replays/arm. Official5/5 pass, outputs equal51 at128/256, maximum32-token relative difference3.5e-5. Required savings >=1us at128 and1.5us at256 fail. This closes the fixed capacity tradeoff, not every deeper/per-job ring. Loss of weight lead is an unisolated hypothesis.

| Tokens/routing | 51 control us | 87 paired delta us | Delta % | Faster seeds/4 |
|---|---:|---:|---:|---:|
| 32 layer | 94.1 | +0.1 | +0.11% | 1 |
| 32 std | 136.3 | -0.4 | -0.29% | 2 |
| 32 skewed | 68.8 | -0.5 | -0.73% | 3 |
| 128 layer | 158.3 | +2.2 | +1.39% | 0 |
| 128 std | 212.6 | +1.5 | +0.71% | 1 |
| 128 skewed | 97.9 | +1.2 | +1.23% | 0 |
| 256 layer | 211.0 | +2.9 | +1.37% | 0 |
| 256 std | 252.9 | +3.9 | +1.54% | 0 |
| 256 skewed | 150.3 | +1.9 | +1.26% | 0 |
[ring4.39804be1a59c429c.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/ring4.39804be1a59c429c.log)

The final skinny FULL-slot test_wait check `9be599cf333a47c4` at036b581 compares sk8w3s base, separately compiled neutral null, test, and test duplicate. Seeds321/322 on1/8/16/32 layer/std,20 cold randomized rounds;64 eager and64 postgraph FP32 checks pass, spin0, identical resources130registers/0local/155520dynamic bytes/1block per SM. Both the >=1% contribution beyond2xnull gate and per-case regression gate fail, KEEP=False. Six-load contribution estimates are null1.000295776, test1.000956258, duplicate0.998638858, assuming128/256 ratio1; these are not official policy scores. No128/256 candidate test was performed.

Stamped wait-region reductions are thread0 CTA averages, not critical-path gains. Prior `87f22aa71e614460` failed a host round-name collision; `252df4a54d764ca5` failed stamp decoding before printing its core summary. Neither preserves a performance conclusion and neither is a kernel loss. The final retry changes host handling, not CUDA. [testwait.9be599cf333a47c4.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/testwait.9be599cf333a47c4.log) [testwait.9be599cf333a47c4.receipt.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/testwait.9be599cf333a47c4.receipt.json)

Count-only job `53c07507b9fd4f8c`, dispatch1bfeec9, tests the proposed paired-job multicast opportunity on16/32 layer/std seeds311/312. Official5/5 pass. The >=10% pairable logical-request savings screen fails6/8 cases:16layer0/0%,16std0/0%,32layer18.18/18.18%,32std0/0%. No copy/multicast kernel ran; HBM/L2 traffic and benefit remain unmeasured. Park this registered scope rather than narrow to32layer after failure. [weight_count.53c07507b9fd4f8c.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/weight_count.53c07507b9fd4f8c.log)

### B: structurally different grouped library schedules now tested at128/256

Standalone INDEXED adapter `8d64807ea7504495` passes5/5 official tests but scores0.911513x. Candidate/SGLang us at1/8/16/32/128/256 are91.01/80.00,167.23/142.91,228.86/187.47,253.86/206.05,448.17/459.39,546.62/612.37. These controls are SGLang, not paired51. The adapter inherited a BF16gamma cast from prefill, while standalone51 uses FP32gamma; preserve that comparison limitation. It remains a losing, unpromoted implementation, not evidence against all grouped/TMA designs.

GROUPED_MASKED job `ae8993f4e3e1435c`, dispatch7785d25, corrects FP32gamma and compares four arms, owned51/duplicate and masked/duplicate, independent scratch.20 randomized cold rounds per128/256 layer/std seeds291/292, skew291 correctness. Official5/5 plus40eager and32postgraph FP32 checks pass. Registered savings >=3us at128 and4us at256, beyond2xnull on both routings, fail by large margins.

| Tokens/routing/seed | Owned51 us | Masked paired delta us (%) | Masked duplicate delta us | Owned-null delta us |
|---|---:|---:|---:|---:|
| 128/layer/291 | 172.86 | +180.16 (+104.22%) | +180.22 | -0.48 |
| 128/layer/292 | 172.90 | +180.00 (+104.11%) | +179.87 | -2.05 |
| 128/std/291 | 204.10 | +184.19 (+90.25%) | +183.71 | +0.86 |
| 128/std/292 | 227.30 | +144.64 (+63.64%) | +143.71 | -2.69 |
| 256/layer/291 | 212.26 | +234.88 (+110.66%) | +235.23 | +0.10 |
| 256/layer/292 | 214.11 | +231.78 (+108.25%) | +232.00 | -0.19 |
| 256/std/291 | 225.86 | +228.70 (+101.26%) | +228.83 | -0.48 |
| 256/std/292 | 217.15 | +202.75 (+93.37%) | +202.24 | -0.86 |
An untimed256/std292 profile reports G1/G2 244.447/136.064us of421.599us total,90.25%; GEMMs alone exceed the same-case owned217.15us. Glue fusion alone cannot close this measured loss. Exact config/CAP/count values fell outside the receipt tail and must remain unknown. No confirmation, fresh promotion gate, local-tile retune or layer integration followed. A remains the working baseline; B's default masked schedule is a tested loss, not an untested family. [humming_masked_moe.ae8993f4e3e1435c.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_moe_w4a16/profiles/humming_masked_moe.ae8993f4e3e1435c.log)

### E: late-tid2 promoted locally after separate correctness gates; no layer result

PRQ address-carry job `c77ce71c81af4fb1`, sourcefdaaa11, all12shapes/two inputs/80rounds, official16/16 and all arms19/19 pass. QK/V/both ratios1.0020/1.0021/1.0004 versus null1.0008 receive substantial credit from unchanged instances. Changed-row causal sensitivity estimates0.999968/1.000749/1.000438 do not establish a useful general gain. Source reverted; possible rider only. At v16sliding control15.58us, QK/V/both +0.03/+0.06/+0.00, null+0.10; v32sliding20.10, +0.06/+0.03/-0.19, null+0.03. These are128/256-token changed rows. v1full19.20 gives-0.22/-0.22/-0.38, null+0.03; v1sliding11.17 gives+0.06/0/+0.13, null-0.03. Full v16/v32 are unchanged and cannot supply this mechanism's gain. [prqcarry.c77ce71c81af4fb1.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/prqcarry.c77ce71c81af4fb1.log)

Matched prologue diagnostic `191e57a0f45a4717` at1f31181,3inputs/100rounds, passes16/16 and real arms11/11. D1full first drained page boundary owned7.04/donor6.02us, narrowing after first tile issue to9.79/9.41; complete owned/control-null17.18/16.96 versus donor16.22/16.48. V1full first boundary5.98/6.18, complete19.17/19.14 versus18.59/18.78. Drained exits destroy overlap and are not additive live-phase costs. This motivated late thread-index scheduling, not a measured pipeline optimization. [prologue.191e57a0f45a4717.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/prologue.191e57a0f45a4717.log)

| Job/source | Repeats | d1full controls and candidates us | Gate/status |
|---|---|---|---|
| b4d75c3185744d0a / fac5188, late-tid1 |3inputs,100rounds|ctrl17.28,ctrl2 17.15,late16.80; -0.48us/-2.78%|-0.51/-0.42/-0.45us perinput, localized signal only;16/16 and19/19 pass|
| ab53721b09854717 / fac5188, broad late-tid1 |3inputs,150rounds|ctrl17.25,ctrl2 17.22,late1/2 16.83; -0.42/-2.43%|All full crosses win, but changed-instance score >2xnull gate FAILS because sliding code also changes and its null shifts;16/16 and19/19 pass|
| a9b416a0c7324ecc /554445d, restricted late-tid2 |3inputs,150rounds|ctrl17.28,ctrl2 17.25,late1 16.90,late2 16.80; -0.38/-2.20%, -0.48/-2.78%|All12full crosses improve; full-only causal sensitivity1.0023928 vs nulls1.0001715/1.0005802 passes2xlognull;16/16 and19/19 pass|

[latetid.b4d75c3185744d0a.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latetid.b4d75c3185744d0a.log) [lateconfirm.ab53721b09854717.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/lateconfirm.ab53721b09854717.log) [latefull.a9b416a0c7324ecc.gate.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latefull.a9b416a0c7324ecc.gate.json)

These scheduling changes affect batch1 full decode, not128/256tokens. Keep broad-value1's failed gate. Restricted-value2 is a separately registered scope, not retroactive rescue. Full-only~0.24% is a calibrated score sensitivity from paired timing, not an official scored delta or isolated producer-consumer pipeline gain.

Fresh gate history must remain visible:

- `3588375a529c4fe8`, candidate554445d plus stress generator5c546fc, failed prepare:17/2000 sliding outputs exceed strict1e-4, maxglobal~4.48e-4. Full was not reached; no score.
- `26c8a3ea28554454` at0d5901e, four independently compiled arms and2000inputs each sliding/full, also failed strict1e-4 after both shapes. Incumbent nulls fail19/2000sliding and10/2000full; candidate nulls16/12; crossfailures11-18/10-14.24FP32samples pass but do not establish every queued output. No score. Both strict failures stay failed.
- NEWprospective TASK-math gate `80415fe736bb4890` atc8e709a passes all4000candidate and4000control outputs against FP32/global/row/cache criteria,6negative controls rejected, both workspace scans clean, canonical references crosschecked.16/16 official pass. The2.623065x score is qualified by validation harness allow_tf32=False; it is not the clean candidate score.
- Pure SHA256cbee36a852fbc62f5491e75f957eeb4298dce2db50947984cdbbcd08aa69b356 check `d331aca81c554d70` passes16/16, checks.ok true/flags empty,2.569431x CI2.530890-2.619575. Exact pure scored `c9bc5fbd85b440e2` passes16/16 at2.515774x CI2.478037-2.553510. Production defaults now late-tid2 after those gates; public best remains older8914751e3b9f468b/2.604290x.

[latestress.3588375a529c4fe8.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latestress.3588375a529c4fe8.log) [latenull.26c8a3ea28554454.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latenull.26c8a3ea28554454.log) [fp32fresh.80415fe736bb4890.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/fp32fresh.80415fe736bb4890.log) [latepure.d331aca81c554d70.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/latepure.d331aca81c554d70.json) [scored.c9bc5fbd85b440e2.json](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/scored.c9bc5fbd85b440e2.json)

No layer port ran. The source estimates0.021980% layer-score sensitivity for0.4us atd1full using actual load weights, below paired-layer noise. This is a forecast from measured component timing, not layer evidence. No new128/256late-tid claim is supported.

### E: mode51 Q8 attribution closes an arithmetic-only direction

`8ac6563343f74bcd` at4f99802,2inputs/120coldrounds, official16/16 control tests pass; two real arms pass targeted correctness. Four wrong-output probes stay probes. All six arms pass the same107104dynamic bytes/CTA and2CTAs/SM capacity gate before timing. Source uses a pinned default0 diagnostic control even though production late-tid2 is now promoted.

| Workload | Control us | Control2 | Read producer, math removed | Zero producer, math removed | No-cache-copy, math retained | Stamped |
|---|---:|---:|---:|---:|---:|---:|
|128tokens,v16full|110.88|110.82|110.69 (-0.19,-0.17%)|74.24 (-36.64,-33.04%)|85.15 (-25.73,-23.20%)|112.70|
|256tokens,v32full|188.38|188.10|197.06 (+8.68,+4.61%)|132.51 (-55.87,-29.66%)|151.17 (-37.21,-19.75%)|190.85|

Consumer-math removal is mixed on128 and slower on both256inputs. Cache traffic/producer pacing merit investigation, but these probes change instruction pacing/compilation and cannot be added or interpreted as realizable savings. Five-repetition stamp medians show full waits13.2/17.5us, producer-empty waits35.1/75.0, partner completion spread12.1/22.7. None is SM idle time. The proposed VT compaction saving was corrected from10240 to4096bytes because query staging aliases it; NS4 requires119392bytes/CTA, exceeding two-CTA capacity. No NS4 build or GPU experiment resulted. Existing mode51 pair-reweight losses remain closed; no new mechanism is justified by aggregate stamps alone. [q8attrib.8ac6563343f74bcd.log](https://github.com/yelgabo/okc/blob/6e078b5/muse_attention_decode/profiles/q8attrib.8ac6563343f74bcd.log)

### Context: unchanged controls and prefill evidence

New standalone MoE control checks/submissions7631974774944a2a/336f3a63bd6a44f3 atd3eeeda score2.193073/2.199082x,5/5each; historical47 repeat116c5185b14b487d fromec565a7 scores2.213803x,5/5. None beats existing2.215712. Attention unchanged PRQ31/HMRG1/pick-off repeatscccc131c55874949 and8914751e3b9f468b atd3eeeda+repeatcomments score2.534850/2.604290x,16/16each. The higher latter score is a repeat, not a new architecture gain. [MoE ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md) [attention ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-attention.md)

Prefill Humming results are separate workload context. They must not turn B green for standalone128/256 or for the decode layer. The user later deferred prefill; completed in-flight evidence remains recorded, no ongoing prefill work implied:

The prefill sequence includes initial ABI failure be3cbc7f5a234c64; passing corrected INDEXED0de84000097946c9 and fresh gateece54526780a4f28; local-M component signal7598f81b9b4645f9 followed by fresh failures ec3618d1604e44ba/d26a87ca2f484145; deterministic passes74eb65dad3fa4e54/da9700816b1641ea and paired confirmation21e98162b7114898; masked16384 failurec1f6023f03d34c72, then I64-address correctionf92ffc2953284a9a and fresh passdefbb042449d44e7. These are documented context, not independently raw-audited here. Final masked fresh score1.300971x remains a prefill result.

[full prefill source/hash and evidence ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md)

No new full ready-job D design, owner-completes C result or combined-family layer score is established by this review. Keep experimental failure scope specific: A ring87 tradeoff; A skinny two-full-slot test_wait; B indexed adapter and default masked backend; E broad late-tid1 gate and strict fresh comparator studies. Late-tid2 is a small standalone scheduling descendant promoted after a different, prospectively registered TASK-math gate, not proof that E's complete pipeline is solved.


### Projection: confirmed C3 router gains; later structural experiments lose

Review covers remote `8f2c89b..6e078b5` without changing the checkout. These are projection results, separate from A–E and whole-layer outcomes. [Projection run ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs.md).

C3 adds owned `r64c8s8fv` (prefetch F + uniform descriptor V) and `r64c8s8ordfv` (memset-zeroed atomic reduction + block order + F/V). It contains no GTICKET, donor code or removed fences. Clean source SHA256 `9fd8e794f7eb989943718348da2bf21ece934be3b4527b349eed34ccbceea4a4`, clean check `61330d87d32c4190`, passes 9/9, score 1.523256530x (95% CI 1.500407682–1.546105378). Initial rk-c source `d3eeeda`; C3 confirmation is recorded by `88860db`. All component FP64/probe/adjacent-size and fresh-input replay gates pass. The following percentages divide paired median deltas by the same-run control median, so they are descriptive component estimates, not aggregate gains. No paired CI is recorded.

| Job / workload | Candidate and control | Paired change | Repeats and nulls |
|---|---|---|---|
| `3170985420f74869`, router128 | FV vs r64c8s8 KLUEYZ, control 7.20 us | -0.42 us (-5.83%); independent FV duplicate -0.46 us (-6.39%) | 100 randomized cold rounds; 88/92 faster; control null +0.03 us |
| Same, router256 | r64c8s8 KLE+GRED+GZERO+RKORD+FV vs r128c8s6 KLUEYZ, control 9.44 us | -0.62 us (-6.57%); order-only -0.46 us (-4.87%) | 91/85 of 100 faster; null -0.03 us |
| `7d2fe593fdfa4720`, independent router128 confirmation | Same FV vs same owned base, control 7.10 us | -0.45/-0.46 us (-6.34%/-6.48%) | Fresh generator/order seeds 5353/4000; 88/87 of 100 faster; null -0.06 us |
| Same, router256 confirmation | Same order+FV vs owned r128c8s6, control 9.31 us | -0.69 us (-7.41%); order-only -0.45 us (-4.83%) | 85/90 of 100 faster; null -0.06 us |

Both benchmark receipts pass 9/9 on their judged paths. F alone at128 is inconsistent (-0.03 us initially, +0.10 us confirmation); FV on production r128c8s6 at256 is near noise (-0.06 then -0.13 us), and the r64c8s8 order path at128 loses +1.18/+1.09 us. Retain this shape specificity. [Initial raw log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/router_rkc.3170985420f74869.log), [confirmation raw log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/router_confirm.7d2fe593fdfa4720.log), [clean receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/cand_c3.61330d87d32c4190.json).

Scored repeats do not establish a new aggregate win: C2 `9eba96894ee1402e`, source `3044ee7` plus repeat comment, passes 9/9 at **1.527806x**, CI 1.504889–1.550724, and remains the recorded best projection entry. Duplicate dispatch `68248e326be14626` is not another measurement. C3 `7a15f46e673b43d5` passes 9/9 at 1.513805x, CI 1.491098–1.536512; exact frozen C3 repeat `2383f5173a934c63` passes at 1.519919441x, CI 1.497120649–1.542718233. Their intervals overlap C2. The paired component gains remain evidence for C3; the lower headlines do not isolate a C3 regression. [C2 receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/scored.9eba96894ee1402e.json), [C3 first score](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/scored.7a15f46e673b43d5.json), [C3 repeat](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/scored.2383f5173a934c63.json).

The previously pending cgemv/rd codec probes are complete. `075dc89d74d144d6`, source `d3eeeda`, passes 9/9 on unchanged control. Pair7 timing model loses qkv1/o1/pooler1/slice1 by +2.50/+2.02/+6.34/+1.34 us and rd pooler8/o8 by +1.23/+0.42 us. Even bytes-only perturbations lose, so this model cannot establish an ideal traffic ceiling. Real codecs were subsequently built and tested, superseding any “unbuilt codec” label. [Model log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/codec_cx.075dc89d74d144d6.log).

| Actual experiment / job | Correctness, control and measured result | Disposition |
|---|---|---|
| Scalar actual pair7 `962d2f9c1d9644f7`; builder introduced after `ed506b0`, C++20 fix `1755795` | Exact stored-byte/escape and exhaustive BF16 roundtrips plus FP64 random/probe pass; 100 randomized cold rounds vs original cgemv. qkv/o/pooler/slice m1 +9.952/+5.344/+24.944/+4.384 us, duplicate null -0.064 to 0. Physical bytes drop only 1–2.9%. | Loses; benchmark-only. Earlier `0ab8f0510515473b` failed compile (C++17), no GPU inference. |
| Vector-scaffold pair7 `4aef50bcd265448b`, builder `9f8e7fb` | Roundtrip/LUT/FP64 checks pass, same 100-round protocol. qkv/o/pool512/pool256/slice +4.960/+3.392/+13.024/+7.568/+1.312 us; null 0 to -0.064. No local-memory allocation. | Loses; no production adoption. |
| TOP7 LUT32 `ee39ec933a8542d8`, experiment `acc2f5a` | Stored-bit/LUT/FP64 gates pass. Same paired control/null protocol. qkv/o/pool512/pool256/slice +4.480/+2.160/+11.344/+8.528/+1.632 us, null 0–0.032. Actual byte saving about 3–3.4%; no spills. | Codec series closed as built, broader codec ideas remain untested. |
| Integrated escape consumer `bac995b3b94541c4`, source `0680370` | Original 12-bit codec; consumer warp0 takes dedicated escape warp's work. Two held-out inputs, 100 cold rounds each, independent old/inline builds. FP64 checks pass; actual CTA residency rises without spills. A/B losses qkv +0.960/+0.960, o +1.504/+1.504, pool +2.272/+2.240, slice +0.448/+0.480 us. Rebuilt-old null -0.128 to +0.032. | Ownership variant closed. The control headline 1.550070x is not the candidate's score. |
| rd16 bounded async ring `b103af15ba564ef6`, exact worker sources `c291a9d`/`dbfb451`, receipt owner `08d8150` | M1/7/15/16 random/probe/fresh-graph gates pass. Removes 24B local allocation but stays 1 CTA/SM; adds 98,304B dynamic shared. Ring loses +0.480/+0.528 us on input0 and +0.528/+0.608 on input1; only 12–17/100 faster. Rebuilt-old gains are noisy (null -0.192 us, MAD .272–.304). | Closed; no retrospective promotion, no128/256 gain. |

All successful projection jobs in this table pass 9 official tests on their unchanged control/C3 path, separately from the named experimental-arm gates. Their official aggregate scores must not be assigned to losing arms. The ring memory probes have different residency, so costs are non-additive. [Scalar log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/pair7_actual20.962d2f9c1d9644f7.log), [vector log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/pair7_vector.4aef50bcd265448b.log), [TOP7 log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/pair_top7.ee39ec933a8542d8.log), [escape receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/cgemv_ie.bac995b3b94541c4.json), [ring receipt](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/rd16_ring.b103af15ba564ef6.json).

Dense qkv256 series uses owned raw128-channel x128-token tiles, four-way K split, three ring stages, 144 CTAs. The residency prerequisite was met: 62 active clusters, two CTAs/SM, 90 registers, zero local bytes, 99,376 dynamic shared bytes. Four stages only supported30clusters and were skipped. Full arms pass FP64/probe at129/200/256 and fresh replay; 128 is not a measured timing workload. These experiments retain C3 judged paths and pass9/9. [Dense result ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs.md).

| Job / evidence revision | Same-run measured result | Conclusion |
|---|---|---|
| `4ca25ee4eccc495c`, `0a75361` | Lt raw26.624 us; full A/B39.472/39.408; paired losses +12.656/+12.816 us (+47.54%/+48.14% of control median),0/100 faster | Actual residency and spill gates pass, performance fails. |
| `e6e5bd6d0a104faf`, `9e10edc` | Lt26.912; full39.536/39.648; no-reduction36.384/36.336; memory-only no-reduction32.224/32.240 us | Ablations are incorrect-output diagnostic probes, not candidate gains. Even removing math and reduction leaves about5.3 us. |
| `d271218896df4a85`, tokenfast builder fixed in `9f8e7fb` | Lt26.592; old full39.328/39.440, tokenfast39.504/39.376; memory probe saves only0.1–0.3 us | Work-order repair fails. Prior `303ba7fc3bb44b9d` failed host harness before CUDA build, not a kernel result. |
| `7803be789c624f8e`, `46b1e45` | One128-row weight TMA replaces two64-row transfers with same bytes/barriers. Lt26.160; old full38.832/38.816, new38.320/38.336 us | About0.5 us saved versus losing experimental control still leaves about12 us to Lt. Local dense series closed. |

No bandwidth counters were captured. H200's4.8TB/s specification corrects an older3.35TB/s H100 premise but establishes no recoverable measured gain. GTICKET rk-d has no new result in this range; C3 deliberately succeeds without it. [Dense raw log](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_dsm.4ca25ee4eccc495c.log), [ablation](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_ablate.e6e5bd6d0a104faf.log), [tokenfast](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_tokenfast_fixed.d271218896df4a85.log), [wide TMA](https://github.com/yelgabo/okc/blob/6e078b5/muse_bf16_proj/profiles/dense_widew.7803be789c624f8e.log).

### Prefill context: measured improvements, now explicitly deferred

Prefill is a different problem with753/3012/16384-token workloads, not128/256 evidence for architecture A–E. User steering now defers all further prefill work. Preserve completed results and failures without listing them as active follow-ups. [Prefill ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-prefill.md), [detailed Humming validation ledger](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md).

- Baseline `09ba15bd18fd4000`, source `e0ce491`, passes5/5 at1.000953x. Scored baseline `181880202c5742db` passes at0.999115x but did not establish a ranked entry in the later board snapshot. Fixed Marlin M16 `e6c12149c99a4a96` and M32 `334149df33f64c54` pass5/5 but lose at0.715181x/0.884745x. These are cross-run whole-prefill results, no same-process fixed-tile gain.
- Capability check `e30ec1bf8e8142df` passes5/5, unchanged baseline0.999768x; installed Humming0.1.12. First indexed attempt `be3cbc7f5a234c64` fails prepare on scale ABI [32,1] vs [32], before GEMM. Corrected `0de84000097946c9`, SHA `bf6d95ae...`, passes5/5 at1.176448889x (CI1.158802–1.194096). Fresh gate `ece54526780a4f28` passes2000 queued/64 eager M32 checks, four synthetic routing banks. Pure-v3 SHA `f3024d1b...` scored `92408f1b65c643c4`:5/5,1.146465578x (CI1.129268595–1.163662562).
- Component `7598f81b9b4645f9`, six arms/two seeds/12cold rounds, passes all arm math checks and5official tests. Estimated-local-M vs routed-M heuristic saves109/162 us at753 and228/111 us at3012, but16384 is unresolved against duplicate noise. Global heuristic `ec3618d1604e44ba` fails unchanged fresh envelope at32 before official tests. Bounded heuristic `d26a87ca2f484145` also fails at unchanged32 control, before changed753/3012; this is sparse replay nondeterminism, not evidence those larger-shape timings lost.
- Disable-StreamK `74eb65dad3fa4e54`:5/5,1.092871039x (CI1.076477973–1.109264104); all ten fresh statistics exactlyzero at32/753/3012,64eager and2000queued each. Deterministic bounded heuristic `da9700816b1641ea`, SHA `29770518...`, passes same gates at1.161679904x. Paired component confirmation `21e98162b7114898`, two seeds/12cold rounds: bounded minus routed -197.9/-191.6 us at753, -399.8/-260.7 at3012; duplicate nulls -0.90/+0.74 and -0.61/-0.83. Component compensation is not an aggregate score estimate.
- Nozero plus filtered SiLU `2bd4f06042864771`, pure SHA `561e4180...`:5official and24arm math checks pass including NaN poison. Four arms/two seeds/12cold rounds, nozero-minus-control -75.808/-80.512 us at753, -314.400/-309.920 at3012, -1705.791/-1752.192 at16384; duplicate nulls +0.672/+0.064,-1.728/-2.208,-3.200/+4.000. Both clearing removal and filtered SiLU change, so credit neither alone. Judged candidate1.262738192x is not the benchmark file's authorization for submission.
- Nozero+noStreamK `36d2347642e14e37`, SHA `a0fe61e7...`:5official and24arm checks pass, fresh stats zero at32/753/3012;1.197430766x. Pure scored `1530ec79d50a4f69`:1.216560352x (CI1.198311946–1.234808757), selected by recorded23:26board. Combined bounded/nozero/noStreamK `e9bd4269a6584df8`, SHA `abce3925...`, completes5official/fresh checks at1.273593087x (CI1.254489190–1.292696983), archived unsubmitted.
- GROUPED_MASKED `c1f6023f03d34c72` passes initial5tests then fails16384 benchmark recheck (residual relativeL2 .6951 vs .0422345 bound), no score. Four widened-before-stride int64 casts correct own Triton overflow: `f92ffc2953284a9a`, SHA `63eb4171...`, passes5/5 plus all8 benchmark rechecks at1.294149997x (CI1.274737747–1.313562247). Poisoned fresh `defbb042449d44e7` subsequently passes5/5 and32/753/3012 unchanged envelope at1.300970887x (CI1.281456324–1.320485450), completed after scope deferral; no promotion. This is envelope-valid, not deterministic arithmetic.

Prefill component timings have no paired CI recorded; do not compute percentage improvements without corresponding exact same-run control medians. Do not import successful prefill Humming results into standalone MoE: interfaces, shapes and precision boundaries differ, and the standalone adapter tests are separate evidence.


## Whole-layer outcome and attribution

Recorded board job `8f2780d7382047e6` scores 2.022870479x, CI 1.992527422-2.053213536.
It contains the HMRG1/PRQ31 port without workspace picking. The first paired comparison
wins at1.0042; the independent repeat at1.0015 is unresolved after all M1 gates fail closed.
The original0.14 gate event remains unexplained despite passing matched diagnostics.
No replicated isolated E gain follows. Late-tid2 is a separate newer standalone scheduling
change and is not in this layer result. Source and diagnostic findings are linked above.

## Decision and next evidence needed

A remains favored over the tested B implementations at128/256. The new masked grouped
schedule passes math but loses substantially; its profiled GEMMs alone exceed the owned
whole call in the measured256/std case. Glue-only fusion cannot rescue that schedule.
A's ring87 and skinny wait changes fail their gates. Multicast opportunity counts fail the
registered scope and measure logical requests, not physical traffic. C ownership remains
retained; D's complete dependency-aware ready-only queue remains untested.

E's full-only late-tid2 change has paired standalone evidence and prospective TASK-math
validation, while strict1e-4 comparator failures remain recorded. Its small layer score
sensitivity does not justify assuming a transferable0.24% gain. The prior PRQ layer port
needs repeatable evidence under matched prepared policies and resolution of the gate event.
Wrong-output Q8 probes point toward producer/copy pacing, but yield no additive gain budget.

The d1/v1 donor comparison and all12-shape timeline are now available. Actual routing
counts from the timeline are invalid; trace timing and reference routing remain qualified
as described above. CUPTI access was denied, so no physical traffic counters were obtained.
Eliminating every uncovered trace interval gives a1.013956 weighted sensitivity only if
activity durations and overlaps stay fixed. That is a conditional calculation, not a ceiling
on kernel improvements. Use matched timing interventions before a production rewrite.

Projection C3 gains are specific router component measurements. Codec, dense qkv256,
escape ownership and rd16 ring losses remain scoped to their tested implementations.
Corrected GTICKET lacks a new timing result. Prefill is deferred context, not an active
A-E candidate. Keep donor F measurement-only and never add component or family scores.

[Layer evidence](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-decode-layer.md),
[MoE evidence](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-moe.md),
[attention evidence](https://github.com/yelgabo/okc/blob/6e078b5/docs/runs-attention.md),
[counter access and sensitivity](https://github.com/yelgabo/okc/blob/6e078b5/docs/steering/2026-11-01-counter-access.md).


## Review verification

New evidence sections received independent source review. Local link targets and55 remote
file objects at6e078b5 were verified. The builder checks timestamp/snapshot agreement;
all five family filters and the tracker fingerprint pass. No layout changes were made.
The local server was unavailable, and browser policy blocked direct file viewing, so this
review has no new rendered-layout check. The generated HTML remains the local fallback.
