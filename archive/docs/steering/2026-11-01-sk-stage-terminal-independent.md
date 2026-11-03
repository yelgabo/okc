# Stage-local terminal result and independent reader review

November1,2026. Owning public projection job **dfcfdd01d6f04525**, mode`run`, exact450781B source SHA256`de1111b98072f5251ff9b2edd5d453cc22eb8fe1710a943c60922909cc796229`. Root owns full receipt `muse_bf16_proj/profiles/stage-xor-release.dfcfdd01d6f04525.receipt.json`, source and canonical CLI profiles. This child made no official calls, GPU run, new source candidate or tuning change.

**Actual qualification passed; the stage-local production variant loses to C2.** No layer port or automatic retry follows. Stage-local packing/shared-helper proofs remain correct; they did not establish a useful consumer on the measured inputs.

## Independent actual result, before invoking the owner reader

`tools/proj_stage_release_independent_analyze.py` reads the owning terminal receipt and archived C2 directly, verifies exact job/board/mode/source, and reuses this child's independently written original score reconstruction. It does NOT import or call the owner's stage analyzer. Official scores reconstruct EXACTLY:

- Stage: **1.4191468393881441x**,CI[1.397859636797322,1.440434041978966].
- Archived C2 **9eba96894ee1402e**: **1.5278064829557674x**.
- Point-score ratio stage/C2 **0.9288786605**.

PASS9official tests,30profiles, checks.ok/no flags; preparation129.385967127s. Startup10overrides/0fallbacks/110keys/160math checks/10captures/30fresh replays. These are actual native stage targets, not fallback-only execution.

| M | stage sum5 adjusted us | C2 sum5 adjusted us | stage baseline sum5 us |
|---|---:|---:|---:|
|1|34.565951|35.182617|60.064001|
|8|40.825728|37.668069|75.647998|
|16|56.389937|46.914076|75.727997|
|32|61.932763|51.720988|76.896001|
|128|68.236245|67.095785|85.472000|
|256|83.533056|82.416369|101.471999|

The sixth root of the product of those six baseline/candidate sums is the official score; individual-cell speedups are not averaged.

## Direct C2 comparison

| target | C2 adjusted us | stage adjusted us | change |
|---|---:|---:|---:|
|qkv8|9.770656|12.789417|+30.896%|
|qkv16|9.740726|12.945855|+32.904%|
|qkv32|10.556945|14.857583|+40.738%|
|o16|5.366495|6.700904|+24.866%|
|o32|5.869364|6.819022|+16.180%|
|pool16|13.800228|16.955878|+22.867%|
|pool32|15.083591|18.664099|+23.738%|
|slice8|4.534868|4.819375|+6.274%|
|unpool16|13.557459|15.568185|+14.831%|
|unpool32|14.720666|16.520413|+12.226%|

All ten targets regress in adjusted AND raw event medians; raw range+5.860–30.694%. Untargeted20split11faster/9slower. Do not attribute those untouched cells to stage repair. Different seeds/weight banks, setup order, baseline/overhead estimates, incumbent tuner choices and nodes remain cross-receipt confounders. Stage and C2 happen to report the same driver580.178.04 and snapshot clocks, but snapshots are not achieved utilization and inputs are still different.

Replace only ten new adjusted medians into archived C2's other20cells and OLD baselines: counterfactual **1.41528114545x**, factor**0.92634843564** versus C2. This is a sensitivity scenario, not a paired result, official hybrid score or causal component attribution. The stage point score is higher than failed global-SX1.394120x; recovery against that loser does not establish a win over C2.

## Independent physical-extent/resource reconstruction

This child's analyzer derives payload/header/arena bytes, X reservation,1024B stage rounding, fixed extras and C2 implied pmax directly from native source configuration fields and printed actual capacities. It does not use owner layout-helper results. Every actual stage vector matches the independent formula. Full/NF control pmax is the same,>=2and even. All40vectors have0local/0static-shared; no treatment block-admission regression.

| target | capacity | padded arena B | stage regs full/NF | dynamic shared B | blocks/SM admission |
|---|---:|---:|---|---:|---:|
|qkv8|10|32|77/78|218352|1|
|qkv16|8|16|68/76|180480|1|
|qkv32|8|16|101/101|210160|1|
|o16|4|16|58/58|65760|3|
|o32|4|16|74/83|90336|2|
|pool16|9|32|61/67|169200|1|
|pool32|9|32|85/84|176352|1|
|slice8|5|16|55/55|120048|1|
|unpool16|6|16|58/58|65760|3|
|unpool32|6|16|74/83|90336|2|

All ten actual arenas fit the prior global-SX stride slack: no extra SST/shared relative to that representation. This now IS owning-H200 resource evidence rather than the earlier CPU natural-bank scenario. It does not quantify the cost of extra shared reads, header/loop instructions, reader-credit timing or reduced escape concurrency. Same admission is not achieved occupancy. No causal stall conclusion or after-A/arena-cap tuning is justified by these tables.

Independent complete30CSV/JSON,6loads,all40vectors and receipt/source pins are in `evidence/2026-11-01-sk-stage-terminal-independent/`.

## Owner analyzer review and corrections

Reviewed `tools/proj_stage_xor_release_analyze.py` against its synthetic tests, exact C2 packing and owning receipt contract. Initial source had two material semantics gaps:

1. C2 pmax was allowed odd/zero, and FULL/NF controls could imply different shared-table sizes despite sharing one pack. The positive fixture used impossible pmax3. Exact C2 pads pmax to even>=2, also needed for16-byte bulk table-copy alignment. Owner corrected delta>=16/multiple16 and FULL/NF equality; positive fixture now uses2.
2. A raw-score diagnostic mixed raw candidate times with adjusted baselines. It now uses `baseline_raw_ns` AND `median_raw_ns`, clearly named raw six-load ratios; official adjusted reconstruction is unchanged.

Owner also rejects failed benchmark status and duplicate/missing profile rows rather than hiding duplicates in a dict. Final reviewed owner commit **ee972f1**, raw analyzer SHA256`d8898e101de7bb8a4622cc44ead11f6439c86e4793a4b180cda0e7543131d085`. Six owner synthetic tests independently PASS in.094s, plus three new independent resource/profile/direct-C2 tests PASS in.113s. Before fixes, an independent mutation explicitly reproduced acceptance of different FULL/NF control pmax. No owner files were edited by this reviewer.

Only AFTER completing the independent actual reconstruction above, ran the fixed owner CLI reader on the terminal receipt. Exact agreement on10summary fields and all30old/new adjusted-median pairs is archived in reader-agreement.json. Source pin/owning mode, terminal clean pass, full30coverage/weight62, nine official tests, source-derived resource extents, explicit fallback totals and per-load score reconstruction are fail-closed. Counterfactual/direct-C2 labels remain separate from paired evidence. No remaining material reader-contract blocker found.

This report does not duplicate parent's canonical receipt/profile archive or runs/ruled-out rows. Root owns final closure and best-candidate steering. No further run, public retry, source tuning or layer port was performed here.
