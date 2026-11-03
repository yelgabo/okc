# SK-XOR single-arm H200 result: correct, slower, closed

November1,2026. Independently reconstructed from owning public receipt **2b537518091248f1**, board`muse_bf16_proj`, mode`run`, source`muse_bf16_proj/experiments/sk_sparse_xor_release.py`,321640B SHA256`685ffab5d1614343d3c46d505967ae6817e3784736297018c78984354333f765`. This is the previously reviewed SINGLE-ARM production qualifier, not private e5d82cf2. Root dispatched/owns the full raw receipt at `muse_bf16_proj/profiles/sk-xor-release.2b537518091248f1.receipt.json`. No run or submission by this reviewer.

**Correctness/resource qualification passed, performance lost. Close this exact BEFORE-A/global-record implementation: no public retry or layer port.** Recommended projection candidate remains C2, and layer607e remains untouched. There is no measured win to transfer from this result.

## Owning result and archive

PASS9official tests,30benchmarks/30profiles, checks.ok, no flags. Preparation123.332824071s. Startup confirms10overrides/0fallbacks,110installed keys,160math checks,10captures and30fresh replay checks. This is real H200 execution of all ten offered cells, not an all-fallback pass. The110keys are still not110independent numerical cases.

Official score **1.3941200708702308x**,95%CI[1.3732082698071773,1.4150318719332842], versus archived C2 job**9eba96894ee1402e**1.5278064829557674x,CI[1.504889385711431,1.5507235802001038]. Point-score ratio0.9124978107. Both scores reconstruct EXACTLY as the sixth root of the product of six per-load sums-of-five-operator baseline/candidate ratios; this is not an arithmetic average of individual cell speedups.

Canonical normalized CLI artifacts copied byte-for-byte from root's owning experiments/profiles to `muse_bf16_proj/profiles/sk-xor-release.2b537518091248f1.{json,txt,log}`. Their hashes and lengths are in the evidence archive manifest. CLI-normalized benchmarks omit only empty`flags:[]` compared with full raw records; profiles match exactly. Full raw receipt stays root-owned and is not duplicated here.

Analyzer: `tools/proj_sk_xor_release_analyze.py --root ROOT --output OUTPUT`. It validates job/board/mode/source, reconstructs both scores, extracts complete four-vectors-by-ten resources, verifies coverage and emits all30rows as CSV/JSON, all6load sums and source/receipt pins. Three CPU analysis tests pass. Evidence directory: `docs/steering/evidence/2026-11-01-sk-xor-release-terminal/`.

## All ten adjusted target medians regressed

| target / M | archived C2 us | SX us | change |
|---|---:|---:|---:|
|qkv_gate8|9.770656|13.332913|+36.459%|
|qkv_gate16|9.740726|13.987165|+43.595%|
|qkv_gate32|10.556945|16.089976|+52.411%|
|o16|5.366495|6.910599|+28.773%|
|o32|5.869364|7.326624|+24.828%|
|pooler16|13.800228|17.675916|+28.084%|
|pooler32|15.083591|19.936242|+32.172%|
|unpooler8|4.534868|4.768748|+5.157%|
|unpooler16|13.557459|15.450684|+13.964%|
|unpooler32|14.720666|15.931540|+8.226%|

All ten raw event medians are also higher, by7.372–35.141%, so the sign is not confined to adjusted medians. The other20cells split10faster/10slower. Those are unchanged source paths with different input banks/setup/tuner/environment, not evidence that SX sped or slowed those cells. All30current/archived adjusted and raw medians, offsets, baselines, seeds and full profile strings are preserved in rows30.json/CSV. Profiles show one actual `sk_kernel` for each SX target; their truncated symbols alone do not expose every template flag, so coverage attribution also relies on the exact pinned dispatch and startup records.

## Six-load reconstruction and constrained counterfactual

| M | current sum5 us | archived C2 sum5 us | current official load ratio |
|---|---:|---:|---:|
|1|34.516743|35.182617|1.730407|
|8|40.467773|37.668069|1.837314|
|16|59.024314|46.914076|1.284081|
|32|64.528090|51.720988|1.191171|
|128|68.591346|67.095785|1.243772|
|256|84.070247|82.416369|1.213842|

Replacing ONLY the ten target adjusted medians in archived9eba, holding its other20cells and its baseline medians fixed, gives **1.3960011414x**, factor**0.9137290337** versus archived C2. This is an explicitly cross-receipt counterfactual, not a paired factor, actual candidate score, predicted repeat or causal estimate. It is included solely to prevent attributing the untouched20cells to this implementation.

Different seeds/weight banks, original C2 tuner choices, first-run order, replay overhead and hardware nodes remain confounders. Nodes differ; recorded drivers are580.159.03(current) vs580.178.04(archived), though both report H200/1980MHzSM/3201MHzmemory. Snapshot clocks are not achieved utilization. Currentsetup first=submission; archivedfirst=baseline. No paired controls, three-held-out/null gate or per-seed causal attribution exists in this ONE-ARM receipt. The official score and broad target regression are sufficient to reject this measured production candidate, but do not quantify which source change caused the loss.

## Actual runtime resources, full/NF

All40vectors report0local and0static-shared bytes. Full/NF dynamic shared, thread counts and admission agree within each side; complete raw vectors/field names are in resources.json.

| target | C registers full/NF | SX registers full/NF | dynamic C→SX B | threads C→SX | resident blocks/SM admission |
|---|---|---|---|---|---:|
|qkv8|78/83|76/77|220992→218352|288→160|1→1|
|qkv16|66/71|72/75|182096→180480|288→160|1→1|
|qkv32|112/112|101/101|219200→210160|288→160|1→1|
|o16|54/61|61/63|69248→65760|256→160|3→3|
|o32|83/92|78/84|100160→90336|256→160|2→2|
|pool16|60/63|62/70|174096→169200|288→160|1→1|
|pool32|85/111|78/81|186496→176352|256→160|1→1|
|slice8|60/60|52/53|118368→120048|416→288|1→1|
|unpool16|54/61|61/63|69360→65760|256→160|3→3|
|unpool32|83/92|78/84|100272→90336|256→160|2→2|

Reduced launch warps/registers/shared in some cells did NOT improve block admission; no achieved occupancy or utilization was measured. Slice8 shared increases because its header/stage layout costs more than its removed correction allocation. No interpretation of the latency loss as a resource-capacity effect follows from these vectors alone.

## Explicit correction to earlier dependency wording

The independent source report's4907addendum said BEFORE-A repair was stronger lifetime ordering and “requires no new producer dependency.” That was a safety/topology statement and was too broad as performance wording. **Corrected:** no new cyclic barrier dependency was introduced, but the global-record chain executes before A loading and before that reader releases its stage. It can defer stage-credit arrival and hence can add producer backpressure. Safety does not imply free overlap. This is a source-supported possible mechanism, NOT measured causal stall attribution from this one-arm timing result.

On the exact pinned native8cdb source, `LATE_ARRIVE = MT==2 && UPW==2`; qkv32/O32/unpool32 release after MMA. Merely moving repair after A loading does not universally move it after release or remove the credit dependency. No post-hoc after-A microtune/public retry is justified here. Root has separately requested feasibility of a materially different stage-local-record representation; that is not implemented, measured or endorsed by this report.

## Stop and handoff

Close this exact sparse global-record/B-fragment BEFORE-A production variant. Preserve source/correctness/bit-transport proofs, resources and losing receipt. No layer port, selective post-hoc public subset, timing-repeat or placement sweep. Parent owns runs/ruled-out/indexed steering rows; this report does not duplicate those rows. Any new representation must first provide distinct source/traffic/lifetime evidence and a frozen matched full-score experiment with controls; current one-arm data supplies no positive portfolio headroom.
