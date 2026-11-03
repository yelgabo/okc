# Public implementation discovery refresh — 2026-11-01, 18:30 UTC continuation

Evidence base: coordinator main `65f8810`; this report follows the 14:30 attention delta and 109-handle metadata survey, rather than repeating their immutable source audits. No kernel adoption, GPU check, registration or public communication occurred.

## Decision

No newly located public implementation has sufficiently strong, source-associated evidence to replace the current top-50 backlog or justify another comparator check. This is a bounded discovery result, not proof that stronger public code does not exist. Preserve the first-two reset investigations and the strongest unexplained existing measured comparators; neither global nor stage-local sparse-XOR losses are reopened by these search results.

## Search coverage and limits

Started with entrant-independent identifiers `muse_decode_layer`, `muse_bf16_proj`, `muse_attention_decode`, `muse_moe_w4a16`, plus challenge/Muse/H200 repository searches. Connected GitHub search returned only20 results per query; switched to authenticated public GitHub REST code search for pagination. Excluded the already inventoried lee101/openkernels and rohanarun/Auto-research-agent repositories explicitly, not all collaborators. Separate fork search found xrey167's fork at the identical rohan commit `018b7a102348ea3b73701e8be52d24f82d8c91ee`.

Code search reached page3 for layer/attention/MoE and page4 for projection. Counts changed during pagination (layer235→266, attention222→231, projection386→396, MoE293→246); therefore pages are not a consistent snapshot and deduplication cannot establish completeness. One MoE request initially hit403 rate limiting and was retried successfully. Final page sizes were66/31/96/46 respectively; there is no1000-result cap in these narrowed queries. Repository searches returned29 challenge,26 Muse-kernel and10 H200 matches, including unrelated Android repositories. Search summaries preserve exact queries, errors and result scope.

Pinned24 repository heads and full recursive trees (none reported truncation), inspected selected READMEs/results/licenses and receipts, including beyond top handles. Public metadata checks found zero forks for the17 first-pass repositories and seven additional repositories; this is not an exhaustive collaborator/gist/branch search. Credited upstream chenwainuo and the xrey fork were checked. Nondefault branches, unindexed files, deleted/private repositories, unpublished submissions and external benchmark claims remain gaps. No Twitter attribution or ranking claim is inferred from search text.

## Useful newly inspected evidence

| Source | Exact pin / evidence | Finding and qualification |
|---|---|---|
| [Madhavan113](https://github.com/Madhavan113/ParadigmKernelChallenge/tree/721c62729fa0b407799692f311790643706f292d) | `721c62729fa0b407799692f311790643706f292d`;110 published normalized profiles | Best archived layer `b73aff83f3e441b6`2.1111985479; attention `ebfbc78dbedb4e57`1.5545463548; projection `a9c00feac4494dfa`1.2094269566; MoE `7d33f1fe3a314fc9`1.7095011294. Author artifacts, not our checks; normalized profiles name a submission path but do not establish its exact current source SHA. Source headers attribute vjeux/Apache2 despite repository API license=null. |
| [iyerramji](https://github.com/iyerramji/open-kernel-challenge/tree/1d57ff7749433d0d18b294744e607cd82e33057f) | Apache2, `1d57ff7749433d0d18b294744e607cd82e33057f` | README separates scored1.1583 (`433a8be9f5c14d37`) and1.1560 (`764ac132f5224473`) from Modal probes. Native lossless codec reportedly28% slower; no superior projection candidate. |
| [operatorstack](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a) | MIT plus SGLang Apache notice | Published layer1.7352/MoE1.8320/attention1.8228/projection1.3066; below existing owned/comparator evidence. Edited publication source requires its supplied executable-AST qualification, not a new same-box claim. |
| [Kushagrabainsla](https://github.com/Kushagrabainsla/muse-spark-open/tree/ca71144194e993c9769f20654fade8008c6868c7) | Same `ca711441...` pin as prior discovery | README actual jobs09b49ea4276e4306 attention1.525;069f543bb3604714 MoE1.704;7a0ba34ed4ee4181 layer1.496;fef2017d39ec4cc2 projection1.190. No new stronger source delta. API license=null; do not presume adoption permission. |
| [ozgureyilmaz](https://github.com/ozgureyilmaz/okc-muse-kernels/tree/74daa3a6990168b52359c80955014a945fc7aa4e) | Apache2, scored `20251c5f78804f1f` | Projection1.1718 versus check1.1804, explicitly distinguished; below current candidate. |
| [lachydotmcg](https://github.com/lachydotmcg/muse-sputter/tree/527bd70472e3842bbd6147f8cc2bba6d6452cf43) | MIT | Intentional143.787× slowdown on RTX3060, not an H200 optimization or challenger despite search relevance. |

Madhavan's layer per-cell published adjusted times offer the only concrete new comparative detail: small/decode rows marginally faster than our607e receipt (e.g. D16/full208.490 versus214.534us), but V16/sliding265.133 versus247.131, V16/full347.668 versus333.218, V32/sliding329.772 versus316.390, V32/full493.648 versus482.657us. These are different runs, not a causal paired experiment. They do not establish a shippable gain. Its current layer commit describes q1 bulk-reduction ticket savings of about0.3–0.5us; insufficient new evidence to reopen our closed QK/work-order families. Published composition history also records losing boundary folds; this is context, not an independent reproduction of our failures.

## Unchanged sources and screened tails

Existing measured comparator sources remain byte-pinned: keell0renz `8e30e1be994db4310794b8d04e90d18d895deb84`, credited chenwainuo `3d418e1625f6de2075b5684a27ddbc3d1442422d`, alygg77 `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`, vjeux `f35a41ee1c13e316dabbf772830d47b23b453393`, zhebrak `ff4d55ab669977f15874943044d20312d98a44d1`. No renewed device audit/check for these immutable files.

Additional tree/metadata screening covered infraax, GiulianoIntelligence, Aravind-11, KingCobra52, Saikat-too, lilaclilac09, ruwwww, Ranoobaba, tommyyliu, theluckystrike, xharshvardhan and Okohedeki. This was uneven depth, not full correctness review. Aravind README1.629 and xharshvardhan's four scores1.1991/1.3749/1.2990/1.2252 are weaker; two projection repositories contain starter-sized submissions. Some sources have no detected license. Inventory preserves all pins; no source is imported into production.

## Next action and stopping criteria

Stop this search round without spending a check. The next source-discovery trigger is an actually changed head or new artifact with exact source/receipt association and materially better affected-shape timings. Before any comparator run: inspect license, precision, synchronization and dispatcher scope, compute actual scoring contribution and preregister fresh-input/null/resource gates. Foreign standalone remains measurement-only; an attributed layer fork still requires its separate audit. This survey changes coverage confidence, not the measured frontier or feasibility cutoff.

Evidence: [search and pin ledger](evidence/2026-11-01-public-refresh-1830). Metadata and excerpts only; no foreign kernel bodies committed. API license=null is an absence of detected metadata, not a legal conclusion.
