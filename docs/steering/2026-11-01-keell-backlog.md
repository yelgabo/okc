# Which pre-existing tasks covered keell0renz's approach?

November 1 retrospective at evidence commit 230e796. The historical cutoff is dba82c6, before the keell0renz comparator entered the top-50 campaign. The [visual audit](2026-11-01-keell-architecture.html) describes the subsequent comparison.

| Earlier task | Coverage before the comparator |
|---|---|
| Family F, exact donor then hybrid | Direct match to the evaluation method. On October 30 we explicitly scheduled complete-file comparisons including packing, preparation and dispatch. The named donor was alygg77. This task gave us a way to discover a stronger complete implementation. |
| Two-phase stream-K and designated finishers | Already implemented and tested. In check 2a4fb39964aa448c our mode 79 split both MoE phases and designated the phase-B finisher. Against mode 51 it cost 16.4, 19.4 and 21.4 µs on layer routing at 32, 128 and 256 tokens. The earlier donor arm beat 51 in those same cells. Parking this owned variant was justified. |
| Family D, ready-job scheduling | Partial match. We identified the software grid barrier after phase B and proposed scheduling ready work. Their MFIN makes a different scheduling decision. Each token can enter final aggregation once its contributing local experts publish completion. A split tile's finisher and a token's final aggregation have different dependencies. |
| Lossless dense compression and per-shape dispatch | Direct overlap on BF16 projections, including large-shape library selection. Our denser BF16 codec proposals targeted projection weights. Their SC6 packs the existing NVFP4 expert block scales losslessly from eight to six bits. I found no explicit pre-comparator task specifying this format and guard or the MFIN × SC6 experiment in the reviewed plans. |
| Attention producer/consumer pipeline, Family E | Shared architecture and already tested. The later audit also identified different verify dispatch and an active diagnostic approximation in the external standalone file. |

The earlier plans identified the right families and even tested one close scheduling analogue. Their biggest omission was a concrete experiment isolating completion at the final aggregation boundary and compression of expert scale bytes. The earlier campaign review also identified disproportionate effort on small isolated gains. The external exact-layer check 3a5f879df0d44825 establishes a performance target. Its MoE comparison changes MFIN and SC6 together, so it cannot assign either feature its own share of the 11.17% historical layer score gap.

The current queue now explicitly calls for the four combinations of MFIN and SC6 on matched inputs with independently prepared controls. Validate output after replay, prove publication and scratch reuse, measure both the MoE and all six layer loads, and stop if the difference fails the duplicate-noise or whole-score gate. Mode 79 alone remains closed. Current fork repairs through 230e796 are separate from experimental attribution. No new GPU run was needed for this retrospective.

Historical sources: [October 29 families A–E](../architecture-families-2026-10-29.md), [October 30 family F registration](2026-10-30-family-f-donor-baseline.md), [projection portfolio](2026-10-30-projection-top100-portfolio.md), [mode 79 run log](../runs-moe.md), [campaign accountability](2026-11-01-campaign-accountability.md).
