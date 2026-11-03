# New NVFP4 engine lead: inspect the selected hardware path first

November 2, 2026, source intake 19:41-19:50 UTC. Starting captain evidence
`3f8ed40e`; incumbent public607 and freshcc474 remain unchanged. Previous goal
turn was progress: published the weight-contract reconciliation. This turn
independently follows a newly found complete-engine integration rather than
repeating unchanged challenge-repository inventories. No GPU allocation.

## Concrete decision

**Do not allocate a SonicMoE/NVFP4 wrapper or target harness from this lead.**
The named native low-precision path is Blackwell W4A4, not Hopper W4A16. The
inspected Hopper-compatible SonicMoE path expands weights to dense for the
callable; the separate packed Hopper path uses Marlin. Neither supplies a new
complete Muse operator with measured substantial layer savings. There is also
an exact intermediate-rounding difference in the generic gated helper, proven
by an independent finite CPU witness. This is a source-screen result, not a
measured loss or a claim that SonicMoE/Marlin cannot be adapted.

## Discovery, pins and reproducibility

Independent primary searches: `FLUTE lookup table quantized matrix multiplication
Hopper GPU github` and `SonicMoE Hopper grouped GEMM github NVFP4`. The second
led to Axolotl's NVFP4 documentation, followed to the immutable repository
`axolotl-ai-cloud/axolotl@58deaee13ce0625e0b431ece379b91c7715b8aae`
(head author timestamp 2026-11-02T17:49:26Z). The initial SonicMoE upstream
metadata pin was `Dao-AILab/sonic-moe@7396f3e604827d8186c2e16e64b28ee33d3defd0`;
no claim of a full upstream source audit or challenge-result association.
FLUTE remains an unreviewed search lead, not a candidate or negative finding.

[intake.py](evidence/2026-11-02-captain-nvfp4-engine-intake/intake.py) downloads
nine selected complete text files through `gh api`, verifies their Git blob
identity, records length/SHA256 and writes third-party bytes only to ignored
`.build/captain-nvfp4-engine-intake`.
[intake.json](evidence/2026-11-02-captain-nvfp4-engine-intake/intake.json) preserves
exact endpoints and hashes. No foreign source is imported or executed. This is
selected-path coverage, not exhaustive repository coverage or license/adoption
clearance. Shell-runtime authenticated retrieval succeeded; the Node tool's
separate gh environment lacked authentication, so its failures are not404s or
absence evidence. An initial Windows wildcard rg error was corrected by using
the directory plus filename filtering; it is not a prior-art search miss.

Primary immutable files are under
`https://github.com/axolotl-ai-cloud/axolotl/blob/58deaee13ce0625e0b431ece379b91c7715b8aae/`:
`docs/nvfp4_lora.qmd`, `src/axolotl/integrations/kernels/libs/sonicmoe/`
`{nvfp4.py,nvfp4_lora.py,nvfp4_quant.py,triton_nvfp4.py,fp4_cute.py}`, the
`scattermoe_lora/marlin_w4a16/{backend.py,prep.py}` pair, and the NVFP4 adapter.
No score is transferred from a project name, paper, hardware table or training
benchmark. This library is not established as an OKC competitor submission.

## Actual complete-path distinctions

1. Documentation says SonicMoE W4A4 native with Hopper W4A16 fallback. Read the
   implementation rather than assigning the native path's properties to both.
   `fp4_cute.py:52-62` permits device major10 or11; its grouped path quantizes
   activations to NVFP4. H200 is not selected by that guard, and activation
   quantization would not preserve Muse's BF16-activation contract anyway.
2. `sonicmoe/nvfp4.py:214-252` dispatches `dequant` to dense weight expansion
   before `torch._grouped_mm` or per-expert `F.linear`. The inspected
   `nvfp4_lora.py:885-889` explicitly dequantizes both base banks and changes
   backend to `torch`. Doing this inside the callable is distinct from the
   prohibited prepared BF16 bank in P1 design B; no blanket illegality is
   assigned. It adds dense materialization, not a demonstrated fused NVFP4
   decoding engine improvement. No physical-traffic/timing claim is made.
3. The separate ScatterMoE Hopper-capable source is Marlin W4A16. Its own
   `MARLIN_TILE=64` and compact/pre-scattered metadata are real configuration
   details, not source-linked proof of faster Muse endpoints. Our baseline
   already uses Marlin, and an older separate W4/Marlin treatment has an owned
   negative c4b1b5366bc54857. Shared family is not proof of identical source,
   selected tiles or speed, so this newer source is not declared measured-lost.
4. The inspected grouped forward performs G1, gated activation, G2, then route
   combination; it does not implement Muse's per-expert full-output norm before
   weighting. `combine_expert_outputs` scales grouped rows and combines them.
   Adding that norm, correct gamma/rounding and scratch/replay/integration would
   be required for a complete comparator; do not treat its two GEMMs as one.

## Independent boundary counterexample

For the generic `concat=True`, no-limit gated helper,
`sonicmoe/nvfp4.py:180-211` computes FP32 SiLU, casts that result to gate dtype,
then multiplies by BF16 up. Our selected expert boundary keeps the SiLU/product
in FP32 until the BF16 intermediate store. These are different operations.

[activation_witness.py](evidence/2026-11-02-captain-nvfp4-engine-intake/activation_witness.py)
independently transcribes these scalar boundaries; it does not import the donor.
Executed using the existing WSL Torch2.10.0+cpu environment.
[Owning output](evidence/2026-11-02-captain-nvfp4-engine-intake/activation-witness.json)
records a finite BF16 gate -7.9375 and up0.75. The FP32-product/single-round
result is -0.0021209716796875; the helper's extra intermediate round produces
-0.00213623046875. Our explicit gate/(1+exp(-gate))*up transcription agrees
with the single-round SiLU-product here, isolating the extra boundary rather
than attributing it to the alternative FP32 SiLU spelling. The finite sampled
scan is not exhaustive, a whole-operator numerical rejection or GPU execution.
Do not adopt this helper as boundary-identical. Fixing that scalar difference
alone would not establish ROI or authorize integration.

An initially suspicious Marlin scale clamp is NOT used as another rejection:
the local generated wide-scale range begins at2^-6, which reaches the source's
threshold after its128 factor when scale_factor=1; the adaptive factor does not
reduce it. No generated-domain counterexample to that clamp was established.

## Next gate and gaps

No kernel/harness/compile/Driver/challenge CLI/POST/install/default/public action.
No numerical/performance qualification and no new shippable candidate. Preserve
the source lead and witness, not a speculative integration backlog assignment.
Reopen on a concrete Hopper BF16/NVFP4 complete callable with compatible
boundaries and a source-associated same-format/shaped cold result supporting
the substantial net-layer budget, or a materially different legal engine whose
removed work and replacement costs can actually be priced. A new package name
or tile setting alone does not meet that gate.

P2 completed and published `ba0e1ff94491b1362c96391fb5a0aeafdab54110` during
this intake: `2026-11-02-research-broad-expert-feasibility.md`. Captain read its
full report and source-only analysis, and directly confirmed that exact remote
main head. Adopt its **REJECT WITH STORED-WEIGHT RULE CONTRADICTION / MISSING
SUBSTANTIAL PATH**. P2 adds an important route-specific correction: MID uses
up-to8-token jobs, WGS uses up-to64-token jobs, while alternative grouped tiles
also repeat work; neither architecture has one universal pass per expert.
Its source/CPU sensitivity rows are requirements with assumed added costs, not
measured speed or exposed savings. No independent rerun of P2's script is
claimed here; our separately executed witness answers a different question.
P2 is no longer a pending-result reason to wait. Both reviewed P1 proposals
and this new source lead fail the current allocation gate; none is promoted
into implementation. The next search must supply the concrete complete-path
evidence described above, not repeat these three unchanged sources or fund
another tiny boundary check. This intake does not reopen frozen attention or
generate another acknowledgment task.
