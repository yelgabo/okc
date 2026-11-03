# What local screening did and did not rule out — November 3, 2026

The user asked why prospects had been cut using RTX 5070 timings after learning those timings do not predict H200 performance. The records checked do **not** establish such a timing-based rejection. They do establish that some prospects were stopped before target measurement using local compiler resource gates. These are different evidence classes, and the latter can exclude a useful candidate without establishing that it is slow.

## Concrete decisions checked

| Prospect | Actual evidence used | What the evidence establishes |
|---|---|---|
| MAIN gamma ON `d6756bb5` | CPU-only NVRTC13/PTXAS13 **SM90a** cross-compilation; stack 8→16 B, spill-store/load totals 20/52→60/88 B; no GPU execution. [Owning screen](../../archive/docs/steering/2026-11-01-gamma13-main-screen.md). | Failed the registered no-worse resource screen. **H200 speed unmeasured**; target NVCC equivalence unresolved. This was not a 5070 timing loss. |
| Corrected-storage G2/24 `ed082519` | CPU-only secondary compiler screen: B4 208 B stack, 384/280 B spill totals; MAIN 224 B, 364/260 B. Also oversized Python and incomplete progress/target gates. [Decision](2026-11-02-g2-compiled-resource-decision.md). | Failed the old zero-stack/spill allocation gate; separate packaging/proof gaps remain. **H200 speed unmeasured**. Compiler totals do not prove executed traffic, target residency failure or net slowdown. |
| Later chunk-phased G2 `53d09576` | Separate target job `e1271ad5e5e34b24`. [Target decision](2026-11-02-captain-g224-terminal-decision.md). | Later implementation received an actual H200 test. Its descriptive negative result does not retroactively measure `ed082519`. |
| Short RMW | Target job `c5e4f89672f743bb`; frozen matched reader valid, promotion gate false, all 12 bank/cross gains subthreshold. [Owning terminal](../../archive/docs/steering/2026-11-01-rmw-legal-fixture-terminal.md). | Actual H200 matched evidence closed this exact RMW promotion. Not a 5070 result or rejection of every RMW design. |
| Fully timed SiLU table | Target job `dc62cff972934f54`, official passing 10910.347534710007 us / 2.073288676x. [Decision](2026-11-02-captain-timed-silu-terminal-decision.md). | Actual H200 single-arm evidence supplied no promising signal. Historical comparison is descriptive, **not isolated matched SiLU causality**. |
| Pre-gamma component | Target job `56e0c8b5634545e6`, strict matched component gate false. [Owning report](2026-11-02-worker-moe-component-terminal.md). | Actual H200 matched component evidence, not a 5070 timing loss. |
| Earlier sparse/stage-XOR RTX helpers | [Stage review](../../archive/docs/steering/2026-11-01-sk-stage-repair-independent.md), [sparse review](../../archive/docs/steering/2026-11-01-sk-sparse-repair-independent.md). | Local integer reconstruction/shared-helper correctness checks. Their declared scope excludes latency, complete projection performance and H200 qualification. |

The three raw target receipts for RMW, timed SiLU and pre-gamma were reread and their exact IDs, boards, done/pass status, return code and score retained in the audit evidence. The historical campaign also explicitly states that Windows RTX 5070 is not used as a performance proxy. These findings are scoped to inspected steering/history/run/ruled-out records, not every prior chat, unpublished file or candidate.

## Accountability and correction

There was an aggressive engineering filter: extra compiler spills could end an allocation before an H200 test. That saves implementation/measurement budget but is a prioritization choice, not a measured performance conclusion. Even SM90a cross-compilation on the local CPU has frontend/toolchain differences; its spill counts do not establish actual H200 time. The old absolute-zero gate was especially weak when the control itself spilled; the later G224 reference decision explicitly reconsidered that gate before target testing.

The local-only gamma and corrected-storage G2 entries must remain **unmeasured/deferred by an engineering screen**, excluded from claims of measured H200 losses, family ceilings or exhausted optimization potential. Their original allocation stops and independent source/packaging/proof gaps are preserved. This correction grants no new experiment, retry, public action, threshold change or ownership transfer.

The new [RTX 5070 profile](2026-11-03-worker-local-incumbent-profile.md), publication `ba48cd8f`, is an actual local baseline of two portable selected stages plus explicitly selected projection fallbacks. It supplies no reason to reject an H200 candidate. Local timing can guide a hypothesis; target performance decisions require appropriate target evidence. Source-correctness, legality and proven target resource limits are separate questions.

## Validation and publication

Read-only audit; no new compiler invocation, GPU execution or judge registration/poll. Current short-load allocation and other owners' jobs remain unchanged. Fresh all-five public snapshots and the promotion audit are retained with this report; no official receipt or release candidate was created. README/index remains with the coordinator for handoff.

[Scoped source search and exact receipt checks](evidence/2026-11-03-worker-local-screen-audit/source-audit.json); [promotion audit](evidence/2026-11-03-worker-local-screen-audit/release-audit.json); [validation](evidence/2026-11-03-worker-local-screen-audit/validation.json). Fresh promotion check: {'promotion_gaps': [], 'leads': 16}.
