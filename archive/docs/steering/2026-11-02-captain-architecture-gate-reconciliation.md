# Captain architecture-gate reconciliation

November 2, 2026, 19:40 UTC. Reviewed checkout/reference commit
`f6fde7ad6e9b3d443921d8c21a5b1a43f5988b32`; P1 architecture evidence
`c8aacece`, attention reconciliation `137a65a9`, human priority `588847c9`,
and P2 allocation `b395d49b`. This is source/coordination work only.
No compilation, GPU/Driver, challenge CLI, POST, public or default change.

## Decision

Retain P1's **NO ADEQUATE PATH FOUND** for the two proposed complete
expert dataflows. Neither is a performance-qualified candidate. Preserve the
human architecture-first pivot; do not fund the frozen attention boundary,
another wrapper-only Humming retry, or a new readiness harness. Public incumbent
`607eb52bdaf24160` is unchanged. This report does not refresh ranks or quotas.

One stronger blocker was found: prepared BF16 expansion is inconsistent with
the pinned repository's weight-storage contract, regardless of exact operand
rounding. The performance argument also needs its existing routing caveat
preserved. These distinctions change how future designs are screened.

## Independent source checks

### Prepared expansion fails the repository contract first

`muse_decode_layer/TASK.md:102` and `muse_moe_w4a16/TASK.md:76` require
NVFP4 weights to remain NVFP4, or smaller losslessly, and explicitly allow
dequantization inside kernels. P1 design B instead materializes persistent
BF16 operand weights during preparation and streams them in the timed engine.
Exact decoded BF16 values do not satisfy the separate storage requirement.
No code was built; do not request a performance check of that design under
this contract. This is a repository-contract finding, not a claim that today's
served rules were independently fetched or an organizer ruling was obtained.
Reopen only on authoritative permission for this precise prepared representation,
or a materially different representation that actually preserves the contract.

Pinned Git blobs at the reviewed commit:

- Layer TASK: `fcf3c2e9c317578fe1b9edee0489d73e9aa85bfe`.
- MoE TASK: `1755b9ecebe4f52825244681682b68822af97b97`.
- MoE reference: `b937352204a206cc2de669d3aa36a014755afe86`.
- Layer reference: `f92aa2db6ba064bc686ed5369f979cfe00d4ff44`.

### Uniform-route arithmetic is not the scored workload

Read the full P1 report and its `analyze.py`, not only the summary. P1
explicitly labels its independent uniform top8 assumption, optimistic one-pass
weight stream, conditional HBM interpretation and unknown achieved bandwidth.
There is no undisclosed measured occupancy claim to correct.

However, the actual MoE generator adds heavy-tailed popularity and rotates the
busiest shard into the local expert range (`reference.py:53` onward). The layer
reference also accounts for busiest load and distinct experts. Therefore the
uniform model's expected local routes/active experts cannot be substituted for
actual scored counts, nor can its conditional stream time prove a universal
runtime impossibility. Keep the logical per-element storage expansion fact
separate from physical bytes read and exposed endpoint latency. The stronger
storage-contract objection above does not require that workload assumption.

### Current Humming remains a distinct but unpriced alternative

P1 pins current `948e6fd547e703065be51529ac1db5569256774e` versus earlier
installed `636ba85648c30ae2c2bfb335c9399593a67ecc1d`. Its complete source review
identifies real engine changes. Historical losses of a different configuration
do not close this newer engine family. Conversely, changed code and generic
foreign benchmark gains do not establish Muse whole-operator savings.
Retain the missing substantial net-layer path as the allocation blocker, not
a claim of measured loss or equivalence to the old engine.

## Attention handoff accepted without reopening

Index the completed worker report
`2026-11-02-worker-attention-continuity-reconciliation.md` at `137a65a9`.
Worker retained canonical source and demonstrated final dispatcher selection
under CPU facades, with an exact explicit-entry guard negative: B33 can reach
the continuity module through an explicitly supplied module because flat launch
geometry alone does not reject that shape. The default final selector chooses
control. This is neither an observed GPU numerical fault nor a default leak.
Preserve `direct_entry_rejected=false`; no repair, validation extension, harness
or target allocation follows under the human low-ROI HOLD. Duplicate draft
remains NOT_SELECTED, not an extra candidate.

## Next owner and decision gate

Existing researcher P2 was delivered through Queue and ordinal 27 started
at 19:33:55 UTC; thread read at this reconciliation reports it running. No final
P2 result has been received or inferred. Its original 25-active-minute scope
remains: independently challenge complete semantics, eliminated work and
substantial layer economics; return ACCEPT FOR ONE SPECIFIED NEXT GATE or
REJECT WITH COUNTEREXAMPLE / MISSING SUBSTANTIAL PATH. No duplicate task or
acknowledgment cascade is necessary.

Captain's next decision consumes that actual result. A proposed next engine
must identify a legal complete callable, actual selected workloads and routing,
required numerical/ordering boundaries, replacement/integration costs and a
credible several-percent first contribution within a top100-capable broader
path. If those are still missing, reject that proposal explicitly rather than
turning speculative source into a target harness. Prior exact closed variants
remain closed; a scoped STOP does not declare the whole search exhausted.

## Validation and gaps

Inspected README, latest human pivot, completed P1 report/analysis, P2 allocation,
local TASK contracts, routing sources and attention handoff; read researcher
thread status. No independent reproduction of P1 arithmetic or CUDA resources
is claimed in this task. No current rank, served-rule refresh, physical traffic,
target timing, numerical qualification or new candidate was obtained. The
unrelated shared WIP is preserved. Publication includes this report and the
captain-owned index only.
