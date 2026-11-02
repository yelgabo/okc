# G2/24 compiled resource decision — November 2, 2026

## Decision and objective

STOP the exact corrected-storage G2/24 implementation under its registered zero-stack/zero-spill resource gate. Do not package, sweep registers/rings/shapes, run a target check or change production. This is not a measured slowdown or a rejection of all two-CTA designs. Incumbent public607eb52bdaf24160 remains unchanged; zero new performance-qualified winners. The last rank snapshot is 07:14UTC in [the objective checkpoint](2026-11-01-captain-rank0714-next-gate.md), not a newly refreshed rank.

## Independently verified evidence

Worker publication 04e08381919c1c1be9b54326f80101bbd71418c8: all28 changed path blobs match authenticated fetched origin/main; all27 manifest entries match canonical Git bytes, SHA256, size and blob identity. [Verification receipt](evidence/2026-11-01-captain-g2-compile-review/verification.json). [Owning report](2026-11-01-worker-g2-24-route-prefix.md), including its final resource addendum, supersedes historical tooling STOP746f937 without erasing it.

Actual generated CUDA SHA256 ed082519a4312bc1012192134b2d2865016789802b0454381984260c7aa5ca6f. Fixed secondary NVRTC13/PTXAS13, not target NVCC or runtime:

| Selection | Registers | Barriers | Stack bytes | Spill stores / loads bytes |
| --- | ---: | ---: | ---: | ---: |
| B4 false and true | 96 | 16 | 208 | 384 / 280 |
| MAIN false and true | 96 | 16 | 224 | 364 / 260 |

Both complete selected families were compiled; B4 failure was not extrapolated to MAIN. These compiler totals are not executed traffic or timing. Both PTX/cubin identities are pinned in the worker manifests. Independent CPU/source regression run:9/9PASS,8.644s. Those tests do not establish full progress, numerical correctness or target admission.

## What changed the decision

The earlier unsupported NVRTC offsetof assertions were an instrument limitation, not an algorithm failure. The narrowly attested compiler-only adapter preserves the actual generated/native assertions and emits five same-Smem layout probes. Both selected families pass wc1248, wcp1248, wcpB2272, sizeof(wc)1024 and sizeof(Smem)23096. This reaches actual resource evidence rather than stopping on a broken witness. The shared layout correction saves640B/CTA; shared fit alone is insufficient.

The captain's [pinned header-equation model](evidence/2026-11-01-captain-g2-compile-review/registers.py) corrects the naive102-register ceiling to96 under its explicit320-thread/65536-register/256-register-per-warp-granularity/four-subpartition assumptions.97..102 round to3328registers/warp and model only one block, versus3072 and two blocks at96. This is conditional CPU accounting, not an authoritative target occupancy query. Raising the cap is therefore not an established rescue.

## Remaining gaps and ownership

Full event/progress/ring-reuse proof remains incomplete. Host NVCC build, numerical/replay qualification, actual reservation and occupancy, and timing are NOT_RUN. Generated Python is1,023,788bytes and fails the archived upload-budget check; no packaging allocation. The cubin shared-section1024bytes has no established interpretation as additional user-static storage versus system reservation here; do not double-count it or declare target residency.

Worker implementation allocation is complete. The existing researcher has a separate bounded20-minute read-only attribution allocation started08:00UTC: map existing spill PCs to source/control flow, distinguish gather/setup from repeated expert-loop traffic, and return one pressure hypothesis only if supported. At this decision the research is still running; preliminary notes are not adopted as final evidence. No compilation, rescue, gate relaxation or GPU authorization follows from that task. No new threads or subagents.

## Priorities and next decision

1. Preserve the exact failed implementation and incumbent; avoid another expensive qualification harness for an unready candidate.
2. Use the already-running attribution to decide whether a concrete pressure-reduction mechanism merits a separately bounded implementation, or whether to retire this branch and allocate the strongest materially different opportunity. Do not infer that spilling necessarily makes it slower.
3. Any reopening requires an explicit new mechanism, selected-path coverage, unchanged numerical/publication/lifetime obligations, actual resource evidence and a complete route to numerical/runtime qualification. No blind register/ring/shape sweep. Full-family adoption and the narrow scalar option remain alternatives, not qualified winners; compare expected weighted benefit and engineering cost before choosing.
4. Target dispatch, scored/public submission and default changes remain unauthorized. No quota was consumed by this review. The objective is unachieved; this engineering STOP is not campaign success.

