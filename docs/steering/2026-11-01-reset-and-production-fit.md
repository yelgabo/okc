# Reset execution repaired; production byte fit under final qualification

November1,2026, continuation from main `65f8810`. The preceding goal turn was
progress: public H200 `dfcfdd01d6f04525` completed and closed the second losing
sparse-repair representation. The top10-50 goal is active and unmet. This turn
has dispatched no GPU job or scored submission. Prefill remains deferred.

## Concrete reset blocker found and fixed

Integrated `d58b6ac` (owner `fc77054`) repairs the shared `_wl_clone` helper
in `tools/dl_g1_layer_owners.py`. The old code rejects actual builtin sets
created by normal G preparation: `compose_active` and nested `bfp.disabled`.
Both complete diagnostic prepare flows reproduce that failure with the old
helper. This invalidates all three old queue byte pins, including the later
four-cell comparison that imports the same clone helper.

The narrow fix creates a fresh memoized set and accepts only exact builtin
scalar/tuple policy members. Tensor members, including hashable tensors in
tuples, opaque owners, frozensets and subclasses still reject. Repeated set
aliases are preserved within one owner; different arms own different sets.
Tensor alias reconstruction, immutable packed weights/code, Lt workspace
cloning, diagnostics, native code, precision, fences and score gates are
unchanged. Root reviewed the source delta and reran all six new TorchCPU
tests successfully (0.624s), with the raw transcript retained.

Those tests execute the complete real `_wl_prepare` and `_sp_prepare` host
control flow, read-only/clone/owner checks, report transport and TF32/cache
finally handling. Stage completes36rows/144normal+144fresh validations/12empty
calls; short completes48rows/96checks. Native/reference arithmetic, graph
capture/replay, profiler/events and device memory are explicit CPU substitutes.
They catch the observed host defect; they are not H200 preparation or math
qualification. The source owner census is an assignment inventory, not a
captured real-device state dump.

Root regenerated every payload twice from integrated tracked dependencies
and required exact byte equality against the revised pins:

|Private diagnostic|New bytes|SHA256|
|---|---:|---|
|Full six-load G stage|907822|aa49352105b39538b79c454f49e9f8e2014fa99b5c8205f145131169f18358e4|
|Short fixed-natural-routing|904020|4a1a68f5824bd51c130e03eb736a2e1fb19e011735169400b0be2c0d17448183|
|Later four-cell verify|969272|195bba34415dd614733100cc4e26022fd25891fad747ee393aaac9ab4c8d4fa0|

Canonical new manifest: `muse_decode_layer/profiles/reset-ready.2026-11-01/manifest.json`.
The old906939/903137/968389-byte files must not be dispatched. Each new file
adds exactly883 helper bytes; no device source changed. The old manifests
and reports remain historical evidence. The existing one-shot heartbeat
`qualify-family-g-at-the-november-1-h200-reset` was updated through the app
tool and read back from its saved configuration: ACTIVE, same16:00PST
(November2 00:00UTC) schedule and original chat, first-two order unchanged.
No duplicate automation or new job was created. Its prompt now requires the
new tracked manifest and explicitly rejects superseded payloads.

The real full-stage guard is300seconds at case boundaries; short/four-cell
guards are295seconds. A guard cannot interrupt an in-flight native compiler
or long region. Combined H200 setup/capture/transport still needs measurement;
the repair is not permission to weaken coverage or retry a timeout blindly.
[Detailed readiness audit](2026-11-01-reset-host-readiness.md).

## New byte plan and qualification in progress

The previous production hybrid exceeds1MB in every nonzero mask, as recorded
in `0879082` and its sixteen-mask inventory. A fresh constexpr-body budget
found only34884 removable raw bytes, insufficient for its minimum72105-byte
gap. That does not justify repeating the old pruning matrix.

A distinct representation now has a positive exact screen: remove ordinary
Python/native comments while retaining source text, whitespace/newlines,
licenses, notices, authored credits and generation markers, then apply the
existing readable exact-string factoring. No encoded/compressed source,
blank-line/docstring deletion, active branch pruning, tuner change or
synchronization/precision change is part of this plan. Preliminary physical
files fit, but their counts and hashes are not frozen at this checkpoint.

Two genuine review corrections prevent premature clearance. The first
keyword-only notice rule missed a handle-only `bet4_claude` credit; the
retention rule was expanded and the old notice conclusion narrowed. More
importantly, Python assembles native host dispatch by replacing comment
markers such as `//@MOE1_DISPATCH@`. Removing these is a functional change
even if raw device tokens/SASS are identical. The owner is preserving marker
and source-transformation dependencies and adding actual build-source
assembly comparisons before final freeze. Earlier in-memory byte pins are
superseded, not candidates for dispatch.

Independent source audit `9b46cf9` (owner `18feac4`) and corrected owned
compiler evidence `22711bf` (owner `21ef9f3`) are integrated. For the reviewed
owned source epoch, ONE L8 modes3/51 and MoE51 normalized SASS/opcodes/operands,
barrier/fence counts and resources match original sources exactly. All three
local compiler outputs show zero stack/spills. This is selected local sm90a
compilation, not all-entry compilation, runtime admission, host-generation
clearance or H200 timing. Final G/native and complete wrapper gates remain.

## Priorities, stop rules and retained candidate

1. Finish and independently review the marker-preserving production file,
   exact source budget, actual generated build-source equivalence, all12
   dispatcher/fallback cases and selected native compiler comparisons. Stop
   for a lost credit/marker, token or code-generation mismatch, cap failure
   or changed active math/ordering. Source fit alone earns no score claim.
2. At the existing reset, run the revised full-stage private job then the
   revised short fixed-natural-routing diagnostic, following own job IDs.
   Keep their original full-score/null/regression/correctness gates. No
   queue reordering or duplicate dispatch.
3. Retain revised four-cell private comparison behind those jobs. A fitting
   production hybrid would remove its deployment obstacle, but it still
   needs actual H200 source/prepare/math/capture/admission and useful paired
   performance evidence, then full-six-load candidate qualification. Do not
   submit the private comparator or turn two-load sensitivity into a score.

The fresh owner-independent source survey `2220a34` (owner `ca020e9`) covers
all four identifiers beyond200code-search results and24pinned repository
trees. Newly inspected receipts are weaker and existing strong source heads
are unchanged; no additional comparator GPU check is justified by this survey.
It is bounded discovery, not proof that no stronger code exists.
[Survey and limits](2026-11-01-public-source-refresh-1830.md).

Known-good public sources remain layer607eb52bdaf24160 at2.126082676x,
projection C2/9eba96894ee1402e at1.527806483x, attention8914751 and MoEabe7651.
Neither projection repair is reopened. Top10/top50 feasibility remains
unsubstantiated; the new work removes concrete execution/packaging obstacles
and supplies no projected or measured speedup.

Root evidence/transcripts: `evidence/2026-11-01-reset-operational-root/`.

## Subsequent eligibility finding: hold the four-cell comparison

The first two G diagnostics remain source-ready at their revised pins. The
969272-byte four-cell file is now explicitly **HOLD**, not dispatch eligible.
Its source construction is repaired, but independent audits found a separate
pre-existing owned-MoE51 progress-contract gap; see
[the exact call chain](2026-11-01-owned-verify-progress-addendum.md) and
[primary-contract audit](2026-11-01-owned-moe-progress-contract.md).

The owned entry passes PDLTrue to a launcher with ordinary triple-chevron or
PDL-only launch attributes. Its software grid and cross-CTA waits assume
resident progress, while bounded timeout sets a diagnostic flag and advances.
An SM-count-sized ordinary grid does not supply a documented co-residency
guarantee. NVIDIA's CUDA12.8.1 [grid synchronization contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#grid-synchronization-cg)
uses cooperative launch plus admission; its
[PDL contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)
does not guarantee concurrent execution. This identifies a missing proof,
not a newly measured hang. Comment packing did not introduce the problem.

A separate repair is underway, reusing the existing checked cooperative
launcher with PDLFalse and reviewing fail-closed timeout handling. It must
preserve all math/fences, prove the active wait dependencies, pass host and
local compiler/resource gates, and receive new exact private/production pins.
No old four-cell or packed-file hash authorizes a GPU run. The canonical
reset manifest and the same ACTIVE one-shot reminder now explicitly hold
that third diagnostic until its repair is reviewed; no new timer or
first-two ordering change was made.

## 19:34 UTC addendum: source fit and separate progress repair cleared locally

Evidence checkpoint `4493fec` integrates the final pure-comment packager
(`b0e46c4`), actual packed factory/dispatcher review (`f51ac3a`), separate
progress repair (`5e59e2f`) and its independent review (`4493fec`). Root
regenerated the pure and repaired files from these tracked dependencies.
Earlier preliminary comment-fit pins are superseded. Final pure mask4 is
907642 B / `3b205d408edd8a238dec686fac306b5a28ac9136b42512939ae80db7c79611ec`;
pure mask15 is951539 B /
`d3f2b31ea0991e8b7e14709913e8a55d1304fe20fc47fd369e8a5f1c0334d689`.
Notices, author credits, both dispatch markers and runtime string anchors
are retained. Actual native source factories and selected sm90a compilation
clear the scoped equivalence gates; existing G spills remain, rather than
being misreported as zero. The old all-nonzero-NOFIT conclusion applies to
the older representation and is superseded by this distinct comment-only
representation. No active math, fence or tuning was removed to fit.

The separate repair adds770 bytes: full four-cell core952309 B /
`81f056f5e495f5cab0a66183a5b56cab783e7fa47732ed58e0ffa4feed5127a1`;
mask4 core908412 B /
`d82f3c9d105a4e66ea338767bb67e6116cb79bda1726db54b12cdf651e6f8ce6`.
It uses the exact existing checked cooperative launcher, PDLFalse and a
fatal trap after the retained timeout diagnostic. All other native bytes
and Python behavior reverse exactly to the preceding source. Independent
[progress review](2026-11-01-owned-progress-repair-review.md) found no scoped
blocker in the active readiness dependencies. Local selected MoE51 remains
96 registers, zero stack/spills and16 barriers; the added cold trap sites
are a real code change, with no measured latency implication. Neither this
review nor cooperative admission proves every retained kernel invariant.

Root's fresh12-test WSL run passed in5.490s: comment lexer/notice/marker and
frozen-manifest gates, exact composition/AST scope and actual reused C++
launcher success/rejection/fatal-timeout mocks. Native GPU arithmetic is
not executed by these tests. Raw root regeneration, composition pins and
test transcript are retained in the evidence directory linked above.

Next priority is a **single fixed-mask15 production measurement**, using
the attributed G backend plus our owned four verify cells. A prepare-only
qualifier is being built around the unchanged timed dispatcher. It must
use a separately prepared private owned state for startup checks: shallow
or deep copying Python functions cannot isolate Lt/WD workspace closures.
All four cells must run through the actual public owned route on normal,
changed-input and graph replay checks with unchanged reference bounds,
cache/routing checks, guard/resource admission and295s total budget.
No private comparison, alternative timing arms, tuner or data-dependent
public selector is eligible for a scored slot. Resource, numerical,
capture, deadline or provenance failure stops dispatch. Source fit alone
does not clear these gates.

Under the user's standing permission to measure with expiring scored slots,
root may use one layer slot before reset only after exact committed source,
CPU orchestration and independent review clear. Such a result is a
single-arm whole-layer observation, not a paired causal speedup. Require
all13 official tests,12 benchmark cells, preparation record and resource/
startup coverage. Reconstruct the six-load46/16 score against its own
baseline, compare all12 raw/adjusted cells with607e, retain the incumbent
on a loss and close the exact tested policy without post-hoc subset salvage.

The private four-cell comparator also has a regenerated repaired core:
970042 B / `094841bd26822392d9bcabbe80297dbd84df10a8f29a9749edc0a9dd143b8f3a`.
This is **not a queued replacement** yet; its canonical manifest remains
HOLD until root finishes the new runtime-harness/source-pin review. The
first two reset diagnostics and the existing one-shot schedule stay in
place. No job was dispatched by this checkpoint, and no score/rank gain
or top10/top50 feasibility is established.

## Subsequent private regeneration check

`reset_payload_regenerate.py --owned-progress` now explicitly composes the
reviewed progress repair after building the private four-cell source. Its
default path still reproduces historical unpatched packages. Root double
regeneration yields the same stage907822/short904020 pins and new verify
970042/094841bd pin above, and records the composition plus normalized
verify-generator dependency hashes. The option does not dispatch a job,
lift the canonical HOLD or claim runtime qualification.

The old slice test initially failed on its historical single-cell byte pin:
the shared set-owner fix adds883 bytes there too. Root verified that
reversing only the embedded owner helper exactly restores850769 B /
305d8e28; its current tracked helper produces851652 B /b8dcb1a9. Updated
only that stale fixture pin. The seven slice/package/owner/transport tests
then pass in7.836s. Both the initial failure and corrected transcript are
retained. This is a tooling correction, not a new kernel experiment or
reason to weaken the queue's runtime gates.

Fresh19:38-named CLI snapshots show no live jobs, layer0 checks/2 scored
slots, projection0/4, MoE0/8 and attention3/7. Re-query before actual
dispatch because quotas are shared. The WSL CLI is `/usr/local/bin/okc`;
the legacy coordinator's `~/.okc/bin/okc` path is not installed here.

The owning-receipt reader `tools/hybrid_layer_measurement.py` is prepared
for the proposed one-arm run. Its three CPU tests reconstruct archived
607e from all12 records and reject wrong job/board/mode, missing records,
bad values/weights/flags, failed correctness or a mismatched whole score.
Startup parsing requires48 model checks/8 captures/24 changed replays,
40 empty checks/8 empty captures/32 restored occupied checks, all3 actual
resource vectors with zero local memory and complete cooperation/residency.
It reports all12 raw and adjusted medians plus six46/16 weighted loads,
and explicitly returns no paired attribution or release decision. Synthetic
negative fixtures are parser tests, not new GPU measurements.
