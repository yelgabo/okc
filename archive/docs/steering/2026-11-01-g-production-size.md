# Whole-production G backend: transparent reduction checkpoint

2026-11-01. Root requested this work at main `4c7233d`, following
`2026-11-01-0802-g-progress.md`. Reviewer base `a79f9aa`. No GPU, scored run,
public action or production-file modification. **Partial packaging progress;
the combined production hybrid does not yet fit.**

## Exact input and reusable generator

Original incumbent G607 source is869076 bytes, SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`,
`muse_decode_layer/cand/g1b4_stage_release.py`, owning public job
`607eb52bdaf24160`. The frozen private968389-byte four-cell package is
unchanged. Its explicit partial preparation is not reused as the full G
production preparation.

`tools/dl_g_production_slice.py::build(original_source)` returns
`(source,manifest)`. It retains the complete original prepare/kernel wrapper
chain, all12 incumbent paths, startup self-test, selectors, packing/fallback
functions, source flags, native string values, rounding and scratch handling.
Versioned eager assignment dependencies preserve earlier functions captured
by wrapper aliases; late function globals resolve against final bindings.
Subscript-container mutation history remains attached to its dependencies.
The generator copies retained statements in original order and text, retaining
all inter-statement comments. This is a source reduction, not minification,
encoded content, opaque code or runtime source reads.

The sole branch simplification proves `GLE_T` has exactly one binding to an
empty dictionary, exactly two reads through `.get` in one short-circuit
condition, no alias or mutation. That condition is false for every input.
Its `_after_attention` wrapper retains the original unconditional parent
return; subsequent closure removes unreachable GLE O/route Python helpers.
The independent ACTIVE `GLE_FRT` pool/router path and all its native source
remain. Mutation/alias or nonempty-flag changes make this proof fail closed.
No selected native kernel, wait, barrier, launch policy or precision changes.

Unfactored result: **854399 bytes**, SHA256
`3bdbaed0ddbe8d8d1e4b63139d91292fea4b154baa47bcdd54363fdee4e63045`.
Manifest lists every omitted top-level definition and all large retained
literal hashes. All original large top-level source literals remain exact.
No inactive CUDA namespace was deleted without compiler proof; this pass
found no supported such deletion.

## Transparent shared CUDA text and byte budgets

The independent attention auditor owns `tools/dl_g_literal_factor.py`, used
separately as `factor(reduced_source)`. At this checkpoint the helper's raw
file SHA256 is `9b6fdccbe5dca08b1d7d3dea41f74eff46a652d1d8e887dec14ad124923a30af`.
It factors exact shared multiline text between main/helper MoE and two glue
strings into visibly named multiline constants plus concatenation. No
encoding/compression is used. Runtime source strings reconstruct exactly,
with outside AST unchanged. Evidence records common bytes, runtime hashes
and source sizes. The generator does not duplicate this independent helper.

Factored full-production G is **720525 bytes**, SHA256
`a6f97289c7abc8e49e7b7943e63f9b8138a8682988bb4c1b2603e3704fdc6464`.
Factoring saves133874 source bytes from the reduced backend. Main MoE source
still hashesc7c201ae; helper0b8ed8543; glue39c020a1/a2ce67a7. The quoted
hashes are fully recorded in `evidence/2026-11-01-g-production-size/`.

The owned worker's separately reported first actual reachability results
were390193 bytes for128-only,346441 for256-only,400126 forboth. These are
worker checkpoints, not independently certified final backends. At those
sizes, before dispatcher, source wrapping or new attribution overhead:

| Backend combination | Bytes | Excess over1MB |
|---|---:|---:|
| Factored full G + owned128 |1110718|110718|
| Factored full G + owned256 |1066966|66966|
| Factored full G + owned128/256 |1120651|120651|

Consequently none fits yet. Even the smallest above requires more than66KB
additional legitimate reduction plus dispatcher allowance. A private source
slice fitting under1MB is not evidence that a production hybrid fits.
Mask0 should remain the exact869076-byte incumbent, as root's dispatcher
design specifies. No nonzero mask is size-qualified by this report.

## Executed checks and limits

Five CPU tests pass (`python -m unittest discover -s tools -p
test_dl_g_production_slice.py -v`,2.719s): eager wrapper aliases/dynamic
globals/container mutation history; retained statement AST/order and every
large native literal identity; no additional unbound module globals;
constant-disabled GLE branch proof with mutation/alias negatives and actual
parent-return facades for all six token counts; pinned budgets and the full
mid-file license/NOTICE block. The source's NOTICE is between pure G and
appended self-tests, not just at EOF. A preliminary uncommitted version
omitted that comment-only block during statement joining; it was caught,
corrected and explicitly tested before this checkpoint. Earlier840235-byte
and848896-byte estimates are superseded, not valid shipping budgets.

No new local native compilation is required to compare identical runtime
source strings; no claim is made that hashing validates the original
kernels' broader safety. Full production preparation and all12 outputs
have not been executed on H200 with this generated backend. Compile time,
capture behavior, memory use, selection/timing drift and the external300s
prepare deadline remain unqualified.

## Next gate and stopping criterion

Root and the owned worker continue the combined byte gate. Any further
native branch/namespace removal needs exact selected-flag and compiler
equivalence evidence; dormant-looking paths reachable through original
prepare/tuners cannot be dropped from this whole backend by profile absence.
Do not replace original preparation with the private four-cell setup while
claiming all12 incumbent paths unchanged. If no honest combined source fits,
record the exact blocker rather than encode source or claim an unmeasured
hybrid is shippable. Runtime/correctness/benefit validation remains separate
after a reviewable fitting source exists.
