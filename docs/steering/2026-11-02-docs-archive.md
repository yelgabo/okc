# Historical documentation archive

Moved 1,229 records, 76,428,199 original bytes, from `docs/` to [archive/docs](../../archive/docs/). The [manifest](../../archive/manifest-2026-11-02.json) pins every original path and SHA256 against evidence commit `9f8fa34a991f63a26f53cccefbeade39b045f353`, plus final hashes and citation repairs. This task started no judge jobs and changed no current kernel source.

## Scope and decision

The cutoff is the first calendar date in a repository path, strictly before `2026-11-02`. Checkout modification times cannot establish record age. The archive preserves the original directory hierarchy outside working `docs/`, and `.rgignore` excludes it from default searches. Explicit historical discovery uses `rg --files --no-ignore archive` or `git ls-files archive`.

Hiding records alone would leave the working directory crowded. Moving every old record would break live readers and operating instructions. This move keeps 282 older files at their current paths because current readers, hash-pinned catalog inputs or policy still depend on them. The manifest records those exceptions. Living undated references, current steering, run logs and selected old evidence groups remain in place. This is a dated archive, not a judgment that earlier evidence is obsolete.

## Verification

- All 1,229 archive hashes match the manifest. Of those files, 1,156 retain their exact bytes. Changes in the other 73 files repair Markdown or HTML document links. Raw receipts, logs and source payloads retain their bytes.
- All 2,270 previously valid local Markdown and HTML link targets checked still resolve after relocation. Current prose citations and tool comments were also repaired. Historical Git paths and transcript commands retain their original meaning.
- Six current submission/incumbent source files match the starting commit byte for byte.
- Fifteen reader, evidence-manifest and compiler-wrapper tests passed. The repository file-discovery evaluation passed all eight fixtures.
- Before adding this handoff, default `rg --files` discovery fell from 1,194 to 887 files, a reduction of 307 files. This measures file discovery, not search latency.

[Validation evidence](evidence/2026-11-02-docs-archive/) contains the test output, discovery fixtures, baseline catalog mismatch and [replayable archive verifier](evidence/2026-11-02-docs-archive/verify.py). The independent folder reviews checked original hashes, eligibility, dependency exceptions and citations; their findings prompted the final citation repairs and one retained report. The final repairs received another verification pass.

`python3 tools/check_cuda_research.py` still exits 1 for `docs/ruled-out.md`. Its catalog hash already mismatched at the evidence commit, confirmed from Git bytes. The archive introduces no additional catalog mismatch. The owning research review must reconcile those claims before updating that hash.

## Limits and next actions

Historical scripts can assume their original root depth. Replay them at the recorded original Git revision or restore their former layout before execution. Their preservation does not establish that execution from an archive path works. Arbitrary user-supplied paths and dynamic readers outside the checked entry points remain a retrieval limit.

Keep new work under `docs/`. Future archive sweeps should use the same reader-dependency check and stop on a missing current input, changed raw payload, new link regression or new catalog mismatch. No automatic daily move is configured. Current worker assignments and their sole job waiters remain with their existing owners.
