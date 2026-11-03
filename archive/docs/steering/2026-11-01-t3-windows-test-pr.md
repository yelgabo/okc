# T3 Windows startup test fix PR

Date: 2026-11-01, America/Los_Angeles; publication November 2 UTC.
Supersedes the no-PR status in the [investigation](2026-11-01-t3-windows-test-failures.md)
after the user explicitly requested creation and separate reproduction/before/after sections.

## Published result

PR: https://github.com/pingdotgg/t3code/pull/28997

Head: `c9736009eccfb214ca50b7f4e468868923f7c7aa`.
Base at validation: `e6e2ab05cceb3f8245e87f8499d4891f27de33cb`.
Branch: `yelgabo:investigate/windows-worktree-tests`.
One test file, 10 insertions / 3 deletions. Production code and assertions unchanged.
PR was linked to the current T3 thread immediately after creation.
The earlier Codex history recovery PR #28962 was not changed.

The PR explains why POSIX handling considers a Windows drive path relative:
only `/` starts a POSIX absolute path, and backslashes are ordinary characters.
It includes the recorded input and malformed cwd-prefixed output, then maps that
to skipped recovery, skipped injected interrupts and different workspace lease keys.
It has explicit Reproduction steps, Before the fix and After the fix sections.

Public evidence: https://gist.github.com/yelgabo/a68b25521807bc96cbe285860da9b766

Verified gist revision: `4e6fd444937c9671a4a9e2301afe987c26dddd3a`.
Anonymous page request returned 200; all 10 raw files returned 200 and matched
the intended bytes. Anonymous PR API read returned 200 and confirmed the exact
prepared body, one changed file, open state and expected head SHA.
No private OKC link is used as reviewer evidence.

## Validation and receipts

Original before/after evidence remains pinned at OKC
`abc66ce18275f9486f7ed8cbb1483d81cd608cfc`: four selected failures on unchanged
T3 `fa31b66b`, four selected passes with the correction, then all 42 tests pass.
Upstream subsequently changed unrelated usage reporting; the affected test file
and production path code were unchanged. The fix was rebased without conflicts.

At final head, run `after-current-full` at 17:18:10 local again passes all 42 tests.
Affected package typechecks (t3, evals, scripts) and server GraphQL generation check
pass. Formatter, lint and both knip passes passed in the commit hook before rebase;
the corrected file blob remains `5eca90709083ca52a65addd3b4dfee631cc02007`.
Final `git diff --check` passes. New logs and publication receipts are in
[evidence/2026-11-01-t3-windows-pr](evidence/2026-11-01-t3-windows-pr/pr-verification.json).

The mandatory `vp run check:changed` was attempted before push. It stalled before
executing checks; inspection of the captured process tree confirmed the pending
`git ls-files --others --ignored --exclude-standard` cache-key scan with generated
source/env pathspecs and the node_modules exclusion. The owned process tree was
stopped after the 90-second bound. The selected checks above were run directly;
the aggregate is explicitly not reported as passing. No unrelated processes were
terminated. No Linux/macOS or full repository suite was run, and no live T3 state
was touched.

## Next gate and stopping criterion

The user's requested PR, explanation, reproduction and public evidence are
published and verified. Maintainer review and CI are the next gate. No merge,
deployment, new production fix, or monitoring was requested; none was initiated.
The primary session owns any requested follow-up. No subagents or GPU work were
used. A later merge decision should consider Windows plus Linux/macOS CI and
review of the narrow service injection, without extending this PR into a
production path canonicalization redesign.
