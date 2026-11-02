# Four T3 baseline failures: Windows test-harness path mismatch

Date: 2026-11-01, America/Los_Angeles.
Evidence commit: `abc66ce18275f9486f7ed8cbb1483d81cd608cfc`.
T3 baseline: `fa31b66bbb2ae5234205a8628a78e038d917202f`, also the fetched
`origin/main` at this investigation. This is separate from recovery PR #28962
and its head `332447eea868c2b89328187309e063c99fce1ebb`.

## Finding

All four observed failures have one demonstrated cause: the test helper installs
POSIX path handling into tests that supply a real Windows filesystem and Windows
paths. A correction confined to the test harness makes all four pass, and the
entire 42-test file passes without changing assertions or production code.

This establishes a Windows test setup defect. These failures do not establish
broken production worktree recovery or an actual startup/cleanup race. Production
uses the native Node path service. They are also independent of the Codex sharing
configuration rejection and stuck-session history repaired in the earlier PR.
No provider configuration toggle is involved here.

## Reproduction and controls

Work took place in `C:\Users\gretc\Documents\GitHub\t3code-sharing-recovery`,
on a separate local branch `investigate/windows-worktree-tests` based on upstream.
The four failures were reproduced with the test file unchanged from that base,
then rerun with the archived [test-only patch](evidence/2026-11-01-t3-windows-tests/test-harness.patch).
Dependencies, runner, host and test assertions were held fixed.

Run identifiers below name the archived receipts, not provider or GPU jobs:

| Receipt | State | Result |
| --- | --- | --- |
| `before-focused`, 16:12:20 local | Unmodified upstream | 4 failed, 38 skipped |
| `after-focused`, 16:12:51 local | Test-harness correction | 4 passed, 38 skipped |
| `after-full`, 16:14:12 local | Same correction, full file | 42 passed, 0 skipped |

The real-Git recovery test includes six scenarios: retained branch, deleted
branch, pushed branch, pushed detached tip, detached HEAD, and detached HEAD
without a remote. Its successful rerun reaches the complete loop. The other
cases exercise injected interrupts/failures and deterministic Deferred-based
startup/cleanup ordering; no retries, longer timeouts or assertion weakening were
added. The 38 other tests retain their virtual POSIX fixtures.

Additional checks passed: server package typecheck (`vp run --filter t3
typecheck`), changed-file formatter, changed-file lint, and `git diff --check`.
Environment: Windows, Vite+ 1.4.0 / managed Node 26.12.1, Vitest 5.0.5,
Effect and `@effect/platform-node` 4.0.5, Git 2.54.0.windows.1. Ambient `node`
is 24.16.0; Vite+ reports its separately managed Node version.

## Mechanism and the four symptoms

`layerWorktreeLocation()` in `ProviderTurnStartService.test.ts:68` unconditionally
merged `effect/Path.layer`. Effect 4.0.5 explicitly implements this as POSIX,
regardless of the host. Three real-filesystem harness call sites obtained their
paths and filesystem from `NodeServices.layer`, then passed only the filesystem
into the harness. Its inner layer silently replaced their native path service.

1. **Recovers the local or pushed thread tip, else the current branch, remote
   default, or detached HEAD.** `makeIsManagedWorktreePath` rejects a Windows
   drive path at its absolute-path guard. Consequently the missing worktree is
   not recreated. The provider-open mock then tries Git in the missing directory.
   The visible error says “Failed to spawn Git process,” but the nested error is
   filesystem `ENOENT` while accessing the working directory. Git installation,
   remote authentication and branch selection are not the cause of this failure.
2. **Preserves recreation interrupts and retries, then fails the final attempt
   before provider start.** The same managed-path guard skips recreation, so the
   deliberately injected `createWorktree` interrupt never runs. The test receives
   `Success` where it expects `Failure`. This is not evidence that Effect loses
   interrupts: the interrupt-producing operation was bypassed.
3. **Coordinates provider startup with cleanup-first.** Startup resolves the
   worktree path using the injected POSIX service; cleanup uses the original
   Windows path. They enter different workspace leases. Recovery also rejects
   the Windows path, and the final assertion sees zero recreation calls.
4. **Coordinates provider startup with startup-first.** The same unequal lease
   keys let cleanup enter while startup holds its lease, failing the assertion
   that cleanup has not entered. This fixture was exercising two locks instead
   of contention for the same lock.

The independent [path-service probe](evidence/2026-11-01-t3-windows-tests/path-probe.json)
shows that `Path.layer.isAbsolute("C:\\...")` is false and resolution prefixes
the checkout working directory. Native `NodePath.layer` and explicit
`NodePath.layerWin32` both recognize the same path as absolute, preserve its
normalized representation and produce the same lease key as cleanup. POSIX
`/tmp/...` fixtures still need POSIX semantics: globally swapping every test to
native Windows paths would introduce a different mismatch.

## Correction and product scope

The patch makes the harness accept an optional `Path.Path` service and forwards
it through `layerWorktreeLocation`. Each of the three real-filesystem call sites
passes its already-acquired native `path` alongside `fileSystem`; the parameterized
ordering call site covers two tests. Virtual fixtures keep the existing POSIX
default. Only `ProviderTurnStartService.test.ts` changes.

Pinned source for the causal chain:

- [Test harness and cases](https://github.com/pingdotgg/t3code/blob/fa31b66bbb2ae5234205a8628a78e038d917202f/apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts): helper at 68, harness wiring at 896, real-filesystem cases around 990–1440.
- [Managed-path guard](https://github.com/pingdotgg/t3code/blob/fa31b66bbb2ae5234205a8628a78e038d917202f/apps/server/src/worktreesDirectory.ts): rejects non-absolute and non-normalized paths before checking managed roots.
- [Startup](https://github.com/pingdotgg/t3code/blob/fa31b66bbb2ae5234205a8628a78e038d917202f/apps/server/src/orchestration-v2/ProviderTurnStartService.ts): recovery guard around 626, lease resolution at 1708.
- [Lease implementation](https://github.com/pingdotgg/t3code/blob/fa31b66bbb2ae5234205a8628a78e038d917202f/apps/server/src/workspace/workspaceLease.ts): semaphore map keyed by the exact path string.
- [Production cleanup](https://github.com/pingdotgg/t3code/blob/fa31b66bbb2ae5234205a8628a78e038d917202f/apps/server/src/storageCleanup.ts): resolves the worktree path at 807 and leases it at 1115.
- [Production platform wiring](https://github.com/pingdotgg/t3code/blob/fa31b66bbb2ae5234205a8628a78e038d917202f/apps/server/src/server.ts): `NodeServices.layer` at 374, provided to runtime at 1193. The installed NodeServices implementation includes native `NodePath.layer`.

Thus production startup and cleanup both normalize with the host-native service;
the demonstrated test-specific POSIX substitution is absent from that wiring.
This is source evidence plus the corrected service tests, not a live installed-app
end-to-end test. Case aliases, junctions and other path canonicalization edge
cases were not exhaustively audited and are not ruled out by this result.

## Repeat the experiment

Use a disposable checkout at the pinned base with its dependencies installed.
On Windows, before applying the patch, run from the T3 root:

```powershell
vp test run apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts -t 'recovers the local or pushed thread tip|preserves recreation interrupts|coordinates provider startup'
```

Expect four failures matching `before-focused.txt`. Apply the archived patch
with `git apply <path-to-test-harness.patch>`, rerun the same command, then run:

```powershell
vp test run apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts
vp run --filter t3 typecheck
vp fmt --check apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts
vp lint --report-unused-disable-directives apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts
```

Expected corrected results: four selected cases pass, then all 42 pass, with the
remaining checks clean. To rerun the archived probe, copy `path-probe.mjs` into
`apps/server` in that disposable checkout and execute `node path-probe.mjs` from
the T3 root so its cwd matches the recorded probe; remove that scratch copy
afterward. The script imports from its adjacent package's `node_modules`.

## Handoff, priorities and stopping criteria

The investigation is complete against the pinned upstream: every observed
failure has a source-level mechanism, a failing baseline receipt and a passing
test-only correction; the full file and scoped static checks pass. Evidence is
under [evidence/2026-11-01-t3-windows-tests](evidence/2026-11-01-t3-windows-tests/validation.json),
including logs, probe, patch and source blob identities.

The standalone patch remains reviewable on the local investigation branch and
in this private repository. No new T3 PR was opened, no T3 branch was pushed,
and the existing recovery PR was not changed. The live installation and session
state were untouched. This private handoff is not claimed as public reviewer
evidence; any later external handoff must publish sanitized evidence and verify
anonymous access, following the prior visibility correction.

Next priority, if a separate fix is requested: apply this narrow test patch to
then-current upstream, run required `check:changed` before a T3 push, and validate
on Linux/macOS CI as well as Windows. That aggregate runner was not rerun here;
its previously observed Windows ignored-file/junction traversal limitation
remains separate from these four tests. No repo-wide suite was run. Owner of
this investigation and any requested follow-up is the primary session; no
delegates, GPU jobs or scored submissions were used. No broader production
change is justified by the evidence gathered here.
