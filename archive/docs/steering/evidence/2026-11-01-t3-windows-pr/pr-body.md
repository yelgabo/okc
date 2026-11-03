## Problem

Four `ProviderTurnStartService` tests fail on Windows because the harness combines
a real Windows filesystem with POSIX path handling. `layerWorktreeLocation()`
unconditionally installs `effect/Path.layer`, whose implementation is
`Layer.succeed(Path)(posixImpl)`. The three real-filesystem call sites obtain a
native path service from `NodeServices.layer` but previously passed only the
filesystem into the harness.

POSIX absolute paths start with `/`; `C:\` is not a POSIX root, and backslashes
are ordinary filename characters. Consequently the POSIX service treats this
valid Windows input as relative:

```text
C:\Users\test\AppData\Local\Temp\t3-start\project\worktree
```

The recorded `resolve()` output was:

```text
<CHECKOUT>/C:\Users\test\AppData\Local\Temp\t3-start\project\worktree
```

`<CHECKOUT>` redacts the actual Windows checkout directory. Native `NodePath.layer`
correctly leaves the input unchanged. The POSIX implementation is behaving as
designed; the test harness supplied it a path from another platform.

## Reproduction steps

1. On Windows, check out upstream `fa31b66bbb2ae5234205a8628a78e038d917202f`
   in a disposable checkout and install dependencies with `vp i`.
2. Run:

   ```powershell
   vp test run apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts -t 'recovers the local or pushed thread tip|preserves recreation interrupts|coordinates provider startup'
   ```

3. Observe the four failures below. Apply this PR's patch and rerun the same
   command, then run the complete test file:

   ```powershell
   vp test run apps/server/src/orchestration-v2/ProviderTurnStartService.test.ts
   ```

## Before the fix

On unchanged upstream: **4 failed, 38 skipped**.

- **Recovery:** the managed-worktree guard rejects the Windows path as
  non-absolute, so recreation is skipped. The later Git failure wraps `ENOENT`
  accessing the missing working directory; it is not a missing Git executable.
- **Recreation interrupts/retries:** recreation is skipped, so the injected
  interrupt never executes; the test receives `Success` instead of `Failure`.
- **Cleanup-first:** startup and cleanup use different lease keys, and the
  managed-path guard skips the expected recreation call.
- **Startup-first:** different lease keys let cleanup enter while startup still
  holds its lease, failing the ordering assertion.

## Change

Allow the harness to accept a `Path.Path` service and supply it through
`layerWorktreeLocation`. Pass the native `path` alongside `fileSystem` at the
three real-filesystem call sites; one call site covers both ordering cases.
Keep the existing POSIX default for virtual `/tmp/...` fixtures.

## After the fix

- Same focused command: **4 passed, 38 skipped**.
- Complete test file: **42 passed**, including all six scenarios inside the
  real-Git recovery test and both Deferred-controlled ordering cases.
- Rebased onto upstream `e6e2ab05cceb3f8245e87f8499d4891f27de33cb` and reran
  the complete file at head `c9736009eccfb214ca50b7f4e468868923f7c7aa`: **42 passed**.
  The affected test file was unchanged between the two upstream bases.
- Existing assertions remain intact; no production code changes.

## Scope and approval

This is a small, focused fix for an obvious test dependency mismatch, under the
contribution guide's exception for fixes that do not require prior discussion.
It changes one test file and restores its intended behavior on Windows.
Production already uses native `NodeServices.layer`; these failures do not
demonstrate a production recovery or locking defect. This is independent of the
Codex history recovery fix in #28962.

## Verification

Windows; Vite+ 1.4.0, Vitest 5.0.5, Effect/platform-node 4.0.5.
Public [before/after logs, path probe and patch](https://gist.github.com/yelgabo/a68b25521807bc96cbe285860da9b766)
include exact source pins and commands. The gist page and every raw file were
verified without authentication.

Also passed:

```powershell
vp run --filter t3 --filter @t3tools/evals --filter @t3tools/scripts typecheck
vp run --filter t3 generate --check
```

The commit hook passed formatter, lint and both `knip:check` passes before rebase;
the final test-file blob is unchanged. `git diff --check` also passed.

`vp run check:changed` was attempted before push but stalled in its ignored-file
cache-key enumeration (`git ls-files --others --ignored ...`) and was stopped
after 90 seconds, before executing its checks. Its selected checks were run
directly as described above; the aggregate command is not claimed as passing.
Linux/macOS and the full repository suite were not run. No live T3 state was used.

Implemented and verified with GPT-6 Astra (high reasoning) through the Codex harness in T3 Code.
