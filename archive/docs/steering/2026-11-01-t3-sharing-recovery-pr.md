# T3 sharing rejection: recover the conversation on the next prompt

Date: 2026-11-01 (America/Los_Angeles). User-authorized implementation, isolated
recovery validation, and upstream PR. This supersedes the reproduction report's
pending-repair next action; the original failure evidence remains valid.

## Published result and pins

- PR: https://github.com/pingdotgg/t3code/pull/28962 (open when created).
- Fork: `yelgabo/t3code`, branch `fix/codex-invalid-history-recovery`.
- T3 head: `332447eea868c2b89328187309e063c99fce1ebb`.
- T3 base: `fa31b66bbb2ae5234205a8628a78e038d917202f`.
- Original reproduction: OKC `aff51efe21c8fe112bd89e1360ef76145375b032`.
- Recovery evidence: OKC `9d15ccf04012d49267215cfc8a1bc9699313ba5e`.
- [Validation manifest](evidence/2026-11-01-t3-sharing-recovery/validation.json)
  pins commands, test outcomes, original/published log hashes, and limitations.

The PR was created through the GitHub connector and linked to this T3 thread.
It is not merged or installed. No CI/review monitoring was requested or started.

## Corrected configuration explanation

The initial provider rejection is expected when the configured runtime emits
input unsupported by ChatGPT sharing. The reliability defect is that a later
ordinary prompt remains stuck on that rejected native history.

The measured process already used `features.multi_agent=false` and
`features.multi_agent_v2=false`, but native agents still ran. Earlier wording
that implied these two values alone explained a user configuration mistake was
incomplete. There is no verified `enable=true` repair. The closed, unmerged
[prevention PR #20270](https://github.com/pingdotgg/t3code/pull/20270) proposes
the additional `agents.enabled=false` override. That prevention change was not
included or independently validated here. The recovery PR explicitly separates
configuration compatibility from the unintended persistent failure.

## Reproduction and measured recovery

Original environment: Windows x64 10.0.26200, T3 0.1.6 build `2359643cffa6`,
managed Codex 0.156.1, provider `openai_token_sharing`, Astra/high.

The authorized disposable T3 thread first answered `BASELINE_OK`, then failed
after one native worker spawn/wait. Both worker and parent received
`subscription_sharing_unsupported_capability` (`param=input`, `willRetry=false`).
Literal `huh` then failed without a new assistant answer or tool call. Native
rollouts contain `agent_message` immediately before rejection. These observations
support the invalid-history mechanism; the exact rejected HTTP wire item was not
captured. See the [original report](2026-11-01-t3-sharing-reproduction.md).

The separate real-provider check used a new temporary `CODEX_HOME`, a copy of
the broken parent rollout, the same managed provider/model/effort, and the same
two disabled feature flags. The existing auth command was used in memory without
publishing it or any credentials. No live T3/Codex state was edited. The native
replacement was an isolated runtime fixture, not another top-level T3 thread.

| Step | Measured result |
| --- | --- |
| Resume copied broken native history, send `huh` | Failed, same unsupported-capability code, no assistant reply |
| Start replacement native session with prior conversation as text; ask for the earlier reply token | Completed, `BASELINE_OK` |
| Send another request in that replacement session | Completed, `FOLLOWUP_OK` |

Runtime capture: `2026-11-01T23:47:26.569110+00:00`. Source native parent:
`01a19974-57c5-7781-ab5c-2f71b614ec36`; isolated replacement:
`01a19982-cb0d-7780-b864-5761711ab648`.
[Sanitized transcript](evidence/2026-11-01-t3-sharing-recovery/runtime.json) and
[check script](evidence/2026-11-01-t3-sharing-recovery/runtime-check.py).

## Implementation and regression evidence

`ProviderTurnStartService.ts` recognizes the persisted sharing rejection in the
latest accepted root attempt for the current native session. On the next user
turn it uses T3's existing fresh-session recovery and portable context handoff.
The visible transcript, provider-thread row identity, provider/model selection,
and original error remain. The failed action is not automatically retried.
History transfer keeps the existing context budget and omission behavior.

Recovery is scoped to Codex and the exact sharing error code. It avoids treating
an older native session's failure as a reason to reset a replacement. Legacy
attempts without native IDs use delivered recovery coverage. Pending background
work blocks replacement; after that work finishes the next prompt can recover.

The regression uses the real orchestrator, SQLite projections, and handoff
delivery with a deterministic provider adapter. Once poisoned, that adapter
rejects every later turn sent to the same native ID. It permits ordinary prompts
on a replacement ID. This tests the production recovery path without incurring
extra model turns.

- New regression against original base service: 4 failed, 3 passed. The live,
  detached, legacy and background-work recovery cases expose the defect.
- Patched new regression: all 7 passed, including unrelated error/provider and
  already-repaired-session controls. Conversation retention, current-prompt
  delivery, stable T3 row identity and successful follow-up reuse are asserted.
- Selected suite: 88 passed, 4 failed across 3 files. All 4 worktree-recreation
  failures reproduce on the original base service, unchanged. They are outside
  the recovery branch modified here. Failure names and outputs are published.
- Changed-file formatting/lint, server GraphQL generation, affected typechecks
  (`t3`, `@t3tools/evals`, `@t3tools/scripts`) and both knip passes succeeded.
  Commit hooks also passed format, lint and knip.
- `vp run check:changed` was attempted but its initial ignored-file scan follows
  Windows pnpm directory junctions, producing `Filename too long`. The run was
  interrupted; its selected check commands were run directly. The aggregate
  command is not claimed to pass.

[Before tests](evidence/2026-11-01-t3-sharing-recovery/before-tests.txt),
[patched focused tests](evidence/2026-11-01-t3-sharing-recovery/after-focused-tests.txt),
[selected suite](evidence/2026-11-01-t3-sharing-recovery/after-selected-tests.txt).

## Priorities, next actions, acceptance and gaps

1. Maintainers review the focused recovery PR. Prevention remains a separate
   compatibility/configuration issue, related to #21061 and #20270.
2. After acceptance and deployment, validate this same failure/ordinary-prompt
   sequence in the shipped T3 client. No installed UI was patched here. The
   current evidence combines real T3 integration tests with a separate real
   provider runtime experiment; it is not a deployed-client end-to-end result.
3. Keep the original error visible and preserve visible conversation records.
   Acceptance requires ordinary follow-ups to finish, recovery not to repeat on
   healthy sessions, and active background work not to be abandoned.
4. Windows check-runner traversal and baseline worktree-test failures remain
   separate tooling gaps. They were documented, not hidden or fixed incidentally.

Stopping criterion met for this authorized task: a pushed focused patch, an open
linked PR with reproducible before/after evidence and explicit limitations, and
this repository handoff. No original live session was repaired in place, no
kernel work or scored jobs ran, and no stopped campaign or automation resumed.
