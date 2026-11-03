# T3 native-worker failure: independent reproduction and upstream report

2026-11-01, investigation completed after 23:34 UTC. Evidence checkout:
`C:/Users/gretc/Documents/GitHub/okc`, main at
`97a91dfdfd254ae8b438e1b0bc67c4bef333bbbc`, origin
`https://github.com/yelgabo/okc.git`. This report follows the earlier
[session audit](2026-11-01-session-provider-failure.md), replacing its
unconfirmed reproduction status with a fresh controlled observation.
No OKC kernel, measurement, submission, or campaign state changed.

Publication reconciliation: concurrent documentation commit `6d6b31c` arrived
during evidence preparation and included the new index pointer. Its backlog
changes are preserved; this task commits only its report, evidence and index
history addition. The technical evidence remains pinned to the snapshot above.

## Outcome

Reproduced on one newly created, user-authorized disposable T3 thread:
plain-text baseline succeeds, native worker spawn fails, and the subsequent
single-word message `huh` fails without an assistant reply. The original
thread independently exhibits the same sequence and three failed follow-ups.
All eight captured raw error notifications (five original, three reproduction)
identify `subscription_sharing_unsupported_capability`, `param: input`,
`type: invalid_request_error`, and `willRetry: false`.

The upstream issue [pingdotgg/t3code#21061](https://github.com/pingdotgg/t3code/issues/21061)
is open as retrieved today. A matching earlier report is
[#17679](https://github.com/pingdotgg/t3code/issues/17679).
A proposed prevention fix, [#20270](https://github.com/pingdotgg/t3code/pull/20270),
is currently **closed and unmerged**, head
`9e7de4f3490a4716bb55a85fec209d467d2be4d2`.
Its description still speaks of pending approval; current metadata takes
precedence over that historical prose. We did not establish why it was closed.

The [ready-to-post update](evidence/2026-11-01-t3-sharing-reproduction/issue-comment.md) is intended for #21061.
It has **not** been posted. The user's unfinished "maybe we can report" request
authorized preparing evidence, not sending an external message.

## Exact environment and ownership

T3 Code desktop/server 0.1.6, installed application build `2359643cffa6).
Windows x64; app-server initialize reports Windows 10.0.26200.
The running managed binary reports `codex-cli 0.156.1`.
Both original and reproduction `thread/start` results identify
`modelProvider: openai_token_sharing`. Reproduction turn/start requests use
`gpt-6-astra`, effort `high`, full-access/default mode.
The thread/start default reports low, but the actual turn/start overrides are
high; these must not be conflated.

Original T3 thread:
<t3-thread://v2/471a5a44-f247-4d6c-b35d-7f45e6f30552/40356bc9-4e84-4f44-81ab-572185175234>

Disposable reproduction:
<t3-thread://v2/471a5a44-f247-4d6c-b35d-7f45e6f30552/mcp:launch:439478037f2ebf14913317acee5b9359e3735c6a34ea557935ae942a42516664>

The reproduction T3 run IDs are the three ordinal entries in
[reproduction-thread.json](evidence/2026-11-01-t3-sharing-reproduction/reproduction-thread.json);
the native provider turn IDs and exact errors are in
[protocol-extract.json](evidence/2026-11-01-t3-sharing-reproduction/protocol-extract.json).
The original thread and installed runtime/configuration were not modified.
The disposable thread is terminal failed with no active run. No further
workers, providers, models, or top-level test threads were launched.

## Registered test and observed result

Question: Does a successful fresh conversation become unable to answer plain
text after one native worker spawn on the same model/provider?
Control: first turn requests exactly `BASELINE_OK` with no tools.
Treatment: request exactly one `collaboration.spawn_agent`, inherited model,
worker instructed to return `WORKER_OK` without tools, followed by
`collaboration.wait_agent`. Follow-up: literal `huh`.
No repository activity or external application work was requested in that test.
Stop after the first fresh failure sequence; no repeated retry loop.

Observed UTC timeline on 2026-11-01:

- 23:31:44.481: baseline assistant reply `BASELINE_OK`; turn completed.
- 23:32:20.753: native `spawn_agent` recorded in parent rollout.
- 23:32:21.282: `agent_message` enters child rollout.
- 23:32:22.832: child rejects input with the sharing error.
- 23:32:23.242: parent calls `wait_agent`.
- 23:32:23.317: `agent_message` enters parent rollout.
- 23:32:24.951: parent rejects input with the same error.
- 23:32:34.521: outgoing parent turn/start contains one text item, `huh`.
- 23:32:35.750: follow-up rejects input; no assistant response or new tool call.

T3 timestamps differ slightly from protocol timestamps because they record
projection/persistence after the underlying event. The timeline above identifies
each source rather than presenting them as one clock origin.

## Mechanism evidence and remaining uncertainty

The real running managed app-server command contains
`features.multi_agent=false` and `features.multi_agent_v2=false`.
It has no `agents.enabled` override. Read-only CLI feature listing with those
same two overrides reports both features false. Nonetheless, native agent
creation occurs in both observed sessions.

The installed `CodexManagedRuntime.ts` bundle and upstream source at
`fa31b66bbb2ae5234205a8628a78e038d917202f` both supply only those two
multi-agent restrictions. Current local model-cache fields for gpt-6-astra show
`multi_agent_version: v2` and `multi_agent_reasoning_effort: xhigh`.
That cache is corroboration, not a captured per-request catalog response.
The prior upstream diagnosis and PR identify catalog-selected V2 as bypassing
those feature flags and propose `agents.enabled=false` in addition.

Native rollout evidence independently identifies `response_item.type:
agent_message` immediately before both child and parent failure, in both
the original and fresh reproduction. The parent retains this record before
the failing plain-text follow-up. These observations strongly support
unsupported native-agent history as the continuing failure mechanism.
The exact Responses HTTP request body was not captured, so the specific
rejected wire item and endpoint implementation are not directly proven.
We did not remove history, patch T3, test API-key authentication, or validate
the proposed override against this installed binary.

The friendly T3 error is a translation by
`apps/server/src/provider/CodexManagedErrors.ts`; the underlying API rejection
explicitly names unsupported input. This is not merely a UI rendering failure.
Prevention and recovery of already affected histories remain separate needs.

## Evidence inventory and disclosure

[Collector](evidence/2026-11-01-t3-sharing-reproduction/collect.py) records only allowlisted protocol fields,
history item metadata, runtime facts, installed source excerpts, and hashes.
The raw shared provider log contains unrelated sessions; it is not published.
Complete local snapshots of that log and four relevant native rollout files
remain under `%TEMP%/t3-provider-repro-20261101/private/`.
Their byte counts and SHA-256 values are in
[snapshot-manifest.json](evidence/2026-11-01-t3-sharing-reproduction/snapshot-manifest.json).

Published evidence includes:
[protocol events](evidence/2026-11-01-t3-sharing-reproduction/protocol-extract.json),
[history structure](evidence/2026-11-01-t3-sharing-reproduction/rollout-structure.json),
[runtime checks](evidence/2026-11-01-t3-sharing-reproduction/runtime.json),
[installed source excerpts](evidence/2026-11-01-t3-sharing-reproduction/installed-source-excerpts.json),
[upstream pins and search receipts](evidence/2026-11-01-t3-sharing-reproduction/upstream-references.json),
and [test actions](evidence/2026-11-01-t3-sharing-reproduction/reproduction-actions.json).
Message content, encrypted payloads, auth files, process auth-command arguments,
and databases were excluded from the public extraction.
Only the intentionally trivial reproduction prompts appear in its T3 receipt.
No raw credential material is needed to reproduce or report this failure.

Search scope: GitHub connector, 2026-11-01; exact error search scoped to
pingdotgg/t3code, broader sharing/agent search, direct issue/comments/PR reads,
and pinned managed-runtime source. The known matching issue makes a duplicate
new issue unnecessary. Search results are not an exhaustive issue inventory.

## Priorities, next actions and acceptance criteria

1. User owns the decision to post the prepared reproduction update on #21061.
   Attach the sanitized evidence or link this committed report; omit private logs.
2. T3 maintainers should establish effective native-tool suppression for the
   installed catalog/runtime combination, considering the existing proposal.
3. Track recovery separately: ordinary text must complete after a failed child,
   with conversation content preserved. Do not silently erase history.
4. Validate a fix with the same plain-text baseline, attempted native spawn,
   and post-failure text follow-up; inspect actual advertised tools and persisted
   items, not only feature-list output. A fresh-thread prevention fix alone
   does not establish recovery of the original affected conversation.

Stopping criterion met for this investigation: one independent fresh reproduction,
owning terminal records, installed/runtime evidence, and verified upstream report.
No more remote test turns or configuration changes are justified merely to
repeat this result. Any repair/recovery experiment needs a separately defined
scope. The original campaign remains stopped.
