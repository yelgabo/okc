## Independent clean reproduction on T3 Code 0.1.6 (Windows)

Reproduced on 2026-11-01 in a newly created disposable thread. A plain-text
baseline completed, one native worker spawn failed, and the next literal
`huh` failed without any assistant response. This confirms the persistent
follow-up failure on 0.1.6 in addition to the earlier reports here.

Environment:
- T3 Code 0.1.6, installed build `2359643cffa6`.
- Windows x64 (app-server reports Windows 10.0.26200).
- T3-managed Codex CLI/app-server 0.156.1.
- `modelProvider: openai_token_sharing`.
- `gpt-6-astra`; actual turn/start effort `high`.

### Reproduction

1. Start a fresh thread and ask: `Respond exactly BASELINE_OK. Do not use tools.`
   Actual baseline included explicit no-agent/no-file instructions and succeeded.
2. Ask for exactly one native `collaboration.spawn_agent`, inherited settings,
   with the worker instructed to reply `WORKER_OK` without tools; then call
   `collaboration.wait_agent`. Do not use T3 `delegate_task`.
3. Child fails, followed by parent failure.
4. Send exactly `huh`. It fails with the same rejection and no assistant reply.

The complete diagnostic prompts and sanitized receipts are in the evidence
bundle. No files, configuration, authentication, or existing history were changed
by the disposable test.

### Captured sequence (UTC, November 1)

```text
23:31:44.481  baseline assistant: BASELINE_OK; turn completes
23:32:20.753  parent rollout: collaboration.spawn_agent
23:32:21.282  child rollout: response_item type=agent_message
23:32:22.832  child: subscription_sharing_unsupported_capability
23:32:23.242  parent rollout: collaboration.wait_agent
23:32:23.317  parent rollout: response_item type=agent_message
23:32:24.951  parent: subscription_sharing_unsupported_capability
23:32:34.521  parent turn/start: one text input, "huh"
23:32:35.750  follow-up: same rejection; no assistant response
```

All three reproduction errors have `type=invalid_request_error`,
`param=input`, `code=subscription_sharing_unsupported_capability`,
and `willRetry=false`. The underlying message lists supported input item
types and asks to remove unsupported input items or use an API key.
This is the same input-item error as the issue body, not the separate
`tool_search` error described in one comment.

### Launch and history evidence

The actual running managed app-server command already contains:

```text
features.multi_agent=false
features.multi_agent_v2=false
```

It has **no `agents.enabled` override**. The same binary's read-only
`features list`, with these two overrides, reports both false. Native
`collaboration.spawn_agent` still executes.

The installed runtime bundle and pinned upstream managed-runtime source at
`fa31b66bbb2ae5234205a8628a78e038d917202f` use those two restrictions.
The current local Astra model cache contains `multi_agent_version: v2`.
This matches the catalog-precedence gap discussed in #17679 and the proposed
`agents.enabled=false` change in #20270. As retrieved November 1, #20270 is
closed and unmerged; I have not tested that patch or established why it closed.

Both this test and the original affected session persist `agent_message`
items immediately before the child and parent input rejections. Retained
unsupported history is strongly indicated, but the outgoing Responses HTTP
body was not captured, so the exact rejected wire item remains unconfirmed.

In the original affected session, changing from Astra to Sol and back did not
recover ordinary follow-ups. No native-agent calls were made by the follow-up
turns. A separate fresh Astra conversation remains functional without native
spawning.

### Expected behavior / requested follow-up

Prevent native tools that the configured sharing route cannot accept from
being exposed or executed. Also provide recovery for already affected threads:
an unsupported child operation must not leave all later plain-text messages
failing. A prevention-only fix does not establish recovery of existing history.

Sanitized evidence: exact errors/timestamps, runtime/build details, rollout
item structure, installed source excerpts, and snapshot SHA-256 manifest.
Raw shared logs, private prompts, credentials, auth-command arguments and
encrypted content are omitted. No provider switch or history edit was used.

Prepared with Codex (GPT-6 Astra) assistance from an actual bounded reproduction
and local read-only protocol/runtime inspection. This draft has not been posted.
