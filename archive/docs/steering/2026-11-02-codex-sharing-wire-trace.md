# Exact T3 / Codex / ChatGPT-sharing input discrepancy

Date: November 2, 2026, America/Los_Angeles. This is a new, deeper diagnosis,
not a repetition of the earlier adjacency-only explanation. Repository inspection
began after 8166b90; current concurrent HEAD at offline validation was
ed6b465d86059b8bed8f9eed26e8d985bbc3da56. Original incident remains the single
failed run in T3 thread 6a582167-166e-49e8-b0c1-29cd50228f79.

## Bottom line

There are TWO input schemas at TWO boundaries. T3 supplies Codex app-server RPC
inputs, not the final Responses HTTP body. T3's normal text input builder is not
sending an invalid HTTP item named text: text is correct for that RPC boundary.
Codex native V2 orchestration subsequently inserts agent_message envelopes into
history and serializes them unchanged into the HTTP input array. The configured
ChatGPT-sharing route rejects that native item family. The route accepts ordinary
messages and supported function/custom-tool history, not every internal Codex
ResponseItem variant. Thus valid Codex-native input can be invalid sharing input.

The first report only established native-history adjacency to the rejection.
This task additionally inspected actual installed T3 code, pinned released Codex
serialization code, and captured an actual localhost POST from the installed
executable using copied ROOT history. This proves the outgoing native-envelope
serialization; it does not retroactively provide the original remote request bytes.

## Boundary 1: what T3 supplies

The installed binCli-C12fvdZY.mjs remains exactly 14696322 bytes, SHA256
472d7e2d82900ae9da669da1dfd9cd4b7075f61e6e6ac9bda9d6dd9644550d25.
Its toCodexInput constructs text items and supported image attachment items.
buildCodexTurnStartParams wraps them with nativeThreadId, model, runtime policy
and context, then calls client.request("turn/start", turnStartParams).
Illustrative shape, not a reconstructed original payload:

~~~json
{"method":"turn/start","params":{"threadId":"parent-id","input":[{"type":"text","text":"the user task"}],"model":"gpt-6.1-sol"}}
~~~

Actual protocol log line374 records thread/start at09:54:44.348UTC with model
gpt-6.1-sol, the OKC cwd, and configuration keys tools.update_plan.enabled,
mcp_optional_startup_grace_ms and mcp_servers.t3-code. Its response selects
openai_token_sharing. Line384 records turn/start but deliberately stores
params={truncated:true,threadId:...}; its actual input bytes cannot be recovered
from that truncated log. Source-derived shapes are not mislabeled as those bytes.

The installed launch builder sets base_url=https://api.openai.com/v1,
wire_api=responses, requires_openai_auth=false and supports_websockets=false,
with the token broker used for authentication. This is the documented endpoint
and transport pattern, not a mistaken backend-api URL. Broker credentials and
raw command lines were not read or published.

## Boundary 2: what the sharing route expects

Official [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
shows an input ARRAY of ordinary role/content messages, with store=false and
stream=true on POST /v1/responses. For example, this is a compatible text-message
shape, not a claimed patch for native multi-agent semantics:

~~~json
{"model":"gpt-6.1-sol","input":[{"role":"user","content":"the assigned task as ordinary text"}],"store":false,"stream":true}
~~~

[Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
also permits local child coordination implemented using supported function/custom
calls. It does NOT imply that the entire Codex-native agent protocol is accepted.
The incident backend explicitly rejects unsupported INPUT ITEMS and lists accepted
message/tool/history families; this is not an error about the task's meaning.

## What Codex supplies instead

Original child rollout at09:55:48.902UTC contains a native agent_message from
/root to /root/diagnostic_crosscheck. Its content has TWO pieces: input_text
containing the NEW_TASK/task-name/sender header, followed by encrypted_content
containing an opaque payload. Actual values of encrypted data are omitted from
published evidence. It is not just a plain assignment string with an unusual role.

The public rust-v0.166.0 tag resolves through annotated tag object
593389380740f7b4762771573c447ccfdc59742f to commit
6ff1e1ca3b30fc8b914720681fcf5fe62e95f856. The installed executable independently
reports0.166.0; it was not rebuilt to attest binary/source equivalence.
Source path (exact excerpts and hashes in source-trace.json):

1. protocol/src/models.rs defines ResponseItem with serde tag=type and snake_case;
   AgentMessage has author, recipient and content, not role and ordinary message
   content. AgentMessageInputContent supports input_text and encrypted_content.
2. core/src/client_common.rs get_formatted_input_for_request clones history and
   only normalizes image details. AgentMessage is explicitly a no-op match arm.
3. core/src/client.rs build_responses_request passes that input directly into
   ResponsesApiRequest. It filters ConfigurationUpdate under one condition, NOT
   AgentMessage. The size-shedding helper only bounds tool-execution metadata,
   rather than converting native agent envelopes to supported messages.
4. codex-api/src/common.rs declares input as Vec<ResponseItem>; the HTTP endpoint
   encodes that request and POSTs it to /responses under the configured base URL.

There is no sharing-compatible agent-message conversion at this inspected request
boundary. The source and runtime capture support a format mismatch, not malformed
JSON or the hypothesis that T3 directly POSTed its RPC type:text items to OpenAI.

## Offline wire capture: actual executable, no model or native agent spawn

Executed [offline_capture.py](../../../docs/steering/evidence/2026-11-02-codex-sharing-wire-trace/offline_capture.py)
against Codex0.166.0 with a fresh temporary CODEX_HOME and a COPY of the failed
ROOT rollout. The only model endpoint is a localhost HTTP fixture with a dummy
token. No auth/config files copied; no live session/database/install edits.
agents.enabled=false is additionally set for safety. The fixture returns HTTP400
on the first POST before any inference, generated response or tool invocation.

[Capture receipt](../../../docs/steering/evidence/2026-11-02-codex-sharing-wire-trace/offline-capture.json):
one POST /v1/responses,49input items, store=false, stream=true. The request still
contains agent_message from /root/diagnostic_crosscheck to /root, with an input_text
content item. This is the original parent's child-failure notification being
replayed, not a new child execution. Native startup/tool events are absent.
The script exit1 is EXPECTED because the fixture intentionally returns400;
it is not a successful inference result or an OpenAI validation verdict.

This replay is not byte-identical to the original HTTP request: it uses CLI exec
resume instead of T3 app-server, adds a synthetic text prompt, disables agents
explicitly and adds fresh startup context. It also emits service_tier=priority,
which is outside the intended sharing configuration; do not attribute that to
T3's original request. The original server's error.param was input. The decisive
measurement is that the current binary really emits a native agent_message over
HTTP from retained history even with agents.enabled=false. Preventing NEW agents
does not repair that existing history.

Failed preflights are retained in source-trace.json: a copied V2 worker could not
be independently resumed without its parent, so no child request was captured;
the capture switched to copied ROOT history. A generated-script newline escaping
error was corrected before execution. Neither reached real inference or spawned
a new native agent. No repeated real-provider reproduction was run.

## Why T3 exposed the incompatible path

The exact released config code establishes the previously inferred precedence:
agents.enabled defaults to true. With multi_agent_v2 disabled, the override is
None unless agents.enabled=false. multi_agent_version_for_model then takes the
model's catalog setting before falling back to the feature flags. The observed
catalog selects V2. Thus both feature flags=false does NOT mean force-disabled.
The installed launch builder omits the independent agents.enabled=false switch.
This is a concrete integration guard gap, not the user's missing filesystem grant.

## Next actions and validation/stopping criteria

1. **Prevention owner: T3 integration.** Retain both feature overrides and add the
   independent disabled-agent override on sharing; reuse existing prevention PR
   rather than creating duplicates. Validate declaration absence in an isolated
   mock-endpoint fixture for the exact binary/catalog. This prevents unsafe native
   spawning; it does not enable working delegation or fix existing transcripts.
2. **Recovery owner: T3 integration.** Reuse the separate existing fresh-native-
   session/text-context recovery work. This actual wire capture explains why merely
   disabling agents after a failure cannot repair native items already in history.
3. **Functional delegation owner: separately scoped integration/provider task.**
   Coordinate independent text sessions via supported tools, or use an explicitly
   selected route supporting native-agent items. Do not silently switch billing.
   A simple rename from agent_message to message is NOT validated: preserve trust,
   author/recipient/lifetime semantics and account for encrypted payloads.
4. **Stop now:** format discrepancy and owning boundaries are established without
   another live failure. No settings patch, new child, PR change or real inference
   is authorized/needed for this diagnosis. Passing a mock capture does not qualify
   an implementation or prove a repaired agent works with the real endpoint.

Remaining gap: the original rejected remote HTTP body is not available, nor the
server's per-item validation trace. Child wire inclusion is source-supported and
matches the original error/history; the original request's exact offending index
cannot be asserted. Original root message, standalone copied-root serialization
and successful fixed-agent execution are distinct evidence categories.

## Research and handoff

Read latest steering index and prior diagnosis; reconcile newer concurrent kernel
commits without altering their work. Followed actual installed bundle rather than
assuming the separate t3code checkout matched it. Source request trace inventory
returned no matching trace bundle in the inspected .codex scope (plugins, skills,
vendor and node_modules excluded); local CODEX_ROLLOUT_TRACE_ROOT was empty.
That scoped miss is not an assertion that no other trace exists anywhere.
GitHub primary-source tag/tree/raw fetches are pinned above; tree was not truncated.
No PR creation, takeover, push, comment, monitor or external public publication.

[Source trace and sanitized RPC/envelope evidence](../../../docs/steering/evidence/2026-11-02-codex-sharing-wire-trace/source-trace.json).
The previous [incident report](../../../docs/steering/2026-11-02-codex-subagent-provider-diagnosis.md) remains
historical evidence. This report supersedes its adjacency-only serialization gap,
not its explicit caveat about absent original HTTP bytes.
