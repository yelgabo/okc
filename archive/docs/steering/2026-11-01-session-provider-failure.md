# Session provider failure after worker spawn

Date: 2026-11-01. Evidence checkout: C:/Users/gretc/Documents/GitHub/okc,
main and fetched origin/main at b6def23b9a0a0a96a69ef97ff6c028047e7a077e.
Remote: https://github.com/yelgabo/okc.git.
Read the steering index and latest orientation before this audit.

Session: <t3-thread://v2/471a5a44-f247-4d6c-b35d-7f45e6f30552/40356bc9-4e84-4f44-81ab-572185175234>
Source: T3 thread activity positions 24-32 and recent run metadata, captured
during this audit. Exact source extract: [session receipt](evidence/2026-11-01-session-provider-failure.json).

## Observed evidence

The initial orientation turn completed. The user then requested a worker.
Run ordinal 2 attempted a native worker spawn; the subagent activity failed
at 23:20:52.416 UTC. The parent failed at 23:20:57.161 UTC with:

> Codex used a feature that ChatGPT sharing does not support. Use another provider for this request.

Ordinals 3, 4 and 5 repeated that exact provider error, including on simple
follow-up messages. Their run IDs are
`run:thread:40356bc9-4e84-4f44-81ab-572185175234:ordinal:3`,
`run:thread:40356bc9-4e84-4f44-81ab-572185175234:ordinal:4`, and
`run:thread:40356bc9-4e84-4f44-81ab-572185175234:ordinal:5`.
Ordinal 2 used gpt-6-astra; ordinals 3-4 used gpt-6.1-sol; ordinal 5 used
gpt-6-astra again. All used the codex provider. Last failure: 23:26:58.892 UTC.

## Interpretation and limits

The recorded immediate failure is a provider compatibility rejection associated
with the worker-spawn attempt. This does not implicate repository code or the
earlier H200 preparation-budget failure. Model changes within the same provider
did not recover this thread.

A retained unsupported event or session state is a plausible explanation for
why later ordinary messages also fail. The activity log does not reveal the
underlying transport payload or exact unsupported feature; this remains a
hypothesis, not a proven internal cause. The failed worker produced no visible
result. No broad claim about all Codex subagent support follows from this case.

## Priorities, next actions and stopping criteria

Use the current functioning session for ordinary work. Avoid repeating the
same native worker-spawn action through this failing session. The error itself
suggests another provider; switching providers has not been tested here and
requires the user's choice. This audit does not change providers, send messages,
launch workers, or resume the stopped kernel campaign.

For a later authorized recovery attempt, validate one simple completed turn
first, then a bounded worker task through the chosen supported integration.
Stop if the same compatibility error returns. App/provider transport logs would
be needed to establish the internal cause; no logs outside the provided session
activity were inspected. No GPU job, kernel edit, or submission was performed.

