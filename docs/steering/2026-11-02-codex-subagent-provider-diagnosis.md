# Codex native-subagent provider failure: recovered diagnosis

Date: November 2, 2026 (America/Los_Angeles); inspection approximately 09:58-10:08 UTC.
Evidence checkout: C:/Users/gretc/Documents/GitHub/okc, origin
https://github.com/yelgabo/okc.git, HEAD 8689680 at inspection.
Scope: diagnosis and repository handoff only. No native agents, delegated tasks,
new inference probes, live configuration edits, application restarts, kernel edits,
or challenge jobs were launched. Existing unrelated work is preserved.

## Conclusion

This is a runtime/provider compatibility defect plus a distinct conversation
recovery defect, not evidence that the user's repository permissions are wrong.
The failed diagnostic agent itself invoked native collaboration.spawn_agent;
the child and then its parent received subscription_sharing_unsupported_capability
with param=input and willRetry=false. Local agent_message records immediately
precede the failures. The actual rejected HTTP body was not captured, so identifying
agent_message as the precise wire item remains a strongly supported explanation,
not a packet-level proof.

Do not generalize this to all forms of subagents or all Codex authentication.
Current official documentation explicitly allows child coordination implemented
through supported function/custom calls. The observed native-agent history format
on this installed ChatGPT-sharing path is the narrower incompatibility.

## Exact incident and identity

Read the failed T3 thread, not merely its title:
<t3-thread://v2/471a5a44-f247-4d6c-b35d-7f45e6f30552/6a582167-166e-49e8-b0c1-29cd50228f79>.
Its only run is ordinal 1, full-access, gpt-6.1-sol, failed.

- Parent native thread: 01a19bae-c402-79c3-a27f-baff9e5fd2ac.
- Parent native TURN, not thread: 01a19bae-c605-7183-bc87-167a80609316.
- Worker native thread: 01a19baf-be3b-7453-8a89-11e9fc085579.
- Worker native turn: 01a19baf-bf44-76f1-bd84-28dc9a76e4e3.
- Spawn call: call_bb3e37f2f0864f708e3dc31d48e7e8ef.

Measured timeline on November 2 (UTC): spawn recorded at 09:55:48.402;
child agent_message at 09:55:48.902; child rejection 09:55:51.978;
parent agent_message 09:55:54.160; parent rejection 09:55:55.952;
T3 terminal error at 09:55:55.960. The child rollout includes inherited parent
session_meta, so its metadata ID is not used to overwrite its filename/protocol
identity. No later prompt was sent to this failed thread during diagnosis.

[Sanitized local evidence](evidence/2026-11-02-codex-subagent-diagnosis/local-evidence.json)
contains selected protocol records, source hashes, rollout structure with all
message/tool-argument bodies omitted, installed launch code, and upstream pins.
Raw logs, authentication files, databases and credentials are not published.

## What is actually installed

T3 remains 0.1.6 build 2359643cffa6. Its binCli-C12fvdZY.mjs bundle SHA256 is
472d7e2d82900ae9da669da1dfd9cd4b7075f61e6e6ac9bda9d6dd9644550d25,
identical to the November 1 reproduction. However, today's native rollout records
Codex 0.166.0 and the running npm executable independently returns 0.166.0.
Yesterday's managed 0.156.1 must not be presented as today's runtime.

The running process and installed managed launch builder both retain
features.multi_agent=false and features.multi_agent_v2=false. The builder has
no agents.enabled=false override. The shared cached catalog selects v2 for both
gpt-6-astra and gpt-6.1-sol; that is cache evidence, not a captured model-catalog
response for the rejected request. Together with the actual spawn, this confirms
that those two feature flags did not prevent native-agent exposure in this run.
It does not prove the exact catalog precedence in every CLI version.

The source of the local npm executable is not established: do not claim who
updated it or which T3 setting selected it. The error is an input-capability
rejection, not an observed OAuth, usage-quota, filesystem, or sandbox denial.

## Prevention and recovery are different

1. **Prevention:** PR [28143](https://github.com/pingdotgg/t3code/pull/28143),
   head 3e12accb4bef36d2d30da3691e4b285956b9aef2, is open/unmerged at inspection.
   It adds agents.enabled=false to the managed sharing launch, retaining both
   feature flags. Its author reports a Codex 0.163.0 mock-endpoint test proving
   removal of native tool declarations. That test was not rerun here and does
   not establish Windows 0.166.0 end-to-end behavior. Earlier PR
   [20270](https://github.com/pingdotgg/t3code/pull/20270) is closed/unmerged;
   a revised PR description is not evidence it shipped.
2. **Recovery:** our existing PR [28962](https://github.com/pingdotgg/t3code/pull/28962),
   head 332447eea868c2b89328187309e063c99fce1ebb, remains open/unmerged.
   It replaces rejected native history with a fresh native session plus portable
   text context on the next prompt. It does not make native-agent input supported.
   The unchanged installed bundle predates it. The
   [November 1 report](2026-11-01-t3-sharing-recovery-pr.md) and its runtime receipt
   already show copied broken history failing and a fresh text-context replacement
   answering BASELINE_OK and FOLLOWUP_OK. Those are previous measured results,
   not new tests of today's client.

Changing approval mode cannot add an unsupported input type. Re-enabling the
feature flags or repeatedly logging in is not an evidence-backed repair. A new
thread with a concise textual handoff avoids reusing the failed native transcript;
this thread's ongoing diagnosis is not a successful native-subagent test.
An API-key provider is suggested by the actual backend diagnostic, but changing
authentication/billing is an explicit user choice and was not performed. Neither
that alternative nor T3-managed delegation was tested here.

## Priority, owner, next gates and stopping criteria

- **Immediate:** current agent remains single-agent; user continues work in this
  clean thread. Do not replay the failing spawn to reconfirm existing evidence.
- **Next implementation priority:** T3 maintainers (or a separately authorized
  local-fix task) should integrate prevention AND fresh-history recovery, rather
  than submit a duplicate recovery PR or change this OKC project's code.
- **Prevention validation gate:** use an isolated temporary home and local scripted
  endpoint with the exact 0.166.0 binary and a v2 catalog. Inspect declared tools
  with/without agents.enabled=false, holding the existing flags fixed. Do not
  spawn an agent or call the real provider. Success means native-agent tools are
  absent while ordinary execution remains available. Stop on any real-provider
  routing or modification of live T3 state. Not executed in this task.
- **Recovery validation gate:** reuse the existing deterministic rejected-history
  tests; any further real-provider recovery check needs a separately authorized,
  isolated copied history. It must preserve the visible conversation and selected
  provider/model without retrying the unsupported action. No new live probe is
  justified just to repeat yesterday's result.
- **Remaining gaps:** exact outgoing HTTP item; fresh isolated 0.166.0 prevention
  probe; installed-client end-to-end recovery; verified functioning native-agent
  route if the user's goal is delegation rather than safe single-agent operation.
  A prevention-only patch deliberately disables native agents and cannot satisfy
  that last goal. No global assertion that Codex cannot run subagents is warranted.

## Research trail and publication

Read docs/steering/README.md and its latest G224 terminal report first, then the
November 1 sharing reproduction, recovery and evidence-visibility correction.
Repository path inventory used rg --files and git ls-files for sharing/codex/T3
candidates. The exact installed runtime was read separately only after those
reports identified its location. One exploratory PowerShell rg wildcard path was
invalid; it was replaced with a directory-scoped search, not treated as a miss.

Official source fetched November 2:
[Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
(local tools and agents; supported input/tool formats) and
[Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
(unsupported-capability means remove the invalid feature/input, not repeat it).
An initial exact-topic official-domain web search yielded no usable tool output;
the connected web tool successfully fetched the direct official pages. GitHub
search query repo:pingdotgg/t3code "agents.enabled" independently found the
current prevention PR; its state and all three cited PR heads were read directly.
No exhaustive absence or universal compatibility claim follows from this search.

This report and receipt are for the private repository handoff, not anonymously
accessible public evidence. Existing public gist and PRs were read only. No PR
was created, pushed, taken over, commented on, or monitored in this task.
