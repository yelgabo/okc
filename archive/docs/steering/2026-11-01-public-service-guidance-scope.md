# Public service guidance: scoped inspection

Inspection completed by the owning commit `7e74778`, recorded at 2026-11-01 23:00:31 UTC. A tool clock reading during retrieval was 22:59:51 UTC; the exact inspection start was not recorded. Read-only public-source investigation; no judge API, CLI status, GPU, submission, login, registration or external message was used.

Timestamp correction: the initial report's estimated 23:05 end was in the future and was incorrect. The times above use the actual tool clock and Git commit timestamp; the scoped findings are unchanged.

## Result

No authoritative kernel-worker maintenance announcement, recovery confirmation, queue deadline or recovery ETA was found in the inspected public material. This is a scoped retrieval result, not evidence that no announcement exists or that workers are unavailable. A reachable public site, an authenticated API response, and one queued job do not establish worker health.

The [official homepage](https://openkernelchallenge.dev/) explicitly discusses waiting for a free node in its **serving track** section. That guidance describes whole-node serving submissions and cannot establish expected queue time for the current kernel challenge. Its illustrative kernel CLI animation includes a queue stage, but supplies neither a current queue measurement nor a service-level promise. The [rules](https://openkernelchallenge.dev/rules) describe the judging hardware and run constraints; the inspected material did not provide kernel queue recovery guidance.

## Source and access ledger

| Public source | Observation | Limitation |
|---|---|---|
| Homepage and rules above | Search-index rendered content and directly retrieved public frontend inspected | Search-index freshness is not live worker telemetry |
| [Homepage frontend asset](https://openkernelchallenge.dev/assets/index-VBg5pvCv.js) | Public application text inspected without executing it or requesting its API endpoints | Framework occurrences of worker/recovery are not operational statements |
| [Possible docs path](https://openkernelchallenge.dev/docs), [possible status path](https://openkernelchallenge.dev/status) | Web retrieval reported inaccessible | Guessed paths; this does not establish whether dedicated documentation/status services exist |
| [Paradigm account](https://x.com/paradigm), linked in public frontend | Retrieval returned 403 | Latest organizer announcements could not be checked there |
| Public website GitHub links | GPU MODE reference-kernels/kernelbot attribution and GitHub device login | These links do not identify an official challenge operations repository |

Official-domain maintenance/queue/status searches returned no relevant status notice. Broader searches surfaced community material and social mirrors; those were not adopted as authoritative recovery evidence. An attempted guessed Paradigm announcement URL was inaccessible and is not treated as an inspected announcement.

## Action and stopping rule

There is no verified public action or ETA to relay from this inspection. Preserve the accepted job and its owning observation history; the coordinator remains the sole owner of permitted status observation. Local observation expiry is not a terminal job result. Do not duplicate or resubmit a job based on these public-search misses. Reopen this narrow investigation on a specific organizer announcement/status URL or other primary-source evidence, rather than repeated speculative endpoint discovery.

The coordinator supplied the operational context: prior `1fad70bbe3854cbc` infrastructure error/refund and accepted `3fd1317143304bd5` queued with no start/cluster metadata. Those facts were **not independently queried here**. This report neither changes their classification nor infers a cause.
