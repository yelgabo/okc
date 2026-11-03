# T3 PR evidence visibility correction

Date: 2026-11-01 (America/Los_Angeles); verification occurred November 2 UTC.
This corrects the public-access implication in the prior recovery handoff,
commit `5705f88e3c0a527a91beacfeb5e9f25ad2e1d1fc`.

## Finding and cause

`yelgabo/okc` is private. Anonymous requests to its GitHub API metadata, the
original report link, and raw evidence links returned HTTP 404. Authenticated
verification in the previous task proved the files existed but did not prove
reviewers could access them. Treating those as equivalent was an agent error.

## Completed correction

At the user's explicit request, created a public gist containing only sanitized
reproduction/recovery evidence and test outputs:

https://gist.github.com/yelgabo/2b98e4c3d9f83835e856f4931478f92e

Verified revision: `e555639aea128fd076c3fbe4eb92c6480e9b5ed1`.
The gist page and all 13 raw files returned HTTP 200 through unauthenticated
requests. Each downloaded file matched the intended uploaded bytes.
The corpus excludes credentials, raw session dumps, unrelated conversations and
private project documents. Rollout metadata retains only harmless disposable
tool arguments (task name and timeout); message bodies are omitted.

Updated https://github.com/pingdotgg/t3code/pull/28962 to replace its private
evidence links with the public gist. An anonymous API read returned HTTP 200,
confirmed the gist link and confirmed no `github.com/yelgabo/okc` links remain.
T3 code is unchanged at `332447eea868c2b89328187309e063c99fce1ebb`; no tests or
model calls were rerun for this publication correction.

[Verification receipt](evidence/2026-11-01-t3-sharing-recovery/public-access-verification.json)
records per-file hashes, public visibility, revision and PR-link checks.

## Next gate and stopping criterion

Public reviewer access is now measured rather than inferred. For future external
handoffs, verify evidence links without credentials before calling them public.
Maintainer review remains the next gate; existing validation and deployed-client
limitations in the PR remain unchanged. No repository visibility setting was
changed, no deployment occurred, and no PR monitoring was started.
