# Repository document discovery failure and regression evaluation

Recorded on 2026-10-28 in America/Vancouver. The final evaluation passed 2 of 8
cases before the fallback and 8 of 8 after it. For the actual local content search
`html`, the baseline did not locate the HTML file. Path enumeration followed by
content verification did. These are deterministic procedure measurements, not
measurements of an assistant learning across sessions.

## Incident and evidence boundary

The user asked for an HTML document in `yelgabo/okc` with diagrams comparing kernel
architectures, without supplying its filename. The prior assistant later reported
that it searched GitHub for `html`, received no results, searched an unrelated
ChatGPT file library, and suggested DeepSeek research instead. It also suggested
the requested document might exist only locally. After the user supplied the
filename, the assistant reported that direct GitHub retrieval succeeded.

The source is the [Find gap analysis document conversation](chatgpt-conversation://6ae2a23b-3fcc-83e8-976a-ae3db3f98c47),
read with `read_thread`, `turnLimit=10`, and outputs enabled. The returned five
turns contained retrospective assistant accounts and the user's requests for
persistent learning and a before/after evaluation. They did not contain raw
GitHub search or fetch logs. Treat the historical tool sequence as reported, not
independently replayed. The prior assistant explicitly said its proposed
five-case evaluation had not been executed.

Today, the chat workspace `/Users/gretchengabo/Documents/ChatGPT/okc` was empty
and `git rev-parse --show-toplevel` exited 128 there. A bounded inventory of the
user's GitHub directories located `/Users/gretchengabo/GitHub/okc`. Its origin is
`https://github.com/yelgabo/okc.git`. The checkout contained
[`docs/gap-analysis-2026-10-28.html`](../gap-analysis-2026-10-28.html), also linked
at [the repository path](https://github.com/yelgabo/okc/blob/main/docs/gap-analysis-2026-10-28.html).
This session verified local contents, not the current remote branch.

In the pinned source, line 1 gives the title "Ours vs the 2.1x Layer". Lines
131-135 describe two kernel-launch timelines, ours and the public entry's.
Lines 140-147 introduce the structural comparison and the attention diagram.
Those passages support the requested document's identity. They describe our
implementation versus a public entry, which is more precise than the recalled
"old and new" wording. This work does not revalidate the performance numbers in
the document or its rendered appearance.

## Failure mechanism and durable correction

The reported reasoning failure was treating a search miss as evidence of file
absence without enumerating repository paths. Searching an unrelated library
then supplied plausible-looking substitutes that did not satisfy the request.
The underlying cause of the historical GitHub miss remains unknown.

The simplest structural change is to remove the dependency on search indexing
for file discovery. Start with a repository inventory, then verify content.
Broadening the content query is another viable approach, tested below, but still
depends on guessing terms in the document. A direct fetch works when the filename
is supplied, so it does not satisfy this filename-unknown regression.

The [search protocol](../repository-file-discovery.md) records checkout validation,
path enumeration, content verification, scope expansion, and handling of errors
and ambiguity. The root [AGENTS.md](../../AGENTS.md) points to it and carries the
core rule directly. That instruction file already existed as an untracked file;
its existing instructions were preserved and a discovery section was added.
This is persistent repository guidance. No model weights or global memory store
were changed.

## Test design

The executable design is [design.json](design.json), and the runner is
[evaluate.py](evaluate.py). Both methods receive this filename-unknown request:

> Find the HTML document with diagrams comparing our old and new kernel architectures.

The runner reconstructs all 21 tracked files under `docs/` from commit
`30b2dfff3758a11cd86e0c1a3b8b685a11e1f3d1` into a temporary directory for each
case. The new instructions, this report, and evaluation outputs cannot leak the
answer into the corpus. Each method sees the same corpus and search provider.
The scorer and fixture setup know the expected path. `discover` and `verify`
receive no expected path, document title, date, or gap-analysis filename.

The baseline searches content and verifies any returned candidates, then stops.
The revised method uses the same search and verification, but on an unverified
result it enumerates HTML candidates under `docs/`, then the whole fixture if
needed. This isolates the fallback's effect. The practical protocol starts with
inventory, removing the initial search entirely. The test retains that search to
compare the failure-recovery step directly.

Both methods accept a candidate only if it is HTML, has a title, and contains
evidence for our implementation, another implementation, kernel architecture,
and diagram markup. The shared regex groups appear in the design. Their synonyms
were chosen after inspecting the real document. This is a targeted regression,
not a blind semantic-retrieval benchmark. Both methods return ambiguity if their
verified candidate set contains multiple files.

The natural-language query is recorded as task metadata. Fixed search terms and
regex groups encode its interpretation. The runner does not parse arbitrary
natural-language requests.

`injected_empty` supplies an empty search response to reproduce the reported
failure condition. It does not execute or simulate GitHub's indexing algorithm.
`local_rg` actually runs `rg --files-with-matches --ignore-case -- TERM docs`.
The `html` term comes from the historical account. The `kernel` control tests
whether both methods can recognize the target when content search supplies it.
The runner raises on command errors instead of counting them as empty results.

A case passes only when both status and the complete expected candidate set
match. An unresolved result is correct when the fixture removes the target.
The baseline uses that conservative status, rather than reproducing the prior
assistant's unsupported absence claim or unrelated-library recommendation.

## Actual outcomes

Final observed results are in [results.json](results.json). Each row includes
commands, exit codes, untruncated stdout and stderr, file reads, content hashes,
matching source lines, returned candidates, and the pass decision.

| Case | Input or mutation | Before | After |
|---|---|---|---|
| Reported empty search | Injected empty response, original corpus | Fail, unresolved | Pass, original HTML found |
| Local `html` search | Actual content search, original corpus | Fail, unresolved | Pass, original HTML found |
| Broad content control | Actual `kernel` content search | Pass, original HTML found | Pass, original HTML found |
| Opaque filename | Rename target to `docs/untitled-7.html` | Fail, unresolved | Pass, renamed HTML found |
| Outside `docs/` | Move target to `archive/untitled-7.html` | Fail, unresolved | Pass, moved HTML found |
| Hidden and ignored HTML | Move target under `docs/.archive/`, ignore `*.html` | Fail, unresolved | Pass, hidden HTML found |
| Absent with decoy | Remove target, add unrelated kernel HTML; keep Markdown sibling | Pass, unresolved | Pass, decoy rejected, unresolved |
| Ambiguous copy | Add a second copy with a different filename | Fail, unresolved | Pass, both candidates returned as ambiguous |

The aggregate is 2/8 before and 8/8 after. Among the six cases with one correct
HTML target, discovery succeeded in 1/6 before and 6/6 after. These correlated
fixtures are checks of specific behavior. The totals are not estimates of
general search accuracy, and the repeated run does not increase the sample size.

The local `html` search returned other documentation files but omitted the
target. A direct `rg -n -i 'html' docs/gap-analysis-2026-10-28.html` exited 1 with
empty output. The source uses HTML markup but contains no literal `html` string.
That explains this local content-search miss. It does not establish why the
historical GitHub search returned zero results.

## Initial failed acceptance check

The first executed design assumed the local `html` search would supply the
target and required the baseline to pass that control. The run exited 1. The
baseline passed 1/7 cases, the revised method passed 7/7, and that control
assumption failed. The original evidence remains in
[results-initial.json](results-initial.json), with
[design-initial.json](design-initial.json) and the exact
[initial runner source](evaluate-initial.py.txt).

The final design preserves the `html` case as a measured search, adds the broader
`kernel` positive control, and requires both methods to pass that control.
The discovery and content-verification functions did not change. The design
changed after seeing evidence, so the final acceptance criterion was not wholly
pre-registered. The first run remains visible rather than being presented as a
passing experiment.

## Commands and reproduction

Requires Python 3, Git, ripgrep, and the pinned commit in the local repository.
The runner uses Python's standard library, reads Git objects, and mutates only
temporary fixtures. It does not change the real HTML or run GPU jobs.

From the repository root, the final recorded execution was:

```sh
python3 docs/file-discovery-eval/evaluate.py --output docs/file-discovery-eval/results.json
```

To rerun without overwriting evidence:

```sh
python3 docs/file-discovery-eval/evaluate.py --output /tmp/okc-discovery-results.json
python3 docs/file-discovery-eval/compare.py /tmp/okc-discovery-results.json
```

Corpus setup uses these commands, once for the inventory and once per returned
path. The evidence stores the actual argument arrays and outputs:

```sh
git ls-tree -r --name-only 30b2dfff3758a11cd86e0c1a3b8b685a11e1f3d1 docs
git show 30b2dfff3758a11cd86e0c1a3b8b685a11e1f3d1:docs/gap-analysis-2026-10-28.html
```

The second command is fixture construction, not a discovery query. Discovery
commands use no expected filename:

```sh
rg --files-with-matches --ignore-case -- html docs
rg --files-with-matches --ignore-case -- kernel docs
rg --files --hidden --no-ignore -g '!.git' docs
rg --files --hidden --no-ignore -g '!.git' .
```

Candidate reading and regex checks run inside Python. Their per-file hashes and
matching lines are recorded under `reads`. Environment records include UTC
execution time, tool versions, and runner and design SHA-256 hashes. The corpus
manifest hashes every source file. Compare logical outcomes and hashes between
runs; timestamps and raw ripgrep output ordering can differ. The comparator
normalizes those fields, requires all other saved data to match, and checks the
saved final and initial runner/design hashes. Tool-version or platform changes
will fail strict comparison even if the case outcomes agree.

## Verification and independent review

A separate reviewer read the procedure, report, and saved evidence and ran the
suite independently. That run exited 0 with 2/8 before and 8/8 after. Its complete
output is preserved as [results-repeat.json](results-repeat.json). The
[comparison output](verification.json) records equality after normalizing time
and file-list ordering, along with source-hash checks. The reviewer confirmed
that expected filenames stay outside discovery inputs and that `discover` and
`verify` stayed unchanged between the initial and final versions.

The review found no blocking issue. It requested explicit wording that fixed
terms encode the natural-language query, and that output ordering can vary.
Those limits are stated above. Review covered retrieval logic and evidence
claims. It did not audit GitHub indexing, future assistant behavior, rendered
HTML, or kernel-performance claims.

## Limits and follow-up

This evaluation exercises a local procedure on one known document. It does not
replay the earlier assistant, run a new blinded assistant session, measure search
latency, or test current GitHub code search. The historical five proposed cases
were broader conversational behaviors, so these results do not certify that
earlier 5/5 proposal.

The verifier checks source patterns rather than meaning or rendered graphics.
Its title and term checks can admit a carefully constructed false positive.
Renaming tests resistance to filename hardcoding, not independence from document
wording. The source corpus contains only the pinned `docs/` tree, with explicit
temporary mutations. Full checkout performance, other refs, permissions,
connector outages, pagination, and truncated inventories remain untested.
Enumeration stops after a verified match in the prioritized directory, so
ambiguity elsewhere can remain undiscovered. Future work that needs all matches
must enumerate the full scope before selecting candidates.

The cheapest test of future assistant behavior is a fresh session given the
filename-unknown request and this repository's instructions, with no target path
in its prompt. That would test instruction use as well as retrieval. This run
establishes only the recorded procedure's results and the saved correction.
