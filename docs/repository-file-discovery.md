# Repository file discovery

When a user describes a repository document without its filename, inventory the
repository before relying on indexed search. A search miss establishes only that
the search returned no matches. It cannot establish that a file is absent.

This protocol comes from the [2026-10-28 discovery failure and evaluation](file-discovery-eval/report.md).
The root [AGENTS.md](../AGENTS.md) loads the correction in future sessions.

## Confirm the search location

```sh
pwd
git rev-parse --show-toplevel
git remote -v
git rev-parse HEAD
git status --short
```

Check that the remote identifies the requested repository. A chat workspace can
be an empty directory with the same name as a real checkout. If it is, locate the
checkout through configured projects or a bounded search of known repository
roots. Record the actual directory and ref. Do not treat an empty chat workspace
as an empty repository.

For a remote-only repository, inspect the directory contents or recursive Git tree
at a resolved ref through an available GitHub tool. Record the ref and any
pagination, truncation, permission, or network errors. Code search and directory
enumeration answer different questions. Change retrieval methods after an empty
search or a user correction.

## Enumerate paths, then read candidates

Start with the likely directory. Here, reports and visual comparisons live under
`docs/`. These commands work without knowing a filename:

```sh
rg --files docs
git ls-files -- docs
rg --files docs -g '*.html' -g '*.htm' -g '*.md'
```

Treat the extension as path metadata. Searching file contents for `html` does not
list HTML files. Keep both HTML and Markdown candidates when the user describes
diagrams. Preserve the inventory before narrowing by topic words such as `gap`,
`kernel`, `layer`, `architecture`, `ours`, or `public`.

```sh
rg -n -i -g '*.html' -g '*.htm' -g '*.md' \
  'gap|kernel|layer|architecture|ours|public' docs
```

Read each plausible candidate's title, headings, and the passages or diagram
labels that match the description. For HTML, check the document source, including
SVG labels or diagram markup. A page containing the word `kernel` alone is weak
evidence. A comparison document should identify the compared implementations and
describe their structural differences. When appearance matters, open the page
and inspect the rendered diagrams as a separate check.

Return the verified relative path plus a usable local or repository link, and
cite the title or matching passage. If several candidates satisfy the request,
show that ambiguity. Do not silently choose the first path or substitute a
related research document from another library.

## Widen before reporting a miss

If the likely directory yields no verified match, inventory the whole checkout:

```sh
rg --files
git ls-files
rg --files --hidden --no-ignore -g '!.git'
```

The last command includes hidden, ignored, and untracked paths. Inventory names
first, then read relevant document candidates. Avoid indiscriminate content
searches through secrets, dependencies, and generated assets. If the inventory
is too large, record exclusions and search the relevant excluded directories
separately. Never describe a truncated inventory as complete.

Distinguish `rg` exit 1 for no match from exit 2 for an error. A missing directory,
unavailable connector, or unresolved ref requires a scope or access explanation,
not a file-absence claim. After exhausting the recorded scope, say which checkout,
ref, and paths produced no verified match. Other branches, untracked files on
another machine, and external libraries remain outside that evidence.

## Regression check

```sh
python3 docs/file-discovery-eval/evaluate.py --output /tmp/okc-discovery-results.json
```

The test uses a pinned repository snapshot, a filename-unknown query, real local
content searches, and explicitly injected empty-search cases. It measures a
deterministic fallback procedure. It does not measure model training or establish
that every future assistant will follow these instructions.
