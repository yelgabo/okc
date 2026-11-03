# Code and context cleanup

November 2, 2026 Pacific. Audited base `3a961a9da35c06adb6eaf04e2531b9d323587933`. User authorized aggressive deletion to reduce context and required comments to explain why in at most two lines.

## Result

Removed 21 retired Python snapshots: **271,541 physical lines and 12,002,399 bytes**. These were old complete layer candidates/releases, obsolete standalone benchmark outputs and retired attention/boundary experiments. The [manifest](evidence/2026-11-02-context-cleanup/manifest.json) lists every file, SHA256, reason and exact Git recovery command. This removes copies from the current checkout; Git history still contains them.

Eight instruction/index files shrink from 72,741 to 16,136 bytes, **77.8% less text**. Root AGENTS is the shared rule source; CLAUDE points to it. Each problem's instruction file points to the task and shared rules. The steering entry point is 37 lines, with the preceding complete index [preserved byte-for-byte](index-history-through-2026-11-02-context-cleanup.md). These are file-byte measurements, not measured model-token usage.

Default ripgrep discovery now exposes 1,193 working files instead of traversing historical profiles/evidence and retired candidate copies. The baseline Git inventory contains 5,316 tracked paths. These counts describe different intended views, not a like-for-like file-deletion percentage. Current source, tests and the scored G incumbent remain visible. `git ls-files`, explicit paths and `rg --files --no-ignore` still expose archived evidence.

## Findings and changes

The main source of wasted reads was duplicated complete submissions and repeated instructions. Removing only comments inside a current kernel would leave those copies in ordinary searches. The existing [comment-fit work](2026-11-01-production-comment-fit.md) already has a qualified lexical/selected-compiler analysis for its exact older source epoch; this audit does not repeat it or apply its conclusions to different code.

The first nine deletions are generated attention/MoE/projection benchmarks and retired experimental submissions. Twelve further deletions are superseded self-contained layer candidates and a historical release copy. Their names have no references in the remaining tracked Python/shell files outside archived profiles/evidence. That is a bounded consumer search, not proof about arbitrary external commands. Referenced historical inputs, including layer_attn31s and layer_lt_boundary, remain because tools still read them. Profiles, receipts and old reports remain.

The projection inliner also copied `pack_router` into the layer despite no load of that name. It now omits the helper. The original function remains in the frozen/current source copies where changing bytes would invalidate existing evidence. This is removal from future generated layer output, not a new runtime optimization.

Six inliners lose long module explanations, historical comment blocks and redundant comments. Their remaining docstrings and newly emitted comment blocks are at most two lines and state a reason. Marker strings, legal attribution and required PDL waits remain. Root instructions retain the two-line rule for future edits. Frozen benchmark sources retain their existing comments so current source pins remain valid.

Generated output paths are ignored to prevent another copy accumulating after a local bench build. Three retained benchmark generators were exercised in temporary roots and produced parseable Python from the current source.

The audit also caught a CP1252 historical Markdown file during the bulk link check. The file was preserved unchanged; root instructions now require preserving original encodings. No removed-source Markdown links required rewriting. Historical plain-text paths can be restored with the manifest's Git commands.

## Verification

[Reproducible generator/source check](evidence/2026-11-02-context-cleanup/verify.py) and [results](evidence/2026-11-02-context-cleanup/verification.json):

- All six actual inliner entry points ran against current source in temporary files. A second run produced identical bytes.
- Generated Python ASTs match the old generators after removing only the unused `pack_router` definition. Native bindings match exactly or have identical primary Clang raw-token sequences after comments/whitespace are excluded.
- Six current submission/incumbent files are byte-for-byte unchanged against the audited base, including public607 SHA256 `ba9e327c`. No current CUDA literal, dispatch, numerical fallback or native build flag changed.
- Every removed source is recoverable from the exact base with its recorded hash. The complete pre-integration steering index matches origin/main at `74f7f1c0`; concurrent steering and new frame-access restrictions were retained.
- 21 existing CPU tests pass: 6 comment/marker fixtures, 3 selected-factory tests and 12 compiler-wrapper/research-checker tests. The document-discovery evaluation passes 8/8 cases.
- Three benchmark generators still emit parseable payloads. Search exclusions, current-source visibility, ignored output paths and the two-line limit in edited generators were checked.

The repository-wide CUDA research catalog check still fails on its pre-existing `docs/ruled-out.md` mismatch. The mismatch already existed at the audited base, where catalog expected `e375f308...` and the file was `b3bc424a...`. Later concurrent negative entries were retained. The cleanup itself does not change the file. Its owning research notes must reconcile the newer negative evidence before the hash changes. No catalog hash was refreshed to hide this failure.

The checks cover CPU source generation and lexical identity. No CUDA compilation, GPU execution or performance test was performed. The frozen runtime source bytes are unchanged. No separate agent reviewer was used under the standing restriction on new project agents.

## Remaining scope

This is a dependency and context audit, not proof that every tool is necessary or that all code is free of dead branches. Dynamic dispatch, factory rebinding, neighborhood fallbacks and pinned experiments require more than name counts. In particular, TypedDict interfaces and judge entry points are loaded externally and are not dead merely because an AST sees no local call.

Retain the frozen active sources and known tool inputs. Further deletion should trace a named source consumer, remove a complete retired dependency group, preserve a recovery pin and rerun the affected generator. Any replacement of the current G source needs its own new source epoch and full selected-factory/correctness gates; it should not silently mutate source owned by an active experiment.

No kernel job, quota, scored submission, public code release or worker ownership changed. The next maintenance rule is to keep generated payloads out of the tracked working tree and keep routine steering in the compact index.

