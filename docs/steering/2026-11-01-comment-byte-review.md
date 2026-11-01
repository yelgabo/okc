# Independent comment-only byte-plan audit — 2026-11-01

Scope: full G production closure and owned verify16/32 closure, without constexpr/body pruning. Audited owner tool `tools/prod_size_comments.py` SHA256 `fe54acb10fd39c09147cdf46521b60ab41463a7e2a564ac511e92dd506bf19c9` in sibling `okc-today-proj`. This is source-semantic clearance of the inspected corpus, not public authorization or CUDA/performance qualification. No GPU or official action occurred.

## Result

No concrete preprocessing, Python semantic, or legal-notice loss blocker found in the actual emitted sources. The independent census checks15 G and9 owned native literals; all24 pass phase2-first token comparison and retain exact newline counts. Seven supplemental fixtures pass raw strings, escaped quotes, line-comment backslash continuation, token adjacency, block-comment newlines, legal contiguous groups and Python string/comment separation.

The owner reported mask4=903809B and all-four=947129B after comment stripping plus existing transparent factoring, versus1072104/1125548 before. Those byte figures are owner's screen results; this review does not independently qualify the final wrapper or its runtime behavior. Final frozen physical hash, escaped embedded byte budget and selected native SASS comparison remain root gates.

## Preprocessing and source behavior

The transform replaces removable C++ comments by a space plus their original CR/LF characters. This preserves adjacent preprocessing-token separation and line numbers. Ordinary and prefixed raw strings are skipped, including comment-shaped contents; escaped quote fixtures pass. Newlines retained mean no new `__LINE__` value, although actual emitted sources contain neither `__LINE__` nor `__FILE__`. Macro backslash continuations exist (G attention2; owned ONE8/host3) and were checked after translation-phase2 splicing, rather than trusting the transform's self-consistency assertion alone.

Important scope limit: this is not a general standards-complete preprocessor. Its initial scanner does not globally perform phase2 before recognizing comments, so splice-created delimiters are a theoretical unsupported lexical case. The added phase2-first check passes the actual corpus. Both scans use the owner's scanner after phase2; this is an independent ordering check, not an independent C++ compiler tokenization proof. A different source pin must rerun the census and selected compiler/SASS gates. No reason to weaken fences/barriers follows from source-size success.

Python removal uses tokenize.COMMENT and checks unchanged AST; literals/docstrings and blank lines remain. No emitted `inspect`, `getsource` or `linecache` references were found. Triton itself may derive compilation/cache identity from function source; stripped comments can change hashes/cache misses and startup cost, but the function AST, decorators, source line numbers and native instructions are not intentionally changed. H200 preparation budget remains unqualified until actual qualification. This review makes no blanket claim about every external JIT implementation.

## Notices

Native contiguous comment groups containing copyright/license/NOTICE/SPDX/attribution/derived-from/GitHub references are retained verbatim. Independent census also searched removed comments for author/adapt/borrow/URLs and found only false-positive mathematical “credit” comments. Python legal comment groups and `# okc:` headers are retained. Author lines immediately adjacent to legal lines survive the group rule, confirmed by fixture. Existing string literals, including LICENSE/NOTICE text and docstrings, are not stripped by Python comment removal. This establishes preservation for these original closures, not that absent upstream notices have been independently reconstructed.

## Evidence and next gate

[Native literal census](evidence/2026-11-01-comment-byte-review/native-census.json) pins original/transformed hashes, lines and legal-byte totals. Reproduction scripts: `tools/attention_comment_byte_audit.py` and `tools/attention_comment_byte_fixtures.py`, run from the attention worktree with sibling root/projection tools available; they intentionally do not load GPU modules. Actual closure inputs were854399B G and399155B owned. Reproduce on the exact frozen tool/source revisions before composition.

Proceed only to frozen artifact/source review and selected compiler checks. Require full legal text, all12 dispatcher/native/fallback selections, per-state workspace ownership and startup guards to survive composition. Byte fit is necessary but provides no correctness/performance result; retain the prior public incumbent until a separately authorized exact candidate qualifies. Stop if phase2/token/AST/notice/hash checks fail, if final payload reaches cap, or if required startup coverage cannot complete within budget.
