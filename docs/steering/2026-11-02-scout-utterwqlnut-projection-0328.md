# Public scout: another projection implementation

November 2 Pacific / November 3 UTC. Evidence base `0585bd216cb980e8b47a66eef0ba4c3b0c5768a3` and [retrieval ledger](evidence/2026-11-02-scout-utterwqlnut-0328/coverage.json). The scout read public sources only. Current kernel workers and their jobs retain their assignments.

## New source and what it establishes

Independent repository-name search found [utterwqlnut/muse_bf16_proj](https://github.com/utterwqlnut/muse_bf16_proj/tree/e5c79030dc251222c4e98cf79af8679c1b0f29c7), outside the earlier 36-repository watch. Its [November 3, 01:08:08 UTC commit](https://github.com/utterwqlnut/muse_bf16_proj/commit/e5c79030dc251222c4e98cf79af8679c1b0f29c7) changes actual source and adds experiment records. This is newly discovered by this scout, not necessarily newly published since the previous sweep.

The 57,776-byte [submission.py](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/submission.py) has SHA256 `28efd0826317814da738e0783072732d716126ce60c115fc30dc70504c0190ca`. Source inspection found preparation, fallback dispatch, a lossless 12-bit single-token GEMV, and a raw BF16 TMA/WGMMA path whose K split reduces through cluster shared memory. Its configured ranges include QKV at 2 through 128 tokens and the unpooler slice at 2 through 8. `prepare` packs weights, compiles and checks candidates, then installs only successful routes. This establishes source structure, not numerical or concurrency correctness.

| Author-hosted receipt | Reported result | Association limit |
|---|---|---|
| [74b36974863c4d1e](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/profiles/submission.74b36974863c4d1e.json) | Passed 9/9; 1.356457x, 95% CI 1.336110 to 1.376804 | JSON names a submission pathname but contains no source hash. Its companion log enables the configured cluster routes; profiles name `wa_gemm`. |
| [9bd833b165bc488c](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/profiles/submission.9bd833b165bc488c.json) | Passed 9/9; 1.310338x, 95% CI 1.290683 to 1.329993 | Earlier receipt lacks the cluster-route log. Different runs do not isolate the cluster treatment's effect. |

The later receipt reports QKV medians of 12.122 us at 16 tokens, 12.817 at 32, and 14.874 at 128. These are that receipt's adjusted timings. They are not a matched comparison with our implementation. Our [projection run log](../runs.md) already records a 1.527806x scored result, so the new aggregate supplies no basis for replacing our base.

The commit also adds [author notes](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/AGENTS.md) and [sweep tooling](https://github.com/utterwqlnut/muse_bf16_proj/blob/e5c79030dc251222c4e98cf79af8679c1b0f29c7/sweep_cuda.py). The notes distinguish local pseudo-judge timing from published judge receipts, report cluster-residency and reduction costs, and describe losses on O, larger unpooler shapes and 256-token cases. Treat those mechanisms and explanations as author reports. Our existing [ruled-out cluster variants](../ruled-out.md) remain closed under their exact conditions.

[Normalized receipts and file hashes](evidence/2026-11-02-scout-utterwqlnut-0328/source-and-receipts.json) retain all 30 cells per receipt. Six downloaded files matched their published Git blob identities. The returned tree had 36 entries and was untruncated; it contained no LICENSE or NOTICE file. Source-to-receipt hash binding, adoption terms, compiled resources, graph lifetime, ordering and fresh correctness/performance remain unchecked. No foreign code was executed.

## Coverage and decision

All five identifiers were searched across public GitHub code, repository names, commits, issues and pull requests. Code pagination reached projection page 4, attention page 3, MoE page 3, layer page 3 and prefill page 1. Totals changed while paging. No new repository/path/blob combination appeared among the returned code hits; repository-name search found the new source above. Commits and discussions used the first 100 newest results per query. Temporary search limits were recorded and failed searches retried after cooldown. This is bounded discovery, not proof of absence.

All 11 previously pinned repository heads were unchanged. Five principal source repositories had their branch, fork and release lists checked; NathenFernandes still uses `solution`. The returned vjeux fork was already known. The separately checked operatorstack, lee101 and rohanarun heads match older project pins. The broader search also found NeuroSpark's mixed FPGA/H200 repository; its returned tree supplied no attributable H200 receipt, so it creates no stronger performance claim.

[Social coverage](evidence/2026-11-02-scout-utterwqlnut-0328/social.json) includes two public Guild feed pages, all five communities and public comments on nine threads. Registered-only bodies stayed unread. Rohan Arun's mirrored post reports physical-bound-guided search and a Lean experiment, without a checked source-bound result. Indexed Reddit/Hacker News queries yielded no verified new competition discussion. LinkedIn and participant blogs supplied background, without a new stronger implementation. X mirrors retain their provenance limits. Kernarc remains inaccessible. The earlier keell0renz caveat is unchanged and receives no repeat alert.

Add this source to the watch for changed dispatch or source-bound receipts. Its priority is reference material, below the current qualified worker assignments. Do not dispatch a comparator or reopen a cluster family from this aggregate. A useful next gate would be an exact source/receipt binding and a named selected-shape advantage over the current implementation, with preparation and reduction costs included. Stop this sweep here; no new GPU allocation, kernel change or public submission follows.
