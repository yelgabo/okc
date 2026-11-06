# Process incidents behind the paper's lessons

The paper's "Process lessons" section cites four incidents whose details were kept in the lead
session's notes at the time rather than in a ledger. This file records them in the repository.
Times are Pacific.

## Shared checkout reset (2026-11-03, 02:05)

Another session committed a merge in the shared main checkout while the lead had staged a
README restructure, a `.gitignore` line and two newly tracked folders. The merge's reset wiped
the working tree. The staged new files were deleted from disk and recovered from dangling blobs,
matched by size, with `git fsck --dangling`. The README and `.gitignore` edits were lost. A
subagent's manifest edit landed in that merge commit instead of its own.

Rule adopted: edit, `git add`, `git commit` and `git push` in one command, and never leave the
shared tree dirty.

## Union merge of a CUDA source (2026-11-03)

Two attention lanes both edited `muse_attention_decode/attn_one.cu`: one changed the PSTEAL
loop, the other renamed the ring-depth stage-run variable. The lead's conflict script kept both
sides of every conflict, which is valid only for append-only ledgers. On the CUDA file it left two
stage-advance lines, one inside the wrong branch, and the result was pushed before anyone noticed.

Rule adopted: keep both sides only for `docs/*.md` ledgers; resolve source conflicts by hand and
note in the ledger that the merged source has not been judged yet.

## A flag that turned off the edge being measured (2026-11-04, 18:55)

`tools/bx_box_run.py --skip-pdl8-gate`, passed by most box A/B scripts, skips `_pdl8_setup`, so the
expert kernel ran without its PDL edge at T = 1, 8, 16 and 32. The judge and EAT's evaluation run
with that edge on. The decode 1 change `ly_f1f` read faster in 7 of 7 pairs with the flag and null
over 3 pairs without it. A file without `ly_f1f` was built and evaluated before a 4-pair run with the
edge on showed it 0.48 / 0.52 us faster (sliding / full, 4 of 4 pairs). Identical code drifts by
0.4 to 0.5 us between files, so 3 pairs cannot call a 0.4 us change null. About 25 minutes of box
time were lost. The EAT expert-kernel worker found the cause.

Rule adopted: measure expert, routing and unpooler changes at T <= 32 with the edge on (EAT's
`tools/integ_ab.sh NOSKIP=1`, or an arm that sets `moe_pdl` on for 1, 8, 16 and 32), and use at least
4 pairs before calling a sub-microsecond change null.

## Lane context size (2026-11-04)

Lane agents reported 250,000 to 500,000 subagent tokens per round and grew slower, costlier and
less reliable. The author asked that lanes stay at about 100,000 to 150,000 tokens.

Rule adopted, in `.claude/agents/*.md`: `autoCompactWindow: 150000`, a checkpoint in
`docs/lanes/<job>.md` after every step, and a stop at about 60 tool calls or 120,000 tokens, after
which the lead relaunches a fresh lane from the checkpoint.
