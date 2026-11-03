# D32-ATTN-01 implementation plan

Updated by okc research for the user's medium-effort execution instruction.
[ D32-04: fixed design and direct execution ](2026-11-02-04-research-plan-worker-execute.md)
supersedes the previous open-ended design selection, independent-review appointment
and Driver measurement handoff. Decode32 Worker retains the complete execution loop;
research supplies the architectural decisions. Preserve current implementation work.

1. Implement the already-selected **within-request next-tile QK score carry** in
   the actual L1-flat full-decode path, against control ba9. Follow D32-04 and
   the existing source/lifetime feedback. Do not reopen cross-request continuity,
   redesign the producer or repeat a broad architecture survey. Preserve online
   softmax order, current-V lifetime, original fences/joins and segment drains.
2. Complete `tools/d32_attn_worker_candidate.py` and focused tests. Generate one
   plain full-layer candidate under `evidence/D32-ATTN-01/`. Build its separate
   module during preparation and select it through metadata only. Check actual
   final bindings, unsupported shapes/scales, missing preparation and direct-entry
   guards; keep every other standard route on the control.
3. Run relevant bounds/slot/selector checks and correct localized implementation
   failures. Use an installed compiler when practical; record device-only or
   unavailable-compiler limits honestly. A missing secondary local compiler is
   not a mandatory Driver relay: the ordinary private judge check performs the
   actual target build. Do not dispatch with a known correctness/progress defect.
4. Check shared quota/live jobs, publish the source pin and **run one ordinary
   private full-layer check yourself**. Follow your own accepted ID and archive
   its full result. No captain/research per-step approval or candidate handoff.
5. Report actual selected treatment, correctness, D32 sliding/full/weighted total
   and all other load results. A promising result earns bounded matched
   confirmation using existing infrastructure; an unpromising valid result parks
   the exact variant without a tuning sweep. Integrate confirmed useful changes
   under existing release permissions. Preserve the control for rollback.

Stop only for a concrete math, lifetime, progress, access or resource blocker;
state the failed condition precisely. Proof of a future speedup is not a
precondition to an exploratory check. Target correctness and performance remain
unmeasured until that check; local compilation or CPU mocks do not establish them.
The previous plan remains in Git history; no historical result is rewritten.
