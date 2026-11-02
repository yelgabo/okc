# D32-ATTN-01 status

Picked up by Decode32 Worker in the "Review decode32 research tasks" chat at
2026-11-02 23:55 UTC. Assignment and source inspection have begun.

Assignment pin: `11bfcd4c5bab62fc8d5e117d4b7236034ef891f1`.
Control: `muse_decode_layer/cand/g1b4_stage_release.py`, SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.

Owned files are new `tools/d32_attn_worker_*` files, this task's dated reports,
and `evidence/D32-ATTN-01/`. OKC Developer retains the MID expert comparison.
OK Driver retains judge dispatch and integration. Existing source, generators,
run logs and other workers' files remain under their current ownership.

The deliverable is one isolated full-layer attention candidate, focused tests
and a measurement handoff, or a concrete correctness/source blocker. The
[implementation plan](2026-11-02-01-full-decode-attention-plan.md) tracks the
source decision and remaining checks. No target measurement has run here.
