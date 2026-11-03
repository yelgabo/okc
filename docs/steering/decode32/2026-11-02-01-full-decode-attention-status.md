# D32-ATTN-01 status

Picked up by Decode32 Worker in the "Review decode32 research tasks" chat at
2026-11-02 23:55 UTC. Assignment and source inspection have begun.

Assignment pin: `11bfcd4c5bab62fc8d5e117d4b7236034ef891f1`.
Control: `muse_decode_layer/cand/g1b4_stage_release.py`, SHA256
`ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`.

Owned files are new `tools/d32_attn_worker_*` files, this task's dated reports,
and `evidence/D32-ATTN-01/`. D32-02 supersedes the initial ownership split. This chat owns D32 checks,
analysis and integration directly. MID is now parked. Other workers retain
unrelated code, jobs and project tasks.

The implementation and local checks are complete. D32-02 supersedes the
source-only endpoint. This chat continues through private GPU measurement. The
[experiment record](2026-11-02-01-full-decode-attention-experiment.md) supplies the exact
candidate, regeneration command, source review, compiler evidence and the
remaining target gates. Candidate SHA256 is
`5a580c7a0fad1246503ebc66cfbeca698c2b011638070f340480e4560a18d3b5`.

Six CPU tests pass. Both selected device instances compile with cached
CUDA 12.8.1, with 128 registers, three barriers and zero stack/spills. Generated
instructions retain next-K load/exponential interleaving; they group the QK
arithmetic separately. GPU numerics, replay, host preparation/admission and
performance remain unmeasured. This chat owns the next exploratory check under D32-02.

D32-04 read at 2026-11-03T00:16:30.312420+00:00. The fixed design is implemented.
This chat proceeds directly to the ordinary private check, without another review
or handoff. Source and accepted job are recorded in [STATUS.md](STATUS.md).
