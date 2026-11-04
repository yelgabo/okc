# Revival lane checkpoint: round 2, launch knobs on ly_stk4

Updated 2026-11-04 13:48 PT. Branch worktree-agent-a4ca890a24d12b144. Round 1 (ly_stk3 sweeps) is in
docs/runs-decode-layer.md, section "2026-11-04 - Per-T knob sweep on ly_stk3 (box)".

Base: muse_decode_layer/cand/ly_stk4.py, SHA256 a054c8d57555352d924bc2c4ae73f3199f05025f40a11300bc301d1655ba927e
(judged 10,046.1 us, the draw base).

Screen file: muse_decode_layer/cand/ly_rv2s.py, SHA256 546866a686d857bfb7d3be2e0126ed05b4520bf39d12823fb18d826c8608117b,
built by muse_decode_layer/profiles/box/rev2/mk_rv2s.py from ly_stk4. With empty knob tables it runs the base's
launches. Knobs (module globals, read at launch, so each ly_arms arm bakes its own value into its graph):
- `ASMEM_CFG[(B, L, sliding)]`: dynamic shared memory bytes of that attention launch (env OKC_ASMEM read in the C++
  `launch<L>`; cooperative grids keep the base value when the larger reservation would break residency).
- `CG_NTH_OVR[(n, k)]`: cgemv threads at launch (T = 1 slice, base (1024, 4096) = 128).
- `V1FS_T[T]`: finish dynamic shared memory per T (base {8: 118784}); T = 1 added.
- `MOE_X["tiny" | "tiny2"]`: replacement define tuple for that MoE build.

Instrument: tools/ly_arms.py, 48 runs, router seed 106218864389449120, judge
.build/judge/g1b4_pdl8d.f931abf568544018.json, GPU lock /root/okc/.build/box/gpu.lock. Box scripts:
muse_decode_layer/profiles/box/rev2/{s1.sh,s2.py}; logs /root/okc/.build/rev2/arms_{s1,s1b,s2}.log.

## Sweep 1 (done): attention reservation, T = 1 cgemv threads, T = 1 finish reservation

All checks PASS. Paired median difference against base in us (negative is faster), faster pairs of 48. base2 is a
second copy of the base arm: its spread (up to +0.58 / -0.29) is the noise floor of this run.

| arm | d1 sliding | d1 full | v1 sliding | v1 full |
|---|---|---|---|---|
| a1s100 (1,1,sl) 100K | -0.34 (31) | -0.53 (36) | +0.02 (23) | +0.29 (19) |
| a1s160 | +0.34 (21) | -0.03 (25) | -0.56 (30) | +0.10 (23) |
| a1s227 | -0.05 (24) | -0.82 (37) | -0.34 (28) | +0.08 (20) |
| a1f100 (1,1,full) | +0.13 (22) | -0.96 (36) | -0.32 (29) | +0.32 (19) |
| a1f160 | +0.11 (22) | -0.37 (27) | +0.13 (23) | -0.05 (24) |
| a8s100 (1,8,sl) | +0.42 (18) | -0.82 (34) | -0.08 (26) | +0.34 (20) |
| a8s160 | +0.14 (21) | -0.34 (28) | +0.58 (19) | +0.32 (17) |
| a8f100 (1,8,full) | +0.40 (20) | -0.38 (33) | -0.18 (26) | +0.27 (19) |
| a8f160 | -0.27 (30) | -0.27 (29) | -0.18 (28) | +0.30 (18) |
| cg64 | +1.18 (13) | +0.35 (23) | -0.83 (32) | -0.13 (26) |
| cg256 | +1.97 (5) | +0.93 (7) | -0.35 (27) | +0.16 (20) |
| f1_60k | -0.06 (25) | -0.32 (29) | -0.32 (28) | +0.43 (20) |
| f1_118k | +0.27 (18) | -0.46 (29) | -0.21 (25) | -0.05 (25) |
| f1_160k | +0.40 (21) | -0.80 (35) | -0.11 (25) | -0.03 (24) |
| base2 | +0.00 (23) | -0.29 (31) | -0.11 (26) | +0.58 (16) |

The decode 1 full column drifts negative for every arm, including verify-only arms (a8*) that cannot touch a
decode 1 launch, so its -0.5 to -0.96 entries are base-arm noise. No sliding cell moves 0.5 us with a margin.

Sweep 1b, decode 16 / 32 (sl = sliding):

| arm | d16 sl | d16 full | d32 sl | d32 full |
|---|---|---|---|---|
| d16f100 (16,1,full) | -0.02 (24) | +0.10 (23) | -0.48 (29) | +0.42 (19) |
| d16f160 | +0.18 (18) | +0.08 (22) | -0.19 (26) | +0.61 (21) |
| d32f100 (32,1,full) | -0.06 (25) | +0.00 (23) | -0.72 (32) | +0.51 (18) |
| d32f160 | -0.19 (25) | +0.14 (19) | +0.06 (23) | -0.06 (24) |
| d16s120 (16,1,sl) 120K | +2.96 (2) | +0.27 (22) | -0.40 (29) | +0.48 (18) |
| d32s120 (32,1,sl) 120K | -0.21 (25) | -0.10 (24) | +4.86 (0) | +0.38 (22) |
| base2 | -0.06 (28) | -0.13 (25) | +0.10 (23) | +0.37 (20) |

Reach: the env knob demonstrably changes launches (d16s120 / d32s120 lose 3.0 / 4.9 us on their own cell, which
confirms ly_a16o2's 81920 two-per-SM choice) and cg64 / cg256 lose at decode 1. Reach at decode 1 / verify 1
attention is unproven: with B = 1 the split grid fits one CTA per SM, so a reservation only changes what can
co-reside under PDL, and no arm moved.

Decision: knobs 1-3 flat, no candidate.

## Sweep 2 (done): MoE TINY / TINY2 define variants, decode 1 / verify 1

All checks PASS. Own cells: TINY arms at decode 1, TINY2 arms at verify 1 (the other load is untouched).

| arm | d1 sliding | d1 full | v1 sliding | v1 full |
|---|---|---|---|---|
| t_wsblk1 | +1.31 (8) | +1.62 (3) | -0.30 (27) | +0.40 (19) |
| t_wsblk4 | +1.12 (7) | +1.44 (5) | +0.18 (21) | +0.24 (21) |
| t_finpf1 | -0.21 (29) | +0.35 (15) | -0.21 (27) | +0.32 (16) |
| t_cpf0 | +1.20 (6) | +1.38 (8) | +0.14 (18) | +0.32 (14) |
| t_l2h1 | +0.86 (9) | +1.04 (11) | +0.02 (23) | +0.10 (23) |
| t_wsbl1 | +1.09 (8) | +0.93 (10) | +0.11 (22) | +0.67 (18) |
| t_fast3 | +1.02 (10) | +1.50 (6) | -0.16 (26) | -0.02 (24) |
| t2_cpf1 | -0.14 (27) | +0.03 (23) | +0.85 (11) | +1.52 (7) |
| t2_finpf0 | -0.13 (29) | +0.21 (21) | +1.84 (3) | +2.10 (1) |
| t2_wsbl1 | +0.21 (22) | -0.06 (24) | +0.99 (10) | +1.42 (8) |
| t2_fast3 | +0.42 (19) | +0.13 (20) | +2.06 (4) | +2.11 (3) |
| t2_wsblk4 | -0.64 (27) | +0.26 (20) | +0.75 (13) | +1.12 (8) |
| base2 | +0.10 (21) | +0.11 (23) | -0.19 (26) | -0.13 (26) |

Decision: knob 4 flat (every change loses on its own cell, TINY FINPF 1 inside noise). Round 2 ends with no
candidate. Run log: docs/runs-decode-layer.md, "2026-11-04 - Launch knobs on ly_stk4, revival round 2 (box)";
ruled-out rows appended.

## Next

None of the four knobs is left. ly_rv2s stays useful as an in-process screen for any further launch-shape or
MoE-define idea (add an arm, no new compile of the layer).
