# Projection: frontier map and the route to top 100 (muse_bf16_proj)

Date: 2026-10-30, about 19:30-20:30 UTC. Owner: projection agent (worktree branch
`worktree-agent-a397d671218195a46`). Evidence commit: main `c98018f` merged into this branch;
tested build `90ee4b9` (parked as `muse_bf16_proj/experiments/qkv_c4_submission.py`); check
`ad6c673d9a444c1e`. Live CLI at 19:35 UTC: 6 of 10 runs and 1 of 20 checks left today
on this problem; reset 00:00 UTC.

## Summary

- Board: we are 1.4976x (rank 314, scored `d85d0081fa8a4a43`). Rank 100 is 1.5560, rank 78
  1.5616, rank 50 1.5692, rank 1 1.6498 (CLI, 19:35 UTC, 1,490 ranked). The median of our last
  six receipts of the current code is 1.4947, so rank 100 needs about +4.1% of real speed.
- Floors (measured streaming line, below) leave 42 us of headroom across the six loads. To
  first order, +4.1% needs about a third of every load's headroom, on average. No single
  mechanism found here gets there. A ranked portfolio that reaches the target only if almost
  every item lands is in the last section; read it as a stretch, not a forecast.
- The largest structural item that fit today's one check, qkv_gate at 8 / 16 / 32 tokens in
  one wave (128 CTAs in 4-CTA clusters instead of 96 CTAs), was built and measured: it LOSES by
  9-12 us at every m (`ad6c673d9a444c1e`), cause not attributed (cluster residency is the
  leading hypothesis). Parked with a reopen trigger.
- With that item gone, the remaining portfolio sums to about 2.7-3.5% if everything lands near
  its prediction, below the +4.1% that rank 100 needs. On current evidence top 100 is not
  reachable by the listed mechanisms. Only a denser lossless weight code (item E), which moves
  the floors themselves, has a ceiling above the target; it is the least developed item.
- Against the 989 TFLOP/s datasheet floor, the m = 256 dense GEMMs hold about 20 us. At the
  700 TFLOP/s that cuBLASLt reaches, two of them already beat the practical floor. Only qkv_gate
  (590 TFLOP/s) and o (560) are clearly below what large GEMMs reach on this GPU.
- Selection on run noise: the board keeps each team's best run. Ten screened top entries
  re-ran 0.5% below their board scores (median, range -1.25% to -0.1%). So part of the gap is
  the field's best-of-N, but most of it (about 3.5%) is real.

## 1. Frontier map (no quota)

Method (tool: `tools/proj_frontier.py`, receipts in `muse_bf16_proj/profiles/`):

- Ours: the median of the judged per-case time (`median_ns`, the judge's SGLang median over the
  paired ratio, after its replay overhead) over six receipts of the current or
  near-identical code: scored `d59bc7af342a465d` (ce25f67), checks `3abb1cd974e2477d`,
  `d7c662296eaa4413` (ce25f67), `25ea2b62ff8a47a7`, bench judged path `4f3ff774981a494c`
  (2265fa2 split barrier), scored `d85d0081fa8a4a43` (2265fa2). The routers at 8-256 changed
  between ce25f67 and 2265fa2 by -0.2 to -0.45 us in process; the six-run median blurs that.
- Bytes as stored: 12-bit code (1.5 B per weight; escape tables under 0.1% of the stream) for
  every m <= 32 BF16 pick, BF16 (2 B) for the m = 128 / 256 cuBLASLt and wd picks, the router as
  BF16 hi + lo (4 B per weight) everywhere. Plus x and the output. Shapes from `reference.py`:
  qkv_gate 2304 x 8192, o 8192 x 1024, pooler 4096 x 8192, unpooler 8192 x 4096 (the 1,024-row
  slice at m <= 8), router 256 x 8192.
- Floor = 1.3 us + max(bytes / 4.05 TB/s, FLOPs / 700 TFLOP/s). The streaming line
  t = 1.3 us + bytes / 4.05 TB/s is the judge's measured cold-L2 line (docs/runs-moe.md:24,
  docs/audit-2026-10-28.md 5.4); 700 TFLOP/s is what the m = 256 cuBLASLt picks reach. The
  `math989` column is FLOPs at the 989 TFLOP/s datasheet peak (without the 1.3 us).
- Fixed cost per kernel, measured from the judge's own replay-overhead fields (median of 180
  records): a graph with one trivial kernel costs 1.09 us more than the overhead the judge
  subtracts, and each extra trivial kernel another 1.09 us. Judged times also run 0.6-1.6 us
  above the profiled kernel time at small T (launch, first-byte latency and tail). The 1.3 us
  intercept covers that floor, so at m = 1-8 a kernel cannot get below about 1.1-1.3 us
  whatever its bytes.
- Score value of one case: delta_t / (6 C_T) (the weight 62 cancels), in % per us below.

Per-load totals (median of the six receipts; score from these medians 1.4947):

| load | ours C_T | SGLang | ratio | % per us |
|---|---:|---:|---:|---:|
| decode 1 | 34.87 | 59.78 | 1.715 | 0.478 |
| decode 16 | 49.52 | 75.86 | 1.532 | 0.337 |
| decode 32 | 53.46 | 77.07 | 1.442 | 0.312 |
| verify 1 (m 8) | 38.42 | 75.24 | 1.958 | 0.434 |
| verify 16 (m 128) | 69.21 | 86.01 | 1.243 | 0.241 |
| verify 32 (m 256) | 83.88 | 101.52 | 1.210 | 0.199 |

All 30 cases ranked by the score value of their headroom (us; `head` = ours - floor):

| load | op | m | ours | min-max | kernel | pick | bw | math989 | floor | head | head % |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
| verify 32 | router | 256 | 9.67 | 9.37-9.90 | 9.05 | wd r128c8s6 | 2.07 | 2.17 | 4.47 | 5.19 | 1.03 |
| decode 32 | qkv_gate | 32 | 11.43 | 11.10-11.69 | 10.75 | sk qkv32 | 7.16 | 1.22 | 8.46 | 2.97 | 0.93 |
| verify 16 | qkv_gate | 128 | 14.74 | 14.35-14.99 | 14.05 | wd b1n128c3s8h | 9.98 | 4.89 | 11.28 | 3.46 | 0.83 |
| verify 16 | router | 128 | 7.17 | 6.28-7.24 | 6.55 | wd r64c8s8 | 2.62 | 1.09 | 3.92 | 3.25 | 0.78 |
| decode 16 | qkv_gate | 16 | 10.45 | 9.78-10.66 | 9.35 | sk qkv16 | 7.07 | 0.61 | 8.37 | 2.08 | 0.70 |
| verify 1 | unpooler (slice) | 8 | 4.42 | 4.28-4.84 | 3.70 | sk slice8 | 1.57 | 0.07 | 2.87 | 1.55 | 0.67 |
| verify 16 | pooler | 128 | 21.36 | 21.29-21.89 | 21.35 | cuBLASLt | 17.35 | 8.69 | 18.65 | 2.71 | 0.65 |
| verify 1 | qkv_gate | 8 | 9.78 | 9.59-10.21 | 8.05 | sk qkv8 | 7.03 | 0.31 | 8.33 | 1.45 | 0.63 |
| verify 1 | o | 8 | 5.88 | 5.51-6.10 | 4.40 | sk o8 | 3.14 | 0.14 | 4.44 | 1.43 | 0.62 |
| decode 32 | router | 32 | 5.48 | 5.22-5.83 | 4.85 | rt | 2.21 | 0.27 | 3.51 | 1.97 | 0.61 |
| decode 32 | o | 32 | 6.18 | 5.54-6.41 | 5.15 | sk o32 | 3.25 | 0.54 | 4.55 | 1.62 | 0.51 |
| decode 32 | pooler | 32 | 15.51 | 15.17-15.87 | 15.90 | sk pool32 | 12.62 | 2.17 | 13.92 | 1.59 | 0.50 |
| verify 32 | qkv_gate | 256 | 17.60 | 17.10-18.21 | 20.05 | cuBLASLt | 10.65 | 9.77 | 15.11 | 2.49 | 0.49 |
| decode 16 | router | 16 | 4.77 | 4.41-4.82 | 4.10 | rt | 2.14 | 0.14 | 3.44 | 1.33 | 0.45 |
| decode 16 | o | 16 | 5.66 | 5.25-6.03 | 4.60 | sk o16 | 3.18 | 0.27 | 4.48 | 1.18 | 0.40 |
| decode 1 | unpooler (slice) | 1 | 3.60 | 3.35-3.82 | 2.80 | cgemv | 1.56 | 0.01 | 2.86 | 0.74 | 0.35 |
| decode 1 | o | 1 | 5.15 | 4.61-5.26 | 4.60 | cgemv | 3.11 | 0.02 | 4.41 | 0.73 | 0.35 |
| verify 1 | router | 8 | 4.20 | 3.93-4.45 | 3.75 | rt | 2.11 | 0.07 | 3.41 | 0.79 | 0.34 |
| verify 32 | o | 256 | 9.01 | 8.83-9.21 | 7.95 | cuBLASLt | 5.31 | 4.34 | 7.44 | 1.57 | 0.31 |
| decode 16 | pooler | 16 | 14.72 | 14.46-14.81 | 14.45 | sk pool16 | 12.52 | 1.09 | 13.82 | 0.90 | 0.30 |
| decode 32 | unpooler | 32 | 14.86 | 14.11-15.24 | 14.30 | sk unp32 | 12.62 | 2.17 | 13.92 | 0.94 | 0.29 |
| verify 16 | o | 128 | 7.10 | 6.88-7.50 | 5.40 | cuBLAS / Lt | 4.73 | 2.17 | 6.03 | 1.08 | 0.26 |
| decode 1 | qkv_gate | 1 | 8.81 | 8.65-9.09 | 8.00 | cgemv | 7.00 | 0.04 | 8.30 | 0.51 | 0.24 |
| verify 1 | pooler | 8 | 14.15 | 14.01-14.45 | 13.90 | sk pool8 | 12.48 | 0.54 | 13.78 | 0.38 | 0.16 |
| decode 1 | router | 1 | 3.55 | 3.12-3.89 | 3.20 | rgemv | 2.08 | 0.01 | 3.38 | 0.17 | 0.08 |
| verify 16 | unpooler | 128 | 18.84 | 18.57-19.08 | 18.50 | cuBLASLt | 17.35 | 8.69 | 18.65 | 0.19 | 0.05 |
| decode 16 | unpooler | 16 | 13.93 | 13.55-14.45 | 13.40 | sk unp16 | 12.52 | 1.09 | 13.82 | 0.10 | 0.03 |
| decode 1 | pooler | 1 | 13.76 | 13.01-13.86 | 13.00 | cgemv | 12.43 | 0.07 | 13.73 | 0.03 | 0.01 |
| verify 32 | pooler | 256 | 24.20 | 23.65-24.59 | 24.60 | cuBLASLt | 18.12 | 17.37 | 25.84 | -1.64 | -0.33 |
| verify 32 | unpooler | 256 | 23.41 | 22.92-23.81 | 23.65 | cuBLASLt | 18.12 | 17.37 | 25.84 | -2.44 | -0.48 |

Notes on the floors. The m = 256 pooler and unpooler are already faster than the 700 TFLOP/s
floor (they run about 735 TFLOP/s); against the 989 TFLOP/s datasheet floor (18.67 us with the
1.3 us) they hold 5.5 / 4.7 us. At m = 128 the BF16 shapes are bandwidth floors; only fewer
bytes lowers them, and the one compressed-WGMMA attempt lost (P3, `4922f366a0c142ca`), which a
public thethings.ai effort also reports for bit-exact compressed WGMMA at m = 128 / 256 (as
relayed by the coordinator; not checked here).

### What +4.1% needs, load by load

Rank 100 is 1.556 / 1.4947 = +4.10%, so the six loads must satisfy
sum_l ln(C_old,l / C_new,l) = 6 ln(1.041) = 0.241. If every load captured the same fraction f
of its headroom H_l: f x sum_l H_l / C_l = f x 0.756, so f = 0.32. Per load, the headroom and the
saving at f = 0.32:

| load | headroom H (us) | H / C | saving at f = 0.32 (us) | spread evenly instead (3.9% of C, us) |
|---|---:|---:|---:|---:|
| decode 1 | 2.18 | 6.3% | 0.70 | 1.36 |
| decode 16 | 5.59 | 11.3% | 1.79 | 1.93 |
| decode 32 | 9.09 | 17.0% | 2.91 | 2.09 |
| verify 1 | 5.60 | 14.6% | 1.79 | 1.50 |
| verify 16 | 10.69 | 15.5% | 3.42 | 2.70 |
| verify 32 | 9.25 (to 700 TF; 20.4 to 989 TF) | 11.0% | 2.96 | 3.28 |

Is it plausible against the floors? It does not require below-floor times. It requires about a
third of all headroom to be captured, spread over at least four loads, while decode 1 has only
2.2 us of headroom (cgemv kernels already sit on the streaming line). That is about 13 us of
real per-step savings over 30 cases. Every past round on this problem gained 0.2-1.5% each. So
the target is reachable only with several structural wins landing together, or with a
lower-bytes weight format that moves the floors themselves (item P-E below).

### Against the independent reference (Nathen v18, job 66d8b007f6c641ee)

Per-case judged us from its `results/v18.json` (relayed by the coordinator; another run,
another SGLang sample, so 1-3% apart before any code difference):

| case | ours | Nathen v18 | ours - ref |
|---|---:|---:|---:|
| router 1 / 8 / 16 / 32 | 3.55 / 4.20 / 4.77 / 5.48 | 3.94 / 4.34 / 4.23 / 5.04 | -0.39 / -0.14 / +0.54 / +0.44 |
| router 128 / 256 | 7.17 / 9.67 | 6.20 / 8.44 | +0.97 / +1.23 |
| pooler 1 / 8 / 16 / 32 | 13.76 / 14.15 / 14.72 / 15.51 | 13.29 / 13.61 / 13.98 / 16.13 | +0.47 / +0.54 / +0.74 / -0.62 |
| pooler 128 / 256 | 21.36 / 24.20 | 20.49 / 24.19 | +0.87 / +0.01 |
| unpooler 128 / 256 | 18.84 / 23.41 | 19.37 / 23.32 | -0.53 / +0.09 |
| o 128 / 256 | 7.10 / 9.01 | 6.89 / 8.98 | +0.21 / +0.03 |

Reading: on unpooler and o at 128 / 256 neither entry gets more than 1.03-1.08x over SGLang,
and we are level with the reference there. Those four cases sum to 58.4 us of our 153 us at
verify 16 / 32. Teams at 1.60+ must gain somewhere we and the reference do not. The only
places with floor room are the m = 256 math (989 TFLOP/s floor) and fewer stored bytes. Both
are hypotheses about the leaders, not evidence. Our router at 128 / 256 trails the reference by
about 1 us, consistent with the in-process gap already recorded (docs/runs.md, round 2).
Nathen's pooler at m = 1 / 8 (13.29 / 13.61) sits on or below our modeled floor
(13.73 / 13.78), so their judge sample ran fast or their stream is slightly smaller; the
difference is inside run-to-run spread.

## 2. Structural options for the top headroom cases

Each option: mechanism, predicted saving with basis, why `docs/ruled-out.md` does not close it,
cheapest decisive test. Predictions are hypotheses.

### A. qkv_gate at 8 / 16 / 32 tokens in one wave (built, top pick)

- Mechanism: 96 row blocks of 24 rows leave 36 of 132 SMs without a stream. qkv runs 28.3 MB
  at 2.8-3.25 TB/s while pool16 / unp16 (128-256 CTAs) run 3.8 TB/s. New configs take 72 rows
  per CTA and split K over a 4-CTA cluster: 32 clusters, 128 CTAs at one per SM, each streaming
  25% fewer bytes and running 8 stages instead of 16. Each rank corrects only the escapes of its
  own K range; ranks 1-3 push partials (with their corrections) to rank 0 by `st.async`, and every
  CTA ends with the split exit cluster barrier (relaxed arrive after rank 0's receive wait), the
  conforming form the coordinator adopted.
- Prediction: if the per-CTA rate stays at today's (34 / 30 GB/s at 16 / 32 tokens), the
  stream time drops to 0.75x, capped at the HBM line: about -1.4 us at 16, -2.3 at 32, 0 to -0.3
  at 8 (already near the line in kernel time); +0.47 +0.72 +0.1 = about +1.3% of score before
  the cluster reduction (the 2-CTA cluster split cost about 0 on pool32c, `a9ff49372ffa4eaf`).
- Why not closed: P2 was declined on its work model, and its row names the reopen condition: a
  geometry with exactly one wave (<= 132 CTAs). The losers it cites ran 144-192 CTAs (qkv16n2 /
  n2b in `01703396ea1a409e`, a second partial wave or 2 CTAs per SM). The 2-CTA cluster split
  on the pooler / unpooler kept the CTA count fixed (128), so it never tested SM fill. The
  coordinator agreed this meets the trigger.
- Main risk: residency. Four 1-CTA-per-SM clusters per GPC need at least 16 free SMs in every
  GPC; a chip with one 14-SM GPC holds only 31 clusters and runs a second wave (the time would
  roughly double, so the bench shows it at once). Second risk: the 16-to-8-stage change also
  halves per-stage fixed costs, so a loss would not separate SM fill from stage count; the
  12-warp `c4w` variant gives one more point.
- Cheapest decisive test: one bench check (`tools/proj_make_bench_c4.py`): arms A.pick,
  A.c4, A.c4w, B.pick, B.c4 (B is a separately compiled and packed extension, the independent
  null), cuBLAS, at m = 8 / 16 / 32, 100 seeded rounds, cold L2. Stop rule: adopt a bucket only
  if c4 - pick < -0.3 us with the A/B nulls within 0.15 us and at least 80 of 100 rounds faster.
- Result (section 4): a 9-12 us loss at every m in the prepare-time tuner of
  `ad6c673d9a444c1e`; stop rule fired, parked.

### B. Router at 128 / 256

- Mechanism: the reference's wgmma router is 1.3 / 2.4 us faster in process at 128 / 256 at the
  same tiling, and none of eight toggled differences closes it (docs/runs.md, round 2). Untested
  differences: its 256-thread CTA with a 4-warp producer group, cluster-scope acquire spin waits,
  scalar output stores. A different mechanism is a 128-CTA decomposition at m = 256 (today 64
  CTAs; r64c8s8 at 128 CTAs lost +5 us, cause unknown).
- Prediction: matching the reference is -1.0 / -1.2 us judged (Nathen table above): +0.24 +0.24
  = about +0.5%.
- Why not closed: the ruled-out rows close single toggles, and explicitly reopen per remaining
  difference.
- Cheapest test: one bench with the three untested differences as separate toggles plus a SASS
  diff of the two main loops (no judge cost for the SASS part).

### C. m = 256 dense GEMMs toward the 989 TFLOP/s floor

- What the numbers say: qkv_gate 256 runs 9.66 GFLOP in 16.3 us (590 TFLOP/s), o 256 4.3 GFLOP in
  7.7 us (560), pooler / unpooler 17.2 GFLOP in 22.9 / 22.1 us (750 / 780). Reaching 800 TFLOP/s
  on all four saves qkv -4.2, o -2.3, pooler -1.4, unpooler -0.6 us: 8.5 us, +1.7% of score.
- What a kernel would need at these shapes: the output is only 256 x N (qkv: 256 x 2304), so
  128 x 256 tiles give 18 tiles for qkv and 32 for the pooler; filling 132 SMs needs a K split
  of 4-7 or stream-K, with an FP32 fixup per split. At one CTA per SM, a 128 x 256 x 64 BF16 stage
  is 48 KB, so 4 stages fit beside the epilogue; two consumer warpgroups (m64n256k16 each) with a
  producer warp, a cluster of 2 multicasting the token tile, and a fixup through DSMEM inside the
  cluster rather than global scratch. Large well-tuned GEMMs on H100-class parts sustain about
  700-800 TFLOP/s, so 800 is the top of what is achievable, not a target for every shape.
  Realistic: qkv and o to about 700 / 650 TFLOP/s, -2.5 / -1.1 us, about +0.7%.
- Why not closed: the CUTLASS sweep (`8bea43d605c74546`) and the cuBLASLt search tested library
  configurations and lost or tied at these shapes; no kernel designed around a one-wave split
  for 256 x 2304 was tested. Stream-K in the MoE layer lost for a different reason (fixup cost
  per CTA).
- Cheapest decisive test (one check, no new kernel): in one bench, time the current Lt pick for
  qkv 256 and o 256 against the same GEMM batched 8x in one cuBLASLt call (stride-batched, same
  per-instance shape) and against one 2048 x 2304 x 8192 GEMM. If the per-instance batched time
  is far below the single call, the loss is ramp, tail and wave quantization (a structural
  one-wave kernel can recover it); if not, the mainloop is the limit and only a better mainloop
  helps, which library sweeps already searched.

### D. Short shapes: o at 8 / 16 / 32 and the m = 8 slice

- o (K = 1024, 256 CTAs) holds 1.2-1.6 us per case at 8 / 16 / 32 and the m = 8 slice 1.55 us. o's
  CTAs each stream only 48 KB, so the time is mostly ramp and tail. A CUDA-core 8-token GEMV for
  the 6.3 MB slice (8 FMA per weight, about 1.1 us of issue spread over 132 SMs) could bring the
  slice from 4.42 toward the m = 1 cgemv time (3.60): -0.6 us, +0.26%. Hypothesis; ruled-out
  closes only the 32-register cgemv at m = 1 and sgemv on the slice, not an 8-token GEMV.
  Test: an arm in the next bench.
- The brief's "one-kernel T <= 8 path without escape-warp startup": at m = 1 the cgemv kernels
  already sit 0.03-0.74 us over the floor; the escape start was fixed in a35709c. Little left.
- Persistent kernels to cut launch cost: each case is a single kernel in its own graph, so
  there is no second launch to remove; the 1.1 us per-kernel cost measured above is the floor.
  Persistence helps only as a schedule (option A, C), not as launch removal.

### E. A lower-bytes lossless weight format at m <= 32 (the floor mover)

- Mechanism: BF16 Gaussian weights have about 10.5 bits of entropy (1 sign + 7 mantissa + about
  2.5 bits of exponent, computed for N(0, sigma^2) binades); the 12-bit code stores 4 exponent
  bits. An average of about 11 bits (for example a 2- or 3-bit exponent code for the most
  frequent binades with escapes, or DFloat11-style Huffman exponents) cuts every m <= 32 stream
  by 8%.
- Prediction: if decode stayed hidden, -2.0 / -2.9 / -2.9 / -2.0 us on decode 1 / 16 / 32 /
  verify 1, about +3.7% of score; realistically half. It moves the floors, so it is the only
  item whose ceiling alone exceeds the target.
- Why not closed: ruled-out closes the old escape-tile lookup design, not a denser code. Risk:
  a 3-bit exponent window leaves about 1.2% of weights as escapes (400 K on the pooler) against
  0.014% today; escape handling cost is what sank the first compressed kernel. Needs a CPU work
  model (escape counts per code choice on the judge's weight statistics, decode instructions
  per weight) before any check.

## 3. Ranked portfolio toward +4.1%

| # | item | cases | predicted us | score % | evidence level | check cost |
|---|---|---|---|---:|---|---|
| 1 | A: qkv one-wave cluster split | qkv 8 / 16 / 32 | predicted -0.3 / -1.4 / -2.3; measured +9.8 / +9.4 / +12.3 | 0 (was 1.3) | measured loss, ad6c673d9a444c1e | parked; a residency diagnostic first |
| 2 | E: denser lossless code at m <= 32 | 20 streaming cases | about -1 to -1.5 per load | 1.0-1.8 (ceiling 3.7) | entropy arithmetic only | 0 for the model, then 2-3 |
| 3 | C: m = 256 one-wave GEMM for qkv / o | qkv 256, o 256 | -2.5 / -1.1 | 0.7 | arithmetic from judged times | 1 diagnostic, then a kernel (2-3) |
| 4 | B: router 128 / 256 to the reference | router 128 / 256 | -1.0 / -1.2 | 0.5 | in-process reference gap measured | 1 bench + 1 confirm |
| 5 | D: 8-token CUDA-core slice GEMV, o ramp | slice 8, o 8-32 | -0.6, -0.5 | 0.4 | hypothesis | 1 bench |
| 6 | pooler 128 Lt variance, o 128 pin | pooler / o 128 | -0.4 | 0.1 | preparations differ by up to 1.1 us | 0 (pin) |
| | sum (items 2-6) | | | 2.7-3.5 | | 6-10 checks |

Before the item 1 measurement the sum was 4.0-4.8%, and reached the target only if items 1-4 all
landed near their predictions. After it, the sum is below the target even if everything else
lands. Item 2 is the only one whose ceiling (3.7%) is large enough to carry the target, and it
is the least developed. Recommendation for the next quota day: build item 2's CPU work model
with no quota (escape rate and decode instructions per weight for 11-bit codes on the judge's
weight statistics), spend one check on item 3's batched-GEMM diagnostic, and one bench on item
4's untested router toggles. A paired independent read is advised before item 2 takes checks:
its ceiling rests on decode staying hidden, which is what sank the first compressed kernel.

## 4. Test of item A

Build `90ee4b9` (no external code). Compile gate: nvcc 12.8 sm_90a, `tools/proj_sass.sh` +
`tools/proj_sass_diff.py` against main `c98018f`: 81 existing kernels instruction-identical
(the shipped sk, cgemv, wd, rt and router instances unchanged), 10 added (5 configs, 12-bit and
raw), 59-108 registers, no spills, no C75xx warnings. Exactness, fail closed in prepare(): the
FP64 self-test, then a match against the unsplit pick (2 ulps on random rows; identical outputs
on one-hot rows at escape columns and K-range edges of every rank), else the config is dropped.
An independent read-only review found the escape ownership, exit-barrier phases, prologue
and balancing sound; its one robustness note (a `__syncwarp` before the escape warps'
`.aligned` cluster wait) was taken, and the SASS re-checked (81 identical).

Result, check `ad6c673d9a444c1e` (1.509x, 1.486-1.532; judged path = production picks, since
the tuner rejected every c4 config). Every c4 config passed the FP64 test, the 2-ulp twin match
and 22 identical one-hot columns at escape columns and K-range edges. The bench's paired timing
loop crashed on a bug in my harness (`torch.sum(..., out=)` on a 0-dim tensor; fixed in
`tools/proj_make_bench_c4.py`), so there are no A/B lines with duplicate nulls. The in-process
tuner (cold L2, 24 alternating reps, one preparation) measured, in us with replay overhead:

| m | pick | c4 | c4w | pick raw | c4 raw |
|---:|---:|---:|---:|---:|---:|
| 8 | 13.9 | 23.7 | 25.5 | 16.4 | 20.8 |
| 16 | 14.7 | 24.1 | 25.7 | 16.4 | 21.1 |
| 32 | 15.6 | 27.9 | | 16.2 | 24.6 |

A 9-12 us loss is about 30 times any duplicate null measured on this problem, so the decision
does not need the missing paired lines. The cause is not attributed. Hypotheses: not all 32
one-CTA-per-SM clusters of 4 are co-resident (a second wave), or the 72-row CTA decodes slower
per byte (the 12-bit c4 loses 3 us more than its raw twin, while the 24-row pick beats its raw
twin). The scope of the loss is this geometry; the idea of filling the 36 idle SMs is not
rejected. Reopen trigger and code path are in docs/ruled-out.md.

## Gaps

- Live leaderboard per-case times of the top entries are not available; the reading of where
  leaders gain is a hypothesis.
- The 4.05 TB/s line was measured on the MoE problem's judge; cases slightly under it (Nathen
  pooler m = 1) show it is not a hard bound.
- Cluster residency on the judge's chip (max active 4-CTA clusters) is not measured directly.
