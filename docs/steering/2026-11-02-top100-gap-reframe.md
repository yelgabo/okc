# Top-100 gap reframe: the score pays for the short loads, and the short loads are latency chains

Evidence commit `b24ba3ed610d7aaa2da3c1cd9182dcda71b302ac`, written 2026-11-02 late Pacific. Prefill is
excluded by the user (it does not count for ranking). Every number below is read from a named receipt or
derived from one by stated arithmetic; derived numbers are labelled.

## 1. Position

| Problem | Public | Job | Bar (rank 100) | Needed |
|---|---:|---|---:|---:|
| Decode layer | 10,639.4 us, 2.1261x | `607eb52bdaf24160` | 9,958.4 us (2.2715x) | +6.8% |
| MoE W4A16 | 5,788.7 us, 2.3232x | `4aa033ef23a142cd` | 5,559.8 us | +4.1% |
| BF16 projection | 3,161.9 us, 1.5278x | `9eba96894ee1402e` | 3,053.3 us | +3.6% |
| Attention decode | 1,446.2 us, 2.6043x | `8914751e3b9f468b` | 1,405.8 us | +2.9% |

The user's table listed MoE at 6,069.6 us; the 5,788.7 us public entry (rank 434 to 373) supersedes it.
The layer already runs the same `c7c201ae` MoE engine as that entry (`2026-11-02-moe-release-diagram.md`),
so there is no MoE port waiting to be done.

## 2. What one microsecond is worth

Every score is a geometric mean over six equally weighted loads, so a saving of dT on a load whose time is
T moves the score by dT / (6 T). Derived from the public receipts:

| Load | Layer endpoint us | % per us | MoE us | % per us | BF16 % per us | Attention % per us (sliding) |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 | 78.5 | 0.212 | 26.7 | 0.62 | 0.47 | 1.76 |
| verify 1 (8 tok) | 116.0 | 0.144 | 67.2 | 0.25 | 0.44 | 1.40 |
| decode 16 | 158.2 | 0.105 | 94.0 | 0.18 | 0.36 | 0.45 |
| decode 32 | 191.7 | 0.087 | 127.6 | 0.13 | 0.32 | - |
| verify 16 (128) | 269.3 | 0.062 | 180.9 | 0.09 | 0.25 | - |
| verify 32 (256) | 359.3 | 0.046 | 217.3 | 0.08 | 0.20 | - |

A microsecond at decode 1 is worth 4.6 times a microsecond at verify 32 on the layer and 8 times on the
MoE. The current owner queues (MoE phase attribution on B4/MAIN at T32+, factorized output for MAIN/B4,
the BF16 register-A brief at m = 128/256, the "longer-token MoE first" ranking in
`2026-11-02-top100-profile-priorities.md`) are concentrated on the cheapest end of this table.

## 3. Where the short loads spend their time

Layer, decode 1 sliding, profile of `607eb52bdaf24160` (`muse_decode_layer/profiles/g1-stage-release.607eb52bdaf24160.txt`, profile 0). Bytes and floors are derived at 4.8 TB/s.

| Kernel | us | Weight bytes | Byte floor us | Over floor |
|---|---:|---:|---:|---:|
| qkv (sk2) | 10.5 | 37.7 MB | 7.9 | 1.3x |
| attention sliding | 5.4 | ~0 | - | latency |
| o (sk2) | 6.0 | 16.8 MB | 3.5 | 1.7x |
| boundary (glue_cluster_boundary) | 3.4 | 0 | - | latency |
| pooler + router (sk2 gle) | 17.9 | 75.5 MB | 15.7 | 1.1x |
| route (_sl_tile_route) | 2.0 | 0 | - | latency |
| MoE (moe_fused) | 29.8 | ~28-56 MB (1-2 local experts) | 6-12 | 2.5-5x |
| unpooler slice (sk2) | 4.4 | 8.4 MB | 1.75 | 2.5x |
| finish (_sl_cluster_finish) | 2.3 | 0 | - | latency |
| sum / endpoint | 81.6 / 76.75 | ~200 MB | ~42 | 1.8x |

Verify 1 sliding (profile 6) has the same shape: MoE 71.6 us, pooler + router 18.7, qkv 9.1, attention
7.4, o 6.0, unpooler 4.7, boundary 3.0, route 1.9, finish 1.7; sum 124.0 against endpoint 115.4.

Reading: the two bandwidth-heavy GEMVs (pooler + router, qkv) are already within 1.1-1.3x of the byte
floor. Everything else is latency: three glue kernels with no bytes (7.7 us), two small GEMVs at 1.7-2.5x
over floor, and the MoE, which at one token moves one or two experts' weights in 29.8 us when the bytes
alone take 6-12 us. Nine dependent launches each pay a ramp to full HBM rate, a tail, and the ~0.7 us
boundary that `ruled-out.md` measured. The weight stream stops at every activation dependency although
weights never depend on activations.

The standalone MoE shows the same thing: decode 1 is 26.7 us against an 11.4 us floor (2.3x, the worst
ratio of its six loads) and verify 1 is 67.2 against 37.3 (1.8x). The Tiny build moves 56 MB at 2.2 TB/s
while its ring (3 slots of 36 KB per CTA, 108 KB in flight per SM) could in principle sustain full rate.
Something other than bytes sets the time at T <= 8, and no receipt says what.

## 4. The structural proposal: one persistent post-attention kernel for T <= 8

At T <= 8 every post-attention stage is the same thing: a skinny GEMM whose weights arrive through a
bulk-copy ring (12-bit BF16 for o, pooler, router, unpooler; 4-bit for experts), separated by tiny
activation-only steps (boundary norm, route, finish). The proposal is to run o, boundary, pooler + router,
route, expert phase A, phase B, unpooler and finish as one launch of 132 persistent CTAs, with:

- per-stage completion counters instead of grid barriers (C-1 lost with two grid barriers at about 1 us
  each; `ruled-out.md` row M-C C-1);
- weight prefetch across every activation wait: a CTA that reaches the boundary wait, the route wait or the
  phase A to B handoff keeps its ring filling with the next stage's weights, because those addresses are
  known at prepare time; the only prefetch that must wait for data is the expert selection after routing;
- the existing per-CTA redundant route prologue, cluster boundary math and ring code reused as device
  functions, with `fence.proxy.async` on the writer side of every generic-store to bulk-load edge, per the
  fence rule in AGENTS.md;
- qkv and attention left as they are (two launches), so the first slice fuses 7 of 9 launches.

Expected effect (hypothesis, derived): the decode 1 chain above is 65.8 us for about 157 MB, 2.4 TB/s; the
verify 1 chain is 107.6 us for about 240 MB, 2.2 TB/s. At 3.0 TB/s those become 52 and 80 us, a saving of
14 and 28 us, worth 2.9% + 3.8% = 6.7% of the layer score, which is the whole gap with decode 16/32 and
verify 16/32 untouched. At 3.5 TB/s the saving is 21 + 39 us, about 9.8%. The published Llama-1B decode
megakernel reached roughly 78% of HBM peak at batch 1 with exactly this structure, so 3.0 TB/s (62%) is a
conservative target.

Why the prior failures do not close this: PDL (`179cecda`, `e8b74a35`, M-B slices) kept nine kernels and
only moved launch latency, which the graph already hides. C-1 fused two stages with grid barriers and a
persistent o loop that lost cgemv's memory-level parallelism. The L2 prefetch rows warmed L2 beside
attention and found the small GEMVs are not byte-bound, which is the same observation as section 3 and the
reason a persistent stream helps: it removes ramp and tail, not bytes. `ruled-out.md` itself names the
revisit trigger as "a layer with fewer, lighter glue kernels" and "a boundary fused into the o kernel
without a grid barrier".

Risks: the 869 KB release file, shared-memory budget of a ring that must serve two codecs, register
pressure if the expert stage keeps 288 threads with mma.sync, and the ordering proofs at each new
generic-to-async edge. This is weeks of work for one owner, not a day.

## 5. Cheapest decisive tests, in order

1. **MoE Tiny build timeline at T = 1 and T = 8** (standalone, one private check). Per-CTA `globaltimer`
   stamps at kernel entry, first ring fill complete, each unit retire, phase A to B handoff, final pass.
   Bounded buffer, one replay per shape. Question: of the 26.7 us at T = 1, how much is ramp, A-to-B
   bubble, phase B tail and final pass? Stop: if the hot loop itself runs under 2.6 TB/s with a full ring,
   the ring design is the problem and section 4 inherits it; if the time is in handoff and tail, both the
   standalone fix and the megakernel are justified. Value: every 1 us at T = 1 is 0.62% of the MoE score;
   26.7 to 18 us would be +6.8%, more than the MoE gap on its own.
2. **Two-stage fusion microbench at T = 1 and T = 8**: o + boundary + pooler/router in one launch with a
   counter wait and the pooler ring filling before the wait, against the three separate launches. Pass if
   the fused pair saves at least 2 us at decode 1 (the three kernels sum to 27.3 us against a byte floor of
   19.2). Fail closes section 4 cheaply: it would mean ramp and tail are not where the latency is.
3. **Expert-to-unpooler fusion** as the first production slice if test 2 passes: CTAs that finish phase B
   early stream the unpooler slice (8.4 MB) and the finish, removing two launches at T <= 8. Expected
   4-5 us at decode 1 (about 1%), and it is the piece the M-B slice 1 PDL attempt could not get because a
   PDL secondary only starts as the MoE's CTAs exit.

## 6. Component boards, same lens

- **MoE (+4.1%)**: tests 1 and 3 above. The T = 32 row has the lowest speedup against SGLang (1.647x), but
  that ratio is not the lever; the lever is our distance to floor times value per microsecond, and T = 1
  and T = 8 win on both.
- **BF16 projection (+3.6%)**: the live brief targets pooler/unpooler at m = 128/256 (0.20-0.25% per us).
  The small cells are worth twice as much per microsecond and sit further from floor: o 5.0-5.9 us against
  3.5, router 3.7-5.5 against 1.8, unpooler slice against 1.75, at m = 1-32. These are ramp-limited GEMVs;
  the fix is issuing every CTA's full load set in the first few hundred nanoseconds, not fewer bytes. The
  one large cell with measured room is qkv at m = 128/256 at 2.76 TB/s (14.7 us against 8.4), where the
  comparator `3c6a2baf` is 2.55 us faster; `ruled-out.md` P2's own reopen condition (exactly one wave,
  <= 132 CTAs or a persistent schedule) is the design to try.
- **Attention (+2.9%)**: decode 1 sliding and verify 1 sliding are worth 1.76% and 1.40% per microsecond.
  They move almost no bytes (a 2048-key window) and take 5.2 and 6.8 us: cache write, 32-CTA tile loop,
  workspace merge through `red.global.add`, ticket finalize, gated norm. One microsecond off each covers
  the whole attention gap. The comparator `cd6805e7` is 0.4-0.5 us faster on both, which bounds what is
  known to be possible. A DSMEM cluster merge that removes the workspace round trip and the ticket is the
  obvious mechanism; the recorded cluster attempt (`f74cbe27`, +21 us) lost by far more than the kernel's
  total, so that receipt describes a broken build, not the mechanism.

## 7. What to stop spending quota on

MoE work at T = 128/256 (0.08-0.09% per us on the MoE board, 0.05-0.06% on the layer), the BF16 m = 128/256
register-A brief as a top-100 path (its own text says it is not one), and any further D32 lookahead work
(closed, P4). Noise: the same layer file drifted 2.126 to 2.109 between runs (`cc474883`); nothing under
1.5% resolves in one check, which is another reason to go after the 3-5% items.

## 8. Gaps

No timeline of the incumbent layer exists (the G-family timeline run `41788b39` failed), so the
boundary cost per launch is the older 0.7 us figure. The layer's routing at decode 1 hits 1-2 local
experts on the profiled input; the MoE floor at that load is therefore a range. The 3.0 TB/s target in
section 4 is a hypothesis with one published comparable, not a measurement of our code.

## Correction, 2026-11-03 01:10 Pacific: what the day's checks changed

Evidence: BF16 793602020fd64ce9; MoE 19128cda79ca4e25, ab54144f6c4d4727, b45cce36b47f4e81, abc831c68f174b68;
layer 4c2d5e3a8e1147bf, 0c35e5f5cc614806, c77b95102baf4232, 43aa191eb94948a4; attention c1becf5a66274d37.

- Section 3's reading that the small GEMVs are "latency" that a persistent stream would remove is only half
  right. A short cell's judged time carries about 2 us of fixed per-launch cost (BF16: the kernel got 0.9 us
  faster in the profile and the judged cell did not move), and inside a graph a boundary costs about 0.7 to
  1 us. Fusing stages replaces each boundary with an in-kernel cross-CTA dependency that costs about 1.2 us
  per global round trip. Measured: fusing o + boundary + pooler/router saved 1.3 us at T = 1 (9 of 9 rounds);
  fusing the unpooler slice and finish into the MoE tail lost 2.6 us because its tail is five dependent round
  trips. Section 4's 3.0 TB/s target is withdrawn: the chain's kernels already stream at about 4 TB/s while
  they stream, and the gaps between them are worth about one microsecond each when fused, not several. The
  persistent post-attention kernel's realistic upside at decode 1 is 3 to 5 us (about 1% of the layer score),
  and it is no longer the structural bet for the deadline.
- Section 3's MoE reading was right about the size of the fixed cost (7 of 22.6 us at T = 1) and wrong about
  its recoverability. The timeline shows the ramp is the first wave's bytes, the routing prologue is already
  overlapped with the first fills, the final pass bounds a ticket finalizer at T = 8, and the 2.2 us A-to-B
  handoff can be recovered but every extra instruction in the unit loop costs 20 to 30% per unit. Remaining
  Tiny-build upside is about 1 us at T = 1 with a cleaner build; the MoE bar (+4.1%) is out of reach by kernel
  work in the time left. Re-draws moved the row from 2.3232x to 2.3438x.
- Section 6's attention ranking by value per microsecond missed the largest pool by room times value: the
  q_len 8 full-context modes run 15% slower per byte than the q_len 1 modes on the same bytes (3.00 vs 3.57
  TB/s at batch 16, 3.32 vs 3.73 at batch 32). Matching the q_len 1 rate is worth about 16 + 20 us, roughly
  3.8% of the attention score, which clears that bar. That is now the attention lane; the batch-1 cluster
  merge is closed (third cluster loss).
- Section 6's BF16 small cells are closed: their gap to the byte floor is the fixed launch cost. The only
  BF16 room with evidence is the m = 128 qkv kernel body (comparator 2.55 us faster); not assigned.
