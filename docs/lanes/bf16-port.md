# bf16-port lane checkpoint

Updated 2026-11-04 18:53 PST. Branch worktree-agent-a2d69d5fe5d6ab73d.

## Base and source

- Base: `.build/repeats/bf16_base.py`, SHA256 5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4 (board row 1.5321x, judge job 9eba96894ee1402e, prepare_s 106.4).
- Layer file: `muse_decode_layer/cand/ly_stk7c.py`, SHA256 920587d5b89153f10addfc50627008611f5f873d440eed89db7e950374887c5e.

## Audit of layer projection changes

- m = 128 / 256: the standalone tuner already picks wd b1n128c3s8h for qkv 128 and cuBLAS or cut elsewhere, which ties or beats the layer's Lt pins (docs/lanes/bf16.md step 1). bfq is the standalone's own wd geometry. Nothing to port.
- CG unpooler: C2's cgemv, already in the standalone.
- PCX_CFG, PDL edges, GLE (o epilogue fusion), router pairing, fused pooler/router glue: layer-only.
- Portable: the layer's sk2 backend (keell0renz's kernel and geometries) at m <= 32, and the narrow qkv build (`_de_source`, ly_q1n plus T 2..8) at m 1 and 8.
- Judge comparison (not paired): keell0renz exact source vs base, base minus keell0renz in us: qkv m 8 +0.71, o m 8 +0.13, router m 8 +0.14, pooler m 1 +0.12, unpooler m 1 +0.16; base wins at m 16 and most of m 32.

## Candidates

- Tool: `tools/dl_bfp_port_patch.py` (anchors match exactly once; appends sk2 extraction and glue, overrides prepare and kernel_fn for the keys in `--use`; prepare drops any key whose output misses the reference by more than 2e-2 relative).
- `muse_bf16_proj/cand/bf16_port_all.py` (`--use all`: 20 keys on sk2, qkv m 1/8 narrow), SHA256 35af2029372bdbafbfb7a15d009660dfdb97ccc96eb20ad16775b135036b5b49, 263,713 bytes.
- `muse_bf16_proj/cand/bf16_port_alls.py` (`--use alls`: qkv m 1/8 on the wide sk2 geometry), SHA256 7f54d91e7449aa0799abe12eb0fdf29abcf845c27a270e287211b46805714eae.

## Box

- okc-h200, files in /root/okc/.build/bfport/, runner /root/okc/.build/box/bfp/ab.sh (bx_box_run.py --runs 60 --lock gpu.lock), results /root/okc/.build/box/bfp/r<N>_<side>.json, compare with /root/okc/.build/box/bfp/cmp.py.
- Round 1 (one run each, all cells check PASS): untouched m 128/256 cells moved by up to 0.8 us between runs, so one run only flags large effects. Port minus base, us:
  - qkv narrow: m 1 -0.16, m 8 -0.64. The wide sk2 geometry at qkv m 1 / 8 is +1.44 / +0.08, so the narrow build is the part that carries.
  - Large losses on sk2: router m 1 +1.63, unpooler m 1 +1.39, qkv m 16 +1.06, pooler m 16 +0.82, unpooler m 32 +0.83, o m 8 +0.72, pooler m 8 +0.74, router m 8 +0.86. Dropped.
  - Within noise, kept for the repeat: o m 32 -0.19, qkv m 32 -0.16, router m 16 -0.14, unpooler m 16 -0.14 (the alls run gave -0.03, +0.03, +0.27, +0.21).
- New candidates: `muse_bf16_proj/cand/bf16_port_q.py` (qkv m 1 and 8 narrow only), SHA256 a8282b3e538cd021c9fd0d5f7eb49230e5ae2a920755bd410a58fec37fd0a106; `muse_bf16_proj/cand/bf16_port_qp.py` (plus o 32, qkv 32, router 16, unpooler 16 on sk2), SHA256 ebc16e6f87a75ca69845800c65b869baec37149dc399fd4ed9a2433e85744e58.
- Rounds 2 to 4 queued 18:52 (orders base,q,qp / qp,q,base / base,q,qp).

## Next

Compare rounds 1 to 4 per shape (base vs q vs qp), pick the final file, copy it to `muse_bf16_proj/cand/bf16_port.py`, report.

## Checkpoint 3 (18:55 PST): round 2 in

Round 2 (base, q, qp; one run each, 0 check FAIL in all three logs). Port minus base, us, base median over r1+r2:
qkv m1 q +0.38 / qp +0.10; verify 1 qkv (m8) q -0.29 / qp +0.03; router m1 +0.58 / +0.26; decode 32 qkv -0.63 / +0.55.
Cells the q file does not touch moved by up to +-0.8 us, so one round decides nothing. Rounds 3 and 4 are running.
Next action: cmp.py over r1_base + r2..r4, pick keys, build muse_bf16_proj/cand/bf16_port.py.

## Checkpoint 4 (19:03 PST): final, no win

Rounds 1 to 4 done (base 4 runs, q and qp 3 runs each, 60 timed runs per cell per run). Every log: 30/30 BOX lines check PASS, 0 check FAIL.
Per-cell medians and deltas: docs/lanes/bf16-port-ab-r1r4.txt (box files /root/okc/.build/box/bfp/r[1-4]_*.json).

- q (qkv m1/m8 narrow `_de_source` build) changes only two cells: decode 1 qkv 8.94 -> 9.31 (+0.38 us), verify 1 qkv 10.08 -> 9.95 (-0.13 us).
- Cells the q file does not touch moved between -0.48 and +0.46 us, so both deltas are inside run-to-run noise.
- qp (q plus the layer picks for qkv 32, o 32, router 16, unpooler 16) is worse: decode 32 qkv +0.55, decode 16 unpooler +0.63, decode 16 router +0.36.
- r1 showed all-key ports (bf16_port_all, bf16_port_alls) at +0.7 to +1.6 us on most m <= 32 cells.

Final file: muse_bf16_proj/cand/bf16_port.py, sha256 a8282b3e538cd021c9fd0d5f7eb49230e5ae2a920755bd410a58fec37fd0a106, 263,279 bytes, byte-identical to the measured bf16_port_q.py.
Verdict: no measurable gain over bf16_base.py (5610b8a6). The standalone base already carries the same launch_sk2 family, and the layer's picks are tuned for the layer's neighbours (L2 state, PDL), not for a standalone launch. Do not spend a judge check on it.
Next action: none for this lane; the lead decides whether the board row needs anything else.
