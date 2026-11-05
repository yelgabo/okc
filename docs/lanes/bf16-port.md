# bf16-port lane checkpoint

Updated 2026-11-04 18:43 PST. Branch worktree-agent-a2d69d5fe5d6ab73d.

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

- okc-h200, files in /root/okc/.build/bfport/. Runs queued 18:43 under gpu.lock: bfp_base1, bfp_all1, bfp_alls1 (logs /root/okc/.build/box/<tag>.log).

## Next

Read per-shape medians, pick per-shape winners, build `muse_bf16_proj/cand/bf16_port.py` with the winners only, then three alternating A/B rounds against base.
