# moe-port checkpoint

Updated: 2026-11-04 18:50 PT (from `date`).

## Task

Port the layer's shared MoE changes (muse_decode_layer/cand/ly_stk7c.py, sha256 920587d5...) into the
standalone muse_moe_w4a16 file. The board row is 2.3685x at rank 341.

## Base

`.build/repeats/moe_base.py`, 320,466 bytes, sha256
5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e (copied from the main checkout).

## Candidate

`muse_moe_w4a16/cand/moe_port.py`, 325,047 bytes, sha256
5542c8a8b3e766e06eaaba374d9bc2e6d0a1aefec9fbef28ae128d0407b8c8d4. It is built by
`python3 tools/dl_moep_patch.py .build/repeats/moe_base.py muse_moe_w4a16/cand/moe_port.py`. Every edit
is region-scoped and asserts its match count.

The edits are:

- The ws_helper phase A doneA credit fix, in MOE_CUDA_SRC and in moe_helper0. The per-lane `__threadfence` before
  `__syncwarp` is gone; lane 0 issues `fence.sc.gpu` before its atomics (see docs/lanes/moe-wsrel2.md).
- MFIN on B4 (ly_gb4) and on the RFIN main build (ly_gbm). The gate is now
  `#if OKC_WGS && !OKC_B4FIN && !OKC_RFIN`, and both MFIN polls are unbounded `while` loops.
- RFIN (ly_rfin) on MAIN: the OKC_RFIN/OKC_RFINQ defaults, WGSREORD `(!OKC_RFIN)`, WGSDFIN
  `(OKC_RFIN ? 3 : 2)`, the flush early return, and the register finish slot block. The Python side adds
  `MAIN_EXTRA_DEFINES = ("-DOKC_RFIN=1", "-DOKC_RFINQ=2")` to the main key, as in layer line 16458.
- The ly_micro2 changes. ftz replaces all 8 `__expf(-gt)` sites. wgspack adds the paired xB store in MOE_CUDA_SRC,
  and wspack the paired inter store in the helper. scl adds the one-PRMT scales in the helper unit loop,
  and TINY gets `-DOKC_NOSCL=1`.

The following were not ported because they are layer-only: the PDL launch (cudaLaunchKernelExC), the unpooler,
and the route kernels.

As a fidelity check, the comment-stripped, whitespace-normalized MOE_CUDA_SRC and moe_helper0 of the port
match ly_stk7c's except for the PDL launch block (14 changed lines each, all in the launch).

## Build map (M to build)

| Shape | M | Build |
|---|---|---|
| decode 1 | 1 | tiny |
| verify 1 | 8 | tiny2 |
| decode 16 | 16 | mid |
| decode 32 | 32 | b4 (now with MFIN) |
| verify 16 | 128 | main (now RFIN+MFIN) |
| verify 32 | 256 | main (now RFIN+MFIN) |

## Box

Not run yet. The paths will be `/root/okc/muse_moe_w4a16/cand/moe_port.py` and `/root/okc/.build/repeats/moe_base.py`.

## Next

rsync to okc-h200, run the check (box_bg.sh, which holds gpu.lock), then run a paired A/B of base against port with 3 or more runs each.
