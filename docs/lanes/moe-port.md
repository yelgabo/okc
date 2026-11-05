# moe-port checkpoint

Updated: 2026-11-04 18:53 PST (from `date`).

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

## Build map (M to build) and expected native build tags

The tags come from the scratchpad script tags.py: md5 of MOE_CPP_SRC + src + repr((defines, 3)), with PRMTSC on all
builds and SC6W on the WGS builds.

| Shape | M | Build | port tag | base tag |
|---|---|---|---|---|
| decode 1 | 1 | tiny | glue_moe_2cb6e07dc691eff1 (NOSCL) | glue_moe_415a2bdc6eb73960 |
| verify 1 | 8 | tiny2 | glue_moe_163b29f2a08931ff | glue_moe_b2495a009a910301 |
| decode 16 | 16 | mid | glue_moe_3687d894b73f8e44 | glue_moe_1b555edc8284ed23 |
| decode 32 | 32 | b4 + MFIN | glue_moe_3a4ede75f863927a | glue_moe_7e37cd900cb88979 |
| verify 16, 32 | 128, 256 | main + RFIN + MFIN | glue_moe_13de9d65ad63a10d | glue_moe_45a005177ff52c30 |

The round 1 port log shows builds of glue_moe_13de9d65ad63a10d and glue_moe_3a4ede75f863927a, which are the
RFIN main build and the MFIN b4 build.

## Box

The A/B driver is `/root/okc/.build/box/moep_ab.sh`. It started at 18:43 PT and runs 4 rounds with orders port/base,
base/port, port/base, base/port. Each run is `bx_box_run.py --runs 60 --lock gpu.lock`, and its output goes to
`/root/okc/.build/box/moep/ab_{side}_{round}.{json,log}`. The driver log is `/root/okc/.build/box/moep/ab.out`.

Round 1 is complete, and both sides have check PASS at all 6 shapes. Each run spends 457 to 580 s in import and
prepare, which is the native build, so every run takes about 9 to 10 minutes. Rounds 2 to 4 should end
around 20:00 PT. Net times in us from round 1 (one pair, port run first):

| Shape | base r1 | port r1 | port - base |
|---|---|---|---|
| decode 1 | 23.54 | 23.66 | +0.12 |
| decode 16 | 90.22 | 88.69 | -1.53 |
| decode 32 | 120.83 | 119.41 | -1.42 |
| verify 1 | 64.03 | 63.89 | -0.14 |
| verify 16 | 175.52 | 172.88 | -2.64 |
| verify 32 | 208.99 | 208.14 | -0.85 |

This is a single pair, so it is not a verdict yet. The changes at decode 32 (MFIN on b4) and at verify 16 and 32
(RFIN and MFIN on main) point the expected way. The decode 16 gain is unexpected: the mid build only gets the
helper changes (scl, wspack, ftz, doneA). Both decode 1 and verify 1 are within noise.

## Next

The lane stopped at 19:02 PT because it reached its tool-call budget. The A/B driver keeps running on the box. To
tabulate it once `=== done` appears in ab.out, run:

```
ssh okc-h200 'cd /root/okc/.build/box/moep && python3 -c "
import json,glob,statistics as S
d={}
for f in sorted(glob.glob(\"ab_*_*.json\")):
    side=f.split(\"_\")[1]
    for r in json.load(open(f))[\"rows\"]:
        d.setdefault(r[\"load\"],{}).setdefault(side,[]).append(r[\"median_net_ns\"]/1e3)
for l,v in d.items():
    b,p=S.median(v[\"base\"]),S.median(v[\"port\"]); print(l,len(v[\"base\"]),len(v[\"port\"]),round(b,2),round(p,2),round(p-b,2))
"'
```

If the port wins at verify 16 and 32 and decode 32, and loses nowhere outside noise, it becomes the standalone
candidate. Before any judge check, a ledger registration line is required.
