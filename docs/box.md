# The RunPod H200 box

A rented H200 SXM (RunPod pod `l667980o376btc`, name `okc-h200`, $5.29/h while running) on which
the four boards' files run end to end the way the judge runs them, with no judge quota spent. It
is the same GPU model as the judge's, but the driver, host and per-run router differ, so read the
parity section before trusting any absolute number. Set up and measured 2026-11-03 by the box
worker; numbers below are measurements on that day.

## Reaching it

`ssh okc-h200` works from the Mac (the `~/.ssh/config` alias for `root@205.196.19.51 -p 11831`,
key auth). The direct address changes if the pod is stopped and restarted; the RunPod proxy
`ssh l667980o376btc-64410c58@ssh.runpod.io` stays fixed. Never stop or modify the pod from a lane;
that is the lead's call.

| item | box | judge (from its receipts) |
|---|---|---|
| GPU | H200 SXM, 132 SMs, 700 W limit | H200, 700 W limit |
| driver | 595.91.07 (CUDA 13.2) | 580.159.03 |
| torch / triton | 2.14.1+cu130 / 3.8.0 in `/root/venv` | `ENV torch=2.14.1+cu130` |
| cuBLASLt | 13.1.1 (pip `nvidia-cublas`) | `cublasLt 130101` |
| nvcc | 13.0.88 at `/usr/local/cuda-13.0` (`/usr/local/cuda` points there); 12.8 kept at `/usr/local/cuda-12.8` | 13 (docs mention "the judge's nvcc13 build") |
| SGLang | a55ab7d5 (from `okc serve init`) editable at `/root/sglang/python`, `sglang-kernel` 0.4.9, `flashinfer-python` 0.7.0.post1, `apache-tvm-ffi` 0.1.11 | a55ab7d5 |

The image's system `python3` still has torch 2.8.0+cu128; always use the venv. The repo copy is
`/root/okc` (rsync of `git ls-files` plus `.build/repeats/*_base.py`; profile JSONs left out). Judge
JSONs used for comparison sit in `/root/okc/.build/judge/`. Extension and Triton caches live on the
local disk under `/root/.cache`; `/workspace` is a slow network mount and is not used.

Sync a changed file before running it:

```bash
rsync -a muse_decode_layer/cand/<file>.py okc-h200:/root/okc/muse_decode_layer/cand/
```

## Running a file

`tools/box_run.py` loads a submission, runs `prepare(weights)` with the problem's
`reference.generate_weights(WEIGHTS_SEED)`, checks one fresh input with the problem's
`check_implementation`, then captures one CUDA graph per shape and replays it `--runs` times. Each
replay gets its own fresh input copied into the graph's buffers and a cold L2 (a 1 GiB read just
before it), timed with CUDA events. `--baseline` captures SGLang's `baseline.py` too and alternates
the order per repetition, which gives the judge's paired ratio and its "yours" column. The fixed
replay cost is measured as the judge does (its `replay_overhead` records satisfy
`overhead = 2 * one_ns - two_ns`, graphs of one and two tiny kernels) and subtracted. `--judge`
takes judge JSONs, reuses their case seeds and prints judge against box per cell. For the layer it
also reads the router_distinct target each cell's `speed_of_light_ns` implies (decode 1: 2.5,
verify 1: 7.5, decode 16: 7.5, decode 32: 9.0, verify 16: 18.4, verify 32: 21.3 distinct experts)
and searches a router seed that meets them, as the judge's `weights_fields` does. The judge also
matches mean routes (router_load), which no receipt exposes, so this match is partial.

```bash
# one job at a time on the GPU; tools/box_bg.sh backgrounds it and logs to .build/box/<tag>.log
ssh okc-h200 'cd /root/okc && tools/box_bg.sh mytag muse_decode_layer muse_decode_layer/cand/<file>.py \
  --baseline --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --router-seed 106218864389449120 --loads "decode 1,verify 1" --runs 60 --kernels'
ssh okc-h200 'grep -E "^BOX|KERNEL|PDL8|Traceback" /root/okc/.build/box/mytag.log'
```

Interactive use: `ssh okc-h200`, then `. /root/okc/tools/box_env.sh` (venv, CUDA 13.0 on PATH,
`cd /root/okc`). Options: `--loads` filters by load name, `--no-check` skips the correctness check,
`--kernels` adds a per-kernel GPU time table (torch.profiler over 10 cold replays; Memcpy/Memset
and the flush excluded), `--router-seed` skips the router search (106218864389449120 is the seed the
search picked for the layer), `--out` writes all runs as JSON (box_bg.sh sets
`.build/box/<tag>.json`). The in-prepare benches print exactly as on the judge (PDL8_AB,
PDL8_STRESS, PDL8D lines) because they are the file's own code; prepare of layer_base.py takes
120 s cold and 14 s with warm caches.

## Parity, 2026-11-03

Net time is median raw replay minus the fixed replay cost, in us. Judge numbers are medians over the
public draws of the same bytes (layer: f931abf568544018 and 4a737387f88b4d7d of g1b4_pdl8d 7cb74240;
MoE: seven repeats of moe_base 5df257af; attention: eight repeats of aa3535d0; BF16: eight repeats
of 5610b8a6), from `.build/repeats/*.json` and `muse_decode_layer/cand/profiles/`. A cell is trusted
when box/judge is within 3%. Every box check passed.

### Layer (layer_base.py = g1b4_pdl8d 7cb74240, router matched on distinct experts)

The judge's shown "yours" is SGLang's median over the paired ratio; the task's quoted numbers
(70.7 / 78.3, 128.8 / 211.2, 150.5 / 302.3, 105.9 / 119.3) are the f931 shown values.

| cell | judge net | box net | box/judge | SGLang judge | SGLang box | SGLang box/judge |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 69.94 | 61.68 | 0.882 | 172.64 | 181.06 | 1.049 |
| decode 1 full | 77.93 | 67.49 | 0.866 | 183.14 | 190.50 | 1.040 |
| decode 16 sliding | 126.62 | 134.00 | 1.058 | 257.38 | 272.94 | 1.060 |
| decode 16 full | 207.84 | 209.34 | **1.007** | 433.89 | 447.78 | 1.032 |
| decode 32 sliding | 149.87 | 153.54 | **1.024** | 261.12 | 271.73 | 1.041 |
| decode 32 full | 304.81 | 298.21 | **0.978** | 619.32 | 617.18 | 0.997 |
| verify 1 sliding | 109.56 | 99.54 | 0.909 | 264.70 | 260.43 | 0.984 |
| verify 1 full | 120.42 | 98.62 | 0.819 | 277.70 | 264.34 | 0.952 |
| verify 16 sliding | 248.52 | 236.37 | 0.951 | 524.42 | 522.72 | 0.997 |
| verify 16 full | 336.64 | 322.02 | 0.957 | 711.01 | 699.78 | 0.984 |
| verify 32 sliding | 313.51 | 295.54 | 0.943 | 717.58 | 679.25 | 0.947 |
| verify 32 full | 477.40 | 458.99 | 0.961 | 1061.08 | 1037.70 | 0.978 |

With the default router (no seed search) the layer gave decode 1 61.97 / 68.06, decode 16
119.79 / 208.43, verify 1 99.79 / 92.86: the router moves decode 16 sliding by 14 us and verify 1
full by 6 us, so the judge's per-run router seed is itself a large source of variation at those
cells.

Per-kernel (`--kernels`) at decode 1 and verify 1, set against the f931 profile: the dense
projections and boundary kernels agree within a few tenths of a us (sk2 qkv 17.97 vs 17.3,
sk2 pooler 9.82 vs 10.2), attention is about 1 to 1.5 us slower on the box under the profiler,
and `okc::moe_fused` is the gap: 18.1 vs 29.5 us (decode 1 sliding), 14.5 vs 22.6 (decode 1 full),
51.8 vs 57.7 (verify 1 sliding), 42.2 vs 87.1 (verify 1 full). The judge profile is one input
under one router, so part of this is routing; the rest is not explained yet (hypotheses: the
judge's router_load target gives more routes per expert than the distinct-only match; driver
595 vs 580; the judge's prepare padding and input copies change page placement for the 800 MB
expert bank).

In-prepare logs print the same lines with the same class of numbers. PDL8_AB T = 1 sliding: judge
off 83.94 / on 77.22, saving 7.07; box 77.70 / 70.98, saving 6.98. T = 1 full: judge 90.85 / 84.48;
box 91.14 / 83.33. T = 16 sliding: judge 173.82 / 168.29; box 185.66 / 181.31. T = 32 full: judge
379.10 / 374.46; box 352.70 / 347.23. T = 8: judge 139.30 / 134.43 (sliding), 126.85 / 122.69
(full); box 120.10 / 114.69, 135.23 / 129.63. Savings agree within 1.5 us (7.07 / 6.98, 6.30 / 7.71,
5.06 / 4.51, 4.44 / 5.20); the absolute arms differ by up to 14% in both directions (hypothesis:
the in-prepare inputs route through the run's router, which differs between judge and box). PDL8_STRESS
and PDL8D_STRESS show the same drift class (box T = 1 sliding off/on 6.98e-04 / 7.55e-04 with 0
int mismatches, judge 9.77e-04 / 1.08e-03), the same "strict FAIL, within the off arm's own drift
pass" outcome, and PDL8D picks all six T = 8 edges, as on the judge. The f931 log has no CG_TUNE
lines and neither does the box log.

### MoE (moe_base.py 5df257af, paired with SGLang, 100 fresh inputs)

| cell | judge net (range, n = 7) | box net | box/judge | SGLang judge (0f5f) | SGLang box |
|---|---:|---:|---:|---:|---:|
| decode 1 | 23.28 (22.94-23.44) | 24.08 | 1.034 | 77.30 | 84.51 |
| decode 16 | 95.42 (93.50-101.97) | 91.46 | 0.958 | 195.39 | 192.45 |
| decode 32 | 126.08 (123.28-130.19) | 124.08 | **0.984** | 205.20 | 207.15 |
| verify 1 | 66.72 (63.79-70.30) | 63.25 | 0.948 | 146.51 | 147.62 |
| verify 16 | 179.79 (176.83-183.23) | 172.58 | 0.960 | 460.82 | 472.94 |
| verify 32 | 215.63 (211.87-216.61) | 208.14 | 0.965 | 618.51 | 619.15 |

A first run that cycled only four fresh inputs gave 0.89 to 1.15; MoE time depends on each input's
routing, so the box must use one fresh input per run (the default now).

### Attention (attention_base.py aa3535d0, paired)

| cell | judge net (range, n = 8) | box net | box/judge | SGLang box/judge |
|---|---:|---:|---:|---:|
| decode 1 sliding | 5.16 (4.77-5.34) | 4.96 | 0.961 | 1.007 |
| decode 1 full | 12.44 (11.94-12.70) | 12.86 | 1.034 | 1.071 |
| decode 16 sliding | 8.22 (7.57-8.54) | 8.10 | **0.984** | 1.084 |
| decode 16 full | 87.94 (84.85-88.32) | 84.22 | 0.958 | 1.012 |
| decode 32 sliding | 10.12 (9.73-10.37) | 10.77 | 1.064 | 1.085 |
| decode 32 full | 168.04 (162.06-168.56) | 161.41 | 0.961 | 1.011 |
| verify 1 sliding | 6.88 (6.45-7.14) | 7.10 | 1.033 | 1.120 |
| verify 1 full | 14.70 (14.54-14.90) | 14.85 | **1.010** | 1.057 |
| verify 16 sliding | 11.58 (11.22-11.78) | 11.54 | **0.997** | 1.064 |
| verify 16 full | 104.48 (101.71-105.36) | 99.18 | 0.949 | 1.010 |
| verify 32 sliding | 15.17 (14.99-15.33) | 15.68 | 1.034 | 1.087 |
| verify 32 full | 187.04 (183.26-191.02) | 179.26 | 0.958 | 1.008 |

The short attention cells are 5 to 15 us, so 3% is 0.2 to 0.5 us; decode 1 sliding and verify 1
sliding fall inside the judge's own draw range although outside 3%.

### BF16 projections (bf16_base.py 5610b8a6, paired; per load, sum of the five layers)

| load | judge net | box net | box/judge | SGLang judge | SGLang box | SGLang box/judge |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 | 34.86 | 36.00 | 1.033 | 59.82 | 61.31 | 1.025 |
| decode 16 | 48.61 | 50.13 | 1.031 | 76.07 | 78.91 | 1.037 |
| decode 32 | 52.03 | 54.13 | 1.040 | 77.08 | 80.03 | 1.038 |
| verify 1 | 37.58 | 38.75 | 1.031 | 75.12 | 76.98 | 1.025 |
| verify 16 | 67.85 | 69.04 | **1.018** | 85.54 | 88.35 | 1.033 |
| verify 32 | 83.90 | 84.45 | **1.007** | 101.77 | 102.90 | 1.011 |

Both sides run about 3% slow on the box, so the board ratio holds while the absolute times run
about 3% high. Per-shape rows are in `/root/okc/.build/box/bf16_base.log`.

### What is trusted

- Within 3% of the judge: layer decode 16 full, decode 32 sliding, decode 32 full; MoE decode 32;
  attention decode 16 sliding, verify 1 full, verify 16 sliding; BF16 verify 16 and verify 32.
- Not within 3%: every other cell, including all layer decode 1 and verify 1 cells (box 9 to 18%
  fast, mostly in the in-layer MoE kernel) and the BF16 short loads (box 3 to 4% slow on both
  sides).
- Use the box for A/B comparisons inside one box run (same router seed, same inputs, paired
  order), for correctness checks, for in-prepare bench logic and for per-kernel attribution of
  the non-MoE kernels. Do not use it to predict a judge score or to rank against the board.
  Whether box deltas carry over to the judge at the layer short loads is untested; the in-prepare
  PDL8 savings (judge against box within 1.5 us) are the one direct comparison so far.

## Profiling

Nsight Compute 2025.3.1 is installed (`/usr/local/cuda-13.0/bin/ncu`, with CUDA 13.0) and runs,
but kernel counter collection fails with `ERR_NVGPUCTRPERM`. The host driver has
`RmProfilingAdminOnly: 1` (`/proc/driver/nvidia/params`), and the container lacks
`CAP_SYS_ADMIN`, so only the host can lift it (driver option `NVreg_RestrictProfilingToAdminUsers=0`).
It cannot be changed from inside this pod; profiling counters need a host or pod type that
allows it.
What does work is CUPTI activity tracing: `--kernels` gives per-kernel GPU time per shape. The
attempted recipe, for when counters are available:

```bash
ssh okc-h200 'cd /root/okc && . tools/box_env.sh && ncu --profile-from-start off -k regex:moe_fused -c 1 \
  --set detailed -f -o .build/box/ncu_layer_d1_moe python tools/box_run.py muse_decode_layer \
  .build/repeats/layer_base.py --loads "decode 1" --profile --no-check --runs 1 --router-seed 106218864389449120'
```

`--profile` runs one warm eager call per case inside `cudaProfilerStart/Stop`, so prepare's kernels
are skipped. Prepare's in-file benches do run under ncu, and their numbers there are invalid
(observed: PDL8_AB T = 32 flips to a loss and a PDL8_STRESS T = 32 FAIL under ncu interception).

## Comparison rule

Every box number is a paired comparison against the public file's bytes run on the box in the same session (box-candidate versus box-control). Never compare a box run to a judge number; the judge decides keeps.
