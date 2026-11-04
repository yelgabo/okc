# cl: MoE cluster rewrite

Plan: [docs/plans/2026-11-04-moe-cluster-rewrite.md](../../docs/plans/2026-11-04-moe-cluster-rewrite.md).
This README covers the bench harness `cl_bench.py` (plan task 0.4, interface in section 3.3).

## cl_bench.py

`cl_bench.py` times one candidate MoE file against a control file at one token count and one routing shape, on the
box, paired. Per the box rule (docs/box.md, "Comparison rule"), the control is always the public file's bytes,
and a box number is only ever read as candidate against control in the same run.

```bash
ssh okc-h200
. /root/okc/tools/box_env.sh
python muse_moe_w4a16/cl/cl_bench.py --kernel muse_moe_w4a16/cl/<candidate>.py \
  --control muse_moe_w4a16/cand/current_control_release.py \
  --T 8 --distinct 7 --inputs 20 --runs 30 \
  --lock /root/okc/.build/box/gpu.lock --out .build/box/cl/<tag>.json > .build/box/cl/<tag>.log 2>&1
grep -E "^CL|Traceback" .build/box/cl/<tag>.log
```

What it does:

- Both files are submission-style modules: the first line is `# okc: problem=muse_moe_w4a16` (the harness refuses
  any other), with `prepare(weights)` and `kernel_fn(data, state)`. Weights come from `reference.generate_weights(5252)`
  once, and both prepares receive the same dict.
- Inputs are `reference.generate_input(bs=T, sharding="ep8", seed=s, dist="normal")` for s = `--seed` (default
  1000), `--seed` + 1, and so on. This is the scored shapes' routing: top 8 of uniform scores plus heavy-tailed
  popularity, with the ids rotated so that experts 0 to 31 form the busiest GPU. The MoE reference has no router
  or `case_fields`, so `bx_box_run.py`'s router-seed search has nothing to match here, and the routing is
  the seed. With `--distinct D`, the harness keeps only the inputs whose distinct local expert count is D, and it
  prints the distribution of every seed it scanned. Without `--distinct`, it takes consecutive seeds and reports
  the distribution it got.
- Each side's `kernel_fn` is captured once as a CUDA graph (two warm calls first, as `bx_box_run.capture` does).
  Each input is copied into both graphs' static buffers. Then come `--runs` repetitions in alternating order
  (candidate first, then control first, and so on). Each replay follows a 1 GiB L2 flush and is timed with CUDA
  events.
- After its replays, each input's graph output is checked for both sides with `reference.check_implementation`.
  That check applies the TASK.md bounds: relative L2 error at most 1.5% against the FP32 reference, any row at
  most 5x its bound, and no NaN or Inf. A candidate failure prints the reason, and the run exits with code 1.
- Per input, the harness prints the candidate and control medians with their interquartile range (iqr) and the
  byte floor for the input's distinct count. The floor is 28.31 MB per expert at 4.45 TB/s, so 6.362 us per
  expert. It also prints the paired delta, which is the median of (candidate - control) over the repetitions,
  with its iqr, and the candidate's net time over the floor. The overall `CLSUM` line gives the median of the
  per-input medians, raw and net. Net is raw minus the replay overhead, measured as the judge does:
  2 x t(1-kernel graph) - t(2-kernel graph). The line also gives the median paired delta with a bootstrap 95%
  interval over inputs, and the number of passing checks.
- `--kernels` adds each side's summed GPU kernel time per cold replay (torch.profiler over 3 replays, with the
  flush, memcpy and memset left out).
- `--flush write` replaces the judge's read flush (`torch.sum` over 1 GiB) with the `fill_` flush used by
  `tools/box_moe_tl8.py`. It exists only to reproduce that tool's numbers (see below). Use the default for every
  A/B.
- `--lock` takes `gpu.lock` in-process and lets it go while an extension or a Triton kernel compiles. It reuses
  `tools/bx_box_run.py`'s `GpuLock`, `unlocked`, content-addressed `load_inline` cache (`/root/.cache/bx_ext`)
  and budget clock, so lock waits do not count against a file's own 295 s prepare deadline. Do not also wrap the
  run in `flock(1)`.
- `--out` writes every replay time, the per-input rows, the summary, both files' SHA256 and the lock events as
  JSON.

Cost: the public file's first prepare builds five extensions in 230 s cold, with the lock released throughout.
With a warm cache, prepare takes under 1 s. A 20-input, 30-run T = 8 run then holds the GPU for about two
minutes.

## Self-check (2026-11-04, box okc-h200)

Candidate and control are the same file, `muse_moe_w4a16/cand/current_control_release.py` (SHA256
5df257afd6ae2863bc706826a9798d97b17e73830f1d10703c81bba21203eb9e). Each run used 20 inputs, 30 paired
repetitions per input, and `--kernels`. All times are in us. Logs and JSON are in `/root/okc/.build/box/cl/`
(`self_t8d7`, `self_t32`, `self_t8d7_wflush`).

| run | distinct (inputs) | floor at mean | overhead | candidate net (iqr) | control net (iqr) | paired delta, 95% CI | per-input delta iqr | kernel time cand / ctl | checks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T = 8, `--distinct 7` | 7 (20) | 44.54 | 4.16 | 56.75 (1.07) | 56.74 (1.15) | +0.01 [-0.18, +0.11] | 1.56 | 56.46 / 56.48 | 20/20 |
| T = 32, consecutive seeds | 10 to 21, mean 15.65 | 99.57 | 4.19 | 115.10 (1.36) | 115.20 (1.22) | -0.25 [-0.37, +0.02] | 1.84 | 114.96 / 114.99 | 20/20 |
| T = 8, `--distinct 7`, `--flush write` | 7 (20) | 44.54 | 4.16 | 62.44 (1.50) | 62.45 (1.55) | -0.10 [-0.24, +0.09] | 1.39 | see below | 20/20 |

The deltas are zero within spread. Every interval contains zero, and every median delta is at most 0.25 us
against a per-replay iqr of 1.1 to 1.8 us. No single input's delta at T = 8 exceeds its own delta iqr. One
input at T = 32 (seed 1018, 21 experts) reads +1.57 against an iqr of 3.50.

At T = 8, 125 scanned seeds gave a distinct-expert distribution of {4: 4, 5: 5, 6: 11, 7: 20, 8: 27, 9: 23,
10: 14, 11: 11, 12: 3, 13: 3, 14: 3, 16: 1}. At T = 32, seeds 1000 to 1019 gave {10: 2, 11: 1, 12: 1, 14: 4,
15: 4, 17: 1, 19: 4, 20: 2, 21: 1}.

### Agreement with docs/runs-moe.md "T = 8 MoE floor and gap"

That section reports, at 7 distinct experts, an instrumented span (first CTA entry to last CTA exit) of
61.66 us (sliding, n 8) and 62.24 us (full, n 9). It also reports a CUDA-event time of an isolated eager launch
of 83.82 / 83.58 us (`muse_decode_layer/profiles/box/tl8/tl8.log`). Its fits give 12.5 + 7.06 x 7 = 61.9 and
11.1 + 7.21 x 7 = 61.6 us.

| measurement at 7 experts | this harness | doc | ratio |
|---|---:|---:|---:|
| default read flush, graph net | 56.75 | span 61.66 / 62.24 | 0.920 / 0.912 |
| default read flush, kernel time | 56.46 | span 61.66 / 62.24 | 0.916 / 0.907 |
| `--flush write`, graph net | 62.44 | span 61.66 / 62.24 | 1.013 / 1.003 |
| `--flush write`, kernel time (3-input check run) | 62.24 / 61.94 / 63.92 | span 61.66 / 62.24 | within 1 to 4% |
| eager launch, CUDA events | not measured | 83.82 / 83.58 | |

With the default flush, the harness reads about 8.5% below the doc's span, which is outside the 5% target. The
cause is the flush, and the test above shows it. `tools/box_moe_tl8.py` flushes L2 with `flush.fill_(1.0)`, a
1 GiB write that leaves L2 full of dirty lines. The write flush costs the kernel 5.7 us (62.44 - 56.75). The
likely mechanism, which has not been isolated, is that L2 evicts those dirty lines to HBM while the kernel streams
its weights. With the same write flush this harness gives
62.44 us net, within 1.3% and 0.3% of the two spans. The 3-input `--flush write` run's kernel time (62 to 64 us)
places the extra time inside the kernel rather than in the graph launch. So the doc's span is a time measured
after a write flush, and the judge flushes by reading (TASK.md: "the judge reads a buffer larger than L2"). That
makes 56.7 us the judge-like number for this file at 7 experts.

The read-flush number also agrees with docs/box.md "Routing", which reports a profiler time for the layer's
`okc::moe_fused` of 55.33 (sliding, n 12) and 55.55 us (full, n 15) at 7 distinct experts under
`bx_box_run.py`'s read flush. That is within 2.1% and 1.6% of the 56.46 us kernel time here. The layer's TINY2
is the standalone's helper0 source (the doc's anchor match), but the layer's file is not this file. The two
numbers therefore agree as builds of one kernel, not as identical bytes.

The doc's eager-launch number (83.8 us) is 22 us above its own span. The doc attributes this to launch and host
work inside the CUDA-event window of an eager call. A graph replay does not pay that cost, so this harness has no
comparable number.

At T = 32 the doc has no reference number. docs/box.md's MoE decode 32 net of 124.08 us comes from 100 inputs of
the judge's case seeds through `bx_box_run.py`. This run used 20 consecutive seeds from 1000, with a mean of
15.65 distinct experts. Those are different inputs, so the absolute values are not expected to match. The
self-check here is the zero delta.
