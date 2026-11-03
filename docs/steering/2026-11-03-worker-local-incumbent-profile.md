# Local incumbent component profile on Gretchie — November 3, 2026

Actual CUDA execution completed on this computer's **RTX 5070 (SM120, 12 GB)**, using PyTorch 2.10.0+cu128 and driver 610.88. The measured artifact is the public layer incumbent `muse_decode_layer/cand/g1b4_stage_release.py`, LF SHA256 `ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab`. Fresh public snapshot still identifies incumbent job **607eb52bdaf24160**, 10639.426326569794 us / 2.1261x. Those official H200 results are separate from these local measurements.

**The complete native incumbent was not runnable unchanged on this GPU.** Both selected native MoE source branches require `prop.major==9 && prop.minor==0 && prop.cooperativeLaunch`; this device reports 12.0. Hopper architecture-specific binaries/PTX also cannot be reused on Blackwell, as documented by [NVIDIA](https://docs.nvidia.com/cuda/archive/12.9.2/blackwell-compatibility-guide/index.html). This report measures the two selected portable Triton stages and explicitly exercises the existing cuBLAS dense fallback. It does not retarget or replace the production MoE/attention/SK2 implementation and does not establish an end-to-end layer time.

## Measured local numbers

Cold measurements: 31 randomized rounds, 256 MiB flush immediately before each single-call CUDA graph replay; events bracket the replay, excluding the flush. Inputs and weights are seeded synthetic tensors at actual short-load stage dimensions. All units below are microseconds.

| Stage | T=1 cold median | T=8 cold median |
|---|---:|---:|
| QKV projection (cuBLAS fallback) | 72.48 us | 94.59 us |
| O projection (cuBLAS fallback) | 38.27 us | 39.17 us |
| Pooler (cuBLAS fallback) | 121.28 us | 132.58 us |
| Sliced unpooler (cuBLAS fallback) | 29.54 us | 39.30 us |
| Routing + pre-expert norm (selected Triton) | 4.45 us | 4.35 us |
| Final residual + norm (selected Triton) | 4.19 us | 5.31 us |

The cuBLAS rows use the incumbent's existing `dense()` branch with a private explicit `mm_pick=True` state. They are **not** the losslessly packed SK2 projection kernels selected in the public H200 run. Pooler here is separate from router; its timing is not the fused pooler/router call. Unpooler uses the short-load 1024-output slice. Attention, MoE, QKV preparation and the first residual boundary are unmeasured locally. Do not add these rows into a purported layer total or scale them to H200.

## Timing detail and profiler evidence

Warm measurements average 64 repeated same-input calls in one captured graph, over 11 observations. The desktop GPU was active without clock locking. Some warm dense observations have large scheduling variance; retain their distributions rather than treating the smallest draw as a gain. Single-call event timing includes graph scheduling effects; the separate profiler column is one kernel-only observation, not a confidence interval.

| Stage | T | Cold p10 / p50 / p90 (us) | Warm p10 / p50 / p90 (us/call) | Profiler kernel duration (us), one observation |
|---|---:|---:|---:|---:|
| QKV projection (cuBLAS fallback) | 1 | 72.03 / 72.48 / 75.20 | 63.00 / 115.60 / 125.39 | 64.691 |
| O projection (cuBLAS fallback) | 1 | 37.18 / 38.27 / 39.84 | 28.78 / 33.86 / 40.35 | 29.292 |
| Pooler (cuBLAS fallback) | 1 | 119.52 / 121.28 / 123.26 | 116.73 / 179.29 / 196.70 | 110.643 |
| Sliced unpooler (cuBLAS fallback) | 1 | 28.48 / 29.54 / 31.14 | 15.87 / 16.08 / 86.19 | 18.227 |
| Routing + pre-expert norm (selected Triton) | 1 | 3.74 / 4.45 / 4.86 | 1.41 / 1.41 / 2.72 | 1.631 |
| Final residual + norm (selected Triton) | 1 | 3.81 / 4.19 / 5.02 | 1.28 / 1.28 / 1.28 | 1.567 |
| QKV projection (cuBLAS fallback) | 8 | 93.12 / 94.59 / 95.49 | 32.05 / 36.62 / 102.68 | 82.918 |
| O projection (cuBLAS fallback) | 8 | 37.50 / 39.17 / 39.65 | 14.21 / 14.21 / 14.24 | 35.527 |
| Pooler (cuBLAS fallback) | 8 | 131.14 / 132.58 / 133.54 | 130.14 / 191.23 / 199.48 | 125.256 |
| Sliced unpooler (cuBLAS fallback) | 8 | 38.08 / 39.30 / 41.63 | 15.55 / 19.09 / 38.32 | 25.614 |
| Routing + pre-expert norm (selected Triton) | 8 | 3.68 / 4.35 / 5.47 | 1.40 / 1.40 / 1.41 | 1.887 |
| Final residual + norm (selected Triton) | 8 | 4.16 / 5.31 / 6.24 | 1.82 / 1.82 / 1.92 | 2.174 |

CUDA profiling captured exactly 12 kernel events: the eight cuBLAS/CUTLASS fallback launches and the four selected Triton launches. [Chrome/Perfetto trace](evidence/2026-11-03-worker-local-incumbent-profile/trace.json), [kernel event list](evidence/2026-11-03-worker-local-incumbent-profile/kernel-events.json), [profiler table](evidence/2026-11-03-worker-local-incumbent-profile/profiler.txt), [full samples and validations](evidence/2026-11-03-worker-local-incumbent-profile/result.json). The profiler's annotation rows duplicate attributed device time; they must not be summed with kernel rows.

## Validation, disposition and next action

All eight projection checks agreed with independent FP32 PyTorch multiplication to relative L2 <= 9.11e-6. Both routing checks matched exact global/local IDs and BF16 normalized input; route-weight maximum absolute error was 1.49e-8. Both final-boundary hidden outputs matched the BF16 reference exactly; FP32 residual maximum absolute error was 1.91e-6. These are eager seeded component checks, not full-layer, exhaustive numerical or fresh-replay qualification. The script then ran the captured graphs and retained their CUDA traces.

Collector 6803 exited 0. No judge registration, polling of another owner's job, public/scored action, production-default edit or optimization occurred. **No savings are claimed**: this is an actual local baseline profile. Whole-incumbent numbers still require a compatible H200/SM90 host or a separately scoped local port with new validation. The existing native guard is preserved.

All five public boards were freshly retrieved with successful CLI exits. Release audit against those snapshots passed with **zero promotion gaps / 16 leads**; this local run adds no official receipt or release candidate. Initial WSL audit could not find `rg`; the Windows Python/native-rg rerun succeeded. Raw observations, architecture scope, transport and hashes are archived beside this report. Reproduce with:

```powershell
wsl --cd /mnt/c/Users/gretc/Documents/GitHub/okc --exec /root/.cache/okc-graph-cu128/bin/python tools/local_incumbent_profile.py
```

README/index remains captain-owned; this report is ready for its index handoff. Other owners' active jobs and source files were preserved.
