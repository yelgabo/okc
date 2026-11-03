# Profiling and experiments

Source sections: Systems, Compute, roofline, microbenchmarks, profiler templates and experimental discipline.

## Reuse the measuring instrument

Read [the run logs](../../runs-decode-layer.md), [hotspot history](../../hotspots.md), [the baseline-aware profiler audit](../../../tools/top100_profile_evidence/audit.json), and the experiment's owning receipt. Run `tools/hotspots.py` with its documented baseline argument before choosing a kernel. Existing SGLang baseline profiles satisfy the need for a baseline; do not spend quota recreating them without a reason.

Record file SHA256, commit, selected route, defines, compiler/toolkit/driver versions, GPU model, clocks/power, shape, actual local counts, cache policy, warmup, repetitions, graph semantics and numerical tolerance. Attribute timing to its own job, source and route. A whole-file grep, imported helper or successful compile does not show which path the benchmark executed.

Use Nsight Systems to identify CPU gaps, synchronization, copies, graph nodes, stream dependencies and overlap. Use Nsight Compute on the selected kernel to distinguish DRAM, L2, shared-memory, issue, Tensor Core and tail limits. The exact metric and section names depend on the installed version. Query them before dispatch. See [Systems CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and [Compute profiling guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

Example commands for a separately available local GPU benchmark, outside `kernel_fn`:

```sh
nsys profile --trace=cuda,nvtx,osrt -o systems-capture python benchmark.py
ncu --list-sections
ncu --query-metrics
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section Occupancy --section SchedulerStats --section WarpStateStats \
  --section SourceCounters --kernel-name regex:SELECTED_ENTRY \
  --launch-count 1 -o compute-capture python benchmark.py
```

`benchmark.py` and `SELECTED_ENTRY` are placeholders. Existing standalone harnesses include [moe_make_bench_ab.py](../../../tools/moe_make_bench_ab.py) and the more recent [complete expert harness](../../../tools/expert_top100_moe_harness.py). Their source contracts differ. Use the actual selected path and caller rather than inventing an executable with these placeholder names. Profilers require a GPU host and may be unavailable on the judge. No profiler was executed by this intake.

## Interpret counters narrowly

Record tensor-pipe utilization, issue-active rate, eligible warps, DRAM/L2 traffic, shared bank conflicts, barriers, scoreboard/MIO stalls, registers, spills and tail behavior. Investigate high stall percentages when schedulers miss issue opportunities. A high percentage alone does not identify exposed time.

If DRAM is near its relevant limit, test fewer packed bytes or better reuse. If L2 limits delivery, inspect tile reuse and shared feed. Low memory and tensor utilization points toward scheduling, dependencies or pipeline bubbles. High tensor utilization supports a compute-limit hypothesis only for that instruction mix and workload. It does not prove the whole application has reached a ceiling.

## Measure the workload we are judged on

The task times graph replays with cold L2, fresh inputs, paired order swaps and a fixed replay-cost adjustment. Eager CUDA-event timings can be diagnostics, but cannot replace that instrument. Events on one stream do not automatically include independent-stream work. Keep profiler overhead and observer-only kernels outside the acceptance timing.

Compare independently prepared duplicate controls and interleaved A/B arms in the same process. Check fresh outputs after capture/replay and state transitions, including empty routes. Use real and held-out legal route distributions. Report latency distributions, null drift and confidence intervals, not a single favorable median. Mismatched seeds or nodes make historical comparisons descriptive.

D32's objective is `46 * sliding_us + 16 * full_us`; the layer score is an equal-six-load geometric mean. If only one load improves by a factor `r`, the layer factor is `r ** (1/6)`. For a fraction `f` of exposed execution improved by factor `s`, Amdahl's factor is `1 / (1 - f + f/s)`. For `f=0.1, s=1.1`, that is about `1.00917`, not a promised gain. Activity totals, logical bytes and peak rates are not exposed removable time.
