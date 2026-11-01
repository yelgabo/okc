# Prefill runs

All times and scores come from the named official job, not extrapolated decode results.
The October31 user goal explicitly includes this board and permits scored submissions.

| Date (PDT) | Job | Kind | Source/change | Score (95%CI) | Notes |
|---|---|---|---|---|---|
|2026-10-31|09ba15bd18fd4000|check|SGLang baseline copied with problem header, baseline_profile/baseline.py at e0ce491|1.000953x (0.985939–1.015967)|Passed; first SGLang hotspot map. Raw receipt and all downloaded profiles retained under baseline_profile/|
|2026-10-31|e6c12149c99a4a96|check|cand/marlin_m16.py; fixed expert M16|pending|NVFP4 weights/BF16 boundaries unchanged; all other baseline calls unchanged|
|2026-10-31|334149df33f64c54|check|cand/marlin_m32.py; fixed expert M32|pending|Same hypothesis, alternative tile granularity|
|2026-10-31|181880202c5742db|scored|Exact checked SGLang baseline plus scored-repeat comment|pending|Establish first board entry, not a claim of optimization|

## First hypothesis: MoE M-tile granularity

Baseline profile09ba15bd18fd4000 puts Marlin MoE at40–67% of profiled time, including
749.8us at turn1 sliding,2383.1us at turn4 sliding,11547us at first-turn sliding.
The baseline selects M64 at large shapes through the token-count heuristic. Fixed M16/M32
test whether reducing expert padding/altering occupancy is beneficial on actual routed work.
This is a hypothesis; no real-routing padding count or occupancy measurement is available yet.
Both candidates keep the same alignment, GEMM1, SwiGLU, GEMM2, per-expert norm and weighted
sum calls, and all activation/weight precision boundaries. No expert is skipped or cached.

Acceptance: all five official tests and scored-shape checks pass, then paired same-input
comparison against M64/current baseline on held-out routing inputs confirms weighted
three-load score gain beyond null variation. Cross-run scores alone cannot resolve small
differences. Stop fixed-tile promotion if no gain survives confirmation; record losses in
docs/ruled-out.md. A heterogeneous per-shape policy needs its own measured justification.
