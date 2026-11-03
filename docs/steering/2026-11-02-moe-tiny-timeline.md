# Tiny and Tiny2 startup, dependency and tail timeline

November2,2026 Pacific. MOE-TINY-TIMELINE-01 implements the short-load assignment in the final-day directive. No new GPU job yet. Public5df257af remains5788.720720us, with user target5559.8us. Prepared-gamma B4/MAIN source is parked rather than consuming this allocation.

## Selected source and question

The final factory selects helperb8ed8543 for T1 and T8, NW8/WS1/WGS0/NS3/WSB1/FAST1=2. Tiny retains two phase-B epilogues and Tiny2 retains one. Their helper streams phase-B weights through the shared ring across the phase boundary already. The missing evidence is whether activation-ready waits, ring waits, compute/epilogue work, inter-CTA completion spread or final output work dominate the exposed time. It would be wrong to implement generic cross-phase prefetch without checking the existing implementation.

This observer adds timestamps to that selected WS helper, not the closed WGS-only MOE-PHASE-01 source. It keeps all math, unit assignments, barriers, proxy fences and retirement counts. Each CTA has a private1184word record with192six-event unit slots and32header words. Grid132 and at most32local jobs bound the combined A/B units to188per CTA. No global queue or loop is added. The four owners each receive an independently guarded fourth scratch allocation; original scratch addresses/offsets stay unchanged. Timestamp fields have a single writer and no reader on the device.

Capture entry, helper range publication, A start/end, B start/end, grid-barrier arrival/release and final completion. For every compute unit capture ring wait entry/exit and retirement; for each new phase-B job capture the activation wait; capture helper epilogue retirements except retained tail units. Validation checks every CTA and every assigned unit before compressing the report. All CTA boundary/wait/compute summaries remain, with full unit records for the last two CTAs. Retained timestamp coordinates round down to10ns. The instrument changes allocation and execution cost, so its time cannot establish a performance gain or a precise unobserved critical path.

## Bounded check and stop

One private check. Four independent owners, two controls and two observers. One fixed normal bank1102031 covers all six scored loads with fresh input updates, empty/occupied transitions, captured-output checks, guards and restoration. This is42observations/24captures/768cold timed replays/6restores. One additional cold replay each at T1/T8 yields the validated timeline from owner2 against both control outputs. Other four native paths remain unchanged. All seven actual native compiler requests must match the frozen manifest. The official benchmark arm is the unchanged control0, not the observer.

Observation-cost screen requires the median observer/control difference inside max(2us,10percent of control), with duplicate-owner timing difference bounded by max(0.5us,2percent of control), separately on all six loads. This is an intrusion bound, not a promotion gate. A cost failure preserves evidence but forbids using it to select a correction. One bank and normal routing cannot establish general performance. Preparation remains295seconds. Any source, coverage, ordering, graph, numerical, guard, budget or serialization failure preserves failed status and bounded cumulative partial rows/traces. Full report must fit12000base85characters after zlib compression; detail compaction makes the report fail. No larger guard, seed sweep, old observer repair or blind repeat.

On valid evidence, implement one concrete correction for the largest exposed T1/T8 bubble, then measure against independent duplicate incumbent controls. Final-day acceptance is at least2us saved on T1 or T8 with other four loads inside noise, followed by confirmation and concrete release qualification. Preserve all six scored loads and the exact BF16 mathematical contract. No public permission beyond already submitted5df257af is inferred. General owns independent row-owned evidence; its existing failed source is not rerun by this worker.

## Qualification

Implementation and CPU/compile checks are in progress. CPU CUDA facades test orchestration, not native math or H200 speed. No independent review was available. Actual target outcome, matched phase evidence and a next kernel correction remain pending.

Six CPU integration/source/trace tests pass, including all six host-facaded loads, exact output forwarding,768synthetic timed replays, wrong-output rejection, cleanup and post-cleanup/final-budget failure retention. Target times and native arithmetic remain untested. Both local CUDA12.8.1 SM90a Tiny/Tiny2 control and observer builds pass. They use145/147registers respectively,3barriers and0spills locally; no target-build parity is claimed. Frozen paired source106a05c54e2ffd8b0f851b142e5d8ef3f74a47e92112d09d145c2017d637ee81/492113bytes, observer helper8324f07988fc396f0fab4137d06ce1823eb2fb8ff59caa4df794fc907f55512e, runtimec80be1d4a327041969c25c92651af8222d5eabbcb8c34a516e70710c0ddc4acd. Source/test/build evidence is in evidence/2026-11-02-moe-tiny-timeline. General has separately picked up validation support; these are owner integration tests, not independent review. Latest shared list has four queued unrelated runs. No timeline job has been posted.

Unit detail uses delta-coded ring-entry, ring-wait, compute, helper-lag, dependency-offset and dependency-wait coordinates, all at10ns. General support owns independent validation; native integration and the only timeline dispatch remain here. New user approval recorded in20a6cfa9 delegates a separate factor-Y clean release to the final-day lead. That release task does not repeat or change this worker's matched treatment decision. No duplicate release action here.

Timestamp interpretation: A/B end fields mark compute-function return, before the helper join. They do not prove global activation publication. Helper unit retirement precedes its final fence; existing doneA publication and the original grid barrier remain authoritative. Barrier entry is sampled before the original thread fence/counter, and release is sampled after the original CTA sync. The raw and rounded observer coordinates cannot establish invisible store completion or an unobserved device causal dependency.

## Dispatch registration

Exact source106a05c5 is remotely verified in05e3e875. Shared public repeats are refilling the queue; an empty global queue is not an MoE dispatch requirement in the current owner directive. Register one distinct private observer behind those accepted jobs after fresh shared-list/quota inspection, preserving all other owners. This changes the earlier queue-empty scheduling assumption, not the frozen source, finite stop or measurement gates. No accepted own ID yet; dispatch-started is persisted before the single POST.

While queued, tools/moe_short_gate.py prepares the next kernel comparison gate only. It requires complete36rows, the same T1 or T8 cell saving at least2us across every bank/distribution and all four owner crosses, resolved cell/whole-six benefit above duplicate noise, and other four cells inside their noise envelope. It does not inherit the large-load affected mask or a fixed1percent whole-six threshold that would silently exclude a2us T8 win. Two CPU tests pass for T1/T8 wins and injected noise, other-load drift, affected regression, one-owner/one-bank failure and incomplete rows. This is arithmetic infrastructure, not a kernel candidate or GPU gain.

Accepted privatef4012a0531af4013 for exact106a05c5/492113bytes. One POST succeeded; source/control provenance is frozen. Persisted accepted ID before any result collection or potentially overlapping measurement. Its sole foreground waiter writes ignored .build/moe-tiny-timeline/status-wait.raw.json.

## Terminal evidence and decision

Own f4012a0531af4013 is complete, PASS5tests/6benchmarks/6profiles, no flags, return0. Sole waiter95489 ended0 before archival. The strict owning reader accepts the frozen source/runtime, all seven actual compiler requests, four independent owners,42observations/24captures/768cold replays/6restores, two validated traces and cleanup. Prepare248.575311seconds. Worst paired global/row error0.000177868060. The complete capsule is8634base85characters, inside the12000bound, with no detail compaction. All six observer-cost gates pass. This is usable bounded observer evidence, not an optimization win.

The officially judged unchanged control0 reports5726.051478us/2.348651606x, CI[2.313421832,2.383881380]. Its lower cross-job point is disclosed and promotion-ledgered as an already-public base-family control. It does not qualify the observer for public use or establish a source gain. Official six-load adjusted medians are below, separate from private cold replay times and instrumented spans.

| Official load | Adjusted median us |
|---|---:|
|decode1|25.204540|
|decode16|93.498825|
|decode32|125.162488|
|verify1|63.954341|
|verify16|179.781396|
|verify32|214.815274|

The observed bank1102031 normal census is2local experts/2local contributions atT1, and10local experts/13contributions atT8. The extra cold trace spans29.024/85.248us respectively. Compute-A leader returns reach16.448/55.168us; compute-B leader returns27.008/82.080us. Grid-barrier release is28.416/84.512us; final tail0.608/0.736us. Coordinates are instrumented and quantized, not exact unobserved baseline phase costs.

| Per-CTA interval or total | T1 median us | T1 maximum us | T8 median us | T8 maximum us |
|---|---:|---:|---:|---:|
|activation-ready waits|4.600|5.950|2.430|4.380|
|ring waits summed over units|2.680|4.190|11.855|13.240|
|compute/partial handoff summed over units|12.060|13.280|45.760|47.680|
|post-compute helper join|0.280|0.410|0.350|2.110|
|grid barrier|1.215|2.430|3.010|4.320|

These are separate per-CTA distributions, not additive exposed-time estimates. For retained T1 CTA2, the last A compute-unit retirement is16.22us, helper retirement20.51us and the first B activation wait16.76–21.18us. CTA3 has16.22/20.67us and the same4.42us wait. This locates a real dependency wait across helper completion/publication. It does not separate helper math, global-store drain, atomic contention or scheduling. No numerical-tie or exact cause claim follows.

Next concrete experiment is the distributed A-ready publication design in2026-11-03-moe-short-ready-flags.md. It replaces contended per-job atomic credit with per-CTA flags and a warp-cooperative acquire, retaining every writer fence, sync and math boundary. This tests one possible cause of the measured wait. Final-barrier-only cleanup has less direct upside than the observed activation wait. No repeat of this completed observer is needed or authorized by its finite stop.

The independent General validator db7bd075 accepts both actual exported cases, all132CTA summaries and the two retained unit streams at each load. Its recomputed observer-cost rows equal the frozen owner report. This adds an independent exported-data/schema check, not a second native build, a causality proof or a target-gain claim. See independent-export-audit.json in the evidence folder and the support report for marker limitations.
