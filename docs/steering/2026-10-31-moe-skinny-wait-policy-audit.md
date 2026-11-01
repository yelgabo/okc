# Skinny full-slot wait-policy audit (offline only)

Evidence pin `e163bbc`; production source unchanged from the known-good owned baseline.
Root requested this audit after the fresh weight-reuse screen failed6/8 cases. No
kernel edit, fence change, compile experiment or GPU dispatch accompanies this report.
Root owns shared-index publication and any subsequent check authorization.

## Finding

A nonblocking full-slot wait is a distinct, source-supported hypothesis. The tracked
MoE logs, ruled-out entries, tools/experiments and relevant git-history search did not
find an actual test_wait-versus-try_wait A/B. This scoped search is not proof that no
unlogged scratch experiment ever existed. Commit3f67a45 deliberately retained try_wait
when repairing wait accounting; it did not test another wait policy. The projection
no-suspend-hint result ba85985b8c5e4969 is a different problem/kernel and does not close
the MoE question.

The recorded waits are nonzero, so a single bounded policy check is defensible. They
do NOT separate unavoidable source-data latency from wakeup/polling overhead. Neither
sparse expert jobs nor loads-only≈computing proves HBM saturation. No measured speedup
or ability to close the17% top10 gap is established.

## Exact primary-source ordering argument

[PTX ISA8.7, section9.7.13.15.16: mbarrier.test_wait/mbarrier.try_wait](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
specifies the same completion test, phase-validity rules and acquire pattern on a
True result for both instructions. The omitted memory-semantic qualifier defaults
to acquire, and omitted synchronization scope defaults to CTA. The barrier resides
in shared::cta independently of synchronization scope. Both parity forms accept0/1
for the current incomplete or immediately preceding completed phase.

The production global helper currently uses:

```
mbarrier.try_wait.parity.shared::cta.b64
```

The proposed full-slot-only helper would use:

```
mbarrier.test_wait.parity.acquire.cta.shared::cta.b64
```

Keep the same address, parity, output predicate, volatile assembly and memory clobber.
No code/scales load may occur until the result is True. All original barrier init,
arrive/expect_tx, consumer rendezvous, slot-retirement proxy fences and refills remain.
The existing unit barrier prevents ring reuse from advancing two full phases while
a consumer is still observing an older phase. Polling more often changes neither
the phase nor the producer's permission to overwrite that slot.

The PTX section also states that try_wait may suspend, while test_wait is nonblocking.
A try_wait suspendTimeHint is an optional nanosecond limit hint, not a guaranteed
latency or timeout. A20ns hint is an alternative later experiment if independently
justified; do not combine it with the first test_wait comparison.

## Call-site scope and what must remain untouched

Line positions are at the audited submission.py and may move as comments change.

| region | current call sites | disposition |
|---|---|---|
|global mbar_try helper|about157–165|Do not edit; fat and skinny both use it|
|fat mbar_wait_to/cyc and producers|about1272–1291 and callers|Unchanged|
|skinny hand_wait|about3917/3919|Unchanged; partial-buffer handoff variants|
|skinny compute_loop full[slot]|about4049/4051|Only eligible sites: initial test and retry|
|skinny compute_loop empty[slot]|about4088/4090|Unchanged; kEarly refill policy|
|skinny done_a acquisition|about4019–4028|Unchanged; global data publication|
|skinny final grid barrier|about4470–4483|Unchanged|

Use a namespace-local full-slot helper in `_SKINNY_BLOCK`; never globally replace
mbar_try or every try_wait token. Restrict the candidate to the selected sk8w3s
configuration (eight compute warps, three slots, kLow0). A separate variant flag may
identify the candidate without changing MAIN_MODES, MAIN_CANDIDATES or the existing
official policy. Baseline must compile through the original helper. Any later
production source needs its own exact-file correctness/fresh gate.

## Existing full-wait evidence and limits

Older counters skipped the first potentially blocking try_wait. The corrected
receipts below start timing before that instruction. Values are mean thread0 full
wait regions per CTA, not sums over the grid or attainable wall-clock savings.

| receipt / instrumentation | slots | tokens / routing | A wait us | B wait us | total region us |
|---|---:|---|---:|---:|---:|
|e937013292b24906, ABSK|4|1 layer /std|2.2 /1.9|0.3 /0.2|2.5 /2.1|
|e937013292b24906, ABSK|4|8 layer /std|3.0 /3.9|0.7 /1.2|3.7 /5.1|
|e937013292b24906, ABSK|4|16 layer /std|3.7 /4.4|1.1 /1.5|4.8 /5.9|
|75bca9775f6f4b0b, ABUNIT|3|8 layer /std|5.92 /8.69|2.12 /3.51|8.04 /12.20|
|1fbb9fbf524f4ed7, ABSK|3|32 layer /std|5.8 /7.1|2.2 /2.7|8.0 /9.8|

The four-slot rows are historical, not measurements of today's selected three-slot
kernel. ABUNIT's per-unit clock/global stores perturb pacing more than aggregate
ABSK, so the two instruments must not be silently substituted. No retained matching
three-slot ABSK row for1/16 was located in the scoped profile search. These region
totals are at most optimistic descriptive headroom if all time inside them vanished
without perturbing anything else; that premise is false for data-arrival time, and
the regions overlap other CTAs and helper work. They are not a rigorous critical-path
ceiling. The recoverable wakeup/poll fraction remains unmeasured.

The same75bca receipt at8 layer measured three-slot issue-to-wait-end3.23us, copy
lead3.00us, and0.19us slot-free-to-refill delay. Thus memory availability and pacing
are substantial parts of the wait. A different wait opcode cannot remove those
bytes. Prior early-refill variants shortened a wait region but lengthened the hold
and lost total time; whole-call paired timing is mandatory.

## Resources, timeout and progress risks

The candidate adds no shared buffer, allocation, global metadata or additional
barrier. That does not prove identical compiled resources: record actual registers,
local-memory bytes, dynamic shared memory and resident blocks/SM for baseline and
candidate before timing. Require equal shared-memory layout and occupancy, no new
local memory, and inspect fence/barrier/arrive counts. If those differ materially,
stop and classify the confound before making a wait-policy attribution. All fat
instance instruction text must remain unchanged.

Eight compute warps busy-polling may consume issue bandwidth and delay the helper
warp, which publishes credits and frees partial buffers. Removing suspension is
therefore not automatically faster or fairer. The wait graph adds no new logical
dependency, but performance under scheduling contention must be measured.

Retain the current first-failed-test branch, retry structure, clock64 checks,
kSpinLimit=2^31 cycles and ctr[8] failure flag. The clock starts after the first False
result as before. A nonblocking first False may start that guard earlier than the
potentially suspended original; do not weaken the guard to compensate. A timeout
must fail the run/gate, never count as a successful latency. No sleep, watchdog-vote
removal, reduced polling-clock frequency or relaxed acquire belongs in this test.

## Prospective single-hypothesis four-arm gate (requires root approval)

- Untouched sk8w3s control and independent-scratch duplicate; full-slot test_wait
  candidate and independent-scratch duplicate. One extension build, shared read-only
  weights, no shared mutable scratch or data-derived cache.
-1/8/16/32 tokens, fixed layer-router8 and standalone routing, fresh held-out321/322.
  Four arms checked against owned FP32 reference both eagerly and after queued graph
  replays. Preserve original numerical thresholds, finite checks and zero spin flags.
- Equal resource/ordering gates before timing;20 randomized cold-L2 paired rounds.
  Original official judged kernel/policy remains the control.280s total prepare
  guard fails incomplete coverage, with all case summaries reprinted at the tail.
- Compute the standalone six-load score contribution using the four measured small
  loads and two unchanged large loads. Pre-register at least1.0% aggregate score
  contribution and greater than2x the larger identical-control log-noise magnitude,
  in BOTH candidate duplicates. No individual small-token regression exceeding
  the greater of0.25us or2x its duplicate-null magnitude. Layer timing is a separate
  consistency check, not a substitute for standalone scoring.
- Stop on incorrect output, nonfinite value, spin timeout, resource confound,
  incomplete coverage, or failure of the registered score gate. A localized faster
  shape does not authorize post-hoc whole-policy promotion. If the gate passes,
  require fresh replay/official exact-candidate validation before a public run.

This is a proposed measurement, not dispatched work or a top10 forecast. Source
supports semantic equivalence on successful waits; available counters do not prove
a material recoverable wakeup cost. No kernel changes are included in this audit.

## Dated addendum: November1 PDT, source prepared off production

Root subsequently approved preparing the one-hypothesis harness, with one check
authorized only after independent read-review by today_proj. Production submission.py
is untouched. `tools/moe_skinny_testwait_check.py` generates a scratch source with flags
0/256/128/384 (baseline, unused-bit independent neutral, test, independent testduplicate).
All original kMode predicates mask only low15 and bits16/32/64; new bit128 controls
only full_slot_ready. Bit256 has no predicate. Two call sites change, all other skinny
code and the entire fat device source are preserved. New modes are excluded from the
original production tuner candidate list. Host export adds resource queries only.

`tools/test_moe_skinny_testwait_source.py` passes: removing the new host resource
function/declaration and local helper, and restoring the two full-slot calls, restores
the complete original CUDA/CPP/skinny strings exactly. No empty/hand/global/fat wait,
watchdog, proxy fence or barrier changes. Generated Python syntax passes.

The harness requires exact resource-vector equivalence before timing,64 eager and64
postgraph mathematical checks across16 heldout cases,20 cold randomized rounds,
the registered1% whole-six-load contribution/2xnull and per-case regression gates.
Additional fully counted stamped2/130 twins run only AFTER paired timing, under a
265s optional-stamp guard; incomplete stamps mean no new attribution claim. Core
coverage remains required under280s. Final compact summary asserts under10KB so all
case rows, resource vectors, math tally, stamp totals and decision survive the16KB
receipt tail. No short hint, ring-depth, prefetch, precision or watchdog sweep.
Independent read-review is pending; no GPU job yet at this checkpoint.

November1 execution addendum: today_proj independently reviewed776b859 and verified
generated SHA25658bce614db8b5baefa5d5c264a87da51515272bd11d1dd577beb146d2eabe99b,
source test, successful-wait semantics, full-only scope, watchdog and helper progress,
private states and resource gate. No blocker found for one diagnostic. Root standing
approval then dispatched check87f22aa71e614460 on muse_moe_w4a16. TWSCORE is explicitly
a fixedsk8/3 contribution with two large loads assumedunchanged, not the actual
official policy score. The timeoutbreak has the baseline's preexisting behavior and
must reject through ctr8; no successful ordering claim applies on a timeout. Pending.

November1 harness correction:87f22aa71e614460 failed a first-case host formatting
TypeError because the round loop variable shadowed built-inround(). No complete
timing result or official score. Root authorized host-only round_idx correction and
one corrected check252df4a54d764ca5. CUDA/CPP/skinny/variant constants remain exactly
equal between frozenoriginal andv2. The CPU scaffold now executes the actual timing
loop/formatter, scoregate and compactsummary using no-op CUDA scaffolding, tests both
winning and losing synthetic decisions, and reproduces the original failure as a
negativecontrol. Three tests pass. No threshold, kernel or hypothesis change; pending.
