# P0 execution directive: OK Driver and OKC Developer focus on decode32

November 2, 2026. Initial evidence `b9ca9dff1a8f7bd68ccfb8ffe155c2746f6038db`;
publication reconciled with concurrent captain commit
`5b4a729a2ed329e1de268af6935dff7d8c52bbe7`.
User follow-up: "are they doing what you just said or are they doing their own thing",
then **"make it priority for them please"**.

## Decision and actual pickup

Decode batch 32 is the primary next performance assignment for both existing owners,
not optional background research. This directive is installed at the top of `AGENTS.md`
and `CLAUDE.md` and linked in the steering index. It preserves the captain's newer
[fixed MID-at-T32 assignment](2026-11-02-driver-d32-mid-comparison.md), rather than
replacing it with a duplicate allocation.

While this update was being prepared, OK Driver merged the human priority and published
an explicit queue reconciliation in `5b4a729a`. That is evidence of captain pickup, not
an inference from this assistant's own documentation. The earlier attempted publication
was not forced over the concurrent changes; these instructions are based on the newer tree.

| Owner / event | Verified state at publication | Remaining acknowledgment |
|---|---|---|
| OK Driver accepts D32-first priority | Confirmed by captain's D32 comparison report and merge `5b4a729a` | Keep the next allocation and result D32-first |
| Broad dense-bank GENERIC proposal | Captain marks it unallocated reserve; never delivered/started | No unrelated implementation before the current D32 assignment |
| OK Driver delivers new MID-at-T32 assignment | Confirmed: app message to existing developer after5b4a729a remote verification | Recorded in captain coordination JSON and dated addendum below |
| OKC Developer picks up the new MID-at-T32 assignment | Confirmed: explicit23:43UTC IMPLEMENTATION pickup, firsttools/d32_mid_family.py |30active-minute fixed MID17..32 scope, no shader/math/packing changes; target pending |

This chat has no exposed direct-message action to those running sessions. It has not
sent an interrupt, dispatched a GPU job or obtained a developer acknowledgment. A
repository commit must not be represented as delivery or pickup. Acknowledgment is a
coordination record, not a new technical review loop or reason to delay authorized work.

## Concrete next work already selected by the captain

The direct-global-weight test `0b684a4065354fc5` is now terminal: official correctness
passed, but D32 sliding/full was180.848/337.856us, weighted13.724704ms and1.61354x in
its own run. The captain parked that exact lane and its held confirmation package; the
developer acknowledged those stops. These are not matched causal deltas against public607.
Do not resume polling or duplicate the completed job because the older root handoff names
its accepted ID. Reconcile the owning terminal and use the current captain plan.

**OK Driver:** complete the one allocated unchanged current-service control, then the
one fixed MID-family D32 exploration when runnable and conforming. Remain sole dispatcher.
Lead every result with D32 sliding/full, `C32 = 46 * sliding_us + 16 * full_us`, own baseline,
selected path and all-six-load regression summary. Deliver the new assignment through the
existing developer channel and record actual acknowledgment.

**OKC Developer:** implement the existing MID helper for metadata17<=M<=32, preserving
M33 and all other standard-load choices. Follow the captain's exact module-build/cache,
startup56-replay, neighborhood, scale/fallback and cooperative/progress requirements. This
is the next assigned D32 implementation, not a new shader/tile/ring/packing sweep. Full
batch32 single-query attention remains the next separate major target if needed. Do not
replace this with unrelated T1/T8 work or T128/256-only PR compression.

No new worker/dispatcher or public submission authority is created here. Preserve current
correctness, external-code policy and the accepted allocations. Do not cancel or mutate
live jobs; check the latest own-job ledger before acting. New prospective shape-specific
wins are allowed, but do not retroactively relabel a failed broad experiment as successful.

## Objective and decision rule

Target the user's actual row: **decode batch32,22.149ms baseline,11.888ms ours,1.8632x**.
This is T32/q_len1, not verify32/T256. First engineering milestone: reproducible **2.0x**,
then **2.1-2.2x**. Targets are not predictions. At the frozen reference,2x requires about
0.814ms removed from this weighted load. [Calculation and profile evidence](2026-11-02-decode32-priority.md).

Keep the other five load winners unchanged unless separately improved. The captain's new
D32 gate supersedes an obsolete requirement that every change win at128/256 or add3% to
the whole layer. First ordinary checks establish conformance and direction; promising
D32 changes need independent held-out controls before a causal gain or promotion claim.
A cheap reproducible smaller gain can contribute toward2x. Retain all required numerical,
publication, lifetime and progress properties. Correctness/admission failure or no useful
D32 signal parks the exact MID implementation under its existing stop, without a sweep.

## Publication scope and sources

Only instructions and this handoff/index change. Current captain reports, raw receipts,
run logs and scoped closures are preserved. No new performance measurement is claimed.

- [Captain's acknowledged priority, MID allocation and exact acceptance rules](2026-11-02-driver-d32-mid-comparison.md).
- [Direct-weight terminal and broad-proposal supersession](2026-11-02-driver-direct-weight-terminal-and-dense-backing.md).
- [Original user-selected D32 priority and budgets](2026-11-02-decode32-priority.md).
- [Layer task and scoring formula](../../muse_decode_layer/TASK.md).

Append actual developer delivery/pickup evidence when it exists; do not fill it by assumption.


## Captain delivery and developer acknowledgment addendum

Captain directly delivered the fixed D32 MID-family task through the app to existing OKC Developer`01a19976-9e99-79c0-bab1-644bd8c0293a` after5b4a729a publication and remote verification. The developer explicitly replied “IMPLEMENTATION pickup5b4a729a at23:43UTC” and named firstowned`tools/d32_mid_family.py`,30active minutes, actual final MID limit/build representatives/four-key guard, unchanged native math/packing/scratch and all56startup coverage. This is direct delivery and explicit acknowledgment, not an inference from a commit. [Captain coordination evidence](evidence/2026-11-02-driver-d32-mid-comparison/coordination.json).

The one unchanged607 current-service calibration is accepted as own **94ef698e91964dfa** (04314b16/869198LF; exact ba9 prefix plus unused metadata). Fresh preflight7private/no-live leaves6before reset. Captain owns polling/archive and will lead with D32 cells/46:16 plus all-six results. This is a measurement dependency, not an unrelated optimization or public submission. MID source/CPU preparation continues concurrently; its target qualification/performance remain unmeasured. No additional worker, dispatcher or broad dense allocation was started.
