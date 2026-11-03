# 2026-10-30: Layer bench arm bias, a four-arm harness, and the next comparisons

Evidence: the bench logs in `muse_decode_layer/profiles/` (50c4c1328ab742f4, 720079090ee44cc3,
aa4276e91c924dad, fdf75ba5a8124544, 29fd004fd58e40ef, and f0ee07c4d4904511, committed with this
report), and `tools/dl_make_bench_pub.py` before and after this change. No GPU run was used.

## Summary

The three-arm layer bench timed the candidate twice (cand and null) and the other file once.
The null shared the candidate's state and compiled module, so it measured only the
candidate's timing noise. An effect specific to the incumbent arm showed up in both
`cand - inc` and `null - inc` and looked like a gain for the candidate. The harness now times
both files twice. It prepares every copy on its own, in a random order, and captures and
replays all four arms in random order. It prints each arm's policies side by side and a
carry-over table. The decision pools the four cross ratios and must clear twice the larger
null. Three bench files are ready: an A/A calibration (release against release), mode 51
against the release, and the M1 runtime build against the release.

## What the logs show (measured)

The v16 argument does not show a bias by itself. Both 29fd004fd58e40ef and
f0ee07c4d4904511 carry mode 51, which changes only the 128- and 256-token MoE (v16, v32). The
null is the candidate file again, so `null - inc` equal to `cand - inc` is what a real gain
produces too. Equal values show that the old design cannot tell a gain from an inc-arm
penalty. They show nothing about which of the two happened.

The evidence of an inc-arm penalty comes from shapes where both arms run identical code:

| run | identical-code shapes | cand - inc us (null - inc) | inc / cand per load |
|---|---|---|---|
| 720079090ee44cc3 (arms differ only in the MoE pick at 32+ tokens) | d1 | -4.1 / -5.7 (-4.5 / -5.4) | d1 1.050 (from the ABTIME medians) |
| 50c4c1328ab742f4 (no d1 code path differs) | d1 | -5.2 / -5.0 (-4.3 / -4.9) | d1 1.065 |
| 29fd004fd58e40ef (mode 51 vs release) | d1, v1, d32 (same MoE and front picks) | d1 -0.3 / -1.7, v1 -0.6 / -0.2, d32 -1.2 / -1.1 | d1 1.0099, v1 1.0023, d32 1.0078 |
| fdf75ba5a8124544 (M1 build vs release) | none (M1's SK_PDL build changes every T <= 32 launch) | | |
| f0ee07c4d4904511 (M1 runtime build vs release) | none | | |

In 29fd004fd58e40ef the three identical-code loads average inc / cand 1.0066, against an
overall 1.0084. If that offset is an arm bias that applies to every load, mode 51's overall
gain is about +0.2% (v16 about +1.1%, v32 about +0.7%). This is an estimate under that
assumption, not a measurement. f0ee07c4d4904511 has no identical-code load, so its +0.87%
cannot be split this way.

The per-round samples were never logged (ABTIME printed medians only), so these logs cannot
test the mechanisms below. The new harness logs per-arm deviations grouped by predecessor and
by position (ABPRED, ABPOS, ABTWIN).

## Asymmetries in the old harness (from the code)

1. **Twin carry-over.** `null` was `(kernel_fn, state)` of the candidate: the same module and
   the same state object, so the same repacked MoE bank, scratch, side stream and compiled
   kernels at the same addresses. inc had no copy. In a random order of three arms, cand and
   null each follow their twin in about a third of the replays. Anything that survives the
   256 MB memset between replays and helps a same-module predecessor (per-SM instruction
   caches, TLB entries for the same pages) could speed up only cand and null. Hypothesis; the
   size is unknown.
2. **Prepare order.** cand's prepare always ran first, then inc's. cand got the first
   allocations and ran its tuners first. The tuner picks differed between the arms in
   29fd004fd58e40ef and fdf75ba5a8124544 (the front at 16 tokens, skf against skf-r). With
   the null on cand's state, that policy noise was invisible.
3. **Capture order.** Graphs were captured in dict order (inc, cand, null) for every input,
   so inc's private graph pool always took memory first after the previous shape's
   `empty_cache`.
4. **Module identity.** inc is exec'd from an lzma-packed, comment-stripped string, with
   `okc_inc_` extension names, built on cand's build threads. Device code is identical by
   construction of the strip. The d1 penalty was 4-6 us in the two runs whose candidate was
   sent unstripped (50c4c1328ab742f4, 720079090ee44cc3) and 0-2.5 us in the three with
   `--cand-strip=1`. aa4276e91c924dad, also unstripped, showed d1 +0.2 / -1.3. This is a
   correlation across five runs that differ in other ways too. I found no mechanism in the
   code.
5. **Shared inputs.** The dense weights are shared by every arm: `weights[name].contiguous()`
   returns the judge's tensor itself. The per-arm input copies and MoE banks are separate.
   This is symmetric and not a suspect.

The A/A file below tests 4 directly: the same release file runs as the top-level module and
as the exec'd module. Items 1-3 are now symmetric or randomized.

## The fix: `tools/dl_make_bench_pub.py`

- Four arms, A, A2, B and B2: with `--other=inc`, A = inc and B = cand. The copies reuse
  their file's module and extensions, so they cost no bytes.
- `--dup=own` (the default for inc): every copy gets its own prepare(), with its own
  allocations and tuner picks, so each null includes run-to-run policy noise. `--dup=shared`
  restores the old sharing (the default for pub, whose prepare() has never run twice in
  one process).
- The prepare calls run in a seeded random order (`--prep-seed`, default `--order-seed`).
  Graphs are captured in a seeded random order per input, and replays are random per round
  as before.
- ABPOL prints each arm's picks side by side: MoE variant and W4 configs per token count,
  front, attention boundary, sk projection, wide picks, M1 policy and stage flags. Each line
  says "same", or DIFF with every arm's value.
- ABTIME prints the four cross paired differences and both nulls. ABLOAD and ABSCORE print
  the four cross ratios, their geometric mean (pooled) and both nulls (A2/A, B2/B).
- ABDECIDE says "B faster" or "B slower" only when all four cross ratios fall on one side of
  1 and |log pooled| is above twice the larger |log null|. Otherwise it says "not resolved".
  Pooling cancels anything the two copies of one file share. The nulls bound what differs
  between two copies of one file.
- ABPRED, ABPOS and ABTWIN: per arm, the median deviation from its per-input median, grouped
  by the predecessor arm and by position in the round, and each arm after its own copy minus
  after the others. These measure carry-over directly.
- The tail is reordered so that the decision, scores, loads, timings, policies and carry-over
  lines come last. ABCAND moves earlier (`--cand-report=0` drops it). ABEND records the
  seconds into prepare() at which the bench finished.

Validation (CPU, no GPU): a smoke test generated a bench from two fake layers and executed
the generated body with fake CUDA timing. It injected a -2 us carry-over for the candidate
module's copies only. ABTWIN reported -1.8% for cand and cand2 and 0.00% for inc and inc2.
ABPOL showed the injected MoE pick difference as DIFF. `--dup=shared` reproduced the old
sharing. All three real bench files parse.

## Prepared comparisons (not run)

All three: loads d1, d16, d32, v1, v16 and v32, 8 rounds, two inputs per shape, `--dup=own`,
inc = the release (`muse_decode_layer/submission.py`, SHA256 b9c23bda..., sent as 84b4cc1e...),
candidate comment-stripped (`--cand-strip=1`). The files are in the coordinator session's
scratchpad, `.../scratchpad/dl5/`.

| file | candidate | seeds | bytes | SHA256 |
|---|---|---|---|---|
| bench4_aa.py | the release itself (A/A calibration) | 180001, order 43 | 858,396 | 9473029f... |
| bench4_g51.py | `muse_decode_layer/cand/layer_g51.py` (dacd4e0d..., mode 51 + decided gate) | 160001, order 37 | 879,146 | e95f3218... |
| bench4_m1rt.py | the f0ee07c4d4904511 candidate, M1 runtime measurement build (d57b5e38...) | 170001, order 41 | 892,279 | 824e1e45... |

The time budget is an estimate. The three-arm benches used about 78 s of prepare()
(29fd004fd58e40ef, fdf75ba5a8124544), with two prepares. Two more prepares and a fourth arm
should land well under the 255 s deadline. ABEND will show the real figure, and ABSTOP
reports any shape the deadline skipped.

## Order, reading, and stopping criteria

1. Run the A/A file first. The harness is symmetric if ABDECIDE says "not resolved", the
   pooled ratio is within about 0.3% of 1, and ABTWIN and ABPRED show no predecessor effect
   above about 0.3%. If the pooled A/A ratio sits outside twice its nulls, a module-identity
   effect (item 4) remains. In that case read (a) and (b) only through their difference from
   the A/A offset, and look for the mechanism next (for example, also exec the candidate
   from a packed string).
2. Then run (a) and (b). Use ABDECIDE and the per-load pooled ratios. The old metric's
   +0.84% (mode 51) and +0.87% (M1 runtime) stay unestablished until these runs land.
3. If ABPOL shows DIFF between the two copies of one file, that load's null includes policy
   noise. Read the gain at that load as within noise unless it clears the null.

## Gaps

- I found no code-level cause for the module-identity correlation (item 4); the A/A run
  tests whether it exists.
- `--dup=own` has never run on the GPU. If a second prepare() of a layer file fails, the
  bench prints ABFAIL and skips the timing. `--dup=shared` is the fallback.
- The decision rule (twice the larger null) is a heuristic. With one null per side, it is
  not a significance test.
