# M1's 1x1 sliding gate failure in d6c30f88c617438b, the route row control, and the after-reset checks

Written 2026-10-30 by the layer agent. Evidence: scored run d6c30f88c617438b (file
`muse_decode_layer/cand/layer_m1rt51.py` at 7cdb755, SHA256 65264f31...; receipt
`muse_decode_layer/profiles/layer_m1rt51.d6c30f88c617438b.log`), the code of that file, and the
earlier runs 6ce5be0ba29642ea and 60cf58b69b75491e. No GPU was used for this note. Statements
are labelled measured (from a run), code (read from the file) or hypothesis.

## Summary

- Measured: 2.013x (1.983-2.043), 13/13, prepare 154.8 s. M1's enforced gate failed at its first
  shape, 1x1 sliding, so M1 was off for the whole run. Mode 51 ran at 32 / 128 / 256 and every
  MoE gate passed.
- Code: at 1x1 the M1-on graph and the M1-off graph contain the same launches. M1 changes
  only the o projection when its pick is an sk kernel and the front when it is skf or skf-r.
  At one token, the o pick is cgemv and the front is fp32, so `_m1_mode` is never consulted
  and nothing launches as an M1 PDL secondary. The 1-in-2000 difference therefore cannot be
  an M1 ordering defect.
- Measured plus code: the difference has the signature of FP32 merge-order rounding, which
  the 1x1 path has in two places (attention's split merge and the fused MoE's FP32 atomics). It
  does not look like a stale read. The 6ce5be0ba29642ea check failed the same shape the other
  way round: there the M1-off control differed and the M1-on arm was bitwise.
- Hypothesis (strong, to be confirmed by the reproduction below): the gate compared two
  samples of one rare noise process. Its envelope criterion (the candidate's largest relative
  error at most max(1e-4, 2x the control's largest)) fails whenever the control happens to
  draw no event and the candidate draws one above 1e-4.
- Route row control: a MoE output row is exactly zero for a token with no local route, and the
  row control copied row 16 of one input's reference over row 16 of another. At 32 tokens both
  were zero rows, so the injection changed nothing. Fixed in `tools/dl_gate_patch.py`; the
  offline test reproduces the miss with the old code and shows none with the new.
- Prepared for after the reset: the reproduction build and a four-arm bench of release +
  mode 51 (no M1) against the release.

## 1. M1 at 1x1 sliding

### What the run measured

From the SUMMARY of d6c30f88c617438b:

```
M1 sk pdl tuner: fallback (RuntimeError: [replay] M1 fresh-input gate: 1x1s FAIL cand rel
3.2e-04/1.9e-04 el 2.2e-03/1.3e-03 >1e-4 1/2000 o1 0 | ctrl rel 0.0e+00)
gate negctl m1 1x1s: hidden none:pass(0e+00,0e+00) tile:R(1e-01,9e-01) 1%:R(1e-01,7e-01)
0.1%:R(5e-02,5e-01) row:R(1e+00,2e+00) elem:R(2e-02,3e-01); stream ... elem:R(7e-03,1e-01)
m1 selftest PDL = plain bitwise on 14 launches
```

Of the gate's criteria, only the envelope on the largest relative error failed: the control's
largest was 0, so the limit was 1e-4, and the candidate's was 3.2e-4. The share of calls over
1e-4 (1/2000 = 0.0005, limit 0.01), the element check (2.2e-3 against a limit of 2^-7 = 7.8e-3)
and order-1 (none) all passed.

### Why 1x1 is tested at all

Code: `M1_BUCKETS = (8, 16, 32)`, and `_m1_mode(tokens)` maps every token count up to 8 to
bucket 8, so bucket 8's policy also governs decode 1 (one token). The tuner times bucket 8 at
1x8 only, but the stress runs at every scored T <= 32 shape, including 1x1, because a
policy chosen at 8 tokens would apply to one token too.

### What launches at 1x1 sliding, with M1 on and off

Code (`kernel_fn`, `_proj`, `_moe_front`) and the run's picks:

| step | kernel at T = 1 | launch | M1 involvement |
|---|---|---|---|
| qkv | cgemv (`skp tune m=1: qkv_gate cgemv:14.1`) | plain | none |
| attention | `attn_one_kernel`, plan (32, 1, 13), built with ONE_TRIG (launch_dependents at entry) | plain (`_ONE_PDL = False`) | none: its successor is a plain launch, so the trigger has no effect |
| o | cgemv (`o cgemv:9.9 o8:10.8`) | plain | none: `_proj` asks `_m1_mode` only when the pick has `.sk`; `_skp_gemv_runner` has neither `.sk` nor a `pdl` argument |
| attention boundary | `attn_boundary_kernel` c8x256 | PDL secondary (release chain) | none |
| front | fp32 (`front T=1 ... fp32 25.6/52.4`): `router_kernel` on the side stream | router is a PDL secondary (release chain) | none: `_moe_front` asks `_m1_mode` only for skf and skf-r |
| pre-expert norm | `pre_norm_kernel` | PDL secondary (release chain) | none |
| MoE | sk8w3s fused | plain at T = 1 (no route variant, so `meta["pdl"]` is unset) | none |
| unpooler | cgemv | plain | none |
| FFN boundary | `boundary_kernel` | PDL secondary (release chain) | none |

So the candidate and the control at 1x1 differ only in the policy dict, which nothing on this
path reads. `_m1_selftest` also skips the fused front at m = 1. The secondaries that run here
are the release's own (audited in docs/review-pdl-secondaries-2026-10-30.md, all sound). The
build differs from the release in ONE_TRIG and in the SK_PDL compile of the sk kernels, and
both arms share those.

### Order-dependent arithmetic on the 1x1 path

- Attention, mode 13 with 32 splits: the split merge is FP32 and its order depends on timing.
  The closed history lead (docs/runs-decode-layer.md, "Closed lead", d93286b5110a4410 and
  2b93c6ecf4f641c6) measured production repeats on one input that were not bitwise (4-5
  distinct outputs in 9 calls), while a fixed-order debug build was bitwise under every
  history.
- Fused MoE sk8w3s: FP32 atomic accumulation. In this run its own stress shows the control at
  one token at relative 8e-5 from the eager reference (`moe gate sk8w3s 8+16+1+32+8 ...`).
- cgemv, the boundaries, the norm and the router: not known to have order-dependent
  arithmetic. The reproduction will show whether any of them ever differs first.

### Signature of the difference

- Element error 2.2e-3 in units of the row maximum is about one BF16 ULP of an element at
  0.28 to 0.56 of the row maximum (one ULP is 2^-8 to 2^-7 of the element's binade).
- Relative L2 3.2e-4 over 8192 elements is consistent with a few dozen elements flipped by
  one ULP, which happens when a small FP32 difference upstream crosses BF16 rounding points.
- A stale read looks different. In the same run, injecting another input's reference at 1x1
  gave relative 5e-2 to 1 for a tile, 1% or a row, and even one stale element gave relative
  2e-2 and element error 0.3 (hidden). The observed difference is 60 to 3000 times smaller.
- 6ce5be0ba29642ea, the same shape with an earlier gate: the M1-off control differed (2.7e-3 /
  2.5e-3) and the M1-on arm was bitwise over 2000 replays. 60cf58b69b75491e (stage
  diagnostic, M1 off / off / on / eager) found every arm alike at every stage, with the first
  differences at attention and the MoE output, and eager calls included.

### Assessment

The question was whether the one-in-2000 replay is a stale or early read at a PDL boundary or a
legitimate nondeterminism that M1 introduces. It is neither. M1 has no launch at this shape,
and the difference matches the merge-order rounding that both arms share. "A difference
with no order-dependent arithmetic in the path points to a read of x before it is complete"
does not apply here, because the path has order-dependent arithmetic and no M1 secondary.

This is a strong hypothesis, not a measurement: one replay was seen, and the stage that differed
first is unknown. The reproduction below tests it directly, and it also tests the shape where
M1 does launch secondaries (1x8 sliding: o8 and the skf front as PDL secondaries). d6c30f88c617438b
never reached that shape.

### What this means for the gate (for the user; not changed here)

The user-approved gate stays as it is: no bound was changed or weakened. The finding is that,
at a shape where M1 changes nothing, the gate compares two samples of one process, and its
largest-error envelope is a noisy statistic when events are rare. With an event rate p per
replay and n = 2000, the chance that the control draws none and the candidate at least one
is e^(-np)(1 - e^(-np)), which is about 0.24 for np between 0.5 and 1 (hypothesis: p is unmeasured;
two of two runs with this gate at this shape so far, 6ce5be0ba29642ea and d6c30f88c617438b,
had an event in exactly one arm). M1 would then be turned off in a large share of runs for a
reason unrelated to it. Possible changes, all needing the user's decision:

1. Skip a shape whose candidate capture has no M1 PDL launch (the reproduction's census
   counts them). The bounds at the shapes M1 changes stay as they are.
2. Judge the envelope on pooled event counts (rate of calls over 1e-4 and of non-bitwise calls)
   rather than on the single largest error, with an A/A arm to calibrate.
3. Make the reference path deterministic for the gate (fixed-order attention merge, as
   ONE_DETM does, and a deterministic MoE accumulation) so that any difference is a defect.

## 2. Route gate row control at 32 tokens

Measured: `gate negctl route concat-r@32: none:pass(1e-06,8e-05) ... row:pass(1e-06,8e-05)` and
the same for skf-r@32. The row injection's relative and element errors equal the uninjected
observation's to the printed digits, so the injection changed nothing. A copied row that
changes nothing must equal the row it replaced. Two different inputs' BF16 MoE outputs cannot
have an equal nonzero row, but a token with no route to this GPU's 32 experts has an exactly
zero output row, which happens with probability (224 choose 8) / (256 choose 8), about 0.34,
per token and input. The control always copied row rows // 2 = 16 of bank entry 1 over bank
entry 0, so both inputs had no local route at token 16 (probability about 0.12). Both
route variants use the same bank (seed 4242 + 32) and route the same, which is why both failed
together. At 128 and 256 tokens row 64 or 128 is live in at least one entry. The zero rows also
explain the 5e+11 element errors of other injections in route gates: a nonzero value against
a zero reference row divides by the clamped row maximum of 1e-12.

Fix (`tools/dl_gate_patch.py`, `_gate_negctl`):

- row: inject the row nearest rows // 2 whose two references differ somewhere by at least
  GATE_ROW_FLOOR = 1/8 of the fresh row's maximum. Where rows // 2 already qualifies (always
  for the layer's hidden and stream outputs), the pick is unchanged.
- unrestricted single element (MoE and route controls): draw among elements where the two
  references differ, so a zero-over-zero copy cannot be drawn. The M1 gate's floored draw is
  unchanged.
- if entries i and j have identical references (one token, no local route in either), use the
  first bank entry that differs; if every entry is equal, report "all references equal,
  nothing to inject" and count no miss, because a stale read of equal data is harmless.

Offline test, `uv run --with torch python tools/dl_gate_negctl_test.py tools 200`, CPU, synthetic
rows of width 8192 at noise 3e-3, 200 trials per setting. It now also has a sparse model (each
row exactly zero with probability 0.34 per bank entry, no noise on zero rows). Results are in
the runs log (2026-10-30 entry for this change). The old code (git HEAD before the change, run
from a copy) misses the row 13 to 29 times in 200 at 8 / 16 / 32 sparse rows. The new code misses
it none of those times, and the uninjected output is never rejected.

## 3. After the reset (00:00 UTC), in this order

Both files are in the scratchpad; neither is a candidate for submission.

1. **M1 reproduction**, one check.
   `/private/tmp/claude-501/-Users-gretchengabo-GitHub/4e9778d1-55dc-4e88-9d29-4844bd573620/scratchpad/m1repro/m1_1x1_repro.py`
   (SHA256 16665ed7..., 863,372 bytes), built by
   `python3 tools/dl_m1_1x1_repro.py muse_decode_layer/cand/layer_m1rt51.py OUT` (byte-identical
   on a second run). It is the file that failed, with `_m1_stress` and `_m1_tune` replaced and
   the prepare-time timeline dropped, and M1 is off for the timed path. Per shape (1x1 sliding with
   the gate's seeds, then 1x8 sliding):
   - census: PDL launches per step that `_m1_mode` grants while each arm is captured, and
     the kernel-name sequence of one replay of kernel_fn's graph with M1 off and on.
   - arms: off, on, off2, on2 (stage replays on fresh inputs), off-kfn and on-kfn (kernel_fn
     itself, as the gate replays it), off-same and on-same (one fixed input), then 500 eager
     M1-off calls; 4000 replays per arm at 1x1, 2000 at 1x8.
   - per replay, on the device: for each of the ten stages, the relative error against that
     input's eager M1-off reference, the number of differing elements, and how many of them
     equal the previous replay's input reference (prev) or a reference not read in the last two
     replays (oth); for the next attention input, the index, observed and reference value of
     the largest difference.
   - per arm: events, the stage where each event first differs, per stage the number of
     replays with a difference, prev / oth totals, and up to four events in full with the ULP
     count.

   Reading:

   | outcome | conclusion |
   |---|---|
   | 1x1: census 0 in every arm and identical kernel sequences; events at similar rates in off / on / off-kfn / on-kfn and present in the same-input arms; prev and oth near 0; differences of one or a few ULPs | the 1x1 failure is shared merge-order noise, not M1; the gate question goes to the user (section 1) |
   | 1x1: census above 0 or kernel sequences differ | my reading of the 1x1 path is wrong; find the launch that differs before anything else |
   | 1x8: events first at o (attention output bitwise) or at xmoe / tw (FFN input bitwise), in the fresh on arms and not in on-same or the off arms | a stale or early read of x by an M1 secondary (the sk kernels are deterministic: the selftest finds the PDL and plain launches bitwise on one x): M1 stays off, and the ordering argument has a hole at that boundary. prev does not apply here, because a GEMV of a partly stale x matches neither reference; it applies to stages that copy values through |
   | 1x8: on arms like off arms | no M1 effect at its live shape at this replay count |

   Budget: about 50,000 stage replays at roughly 0.3-0.5 ms each (estimate) plus captures, so
   about 20-30 s on top of a prepare without M1 timing (d6c30f88c617438b's prepare was 155 s).
2. **Four-arm bench of release + mode 51, no M1**, one check.
   `/private/tmp/claude-501/-Users-gretchengabo-GitHub/4e9778d1-55dc-4e88-9d29-4844bd573620/scratchpad/m1repro/bench6_g51b.py`
   (880,647 bytes): `tools/dl_make_bench_pub.py OUT muse_decode_layer/submission.py --other=inc
   --loads=d1,d16,d32,v1,v16,v32 --cand=muse_decode_layer/cand/layer_g51b.py --cand-strip=1
   --rounds=8 --seed0=210001 --order-seed=59 --dup=own`. inc is the release (b9c23bda...,
   sent 84b4cc1e...), cand is `muse_decode_layer/cand/layer_g51b.py` (acd2385c..., sent
   87af0369...). Read it as docs/steering/2026-10-30-layer-bench-arm-bias.md says: ABDECIDE,
   per-load pooled ratios against the A/A run 724aa0dd6ed140ca, d1 left out. The earlier
   g51 run 2dc566298be1498f read v16 1.0080 and v32 1.0107 and did not resolve.

3. **Four-arm bench of M1 + mode 51 with the census skip**, one check (added after the
   user chose option 1 below).
   `/private/tmp/claude-501/-Users-gretchengabo-GitHub/4e9778d1-55dc-4e88-9d29-4844bd573620/scratchpad/m1repro/bench7_m1rt51c.py`
   (895,762 bytes): the same command with `--cand=muse_decode_layer/cand/layer_m1rt51c.py
   --seed0=220001 --order-seed=61`; cand sent as 4a140c81.... Read it against both the A/A run
   and bench6_g51b: the difference between the two is M1's share. Prepare time is an estimate:
   40b9b3f795f84073 reached ABEND at 166 s with two M1 prepares at 1000 report-only replays;
   the enforced gate runs 2000 per arm at six judged shapes, so expect roughly 180-200 s
   against the 255 s deadline (ABSTOP reports any skipped shape).

layer_g51b is the release through the current `tools/dl_gate_patch.py` (row-control fix, W4
GEMM2 wait, SUMMARY after the timeline) with layer_g51's MoE regions (mode 51, the
dl_moe_inline.py output from the 6bbb252 standalone). Its device code equals layer_g51's
except the W4 source, which equals layer_m1rt51's, whose SASS was checked and which ran in
b68bdcd7a1ba464c and d6c30f88c617438b. No new compile is needed. It is the file to ship if the
bench confirms mode 51 and the user agrees.

## Gaps

- The stage that differed in d6c30f88c617438b's event is unknown; the reproduction records it.
- Whether the event rate at 1x1 is high enough to make the enforced gate fail often is an
  estimate from two runs.
- The reproduction runs one build. A release-build arm would need the stage recorder ported
  to a file without M1 code; not done.

## Addendum (2026-10-30, later): user decision, option 1

The user chose option 1 (relayed by the coordinator): the M1 gate judges only shapes whose
captured candidate graph makes at least one M1 PDL launch and logs the others as "skipped: no
M1 launch"; the bounds stay as approved; a failed census judges the shape. Implemented in
`tools/dl_gate_patch.py` and tested offline (`tools/dl_m1_gate_census_test.py`); candidate
`muse_decode_layer/cand/layer_m1rt51c.py` is layer_m1rt51 with only this change. The
reproduction (item 1 of section 3) still runs first: it checks the census claim at 1x1 on the
GPU and looks for stale reads at 1x8, where the gate still judges.
