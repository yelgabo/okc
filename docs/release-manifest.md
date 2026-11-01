# Decode-layer release manifest

## Board entry verified October31 PDT: attention port on stable M1+51+C1

Job `8f2780d7382047e6`, scored **2.022870x**,95%CI1.992527–2.053214,13/13tests.
Official public board was read and selects this job, replacing7b84ed4346d5497b (2.018705x).
Candidate `muse_decode_layer/cand/layer_attn31s.py`, SHA256
`585d46e2cd059ea7ea76c25110fd199bb228d2ee7f96239b44ae0bfb8025893b`, sourceaea73a1.
Only the owned attention device source was refreshed from the stable prior candidate;
HMRG1/PRQ31, no workspace placement pick, host PDL/precision/gates retained.

Paired gatec0402e4c92bc4b20: all24checks perarm, all6loads, independentprepared copies,
pooledinc/cand1.0042, allcrosses1.0036–1.0048, maxnull0.0007log. M1policy differs at32
in one incumbent arm, so this is the whole prepared candidate's gain, not isolated attention
instruction attribution. Second paired9414182118414889 is NOT RESOLVED: pooled1.0015,
maxlognull0.0011,allM1{} after gatefallback. First regime gain was not independently repeated.
Public score movement is within runnoise; do not infer a0.42%official-score guarantee.

Scored policy: M1{8:o+front,16:o,32:o}, open (8token saving2.94us), M1PDL=plain bitwise
on14launches; unchanged-shape1x1 gates skipped as authorized. SUMMARY reports3non-promotional
issues: sensitivity control on unpicked main-m2-ar2-ub@256, sensitivity route-side@128,
and skf-r@1 fresh-input rejection (T1 usesfp32). Do not erase these by saying every diagnostic
passed. Selected official path passes; full receipt retained in profiles/scored.8f2780d7382047e6.json.
Main submission.py remains the older clean control; use the explicit candidate for this entry.

The scored layer files, identified by full SHA256 of the file as submitted, with the source
commit, build switches, the policies prepare() selected in the receipt run, and the receipt.
Main's `muse_decode_layer/submission.py` was the M1 experiment (6377b5a, `PORT_SK_PDL = True`)
from 6377b5a until this branch's 2cee446, which makes it the clean file below again. The M1
experiment stays reachable as `git show 6377b5a:muse_decode_layer/submission.py`, with the
diagnostic builds made from it by `tools/dl_m1_diag.py`.

Policies come from each receipt's SUMMARY lines (`muse_decode_layer/profiles/<file>.<job>.log`);
prepare() picks them per run, so another run of the same file can pick differently where the
bench margins are small (the MoE pick at 32 tokens and the front pick at 8 tokens have flipped
between runs of one file).

## Validation policy (2026-10-30)

The user approved replacing the 1e-4 replay-vs-eager bound with a fresh-input gate on
2026-10-30 (relayed by the coordinator). Scope: M1's replay stress and the shipped MoE replay
self-tests (`_moe1_stress`, `_moe1_pdl_self_test`); every other check and the judge's
tolerances unchanged. Definition and thresholds: docs/runs-decode-layer.md, "Validation policy
change: fresh-input replay gate". The board entry below predates it and still carries the
1e-4 bound; a file built with `tools/dl_gate_patch.py` carries the gate.

Later on 2026-10-30 the user chose option 1 of docs/steering/2026-10-30-m1-1x1s-gate-failure.md
(relayed by the coordinator): the M1 gate judges only shapes whose captured candidate graph
makes at least one M1 PDL launch, skips the others as "skipped: no M1 launch", and judges a
shape as before when the census fails; bounds unchanged. Files built with
`tools/dl_gate_patch.py --m1` at or after this commit, and layer_m1rt51c, carry it.

Later the same day the user decided clause (b) is order-1 = relative L2 >= 1e-2 per output, with
routing-id flips judged against the control's envelope, and extended the gate to
`_route_variant_check`'s replay part. A per-call element check (row-maximum units, within 2x
the control's envelope) and report-only negative controls were added at the researcher's
caution; see the addendum in docs/runs-decode-layer.md.

## Candidate (2026-10-31): layer_m1rt51s (not submitted)

layer_m1rt51c plus the stable prepare-time picks (docs/plan-2026-10-30-tuner-stability.md,
design 1; A/A e582fde557114062), with a preference table that matches this candidate.

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_m1rt51s.py`, SHA256 `f12353566c5c665dda603a495462922778a461f9bfb3675bc65e95b2361bdd33` |
| made from | `tools/dl_stable_tuner_patch.py muse_decode_layer/cand/layer_m1rt51c.py OUT "HEADER" --pref=moe:32=main-lean-grp-finss64 --pref=moe:128=main-lean-grp-finss64 --pref=moe:256=main-lean-grp-finss64 --pref=m1:8=o+front --pref=m1:16=o --pref=m1:32=o+front --m1` at this commit. Byte-identical on a second build |
| preference table | MoE sk8w3s at 1 / 8 / 16, mode 51 (`main-lean-grp-finss64`) at 32 / 128 / 256 (every copy of layer_m1rt51 / 51c picked it in d6c30f88c617438b and 4d4ea5db46db4a10); front fp32 / skf-r / skf-r / skf-r / concat-r / concat-r and attention boundary c8x256 / c1x1024 / sglang / sglang / sglang / c1x512 (the release's receipt); M1 o+front at 8, o at 16, o+front at 32 (4d4ea5db46db4a10's cand copy; its cand2 picked o at 32, a tie) |
| against layer_m1rt51c | Python only, every `_*SRC` region equal, so the SASS is layer_m1rt51c's = layer_m1rt51's (counts in that section). Changed: `_moe1_tune`, `_front_tune` (the route variant keeps the fresh-input gate), `_abnd_tune`, `_m1_tune` (whole-step loads per mode in 9 rotated rounds, `_stable_pick` per bucket; the 8-token stop rule unchanged), `_selection_summary` (one `stable` line per tuner and token count); added `STABLE_*`, `_stable_pick`, `_m1_rounds`. The M1 gate region equals `dl_gate_patch.M1_SRC` except `_m1_tune`; the shared gate code equals `GATE_SRC` |
| offline tests | `python3 tools/dl_stable_pick_test.py tools muse_decode_layer/cand/layer_m1rt51s.py`: all cases pass, `_stable_pick` equals the tool's, M1 ties kept on the preferred mode 497-500 of 500, a 3 us win switches, `_m1_rounds` rotates as designed. `uv run --with torch python tools/dl_m1_gate_census_test.py tools muse_decode_layer/cand/layer_m1rt51s.py` (gate code taken from the file): all five cases as expected |
| build switches | as layer_m1rt51c: M1 runtime order, `ONE_TRIG`, `_ONE_PDL = False`, `MOE1_HELD = ()`; no workspace pick (`_ONE_WS_PICK` absent), no donor code |
| evidence | four-arm bench 3aee22ba4a404c8a against release_s (both arms stable): **cand faster**, pooled 1.0052, cross 1.0036-1.0068, nulls 0.08% / 0.24%; v16 +1.6%, v1 +1.5%, d16 +0.7%, d32 +0.5%, v32 -0.1%, d1 -1.1% (one cand copy slow at d1s). Policies in both cand copies: MoE 51 at 32 / 128 / 256, M1 `{8: 'o+front', 16: 'o', 32: 'o'}`, front and abnd the release's. Every gate passed, 24 of 24 checks per arm. Prepare 62-64 s per copy in the bench |
| unpatched judge check of this exact file | none (the bench ran it comment-stripped) |

## Candidate (2026-10-31): release_s (not submitted)

The board entry plus the stable prepare-time picks with the release's own preference table:
the incumbent arm for comparisons with stable candidates, and a candidate for a more stable
board entry.

| field | value |
|---|---|
| file | `muse_decode_layer/cand/release_s.py`, SHA256 `a45fae28aed1a7f2e47565301270487130f904b912acb09664d6d5202aba6ae6` |
| made from | `tools/dl_stable_tuner_patch.py muse_decode_layer/submission.py OUT "HEADER"` (the release, b9c23bda...). Byte-identical on a second build; differs from `muse_decode_layer/cand/layer_stable.py` (the A/A file of e582fde557114062) only in line 2, so both send the same comment-stripped code (SHA256 11420923...) |
| evidence | A/A e582fde557114062: every tuner but W4 picked the same in all four copies, nulls 0.12% / 0.03%; check score 2.001x (1.971-2.031) as a bench, not a direct check of this file |
| build switches | as the board entry |

## Candidate (2026-10-30): layer_hp (not checked, not submitted)

Release + the standalone attention of rounds 2 and 3 (HMRG, PRQ = 31) + the workspace placement
pick; nothing else.

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_hp.py`, SHA256 `0efd553c60815db9c2f6650ddbc3a9e6762d05fa81ff56d802cd9cc12cb2ee6f` |
| made from | a copy of the release (b9c23bda...) through `tools/dl_one_inline.py muse_attention_decode/attn_one.cu OUT` (attn_one.cu at 1faadbc) and `tools/dl_one_port_patch.py OUT HEADER`. Byte-identical on a second build |
| against the release | `_ONE_SRC` (HMRG on q_len 1 modes 1 / 3 / 11 / 13, PRQ 31 on q_len 8 modes 1 / 3 / 11 / 15; switches off gives SASS text-identical to the release); Python: `_one_ws_alloc` / `_one_ws_pick` in `_one_setup`, four more asserted attention stresses, `_attn_one(plan=)`, SUMMARY `attn ws pick`, line 2 |
| build switches | as the board entry: `SK_PDL` 0, `ONE_TRIG` 0, no M1, `_ONE_PDL = False`, `MOE1_HELD = ()`, MoE mode 47 |
| SASS | docs/runs-decode-layer.md, layer_hp build row: ordering opcode counts equal to the standalone except the inliner's ACQBULK; no spills |
| expected effect | hypothesis from the standalone benches only: HMRG -0.5 / -1.4 / -1.4 / -0.8 us cold at d1 sliding, d1 full, d16 sliding, d32 sliding (ce92d32e26024bd2, d0819bb26907426b); PRQ -0.6 / -0.5 / -1.0 us at v1 full, v16 sliding, v32 sliding (three benches, round 3); the pick 0.3-0.4 us at d1 full when the first allocation lands slow. Inside the layer these are a fraction of a percent per load; whether the layer keeps them is what the four-arm bench measures |
| bench | four-arm against the release, scratchpad `dlp/bench8_hp.py` (885,566 bytes, SHA256 a6fbe13f...), seeds 230001+, order seed 67, `--dup=own` |
| unpatched judge check of this exact file | none |

## Candidate (2026-10-30): layer_m1rt51c (not checked, not submitted)

layer_m1rt51 plus the M1 gate's census skip (user decision below) and the negative-control fix (rebuilt at the coordinator's request so its route gates at 32 tokens do not report the d6c30f88c617438b sensitivity miss again); nothing else changes.

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_m1rt51c.py`, SHA256 `1eda8ec96ded87b1298a8fed54e93583db73b85894eb1400fe4639680659d801` (the first build without the negative-control fix was 525b85b0...) |
| made from | `tools/dl_m1_census_patch.py muse_decode_layer/cand/layer_m1rt51.py OUT "HEADER" --gate-src` at this commit: the M1 gate region replaced by `dl_gate_patch.M1_SRC`, the shared gate code by `dl_gate_patch.GATE_SRC` (verbatim), the PDL launch counter added before the sk and fused runner launches, line 2 replaced. Byte-identical on a second build |
| against layer_m1rt51 | Python only (every `_*SRC` region equal, so the SASS is layer_m1rt51's): the census skip and the negative-control fix (row nearest rows // 2 whose references differ, unrestricted element drawn where they differ, an equal bank entry replaced). In this code a negative-control miss in a route or MoE gate is logged as a sensitivity failure only; it drops no variant (the verdict comes from `_gate_verdict`). In the M1 gate a miss fails the shape, as approved |
| M1 gate | as layer_m1rt51 at shapes whose candidate capture makes an M1 PDL launch; shapes with none (1x1 sliding and full with the current picks) are skipped and logged; a failed census judges the shape |
| evidence | offline test `tools/dl_m1_gate_census_test.py`; no GPU run yet. Four-arm bench against the release prepared for after the reset, third in order |

## Candidate (2026-10-30): layer_g51b (not checked, not submitted)

Release + MoE mode 51 + the current gate, no M1: the file to ship if the four-arm bench
prepared for after the reset confirms mode 51 on its own (docs/steering/2026-10-30-m1-1x1s-gate-failure.md, section 3).

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_g51b.py`, SHA256 `acd2385ca9af687a2f53b8412c65cf83ea280cc0d4db8eb52bb5c2ad75dab222` |
| made from | the release (b9c23bda...) through `tools/dl_gate_patch.py` at this commit (no `--m1`), then layer_g51's `_MOE1_SRC`, `_MOE1_VARIANTS`, `_MOE1_MAIN_MIN_M` and repack region (the `dl_moe_inline.py` output with LAYER_MODES + 51 from the 6bbb252 standalone), line 2 replaced by the candidate header. Byte-identical on a second build |
| against layer_g51 | device code equal except `_W4_SRC` (the W4 GEMM2 early exit waits first, as in layer_m1rt / layer_m1rt51, whose `_W4_SRC` it equals); Python: the negative controls always perturb (row-control fix), the SUMMARY prints after the timeline |
| build switches | as the board entry: `SK_PDL` 0, `ONE_TRIG` 0, no M1, `_ONE_PDL = False`, `MOE1_HELD = ()` |
| SASS | MoE as layer_g51 / layer_m1rt51 (kernel 51: instr 17268, 96 registers, no spills); W4 as layer_m1rt (14 GEMM2 instances gain one ACQBULK); everything else the release's |
| evidence so far | the same device code except W4 ran as layer_g51 in a438b82ff2c141c2 (2.022x) and 2dc566298be1498f (four-arm, not resolved); d6c30f88c617438b (2.013x) is release + 51 in the SK_PDL / ONE_TRIG build with M1 off |
| unpatched judge check of this exact file | none |

## Candidate (2026-10-30): layer_m1rt51 (scored d6c30f88c617438b, 2.013x, M1 off by its gate)

Scored run d6c30f88c617438b: 2.013x (1.983-2.043), 13/13, prepare 154.8 s. M1's enforced gate
failed at 1x1 sliding (one replay of 2000 at relative 3.2e-4, control bitwise), so M1 was off
for the run. At one token M1 changes no launch; analysis and the after-reset reproduction in
docs/steering/2026-10-30-m1-1x1s-gate-failure.md. Mode 51 ran at 32 / 128 / 256 and every
MoE gate passed. The board entry is unchanged.

layer_m1rt plus MoE mode 51: the combination four-arm bench 40b9b3f795f84073 measured as "cand
faster" (coordinator's request after layer_m1rt went out as scored run b68bdcd7a1ba464c).

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_m1rt51.py`, SHA256 `65264f3193f952177e8ee1c7aadda45bf5c3d7c6965e9268c6e0f78a459d64a3` |
| made from | `tools/dl_gate_patch.py --m1 --m1-runtime` at this commit on 6377b5a's file (layer_m1rt's recipe, plus the SUMMARY-last change below), the header line, then `tools/dl_moe_inline.py` (LAYER_MODES (0, 16, 26, 28, 37, 47, 51), MAIN_NAMES 51 "main-lean-grp-finss64", as at this commit) with `muse_moe_w4a16/submission.py` at 6bbb252 (the commit the port guide names for 51). The whole recipe gives a byte-identical file on a second run. Current main's standalone (with the timing twins 52-59) gives a different `_MOE1_SRC`, so it was not used |
| against layer_m1rt | `_MOE1_SRC`, `_MOE1_VARIANTS`, line 2, and prepare() printing the timeline before the SUMMARY: scored run b68bdcd7a1ba464c's 16 KB tail held only timeline lines (M1 adds v1s and v1f to it), so its SUMMARY, including M1's gate verdict and policy, was lost. No device code differs |
| against the timed build | the device code and the C++ host code equal the 40b9b3f795f84073 candidate (m1rt_meas, d57b5e38...) except the W4 GEMM2 early-exit wait; the Python differs only by the enforced gate (2000 replays, raise, negative controls) and the header; `_MOE1_*` regions equal layer_g51's (checked in a438b82ff2c141c2) |
| SASS, MoE (`dl_moe_sass_counts.sh FILE moe`) | the release's ten MoE kernels have identical counts (instruction totals included); kernel 51 adds instr 17268, FENCE.S 22, FENCE.G 11, MEMBAR 43, SYNCS 202, BAR 16, ATOMG 148, REDG 9, UBLKPF 1, ACQBULK 1, LDL 0, STL 0, exactly the port guide's standalone figures; 47 unchanged (16354, LDL 22, STL 1 as in the release) |
| registers (`dl_moe_compile_check.sh`) | kernel 51: 96 registers, 0-byte stack frame, no spills, no C7510-C7515 |
| other SASS | projections, attention and W4 as layer_m1rt (the extensions' sources are identical) |
| expected effect | 40b9b3f795f84073 measured this combination (pooled 1.0162, +1.2% net of the A/A offset; M1 about +0.4%, 51 about +0.7%). One scored run can resolve neither part |
| prepare time | not measured; the timed build reached ABEND at 166 s with two prepares of it plus two of the release, and the enforced gate doubles M1's replays |

## Release candidate (2026-10-30): layer_m1rt (scored run b68bdcd7a1ba464c, 2.013x; board entry unchanged)

Scored 2.013x (1.983-2.043), passed 13/13, prepare 143 s, against the board entry's 2.014x. Its log tail lost the SUMMARY (see layer_m1rt51), so whether M1 was on is unknown. The build recipe below is `tools/dl_gate_patch.py` as at dd1777e; at later commits the tool also moves the SUMMARY after the timeline.

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_m1rt.py`, SHA256 `d695844971f2bc7ab85720ad3a464c3464fc7efb01182746b4a96bcf14be7269` |
| made from | 6377b5a's `submission.py` (the board entry's file b9c23bda plus M1: `PORT_SK_PDL = True`, `ONE_TRIG` on the q_len 1 and batch-1 q_len 8 attention instances, `_m1_tune`; the diff against b9c23bda is M1 code only) through `tools/dl_gate_patch.py IN OUT --m1 --m1-runtime` at this commit, plus one header comment line. The patch output is byte-identical on a second run |
| what it adds to the board entry | M1 with the runtime issue order (PDL order only on launches made as PDL secondaries, flagged in bit 30 of M), the reader-side `fence.proxy.async.global` after `griddepcontrol.wait` in thread 0 and in the producer lane, `ONE_TRIG`; the fresh-input gate on the MoE replay self-tests, the route-variant replay check and M1's replay stress |
| M1 gate, enforced | `_m1_stress` at the eight T <= 32 shapes (1x1, 1x8, 16x1, 32x1, sliding and full): 2000 fresh-input replays with M1 off (control) and 2000 with M1 on (o + front), each checked on the device against that input's M1-off eager reference; order-1 = relative L2 >= 1e-2, the row-max element check within 2x the control's envelope (floor 2^-7), the candidate's relative error and share over 1e-4 within 2x the control's, ids flips within 2x the control's rate plus two. Then negative controls on an eager M1-on output, hidden and stream: a 64 x 128 tile, 1%, 0.1%, one row and one element of another input's reference must all be rejected and the uninjected output must pass. Any failure raises; `_m1_enable` then turns M1 off for the whole run (the layer runs as the board entry plus the gate) and the failure is listed with its class in the SUMMARY manifest. The measured build (40b9b3f795f84073) ran this gate report-only at 1000 replays and passed at all eight shapes |
| single-element control | drawn among the elements where the two inputs' references differ by at least 1/8 of the row maximum. Offline (`tools/dl_gate_negctl_test.py`, CPU torch, synthetic rows of width 8192 at noise 3e-3): drawing among all elements missed 5/150, 4/150 and 6/150 trials at 1 / 8 / 32 rows (a stale value close to the fresh one), which would turn M1 off on about two runs in five across its 16 checks; with the 1/8 floor 0 misses in 600 trials, and the uninjected output never rejected. The MoE and route negative controls keep the unrestricted draw and stay report-only |
| W4 GEMM2 early exit | at the coordinator's request after the PDL-secondaries audit (docs/review-pdl-secondaries-2026-10-30.md, e07a9a2): `w4_gemm_kernel` KIND 2 returned at `if (q >= total)` before its `griddepcontrol.wait`, so with no local routes every CTA could exit without waiting and the kernel could complete before its predecessor. The exit now executes the wait first when KIND is 2 (`dl_gate_patch.py`, always applied). Nothing is written on that path, and the W4 chain is not on the scored path while a fused MoE variant is picked |
| unchanged from the board entry | MoE: mode 47, `LAYER_MODES` as the release, `_MOE1_SRC` byte-identical to b9c23bda's; `_SKP_CONFIGS`, attention plans, fronts, boundaries and every other check; `_ONE_PDL = False`, `MOE1_HELD = ()` |
| inliners | at the release's source commit 2cee446 (`muse_moe_w4a16`, `muse_bf16_proj`, both inliners), the release and 6377b5a's file go through `dl_proj_inline.py` and `dl_moe_inline.py` unchanged, twice. At current main the MoE standalone carries mode 51's shared helpers, so re-inlining from main changes `_MOE1_SRC` even without mode 51; build from 2cee446's sources or from this file |
| SASS (CUDA 12.8.1, sm_90a, `tools/dl_moe_sass_counts.sh`) | Projections (`skp1` against the release's `skp`, 39 kernels): the 23 single-CTA sk instances gain the wait (ACQBULK 0 -> 1) and the two reader-side fences (FENCE.G 0 -> 2); their SYNCS, UBLKCP, FENCE.S, MEMBAR and CCTL counts rise because the runtime build compiles both issue orders into each PDL instance; no opcode count of any kernel falls below the release's or 6377b5a's, and LDL / STL are 0 in all 39. The candidate's projection SASS counts are byte-identical to those of the build the four-arm bench timed (40b9b3f795f84073). Attention (`one1` against the release's `one`, 13 kernels): every fence, barrier, mbarrier, atomic and local-memory count equal, instruction totals move by -2 to +3 in 7 instances (ONE_TRIG's launch_dependents), LDL / STL 0; equal to 6377b5a's byte for byte. W4 (`w4`, 35 kernels): the 14 GEMM2 instances gain one ACQBULK and 2 instructions, nothing else changes; the 3 instances with LDL 1 / STL 16 are as in the release. MoE: `_MOE1_SRC` and `_MOE1_HOST_SRC` are byte-identical to the release's, so its SASS is the release's |
| compile | `DL_DEFS=ONE_TRIG tools/dl_compile_check.sh FILE _ONE_SRC`: the attention extension compiles with ONE_TRIG (350 s, warnings only). The projection extension is assembled at run time, so its compile check is the SASS build above, which compiles device and host code with `SK_PDL` |
| ordering | docs/steering/2026-10-30-m1-ptx-ordering.md (with the runtime-order addendum); independent review docs/review-m1-ordering-2026-10-30.md (fba35bb): sound with caveat, keep the reader-side fences, no code change; the caveat's audit of the other PDL secondaries is docs/review-pdl-secondaries-2026-10-30.md (pending at the time of writing) |
| expected effect | from 40b9b3f795f84073 read per load against the A/A run 724aa0dd6ed140ca: M1's own shapes give about +0.4% of the score (v1 +1.2%, d16 +0.6%, d32 +0.5%). The bench's +1.2% net includes mode 51 (about +0.7% at v16 / v32), which this file leaves out. +0.4% is under the 1-2% run-to-run swing of scored runs, so one scored run cannot confirm or refute it; it can show a regression only if it is large |
| prepare time | not measured for this file. Estimate: a438b82ff2c141c2 (release + gate + 51) took 146 s; the M1 tuner and the SK_PDL / ONE_TRIG builds add about 40-50 s by the bench's ABEND (166 s with two M1 prepares against 72 s with none), and the enforced gate runs 2000 replays per arm instead of 1000. Expected under 220 s against the 300 s limit |
| unpatched judge check of this exact file | none yet |

## Candidate handed to the coordinator (2026-10-30): layer_g51 (not submitted; updated to the decided gate)

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_g51.py`, SHA256 `dacd4e0df21493b28aceb3fcde3cd1cb03c4ce89e3b29632c9908eee0184b49e` (the decided gate, d9b7d67). The first version, SHA256 c8a843cf... (first gate definition), is the file of check a438b82ff2c141c2 |
| made from | the board entry's file (b9c23bda) through `tools/dl_moe_inline.py` (LAYER_MODES + 51, `main-lean-grp-finss64`; inliner byte-identical on a second run) and `tools/dl_gate_patch.py` (fresh-input gate on `_moe1_stress` and `_moe1_pdl_self_test`), plus one header comment line |
| build switches | as the board entry: `SK_PDL` 0, `ONE_TRIG` 0, no M1, `_ONE_PDL = False`, `MOE1_HELD = ()` |
| check | a438b82ff2c141c2, **2.022x** (1.991-2.052), 13/13, failed 0, every MoE gate line order-1 free |
| policies in that check | MoE sk8w3s 1 / 8 / 16, mode 51 at 32 / 128 / 256; fronts and attention as the board entry |
| same-process timing | bench 29fd004fd58e40ef (this file, comment-stripped, against the release, null arm): cand / release 1.0084, null 0.9997; v16 -5.1 / -5.3 us, v32 -6.5 / -1.1, every other shape within the null spread. Earlier, before this file existed: Bench fdf75ba5a8124544 (the M1 measurement build, which differs from it at T <= 32 only) against the release: v16 -4.7 / -3.2 us, v32 +1.2 / -4.6, null within 0.6 of cand on those shapes; mode 51 is the only difference at 128 / 256 tokens. **Not established** (2026-10-30): the null shared the candidate's state, and identical-code loads in 29fd004fd58e40ef read inc / cand 1.0066, so part of the 1.0084 may be an inc-arm bias; four-arm rerun prepared, docs/steering/2026-10-30-layer-bench-arm-bias.md |
| what the gate change removes | the shipped 1e-4 replay bound dropped unused MoE variants on FP32-atomics noise twice (6372154fcd7a4fa9, 3e2aa83358dd4f60); with the gate, 0 variants dropped in a438b82ff2c141c2 and fdf75ba5a8124544 |
| gate coverage | MoE stress and route PDL test, and `_route_variant_check`'s replay part (user decision); still on 1e-4: `_one_stress`, the norm and route side-stream tests |
| unpatched judge check of this exact file | none (the bench ran it comment-stripped; compiled code identical). a438b82ff2c141c2 checked the first gate version |

## Current board entry: 3fbb3ad027e84c6b (2.014x)

| field | value |
|---|---|
| receipt | scored run 3fbb3ad027e84c6b, **2.014x** (1.984-2.045), passed 13/13, failed 0; submitted by the coordinator 2026-10-30 |
| file SHA256 | `b9c23bda16bb9d6311f025062e415841c67b16284737172badaa190fdc506a49` (judge content hash `c4b9a3d355194ebd...`, the same payload as check c2ba945496ab47b4) |
| source commit | 2cee446 `muse_decode_layer/submission.py` (byte-identical) |
| made from | the 1.982x release below, through `tools/dl_proj_inline.py` (muse_bf16_proj at main: LATE_ARRIVE on MT 2 / UPW 2, o16 and unp32 geometry) and `tools/dl_moe_inline.py` (`LAYER_MODES` + 47); nothing else differs (`_MOE1_SRC`, `_MOE1_VARIANTS`, `_SKP_SRC`, `_SKP_CONFIGS`) |
| build switches | `SK_PDL` 0 (the macro's default; no `-DSK_PDL`), `ONE_TRIG` 0, no `PORT_SK_PDL` / M1 code, `_ONE_PDL = False`, `MOE1_HELD = ()` |
| source digest (SUMMARY) | `01422b6e6f3dd43b`; components main=24ce6ad85709 one=e3b9a3e6b098 w4=069518f7252d moe1=34efe45010f0 skp=cc01d2410800 |
| MoE per token count | 1 / 8 / 16: sk8w3s; 32 / 128 / 256: main-lean-grp-nohoist (mode 47; 99.7 / 177.4 / 229.3 us against mode 37's 101.5 / 180.3 / 231.6) |
| front | 1: fp32; 8 / 16 / 32: skf-r; 128 / 256: concat-r |
| attention boundary | 1: c8x256; 8: c1x1024; 16 / 32 / 128: sglang; 256: c1x512 |
| wide projections at 128 | qkv_gate b1n128c3s8h, pooler b1n128c2s8h |
| attention plans | d1 (32,1,13) / (252,8,11); d16 (16,1,1) / (264,1,49); d32 (8,1,1) / (264,1,49); v1 (33,1,15) / (252,8,11); v16 (8,1,3) / (264,1,51); v32 (4,1,3) / (264,1,51) |
| per shape us (s / f) | d1 77.3 / 82.9, d16 140.8 / 215.7, d32 159.1 / 311.3, v1 121.6 / 130.3, v16 264.0 / 352.5, v32 358.9 / 518.7 |
| other readings of this file | checks c2ba945496ab47b4 2.006x, 86ced49f092a48d8 1.991x (bench cand arm), 50c4c1328ab742f4 2.003x (bench cand arm), f5ee8632e1f64bf2 (bench cand arm); same process against the release +1.8% (null 0.9987, 50c4c1328ab742f4) |

## Previous board entry: edd71d3152a946fe (1.982x)

| field | value |
|---|---|
| receipt | scored run edd71d3152a946fe, **1.982x** (1.953-2.012), passed 13/13, failed 0 |
| file SHA256 | `1355d39f6b8ddebcdae1cdee156a34873b647c26f4724cd8a3c049780f7655fb` (check 261a76ac7b364091's file) |
| source | not a commit of `submission.py`: d006531 plus the stress fix (ae171e6's change) and the FP32 fallback diagnostic (b13ecd9's change) without M1 (b1e620a) or the o16 / unp32 geometry (c365835), plus one comment line. Archived byte-identical as `muse_decode_layer/release/edd71d3152a946fe.py` |
| build switches | no `SK_PDL` / `ONE_TRIG` in the source at all (it predates them), `_ONE_PDL = False` |
| source digest (SUMMARY) | `e4dec04dcbf67ab1`; components main=24ce6ad85709 one=e3b9a3e6b098 w4=069518f7252d moe1=255f6073718d skp=6c5960c6dac9 |
| MoE per token count | 1 / 8 / 16: sk8w3s; 32 / 128 / 256: main-lean-silu-bar (mode 37) |
| front | 1: fp32; 8 / 16 / 32: skf-r; 128 / 256: concat-r |
| attention boundary | 1 / 8: c8x256; 16: sglang; 32: c1x1024; 128: sglang; 256: c1x512 |
| per shape us (s / f) | d1 76.8 / 84.6, d16 138.2 / 221.2, d32 171.3 / 327.9, v1 122.6 / 122.4, v16 270.0 / 356.4, v32 366.5 / 514.3 |
