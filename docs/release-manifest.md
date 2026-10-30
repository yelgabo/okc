# Decode-layer release manifest

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

Later the same day the user decided clause (b) is order-1 = relative L2 >= 1e-2 per output, with
routing-id flips judged against the control's envelope, and extended the gate to
`_route_variant_check`'s replay part. A per-call element check (row-maximum units, within 2x
the control's envelope) and report-only negative controls were added at the researcher's
caution; see the addendum in docs/runs-decode-layer.md.

## Candidate handed to the coordinator (2026-10-30): layer_g51 (not submitted; updated to the decided gate)

| field | value |
|---|---|
| file | `muse_decode_layer/cand/layer_g51.py`, SHA256 `dacd4e0df21493b28aceb3fcde3cd1cb03c4ce89e3b29632c9908eee0184b49e` (the decided gate, d9b7d67). The first version, SHA256 c8a843cf... (first gate definition), is the file of check a438b82ff2c141c2 |
| made from | the board entry's file (b9c23bda) through `tools/dl_moe_inline.py` (LAYER_MODES + 51, `main-lean-grp-finss64`; inliner byte-identical on a second run) and `tools/dl_gate_patch.py` (fresh-input gate on `_moe1_stress` and `_moe1_pdl_self_test`), plus one header comment line |
| build switches | as the board entry: `SK_PDL` 0, `ONE_TRIG` 0, no M1, `_ONE_PDL = False`, `MOE1_HELD = ()` |
| check | a438b82ff2c141c2, **2.022x** (1.991-2.052), 13/13, failed 0, every MoE gate line order-1 free |
| policies in that check | MoE sk8w3s 1 / 8 / 16, mode 51 at 32 / 128 / 256; fronts and attention as the board entry |
| same-process timing | bench 29fd004fd58e40ef (this file, comment-stripped, against the release, null arm): cand / release 1.0084, null 0.9997; v16 -5.1 / -5.3 us, v32 -6.5 / -1.1, every other shape within the null spread. Earlier, before this file existed: Bench fdf75ba5a8124544 (the M1 measurement build, which differs from it at T <= 32 only) against the release: v16 -4.7 / -3.2 us, v32 +1.2 / -4.6, null within 0.6 of cand on those shapes; mode 51 is the only difference at 128 / 256 tokens |
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
