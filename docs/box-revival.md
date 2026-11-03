# Box revival lane, 2026-11-03

Experiments the judge quota cut short, re-run on the rented H200 (docs/box.md) against the public layer
file `muse_decode_layer/cand/g1b4_rs.py` (SHA256 01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07,
verified on the box). Every number here is box net us (median cold replay minus the fixed replay cost),
candidate against control in one session on the same inputs. No number here predicts a judge score.

Settings for every run: case seeds from `.build/judge/g1b4_pdl8d.f931abf568544018.json`, router seed
106218864389449120, per-input local_start, `--runs 60`, box_run's reference check on every arm and cell.
A load's time is the weighted sum over its two shapes (46 x sliding + 16 x full, TASK.md); the
six-load sum adds the six load times.

## Finalists

None yet.

## Tools

- `tools/rv_ab.py`: in-process paired A/B. Every arm is prepared in one process; the arms' replays
  rotate order per repetition on the same fresh inputs with a cold L2 each. It reuses box_run.py's
  loader, timer, router and case seeds unchanged. It prints each arm's per-cell median, the paired
  median difference against the first arm, load times and the six-load sum. It records each arm's
  PDL8 enable set (`state["pdl8_on"]`) and re-prepares an arm (`--pdl8-tries`) until its set equals
  the control's, which in turn is held to `--pdl8-want`.
- `tools/rv_bg.sh`: rv_ab.py in the background under the same `gpu.lock` as box_bg.sh.
- `tools/rv_knob.py`: one-knob variants of g1b4_rs.py (`rv_k_<name>.py`), each edit an exact string
  that must occur once.

## The PDL8 gate is part of the control's variance

g1b4_rs's in-prepare PDL8 gate picks its enable set per prepare, and the same bytes come up with
different sets. Of eight prepares of these bytes this session, six chose {1: on, 8: off, 16: on, 32: on}
(the set of judge check f8a55f1043804158) and two chose {1: on, 8: on, 16: off, 32: off}.
The set moves whole cells: in-process A/A `rv_aa1` (g1b4_rs against a byte copy, the two arms on
different sets) read decode 1 -0.08 / -0.38, decode 16 +7.49 / +6.99, decode 32 +6.54 / +7.12, verify 1
-6.05 / -7.98, verify 16 and 32 within 0.7 us (paired, ctl2 - ctl, sliding / full). Decode 1 (T = 1
on in every set) does not move. So every comparison below is between arms on the same set, and the
spread table uses controls on one set only. The enable set is listed with each run.

## Control-vs-control spread

Three separate box_run.py processes of g1b4_rs (`rv_ctl_a`, `rv_ctl_b`, `rv_ctl_c`), all six loads, all
three on the set {1: on, 8: off, 16: on, 32: on}. Spread is max minus min of the three.

| cell | rv_ctl_a | rv_ctl_b | rv_ctl_c | spread |
|---|---:|---:|---:|---:|
| decode 1 sliding | 66.18 | 66.29 | 66.22 | 0.11 |
| decode 1 full | 71.95 | 72.08 | 72.03 | 0.13 |
| decode 16 sliding | 130.72 | 130.53 | 130.38 | 0.34 |
| decode 16 full | 202.90 | 203.46 | 204.46 | 1.57 |
| decode 32 sliding | 146.99 | 147.28 | 147.01 | 0.29 |
| decode 32 full | 298.16 | 298.62 | 298.69 | 0.53 |
| verify 1 sliding | 111.79 | 112.30 | 112.50 | 0.70 |
| verify 1 full | 120.11 | 120.67 | 120.80 | 0.69 |
| verify 16 sliding | 244.10 | 244.91 | 243.76 | 1.15 |
| verify 16 full | 328.77 | 327.68 | 328.16 | 1.09 |
| verify 32 sliding | 296.26 | 298.34 | 298.13 | 2.08 |
| verify 32 full | 462.00 | 462.54 | 461.63 | 0.91 |

| load (46 s + 16 f) | rv_ctl_a | rv_ctl_b | rv_ctl_c | spread | bar (2 x spread) |
|---|---:|---:|---:|---:|---:|
| decode 1 | 4195.3 | 4202.5 | 4198.8 | 7.2 | 14.4 |
| decode 16 | 9259.5 | 9259.6 | 9269.1 | 9.6 | |
| decode 32 | 11532.2 | 11552.9 | 11541.4 | 20.7 | |
| verify 1 | 7064.2 | 7096.7 | 7107.6 | 43.4 | 86.8 |
| verify 16 | 16488.7 | 16508.8 | 16463.5 | 45.3 | |
| verify 32 | 21019.8 | 21124.2 | 21100.0 | 104.4 | |
| six-load sum | 69559.7 | 69744.7 | 69680.4 | 185.0 | 370.0 |

In weighted units: the decode 1 bar is about 0.23 us per cell, verify 1 about 1.4 us per cell, the
six-load sum about 6 us per cell over all twelve. In-process nulls (the control's byte copy `rv_ctl2.py`
as a second arm, same set) are listed with each run; so far they are within these bars.

## Runs

| experiment | box log tag | arms (set) | cell numbers, box net us | decision |
|---|---|---|---|---|
| UNP mechanism: g1b4_unp_release.py 3d7f252f (unpooler slice + finish fused into the MoE tail; four mma chains, 16-byte x chunks, `UNP_LOSS_US = 3.0`) against its base g1b4_stage_release.py ba9e327c | rv_unp_mech | release has no PDL8 gate | decode 1: ctl 77.34 / 83.01, cand 78.62 / 84.42, paired +1.18 / +1.62; verify 1: ctl 112.24 / 121.58, cand 112.40 / 121.02, paired +0.80 / +0.19. In-prepare UNP_GATE ok at T = 1 and 8 (tails hot sep 74.1 / fused 75.0 us at T = 1) | Mechanism loses at decode 1 and is flat at verify 1, as the judge's third check read it (c77b95102baf4232, decode 1 80.67 / 85.35) |
| UNP port: rv_unp.py 1072c330 (tools/dl_unp_patch.py on g1b4_rs, all anchors apply) | rv_unp_port | ctl and ctl2 {1,16,32 on}; cand {1,8,16,32 on}: mismatched | decode 1: ctl 67.44 / 73.74, ctl2 67.25 / 74.14, cand 68.10 / 74.29, paired d_ctl2 -0.27 / +0.35, d_cand +0.56 / +0.61. verify 1 -7.26 / -8.50 is the T = 8 PDL set, not UNP | Decode 1 loss; verify 1 re-run on a matched set queued (rv_unp_port2) |
