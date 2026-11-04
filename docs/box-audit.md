# Box run efficiency audit, 2026-11-03

Audit of how the lanes use the rented H200 (`okc-h200`, RunPod pod l667980o376btc, $5.29/h, see
[box.md](box.md)). Pod start (`/start.sh` start time) 2026-11-03 12:30:18 PST; rented through
about 07:00 PST 2026-11-04, 18.5 h or about $98 in all. Every number below is a measurement on the
box unless it is labelled a hypothesis. Raw material: `/root/okc/.build/box/*.log|json` (birth
and modify times from `stat %W %Y`), `/root/okc/.build/bx_audit/` (this audit's runs, nvidia-smi
samples every 0.5 s and process samples every 2 s from 14:56 PST). Box clocks are UTC; times
here are PST.

## Summary

| # | finding | measured size | fix |
|---|---|---|---|
| 1 | The GPU is idle about half the time even with a full queue | 53% of 0.5 s samples had any kernel running over 14:56-15:07 PST, while 97% of 2 s samples had at least two jobs waiting on gpu.lock | items 2-5 |
| 2 | Building fresh inputs is the largest phase of a run | 30.2 s of a 52.8 s six-load g1b4_rs run (57%); timed replays 1.23 s (2.3%) | `--ls-cache` stores each input's local_start (ref_front is 90% of the phase); fewer `--runs` |
| 3 | Extension builds thrash between file families and run under the GPU lock | `muse_att_v298w1pf_90a` and `dl_okc_bf16proj_v86y4_90a` rebuilt 5 times between 23:03 and 23:12 UTC; prepare 63-85 s instead of 14 s; 0% GPU for 3.3 min while held | `--ext-cache` (content-addressed build directories) |
| 4 | The in-prepare PDL gate is a coin flip that changes the timed path between runs of the same bytes | 4 distinct enable sets in 35 logs; 5 to 8 us per cell at verify 1, decode 16 and decode 32 between two runs of g1b4_rs | `--pin-pdl8`, and `--skip-pdl8-gate` saves its 8.3 s |
| 5 | rs2 lane ran GPU work outside the lock | box_rs_check.py jobs at 22:50-22:56 UTC overlapped flocked timed runs | lane fixed by 15:20 PST (coordinator) |
| 6 | `--runs 60` is more than the paired resolution needs | paired per-input SE of the median 0.10-0.36 us at the short loads with 60 runs | `--runs 20` to `30` for A/B screens |

## 1. Anatomy of one six-load run

Run: `tools/bx_box_run.py` (box_run.py plus phase timers, nothing else changed in the timed path)
on `muse_decode_layer/cand/g1b4_rs.py` (SHA256 01426b891ea12af1..., checked on the box),
`--baseline --judge g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --runs 60`,
all six loads (12 cells), warm extension cache, through flock, stdout timestamped per line.
`.build/bx_audit/warm_six.log`, `warm_six.json`; 23:04:42-23:05:36 UTC.

| phase | s | share |
|---|---:|---:|
| python start + `import torch` | 1.19 | 2.3% |
| reference load + CUDA init | 0.29 | 0.5% |
| weights (generate_weights with the router seed; no router search) | 0.24 | 0.5% |
| prepare g1b4_rs (warm) | 16.46 | 31.2% |
| prepare SGLang baseline | 0.04 | 0.1% |
| fresh inputs: 60 per cell x 12 cells, case_fields + generate_input each | 30.21 | 57.2% |
| correctness check, one input per cell | 1.20 | 2.3% |
| graph capture, both sides, 12 cells | 0.38 | 0.7% |
| fixed-cost graphs (400 cold replays per cell) | 1.29 | 2.4% |
| timed replays (60 x 2 sides x 12 cells, 1 GiB flush each) | 1.23 | 2.3% |
| cell cleanup, JSON | 0.24 | 0.5% |
| total | 52.78 | |

The job held the lock 54.0 s and had waited 416 s for it (enqueued 22:57:46, lock 23:04:42 UTC).

Inside prepare (line timestamps from the start of the process):

| segment | s |
|---|---:|
| import, extension loads from cache, MoE scale / route setup (to `G1b4 backend` line) | 4.9 |
| G_RELEASE_STARTUP_OK self-test (56 verified replays) | 4.9 |
| PDL8 gate: PDL8_AB T = 1 / 16 / 32, PDL8_STRESS (2000 replays per arm), PDL8D T = 8 stress and edge search | 8.3 |

The PDL8 gate is half of a warm prepare. Its share depends on the draw: when the T = 8 all-edges
stress passes, the edge search stops early; in this run it failed and tried 8 more edge sets
(0.44 s each).

Fresh inputs scale with batch and with full attention, since `case_fields` runs the reference
front (FP32 attention over 80,000 cached tokens) and then `generate_input` builds the whole FP8
KV cache a second time. Per cell (seconds from the previous cell's line): decode 1 0.6 / 0.5,
verify 1 0.5 / 0.6, decode 16 1.3 / 4.0, verify 16 1.4 / 4.6, decode 32 2.2 / 7.6, verify 32
2.3 / 8.9 (sliding / full). The split between generate_input and ref_front is in
"Input generation" below.

History: before the per-input `case_fields` change (about 15:00 PST, docs/box.md "Routing") a
12-cell run with `--kernels` and an 11 s router search took 35 s birth-to-end (`layer_rt.log`);
afterwards the same shape of run takes 51-53 s (`rs_ab_*`, `rs2_ab_*`, this run). The per-input
routing fix is correct and must stay; its cost is what `--ls-cache` removes.

Cold prepare: 119.4 s for layer_base.py at 12:55 PST with an empty cache (`layer_base.log`); 63-85 s
for files whose extensions were evicted by another family (section 3). Warm: 14.0-16.5 s.

`--kernels` adds 10 profiled cold replays per cell; in `rs_k_*.log` (2 cells, with kernels) runs took
17-24 s birth-to-end against 14 s prepare, so the profiler pass is a few seconds and is not a
priority.

## 2. GPU idle fraction

Job-level: the union of [birth, end] over every `.log` in `.build/box` (48 logs) from pod start to
14:55:43 PST is 3,581 s of 8,125 s, 44%. The longest idle gaps were 29 min (13:28-13:57 PST,
before the routing-slack lane started), 14 min and 9 min. This counts a job as busy for its whole
life, including compile and input building.

Kernel-level, with the queue full: nvidia-smi `utilization.gpu` every 0.5 s over 14:56-15:07 PST
(1,276 samples): any kernel running in 53% of samples, mean utilization 44.5%, mean power
212 W. Over the longer window 14:56-15:39 PST (5,197 samples) the figure is 47.8% (mean utilization
40.7%), with at least two jobs on gpu.lock in 99% of the process samples (mean 6.2 flock wrappers). Over the same window the process samples show at least two jobs queued on gpu.lock in 97%
of samples (mean 4.05 flock wrappers, max 6). So with work waiting, the lock holder kept the GPU
idle about half the time. The idle stretches line up with prepare compiles (15:00:30-15:03:45 PST,
utilization 0-9% while an rv_ab job on g1b4_stage_release then g1b4_rs held the lock) and with the
host-side parts of input building.

`utilization.gpu` counts any kernel, so input building, reference checks and prepare's own
benches count as busy. The share of rented time spent in timed replays is far lower: 1.23 s of a
54 s job, about 2.3%.

Rented-time view, hypothesis built from the two numbers above: at about 50% kernel occupancy
under a full queue and 44% of wall time with a job alive, the box has delivered roughly a quarter
of its rented hours as GPU work so far, and about 1% as timed replays.

## 3. Queueing

flock(1) grants the lock in no fixed order, so jobs are not served first-come. My anatomy job
waited 416 s behind about five others, and lanes' chained A/B scripts (control, candidate,
control, candidate) are interleaved with other lanes' jobs: pdl8h's `ab_cand1` ended 205 s after
`ab_ctl1` though one run takes about 50 s. Later in the afternoon the queue grew to 9 flock wrappers (15:16 and 15:24 PST) and this audit's
52 s control run waited 1,084 s for the lock (enqueued 23:09:22, lock 23:27:25 UTC); the rv lane's
multi-arm rv_ab jobs held it for 3 to 4 minutes each. The expected wait is (jobs ahead) x (mean job time);
with a mean of 3 waiting and 50-90 s jobs that is 2.5 to 4.5 min per run, and up to several
minutes more when the holder is compiling.

Options compared, cost per candidate in lock-held seconds (measured parts, combined by arithmetic):

| option | saves per candidate | note |
|---|---:|---|
| one job per candidate with both short loads (decode 1 + verify 1) instead of two | 1 prepare + start, about 18 s | lanes already do this (`--loads "decode 1,verify 1"`) |
| several candidates in one process (tools/rv_ab.py, rv lane) | the later arms' prepare drops to 7.5-8 s (first arm 14 s: one-time loads), and the fresh inputs are built once for all arms | the strongest structural option; already written by the rv lane |
| content-addressed extension cache | 50-70 s each time a family switch would rebuild | section 3 finding; `--ext-cache` |
| skip the in-prepare PDL gate when the A/B pins PDL | 8.3 s per prepare, plus every rv_ab re-prepare ("attempt N") that only re-rolls the gate (7.5 s each) | `--pin-pdl8 --skip-pdl8-gate` |
| cache each fresh input's local_start across runs | about 26 s of the 30 s input phase per six-load run | `--ls-cache` |
| batching several candidates' prepares | none beyond the in-process option: prepare is per file and its compile is per source | |

The extension cache: `/root/.cache/torch_extensions/py312_cu130` holds 119 MB (the 1.7 GB under
`/root/.cache` is mostly pip's 1.5 GB). torch's JIT cache is keyed by extension name only and
rebuilds in place when the source hash changes. g1b4_stage_release.py, g1b4_unp_release.py and
g1b4_rs.py compile `muse_att_v298w1pf_90a` and `dl_okc_bf16proj_v86y4_90a` with different
sources, so every switch between them rebuilds both: .so build times 23:03:33, 23:04:00, 23:10:49,
23:11:12, 23:11:48, 23:12:13 UTC. The rest of the cache was stable (no rebuilds 21:23-23:00 UTC).
The Triton cache (`/root/.triton`, 59 MB) is keyed by content already.

## 4. In-prepare work and the PDL coin flip

g1b4_rs.py reads one environment variable, `SK2_DELAY_ESC` (line 7177), and nothing that skips
its self-test or benches. So no env flag the file already honours can skip them.

The PDL8 gate is not redundant on the box in the way the task assumed: it decides the timed path.
`prepare` sets `state["pdl8_on"]` from the gate, and `kernel_fn` reads it on every call (line
17991). The gate's stress compares PDL-on drift with PDL-off drift on one input, and that drift is
FP32 merge-order noise (ledger, 14:40 and 14:47 PST), so the same bytes come out with different
enable sets. Tally over the 35 box logs that print the gate result:

| enable set | logs |
|---|---:|
| {1: on, 8: off, 16: on, 32: on} | 16 |
| {1: on, 8: on, 16: off, 32: off} | 9 |
| {1: on, 8: on, 16: on, 32: on} | 6 |
| {1: on, 8: off, 16: on, 32: off} | 4 |

g1b4_rs.py alone (16 logs) produced all four sets (7 / 4 / 3 / 2); layer_base.py (pdl8d, 9 logs)
three of them. The effect on a box A/B, from four 12-cell runs of g1b4_rs with the same seeds
(`rs_ab_cand1/2`, `rs2_ab_cand1/2`): verify 1 sliding 112.48 / 105.34 / 112.43 / 105.17 us and
decode 16 sliding 132.13 / 137.62 / 131.10 / 137.79 us, the two values tracking T = 8 and
T = 16 on or off in that run's log exactly. The rv lane's in-process A/A of g1b4_rs against a
renamed copy (`rv_aa1.log`, enable sets differing at T = 8 / 16 / 32) shows paired differences
of +7.49 / +6.99 at decode 16, +6.54 / +7.12 at decode 32 and -6.05 / -7.98 at verify 1 between
identical kernels. Any box A/B on g1b4_rs or a derivative that does not match the control's and
the candidate's enable sets measures the coin, not the change, at verify 1, decode 16 and
decode 32. Decode 1 is unaffected (T = 1 was on in every log but two rv_ab arms).

Fix for A/B work: set the same enable set on both arms after prepare (`--pin-pdl8`), which
changes no kernel and no launch, only which of the file's two existing paths the graph captures,
and then the gate's 8.3 s can be skipped (`--skip-pdl8-gate`). The judge still runs the gate, so
a pinned A/B answers "is the change faster with PDL in configuration X", which is the question an
A/B needs; the judged file's own enable set stays a separate draw. The G_RELEASE_STARTUP_OK
self-test (4.9 s) guards correctness of the build and stays.

## 5. Repeats and resolution

From the four g1b4_rs runs above (60 fresh inputs per cell, same seeds in every run;
`.build/boxaudit/se.py`):

| cell | median us | bootstrap SE of one run's median | paired run-vs-run diff (median) | bootstrap SE of the paired-diff median | runs for SE <= target |
|---|---:|---:|---:|---:|---:|
| decode 1 sliding | 70.96 | 0.21 | -0.18 | 0.21 | 10 (0.5 us) |
| decode 1 full | 77.01 | 0.20 | -0.22 | 0.19 | 8 (0.5 us) |
| verify 1 sliding | 112.48 | 1.55 | +7.30 (PDL T = 8 differs) | 0.21 | 10 (0.5 us) |
| verify 1 full | 120.56 | 2.25 | +8.46 (PDL T = 8 differs) | 0.10 | 3 (0.5 us) |
| decode 16 sliding | 132.13 | 1.66 | -6.48 (PDL T = 16 differs) | 0.22 | 2 (1%) |
| decode 16 full | 202.26 | 2.58 | -7.22 (PDL T = 16 differs) | 0.36 | 2 (1%) |
| decode 32 sliding | 152.75 | 1.11 | -0.69 | 0.28 | 2 (1%) |
| decode 32 full | 306.48 | 1.46 | +0.10 | 0.24 | 1 (1%) |
| verify 16 sliding | 244.13 | 1.91 | -0.90 | 0.13 | 1 (1%) |
| verify 16 full | 327.73 | 1.19 | +1.36 | 0.73 | 3 (1%) |
| verify 32 sliding | 296.86 | 1.58 | -1.14 | 0.28 | 1 (1%) |
| verify 32 full | 461.79 | 1.19 | -0.98 | 0.33 | 1 (1%) |

"Runs for SE <= target" scales the paired SE as 1/sqrt(n): n = 60 (SE / target)^2. The unpaired SE
of one run's median is 1-2.6 us at the batched and verify cells because routing varies per input;
the per-input pairing (same seed, same input in both runs) removes most of it. So the paired
comparison already resolves 0.5 us at the short loads with 10 inputs and 1% at the long loads with
1 to 3, and 60 inputs is 6x more than the short loads need. The bootstrap at n below about 20 is
itself unreliable and medians of skewed routing draws need the inputs to cover the routing mix
(2 to 4 distinct experts at decode 1, docs/box.md), so the recommendation is `--runs 20` for screens
and 60 only for a result that goes in a ledger. Run-vs-run differences larger than these SEs
(decode 1: about -3.4 us between the rs_ab pair and the rs2_ab pair, made 10 minutes apart) are a
between-session drift that more repeats inside one run do not remove; in-process interleaving
(rv_ab.py) or alternating control and candidate jobs does.

Because fresh inputs are 57% of a run and scale linearly with `--runs`, `--runs 20` cuts a
six-load run by about 20 s of the 30 s input phase, more than any other single option.

## 6. Other checks

- The 800 MB expert bank is never rsynced: `generate_weights` builds it on the GPU in 0.24 s.
  Only source files are rsynced (g1b4_rs.py is 912 KB).
- `/workspace` (network mount) is not used by any job (no file there newer than pod start). Caches
  and repo are on the local overlay disk. 2 TB of host page cache is free; no page-cache effect is
  visible in prepare times.
- Router-seed search: every lane passes `--router-seed`; the search ran once (`layer_rt.log`, 11 s).
- The 1 GiB L2 flush per replay: about 6,200 flushes per six-load run (400 fixed-cost replays plus
  120 timed per cell) inside the 2.5 s of timed and fixed-cost graph work. A 128 MB flush (2.5x the
  50 MB L2) would save about 2 s per run (hypothesis, not measured) and change the cold state the
  parity table was measured under; not worth changing.
- `box_run.py` generates the weights twice when `--router-seed` is given (default seed, then the
  router seed); 0.24 s for both, not worth changing.
- Concurrency outside the lock: the rs2 lane's `.build/box/rs2r/var.sh`, `run.sh` and `model.sh`
  called python directly; `box_rs_check.py` (pid 26302) ran from 22:52:49 UTC beside flocked timed
  runs. Reported to the coordinator at 14:57 PST; all jobs were under flock by 15:20 PST.

## Ranking (GPU-hours saved per hour of work)

Remaining rental about 15.7 h at 15:17 PST. Estimates combine the measured phase costs with the
current job mix (mostly six-load or four-load layer A/Bs, 50-90 s each, plus rv_ab multi-arm jobs).

1. `--ls-cache` plus a smaller `--runs`: input building is 57% of a job and 90% of it is
   `case_fields`' reference front, whose answer (local_start) is the same for every run with the
   same router seed. Caching it cuts about 26 s of a 53 s six-load run; `--runs 20` cuts two thirds
   of what is left. One hour of work; it applies to every job.
2. `--ext-cache`: removes 50-70 s rebuilds on family switches, which happened 5 times in 10 min
   while the rv lane alternated families. Half an hour of work.
3. `--pin-pdl8` with `--skip-pdl8-gate`: 8.3 s per prepare plus the rv lane's re-prepares, and it
   removes a 5-8 us confound from every g1b4_rs A/B at verify 1 / decode 16 / decode 32. Half an
   hour of work; the validity gain is larger than the time saved.
4. Release the lock while a compile runs (the runner takes the lock itself and drops it around
   load_inline). Not done: it needs the lock moved into the runner and a re-acquire path, and once
   2 removes the family thrash, compiles happen only for new sources.

## Input generation

`.build/bx_audit/gen_bench.py` (one lock hold, 8.3 s, 23:54:23 UTC): per cell, 5 fresh inputs,
time per input in ms:

| cell (sliding / full) | generate_input | ref_front | case_fields (generate_input + ref_front) | KV cache GB |
|---|---:|---:|---:|---:|
| decode 1 | 1 / 1 | 2 / 3 | 2 / 5 | 0.02 |
| verify 1 | 1 / 1 | 2 / 4 | 3 / 5 | 0.02 |
| decode 16 | 4 / 4 | 8 / 49 | 12 / 54 | 0.35 |
| verify 16 | 4 / 4 | 8 / 60 | 13 / 64 | 0.35 |
| decode 32 | 8 / 8 | 14 / 98 | 22 / 106 | 0.70 |
| verify 32 | 8 / 8 | 15 / 119 | 23 / 127 | 0.70 |

Summed over the 12 cells, one input set costs 488 ms, 60 sets 29.3 s, which matches the 30.2 s
phase. 433 ms of the 488 is `case_fields`, and of that the reference front (FP32 attention over
the full 80,000-token cache at the full layers) is nearly all; building the KV cache twice costs
only 52 ms per set. So the duplicate `generate_input` I first targeted is worth 3 s per run, and
the reference front is worth 26 s.

The reference front's only use in the runner is `local_start`, a deterministic function of
reference.py, the weights (weights seed and router seed), the case and the input seed. Every run
with the standard router seed recomputes the same 720 values. `--ls-cache FILE` keeps them in a
JSON file keyed by those inputs (reference.py SHA256 prefix, router seed, weights seed, case
fields, input seed), computes a missing one with `ref.case_fields` exactly as before and merges
new values into the file by atomic rename, so lanes can share one file.

A side observation: building an input once and filling local_start afterwards did not give
byte-equal tensors at decode 1 full in one of 5 inputs (`gen_bench.log`, "reuse identical False"),
and a runner variant that asserted equality failed there (`v_bx1.log`). One of the tensors
`generate_input` returns differs between two builds of the same seed in that cell, or ref_front
writes into its input; not traced further, and the cache path does not depend on it because it
still builds each timed input with one `generate_input` call, as box_run.py does.

## Changes and validation

All changes are opt-in flags in `tools/bx_box_run.py`, a copy of `tools/box_run.py` (box_run.py
and box_bg.sh on the box are untouched). Without flags it behaves as box_run.py and adds three
things: per-phase timers (a `BXPHASE` line and `phases` in the JSON), each side's PDL8 enable set
as prepared and as timed (`BX submission pdl8 ...` line, `pdl8` in the JSON), and `argv` in the
JSON.

| flag | what it does | why it keeps the timed path |
|---|---|---|
| `--ls-cache FILE` | reads each fresh input's local_start from FILE, computes a missing one with `ref.case_fields` and merges it into FILE (atomic rename) | each timed input is still one `generate_input` call with the same seed and the same local_start |
| `--ext-cache DIR` | wraps `torch.utils.cpp_extension.load_inline` so every call without its own build_directory builds in DIR/<name>-<sha256 of all its arguments and TORCH_CUDA_ARCH_LIST> | same sources, same flags, same compiler; only the directory differs, so two families no longer evict each other |
| `--pin-pdl8 T,...` | after prepare sets `state["pdl8_on"]` to exactly these T and, with 8, `state["pdl8_edges"][8]` to all of `PDL8D_EDGES` | picks one of the file's own two paths per T, identically for every arm; no kernel or launch changes |
| `--skip-pdl8-gate` | with `--pin-pdl8`, replaces the file's `_pdl8_setup` (the PDL8_AB / PDL8_STRESS / PDL8D gate) by a no-op | the gate only writes `pdl8_on` / `pdl8_edges`, which the pin then sets; G_RELEASE_STARTUP_OK still runs |

Validation on g1b4_rs.py (01426b89) against itself, six loads, `--baseline --judge
g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120`, all in one lock hold
(00:00:14-00:02:29 UTC; `.build/bx_audit/v_*.log|json`): `v_cur` and `v_cur2` are the current
runner; `v_bx2` (empty local_start cache), `v_bx3` (warm cache) and `v_bx4` (warm, `--runs 20`)
use all four flags with `--pin-pdl8 1,16,32`, the set `v_cur`'s own gate picked. `v_cur2`'s gate
picked {1, 8}, so it is the coin-flip example. Paired differences are medians over the same
inputs (input i of both runs has the same seed), net us; every check PASS.

| cell | v_cur | v_cur2 | v_bx2 | v_bx3 | v_cur2 - v_cur | v_bx3 - v_bx2 | v_bx2 - v_cur | v_bx3 - v_cur | v_bx4 - v_cur (20 inputs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode 1 sliding | 67.52 | 67.46 | 67.39 | 67.86 | +0.22 | -0.02 | -0.24 | +0.19 | +0.37 |
| decode 1 full | 73.76 | 73.92 | 73.46 | 73.50 | +0.42 | -0.06 | -0.02 | +0.13 | -0.40 |
| decode 16 sliding | 131.68 | 137.66 | 130.74 | 131.31 | +6.40 (PDL T = 16 off) | +0.21 | -0.38 | -0.11 | +0.19 |
| decode 16 full | 203.09 | 210.03 | 202.96 | 202.72 | +6.99 (PDL T = 16 off) | +0.34 | -0.29 | -0.19 | +1.12 |
| decode 32 sliding | 146.93 | 153.71 | 147.70 | 147.44 | +6.69 (PDL T = 32 off) | +0.64 | +0.30 | +0.96 | -0.30 |
| decode 32 full | 298.59 | 305.30 | 297.74 | 298.37 | +6.67 (PDL T = 32 off) | +0.94 | -0.91 | -0.21 | -0.72 |
| verify 1 sliding | 112.75 | 105.33 | 111.60 | 111.74 | -7.46 (PDL T = 8 on) | -0.14 | -0.75 | -0.83 | -0.06 |
| verify 1 full | 120.53 | 112.90 | 120.13 | 120.80 | -8.19 (PDL T = 8 on) | +0.56 | -0.51 | -0.14 | -0.03 |
| verify 16 sliding | 243.92 | 244.46 | 243.66 | 244.37 | +0.16 | +0.03 | +0.72 | +0.53 | +0.11 |
| verify 16 full | 327.79 | 326.62 | 326.05 | 326.40 | -1.26 | -0.48 | +0.10 | -0.62 | -0.43 |
| verify 32 sliding | 297.97 | 298.64 | 297.39 | 298.45 | +0.86 | +0.45 | -0.48 | +0.30 | +0.30 |
| verify 32 full | 462.75 | 461.73 | 462.56 | 461.82 | +0.26 | -0.02 | -0.05 | -0.19 | -0.51 |

On the cells where the two current-runner runs timed the same path (decode 1, verify 16, verify
32) their control-vs-control differences span -1.26 to +0.86 us; the flagged runs against the
current runner span -0.91 to +0.96 us over all 12 cells, and the two flagged runs against each
other -0.48 to +0.94. So the flagged runner gives the same per-cell answer as the current runner
within the current runner's own run-to-run spread. The coin flip moves `v_cur2` by 6.4 to 8.2 us
at six cells; the pinned runs do not move.

`--runs 20` keeps the paired answer (last column) but its medians are over the first 20 inputs,
which is a different routing mix: verify 1 full reads 126.14 at 20 inputs against 120.80 at 60
(`v_bx4` against `v_bx3`). Compare runs only at the same `--runs`.

Wall clock of the lock-held part of a standard six-load run (12 cells, 60 inputs, with SGLang),
from the stamps in `.build/bx_audit/val.times` and `a1.times`:

| run | s |
|---|---:|
| box_run.py, warm torch extension cache (`warm_six` instrumented copy, `v_cur`, `v_cur2`) | 54.0 / 52.1 / 52.0 |
| bx, all flags, empty local_start cache (`v_bx2`) | 47.0 |
| bx, all flags, warm (`v_bx3`) | 19.5 |
| bx, all flags, warm, `--runs 20` (`v_bx4`) | 16.7 |
| bx, first build into an empty `--ext-cache` (`v_bx1`, prepare only) | 95.5 s prepare (compile) |

Warm phases of `v_bx3`: prepare 8.05 s (gate skipped; 16.5 s with it), fresh inputs 3.95 s
(30.2 s before), import 1.56, check 1.26, fixed-cost graphs 1.29, replays 1.23, capture 0.38.
A six-load A/B pair (control and candidate) drops from about 105 s to about 39 s of GPU lock. The
local_start cache for the standard judge file and router seed is pre-filled at
`/root/okc/.build/box/ls_cache.json` (720 entries), and `/root/.cache/bx_ext` holds g1b4_rs's
extensions.

### Commands for the lanes

Single file (in place of `tools/box_bg.sh <tag> ...`), with the same control and candidate pin:

```bash
ssh okc-h200 'cd /root/okc && . tools/box_env.sh && nohup flock /root/okc/.build/box/gpu.lock \
  python tools/bx_box_run.py muse_decode_layer muse_decode_layer/cand/<file>.py --baseline \
  --judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 \
  --loads "decode 1,verify 1" --runs 60 \
  --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --pin-pdl8 1,8,16,32 --skip-pdl8-gate \
  --out .build/box/<tag>.json > .build/box/<tag>.log 2>&1 &'
```

- Pin the same set on every arm of one comparison. `1,8,16,32` is pdl8h's build-time set; use
  `1,16,32` to match the file's most common gate outcome. Omit both PDL flags only when the
  question is the gate itself (pdl8h's own dry runs).
- `--ext-cache` and `--ls-cache` are safe for every layer file; `--ls-cache` keys on the judge
  file's case seeds and the router seed, so a different `--judge` or `--router-seed` just fills new
  entries.
- For tools/rv_ab.py (rv lane): it calls `br.capture` and its own fresh-input loop, so the flags
  do not reach it. The equivalent change there is three lines: call
  `bx_box_run.content_addressed_extensions("/root/.cache/bx_ext")` before loading arms, set each
  arm's `state["pdl8_on"]` after prepare instead of re-preparing until the gate agrees ("attempt
  N"), and take local_start from `bx_box_run.LocalStartCache`.

Lead option: `tools/bx_box_run.py` can replace `tools/box_run.py` as is (same default behaviour,
extra log lines and JSON keys only); box_bg.sh then gains the flags by passing them through.
