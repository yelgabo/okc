# Owned multiply decoder: static proof and registered six-load comparison

November1,2026. No donor code/table imported. Coordinator supplied independently
searched multiplier16648 (0x4108), input mask0x001b001b and rotations10/13.
`tools/moe_multiply_decode.py` derives the packing table by mapping the owned
register/half E2M1 target bits{6,7,8,15} against mask0x81c081c0. The32 source bits
form a bijection; all256 combinations of multiplier-selected bits are carry-free
at the target mask. This is a lossless bit permutation, not requantization.

Input is today's tracked owned standalone source, LF299717B,
SHAc7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7.
It is NOT the historical ec565a7 public file235700B/b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14.
Only the published `abe7651dbaa44458` dispatch policy is frozen here: skinny8/3
for1/8/16/32 and main47 for128/256. Identity of all active device bodies between
today's tracked source and historical public bytes has not been established.

Three actual TorchCPU tests pass:1024 E2M1-pair fixtures,256 carry subsets,
and all150994944 code positions from full8192x4096 gate/up and4096x4096 down
tensors through actual unit/fat/legacy packers and both decoders. These are seeded
real CPU tensor bytes, not a claimed copy of the judge's inaccessible raw bank.
Scale bytes and resulting four raw BF16 fragments match exactly. The H200 harness
also verifies the actual supplied weight packs before timing. Source reversal
proves only decoder statements and derived W4_BITPOS change, including fallback.

## Actual sm90a CUBIN evidence, not a PTX instruction estimate

Local NVRTC12.8 + PTXAS + nvdisasm compile the actual owned device bodies, with
torch/host wrappers removed and disclosed standard-header adapters. Both original
and multiply CUBINs/SASS are archived gzip-compressed under
`muse_moe_w4a16/profiles/multiply_decode.{control,multiply}.2026-11-01.*`.
Manifests pin source/device/PTX/CUBIN/SASS and compiler options. No GPU ran locally.

| actual entry/loop | control | multiply |
|---|---:|---:|
|main47 whole static entry instructions|16480|16224|
|main51 whole static entry instructions|17424|17168|
|main47/51 innermost WGMMA stages, N24..64|390|374|
|HGMMA per such stage|16|16|
|main47/51 registers|96|96|
|main51 spill stores/loads|0/0|0/0|
|main47 spill stores/loads, outside hot loops|12/112B|12/112B|
|main47/51 wait1 / wait0 instructions|96/12|96/12|
|skinny8/3 whole entry instructions|3000|2976|
|skinny innermost HMMA-loop instructions|833|801|
|skinny HMMA per loop|32|32|
|skinny registers / spills|140 /0|128 /0|

Counts include NOPs and static cold paths in the identified backedge range; they
are not dynamic retired instructions or throughput measurements. Both main entries
keep192 HGMMAs total and16 barriers, with no C75xx warnings. No local accesses
appear inside the measured hot loops. Stage code shrinks4.10%, but this does not
forecast4.10% latency: memory, dependencies and other phases may dominate.

Earlier mode80 did not test this arithmetic; its proposed decoder was already
the existing7ALU+2IMAD lowering and the broad proposal was not built. Mode84's
NOSCALE-like ablation serialized the WGMMA pipeline and was never a valid runtime
decode ceiling. Modes83/76 yielded small gains and do not justify generic
instruction-count predictions. This new result changes actual decode lowering,
reduces skinny register pressure and retains the original pipeline, enough for a
bounded measurement—not enough to claim a several-percent score improvement.

## Common admission/precision repairs and frozen check artifact

Root required both conditions to use checked cooperative admission, rather than
qualify an ordinary software-grid-barrier launch. Only the two reachable host
wrappers change: main47 and skinny8/3. They require SM90 cooperative support,
grid<=deviceSMcount and136,512-token launch bounds, local0/32, successful shared
attribute/occupancy queries and a whole-resident grid; then use
cudaLaunchCooperativeKernel without PDL. Existing chunk splitting is unchanged,
so official257-token tests and512-token chunks are allowed. Actual wrapper C++
CPU mocks pass M257/M512 plus ten rejected/fatal admission/API conditions.
Source reversal confirms no device body changed in this common repair.

Both conditions also bind BF16(1+raw BF16post_norm_w), converted back toFP32 in
prepare for the unchanged float* ABI. This matches the official baseline's
effective parameter; historical unrounded FP32gamma is not silently retained.
The independent mathematical reference staysFP32. A CPU fixture proves the
common gamma round is observable and exactly typed. These common changes mean
the control differs from historical public bytes and launch/parameter policy.

Frozen CHECK ONLY file `.build/today-moe/multiply_check.py`:628518B,
SHAb1e6e0886fe0058b9c5719950fb4dd06eef27b8f69975034affa234c440e98f3.
Generator `tools/moe_make_multiply_check.py`; harness
`tools/moe_multiply_check_body.py`. Four private namespaces, independently packed
weights/scratch/inputs/captures; two distinct binaries, duplicate states per binary.
Only CPU extension builds overlap; no GPU tuner, reference or packing overlaps.
TF32 disabled for preparation reference and restored in finally on success/error.

Registered fixed normal seeds991/997/1009 across six loads (18cases). All arms
compare to FP32 at global.008/row.075; same graphs get all-three fresh inputs and
32 replays; every load also checks skew and unique nonlocal global IDs32..39 with
nonzero output poison; restore original input and recheck after cold timing.
Actual256-expert weights and real global IDs, no layer/prefill quota workaround.
Resources allow improved registers/residency but reject worse spills/geometry.
Every one of four control/treatment crosses must gain>=1% whole-six-load and
beat2x within-condition duplicate null BOTH pooled and on EACH held-out seed;
every case regression<=max(.5us,1%). No fastest-arm/seed policy. Incomplete cases,
spin flags, numerical/resource/timeout failures reject;280s total prepare guard
checks between cases and cannot interrupt nvcc. Full host scaffold CPU execution,
fresh-stale negative and actual summary win/loss/null/unlucky-seed branches pass.

Root independently verified the final pin and host tests and authorized ONE free
off-board MoE check. No public action is authorized. Any gain still needs exact
pure-source qualification and replay coverage before a submission decision.

## Terminal H200 outcome: correctness passes, registered speed gate fails

Exact owning check `835253441a214a1c`: board=muse_moe_w4a16, mode=check, H200,
status=done/ok, all5 official tests pass. Source preregistration574cc1c and dispatch
61adcec; frozen628518B/b1e6e0886fe0058b9c5719950fb4dd06eef27b8f69975034affa234c440e98f3.
Official COMMON CONTROL score2.189649236x,95%CI2.156804498–2.222493975;
prepare46.253s. This is the cooperative/BF16-effective-gamma control, not the
treatment score, and not a causal comparison against historical public bytes.

Actual supplied-weight validation checks3221225472 code positions before timing.
Reported tallies all pass:72normal,72fresh-captured-input,24skew,24exact-empty,
72post-timing comparisons. Original inputs are also restored/rechecked before
timing (not added to those counters). Worst FP32-reference relative error is
.004383652 global and.004751944 row. Spin flags remain zero. CUDA graph capture
and cooperative launches execute successfully on this H200; the earlier host mocks
alone did not establish that runtime fact. No tolerances or gates were relaxed.

| M | control mean of6 arm/case medians,us | multiply mean of6 medians,us | balanced mean-based ratio |
|---:|---:|---:|---:|
|1|37.541333|37.490666|1.001351|
|8|71.709334|72.226667|0.992835|
|16|81.682667|82.218665|0.993481|
|32|146.536000|146.229334|1.002097|
|128|184.048002|181.893334|1.011849|
|256|255.288000|252.471998|1.011153|

These rows use the same18 input cases within each condition; they are not pooled
raw event samples or independently measured official-shape scores. Exact18x4
medians are retained in the receipt/summary. The four registered whole-six-load
ratios (B0/T0,B0/T1,B1/T0,B1/T1) are1.001247993/1.002203525/1.001995933/1.002952178,
against duplicate null.008029444 (0.803%). None clears1%, let alone2xnull.

| held-out seed | four six-load cross ratios | seed duplicate null |
|---|---|---:|
|991|1.001275/1.000400/1.004167/1.003289|.007338|
|997|.998835/1.003684/.999263/1.004115|.008219|
|1009|1.003760/1.002462/1.003260/1.001963|.008531|

The per-seed gate and per-case regression gate both fail. For example at M16,
seed997, a control median94.992us versus treatment96.784us is a1.792us loss,
above its.950us allowance. Do not select the other duplicate or seed afterward.
The observed large-load advantage is real in these case medians but too small
to establish the registered whole-score win. Holding the four small-load ratios
at1 and substituting only the two balanced large-load ratios gives a calculated
six-load sensitivity1.003819, still below1%; this is not a measured hybrid run.

Actual H200 resources differ from the local NVRTC frontend's register prediction:
skinny control130registers versus multiply128, both0local/0static/155520dynamic
shared bytes,1residentCTA,288threads. Main47 is96registers/8local/0static/
228736dynamic/1residentCTA/576threads for all four arms. The getter's local bytes
must not be relabeled as measured H200 spill-store/load counts. Occupancy does
not increase. Local140→128register evidence remains valid for its archived CUBIN,
but the actual H200 comparison is130→128. These are resource/admission queries,
not measured SM utilization or HBM throughput.

Official common-control median latencies at1/8/16/32/128/256 on separate fresh
official cases are28.656/66.704/95.364/132.629/199.271/239.468us. Preserve their
distinction from the matched table above. Current public owned47 remains the
submission recommendation; no promotion or public action occurred.

Artifacts: `profiles/multiply_decode.835253441a214a1c.receipt.json` and summary;
CLI log/profile files are named `profiles/multiply_check.835253441a214a1c.*`.
Independent today_proj reviewed the final exact pin and independently reran both
cooperative source/C++ tests plus all three real TorchCPU host tests, finding no
blocker. Known-good source bytes remain unchanged.

**Decision/stop:** close this multiply-layout-only variant as a whole-six-load
improvement on the measured controlled owned dispatch. Do not spend a second
check retuning this multiplier or selectively adopting M128/256. This does not
rule out a different decoder/packing architecture or combination with a genuinely
different, measured pipeline; reopening requires a new mechanism and sufficient
whole-score headroom. The experiment demonstrates why fewer static instructions
and registers alone are insufficient. No generic per-stage/fence sweep follows.
