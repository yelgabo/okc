# Owned cached band32 schedule — bounded representation hypothesis

2026-11-01. Source/CPU/local sm90a only; no GPU dispatch or production changes.
Parent terminal a5c7c7e3fe344919 closes the direct band32 implementation: PASS5/all264
checks, all18 records, six-load factors .9422–.9460, target16 +15–21% latency and32
+17–19%, all three held-out inputs lose. Actual control/treatment both130registers,
0local,155520B shared,288threads,1CTA/SM admission. This report does not reopen a
band-size/cache-hint/identity sweep or attribute that loss to arithmetic.

## Distinct question and exact source

Can the SAME physical order/credits be represented as one small shared table, avoiding
repeated issuer/compute/helper modulo/division? No weight/activation traffic or credit
counts are improved relative to direct band32. The same bandwidth/cache/credit risks
remain. A six-arm causal comparison must retain original duplicates, cached duplicates,
and direct duplicates; cached beating direct alone is not a candidate win.

`tools/moe_skinny_cached_band.py::patch(source,cooperative=True)` applies the reviewed
fc28a77 band transform/common cooperative admission then the representation change.
Input exact owned LF c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7.
Raw generated source302476B SHA
`71e9a5b0b1e14e984928f740a9998e9f3901ca50760d46e70b72d5f32eba396b`.
Known-good public source untouched. Treatment only16/32, unchanged control for other
loads; host gate applies actual launch `p.M<=32 && sms>=132`, plus common device/admission
checks. A wrapper bypass or lower-grid fallback is forbidden.

## Rigorous capacity

M<=32 gives at most R=256 selected routes; at most32 local experts. For a positive count c,
ceil(c/8)<= (c+7)/8. Thus jobs J<=floor((R+7*active)/8)<=60. This loose bound even permits
duplicate IDs; uniqueness is not needed for allocation safety. Each A job has512 units,
B256. A floor-partitioned interval on G>=132 CTAs is at mostceil(60*512/132)=233 units;
B at most117. Sum<=350. Canonical A v<=30719 and B v<=15359 fit unsigned16; local A
credit count<=233 also fits16. Empty routing executes no mapping entries.

One packed unsigned32 per CTA-local q stores canonical physical v in low16 and the
last-occurrence A credit count in high16 (zero for B). 350 entries=1400B. Existing
128-byte struct alignment predicts155520→156928B, +1408B. This is far below the
Hopper block-shared limit, but actual H200 attribute/occupancy/cooperative admission
must still succeed; it is not a utilization measurement.

## Initialization/publication/progress

The sole helper warp creates job metadata as before. Lane0 writes phase ranges as before;
an added full helper `__syncwarp` publishes those ranges to all helper lanes. Each lane
fills q=lane, lane+32, ... using the exact same band mapper and clipped last-credit
function. Unique q ownership makes writes disjoint. A second helper `__syncwarp`
publishes the whole table before lane0's initial weight issues. Existing CTA join after
position_pass publishes to compute/helper loops; position_pass does not mutate the
schedule or cnt/off/job metadata. No added CTA barrier or inter-CTA dependency.

Weight issuer, compute warps and helper load the same canonical v from their q entry;
helper loads the exact old credit count. Both phases retain virtual q ordering, ring
slot/parity, reduction buffer handoffs, named barriers, proxy fences, all credit fences,
all BF16/FP32 arithmetic, A-before-B completion, generation/reset/grid joins and final
reads. Table fill is finite independent arithmetic before all A/B computation. Thus
the previous last-occurrence publication/progress proof carries over exactly. No new
persistent counter/scratch or cross-replay state is introduced.

CPU2 tests pass: all route/active-count capacity bounds,100 fixed arbitrary count banks
at actual132CTA intervals, packed-field/credit equivalence and empty cases; exact source
guard, setup-only mapper calls, four low16 consumers and sole high16 credit load, pre/post
warp publication placement. Prior band2tests still prove mapping/last counts. Independent
audit of actual source cleared metadata, both warp synchronizations, capacity and host
actual-grid gate. CPU proofs are not H200 math/capture/admission evidence.

## Actual local CUBIN evidence

Same isolated NVRTC/PTXAS12.8 sm90a extraction as direct-band compile; no GPU execution.
Actual skinny entry128registers,0stack/0spill,16barriers. Local original140/direct130;
actual previous H200 original/direct were BOTH130, so local predictions are not H200
resource measurements. Unchanged main47/51 compile96registers; existing main47 spill
12/112B remains. Shared layout alone adds the bounded schedule; no accumulator retained.

Source-line annotated actual SASS associates direct mapper with435 static instructions
and12 MUFU.RCP instances versus cached75 and2; prefix56→54 static instructions/oneRCP.
These are static inlined-copy/source-line associations, not dynamic counts or cycle costs.
The table moves map/prefix calculation into a single parallel setup loop. All issue,
compute and helper mapping calls are removed from hot loops by exact source guards;
they load a packed shared word instead. Existing helper SiLU reciprocal instructions
remain. Earlier18% loss could have other causes; this proves representation elimination,
not that divisions caused it or that setup/shared loads are free.

Compiler manifests/logs and line-opcode census:
`evidence/2026-11-01-owned-cached-band32/`; raw exact CUBIN/SASS/annotated SASS at
`/root/.cache/okc-owned-cached-band32/`. Direct comparison raw at
`/root/.cache/okc-owned-band32/`. Added setup may delay first weight issue, activation
reloads and grid credit inflation persist, shared footprint grows; all are explicit risks.

## Bounded next gate

Parent authorized source development only. Proj owns transparent six-arm private harness,
held-out993727/993733/993739; original0/1,cached2/3,direct4/5, independent packs/scratch,
three binaries, fixed16/32 treatment and common BF16 gamma/cooperative admission. Actual
H200 attributes/admission fatal if worse than preregistered gate. Strict finite global.008/
row.075 FP32, fresh/skew/poisoned remote exactzero/restoration, capture/state and full18-case
coverage required. Cached must beat ORIGINAL, all four crosses/each seed >=1% whole-six-load
and >2x null, with per-case regression gate. Direct comparison is attribution only.
Root owns remaining quota/dispatch. No second representation/band/cache sweep on failure.
