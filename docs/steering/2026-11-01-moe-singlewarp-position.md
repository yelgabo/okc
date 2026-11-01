# Skinny stable route-position map: one registered mechanism

Evidence pin ba9e538 and this report's commit. Standalone MoE only; no prefill,
production edit or public submission. Root authorizes at most one check after
CPU/source validation and independent review. Parent owns index/publication.

Historical phase receipt e937013292b24906, `bench_sk3` log ABSK, records generic
positions at M1/8/16 std1.6/1.8/1.9us (layer1.7/1.9/1.9). This is the old skinny
4-slot stamped path, not a new current-kernel measurement. Current code still
uses the identical generic position_pass: per-warp histogram, CTA barrier,
warp0 prefix across all9warps, CTA barrier, positions and final CTA barrier.
For E<=128 this is an opportunity to avoid two cross-warp prefix barriers.
The initial routing histogram, job order and first weight copies remain unchanged.

At current unchanged-control official medians27.401/68.636/95.941us from
9be599cf333a47c4, algebraically subtracting those entire historical regions
would produce about1.8% six-load improvement. This is an optimistic screening
calculation, not a measured attainable bound: a replacement still costs time,
stamped costs perturb execution, and initial copies can overlap positions.
M1 alone barely clears1%, so root rejected a check on M1-only specialization.
All three loads must contribute meaningfully; no threshold weakening.

Mechanism: at M<=16, warp0 scans successive32-slot chunks. For each valid expert,
its position equals off[expert] + count in prior chunks + count of same-expert
predecessor lanes in this chunk. match_any_sync supplies equal-expert peers;
one leader updates the per-expert prior count. Thus output is identical to
the original stable expert grouping, including duplicates within/across chunks,
invalid/nonlocal slots and partial final chunks. Other warps do no prefix work.
M>16 retains the original code; M32 is a same-source negative control.

Ordering proof: all32 lanes of warp0 execute each match and each warp barrier,
including invalid lanes (eid=-1); no collective sits inside the eid-valid branch.
Initialization stores to wc[0] precede a warp barrier; current-count reads precede
a second warp barrier before leader updates; a third warp barrier publishes
updates before the next chunk. Each shared count has exactly one writer/chunk.
One final CTA barrier publishes all pos/ent writes before any compute/helper uses
them. Initial eid/off publication remains the original preceding CTA barrier.
The two removed generic prefix barriers served communications between warps;
the new algorithm has no such intermediate communication. No weight/intermediate
proxy fence, mbarrier, completion credit, watchdog, grid barrier or precision changes.

Primary proof is [CUDA12.8 Programming Guide7.6](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#synchronization-functions):
syncwarp orders participating-lane memory and syncthreads publishes preceding
shared writes to the CTA. Section7.20 requires all named lanes at match_any and
explicitly does not grant it memory ordering; actual syncwarp supplies ordering.
New wc[0] values are private to this algorithm; no later skinny code reads wc.

Four computing arms modes0,256,512,768 use private workspaces and shared read-only
weights.256 is a neutral independent compile,512 selects only this mechanism,
768 its duplicate. Original kernel_fn policy excludes all new modes; official
score is the unchanged owned control. Resource vectors must match exactly before
timing. Fixed fresh seeds441/442, layer/std at1/8/16/32,64 eager+64 postgraph
FP32 checks with original .008 global/.075 row gates and spin0,20 randomized
cold rounds. Gate remains >=1% fixed-small-candidate six-load contribution AND
>2x null log difference for both duplicates, with no per-case regression above
max(.25us,2xabs null paired delta).128/256 assumed unchanged; this is not the
actual official policy gain. No optional stamps or knob sweep.280s guard fails
closed, compact complete summary retained. On failure close; winner would still
need root-approved fresh candidate promotion validation, never submit this harness.

Three CPU tests execute duplicate/partial/all-nonlocal bijection fixtures (all
lengths1..128 plus explicit extremes), exact source restoration/fat exclusion,
and actual20round host formatter/16case scoregate with synthetic winner/loser.
CPU fixtures are not GPU measurements. today_proj independently reran all3 tests
and cleared exact a776f32 generated SHA256
f5762dbdaebd9f646781fd5eba6e931246ad5228ddef5841fd68b4732687902f,
including invalid-lane participation, duplicate ranking, shared publication and
unchanged wait/proxy/watchdog paths. Root's conditional approval dispatched
`85a113a3f11d47a5` on muse_moe_w4a16. Pending; no speed/correctness claim yet.
