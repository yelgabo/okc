# Owned complete-output q8 sliding attention: prospective experiment

2026-11-01. Baseline production attention `6da000f`, unchanged through inspected main `8e3b681`. This report accompanies the tools and device compilation manifest; no GPU result exists yet.

## Opportunity and prior loser

Exact comparator `4862f8e0dc8a4b5a` measured B1/q8/sliding control11.10us versus VP2 10.34/10.37us. Restricted prospective transfer is approximately1.086% six-load score, not a prediction for owned code. Original combined matrix failed due to full16 WST loss. Saved rank50 gap4.0322% cannot be closed by this row alone at comparator speed.

Owned `f74cbe27819147c1` lost32us versus11us: both heads per cluster, remote partial pulls and global finalize. This new design gives each cluster a complete KV-head context and pushes partials into distributed output owners, removing global partials and finalization. Do not retune f74.

Independent Nov1 problem/challenge searches screened official challenge/rules, vjeux GitHub and nonprimary anecdotes; no new faster pin found. Reuse coordinator discovery report `2026-11-01-discovery-and-reasoning.md`. The keell lead was root/user supplied. Exact pin `keell0renz/muse-open-kernel-challenge@8e30e1be994db4310794b8d04e90d18d895deb84`, standalone SHA496d460e0bb1e7489684de2742f1ff22f3959d9d5fe30674199e74b3d002d651. Public measurement used SCH0. No donor CUDA copied.

## Ownership, ordering and precision

`tools/attn_headcluster_patch.py` changes generated private source only. Flag defaults0. Supported B1/q8/sliding2048/page1/unitKVscale,16Q/2KVheads,D64,32CTAs as2clusters of16,128threads. Other geometries use control. Each cluster covers WHOLE context for one KV head, balanced64-key tiles across16ranks. Existing PRQ31 math remains; query skips only whole other-head warps.128threads use two passes for newKV/RoPE. CPU bijection proves every cache/output coordinate has exactly one writer, including empty partitions.

Aggregate cache bytes retain equivalent head coverage: each cluster copies only its64bytes/head. Query duplication and dependencies change; this is not an additive bandwidth estimate. Existing WGMMA/fences/waits remain. cp_wait0/CTA sync retire raw readers, first cluster barrier establishes every destination alive and ready before generic DSM pushes reuse dead rawKV storage. Each sender writes unique slots in distributed four-row output owners. Second cluster barrier publishes and retires remote stores. Finalizers then read ONLY LOCAL storage; peer exit cannot invalidate another CTA's reads. Empty CTAs join both barriers and publish zero partials. No global partial/counter/finalize access remains in instantiated HC path.

CUDA12.8.1 [DSM guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#distributed-shared-memory) gives initialization/retirement pattern; PTX8.7 [barrier.cluster](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster), section9.7.13.3, supplies default release/acquire and prior nonasync completion. Existing async waits remain explicit. No intercluster dependency/cooperative grid launch.

Attention value is explicitly BF16 RNE then widened BEFORE RMS/gate, matching SGLang standalone baseline.py129 (judge reference.py336 instead RMSes the FP32 attention value). Existing owned FP8 probability conversion unchanged; no new approximate probability conversion. New queued FP32 reference also includes this BF16 boundary. Actual Torch2.10 CPU midpoint/tie fixture passes. Independent scalar versus vector reference relative errors9.19e-5(BF16 query),0(FP8 query). Five CPU ownership/precision tests pass.

## Compilation and prospective check

Actual NVRTC12.8/PTXAS sm90a passes:96registers,0stack/spills,16barriers,128threads,min2CTA. Exact sourceSHA457f3c9199c0472f4d3ce02b813dce0e2989e9c5bb9952217f573b1bd20588f6. Logs/manifest in `muse_attention_decode/profiles/headcluster-local-2026-11-01`. Device extraction removes host headers, supplies metadata type_traits and exact negative-infinity declaration adapter; attention arithmetic retained. This does not validate host APIs or H200 occupancy. Runtime must report zero local bytes,dyn90704,128threads,>=2blocks/SM and>=2active16CTA clusters before timing; checked opt-in/launch errors fail closed.

First authorized check:4independent modules/states,3heldout inputs,all12scored shapes,120random cold rounds. Generated benchmark SHA4559fc2ff4541d2db6f0aa7e9b7da75fcccad225d06c551c04c58e20ec5120f5. All arms official-style independent FP32 global/row/cache checks; short contexts0/10;64distinct freshQKV x4graphs =256checked outputs plus3corruption negatives. Errors/resources printed at tail.

Preregistered acceptance: v1sliding faster in all12crosses(2candidates x2controls x3inputs). Score counterfactual credits ONLY changed row using six equal loads, each46sliding+16full. Require >=1% aggregate factor and loggain>2xindependent control/candidate nulls. Unchanged rows sensitivity only. Any accuracy/order/resource failure closes candidate; performance loss closes this architecture. Only repeatable win merits second authorized candidate-only check with actual judged path/private state and expanded fresh coverage. No public submit here.

Runtime residency, DSM cost and old PRQ31 efficiency remain unmeasured. Future pure candidate must give concurrent calls separate state or explicit serialization; benchmark arms already private. No top50 claim before actual candidate score.

## Dispatch qualification and correction

First screen dispatched `eac485ea0d554e50` from `eee3b5b`, normalized LF SHA4559fc2ff4541d2db6f0aa7e9b7da75fcccad225d06c551c04c58e20ec5120f5; physical CRLF upload SHAfdd91207817299a1db9d72aaca36bedf154681e5133fd27113760f4ecc327e45. Final source read caught inherited fresh loop sharing input/cache tensors across arm graphs, although modules/scratch were private. Root explicitly retained the job as performance/math SCREEN ONLY; this does not prove independent fresh-cache correctness. No second check if it loses performance.

Prospective harness now clones all data/cache tensors/config per arm, captures using that arm, and feeds all64banks to each private QKV. Actual extracted AST CPU fixture verifies pointer distinction and mutation isolation. Six CPU tests pass. Prospective artifact `.build/today-attn/headcluster-private.py` uses write_bytes, physical=normalized SHA1e1ab4c70fb8116d1770548fa442720cb7ee9863a0283c8d461834af4981a8a8. This corrected artifact has NOT been dispatched.

Legacy production finalizers attn_one.cu850/946 retain FP32 attention values through RMS; new HC rounds BF16 per SGLang baseline129. Thus A/B timing includes both ownership change and SGLang-parity BF16 correction. It does not isolate ownership alone. Probability conversion unchanged. No promotion based on shared-cache screen.


## Reference attribution correction

SGLang baseline.py129 rounds attention value toBF16. Official judge reference.py336 RMSes the FP32 value. These are distinct: new HC preserves SGLang BF16 parity, not exact judge-reference arithmetic. Both must pass official output tolerance. Screen eager `_ob_check` uses unrounded FP32 references; its queued BF16 reference is only SGLang-parity coverage and must not be labeled exact every-fresh judge parity. Prospective harness now retains FP32 default queued references/bounds, optional BF16 CPU parity fixture, and unchanged FP32 `_ob_check`. Latest undispatched private artifact physical=LF SHA8c549b5bd17cb28786db2da8c893b53c36f16e7aa94be04b7ea9fe4070c40ab7 supersedes earlier private hashes. Six CPU tests pass. No second dispatch without screen performance win.

## Terminal screen: close this variant

`eac485ea0d554e50` DONE. Judged unchanged control passes16official tests,2.49015534234613x CI[2.452803012210938,2.5275076724813217], checks.ok true, noflags. These official tests/score are CONTROL, not the HC judged path. Every privatearm passes21eager official-style math/cache/row cases. Qualified shared-cache/BF16-parity freshscreen256checks/3negatives passes; maximum relative.0765823,row.3389804,newK/newV0,boundmax.15. This is not independent-cache/everyfreshjudge proof.

Runtime both HC builds96regs,0localbytes,90704dynamicshared,128threads,2blocks/SM,cudaOccupancyMaxActiveClusters14 under actual16CTA config. No resource-gate failure. Exact artifacts/logs/receipt and decision JSON preserved under muse_attention_decode/profiles/eac485ea0d554e50.

Performance is decisive loss: pooled cold v1sliding control11.10us versusHC1 14.98/HC2 15.01us; warm6.87 versus10.72/10.69. Three fresh timing inputs HC1deltas+4.45,+3.68,+3.87us;HC2+4.45,+3.94,+3.84,controlnull0,−.03,0. No gain/null ambiguity. Raw allrow ABSCORE.9672/.9670 is diagnostic only. Exact affected-row-only six-load factors using observed rounded pooled cold medians and held full18.94us are in decision.json (~.9674). Unchanged full16/full32 placement noise gets no credit.

Prospective gate fails all timing crosses and1%gain; NO secondcheck/purecandidate/no publicsubmit. Close this owned64-key PRQ31/generic-DSM complete-head architecture. Do not generalize to a blanket closure of VP2's still-positive N128/asynchronous-push implementation. Missing decision is whether its wider tile/producer-to-owner communication can be implemented efficiently with owned math; this run does not attribute the3.9us regression to any one part. No source-specific followup is authorized by this report. Best submission remains incumbent default2.

Prospective private-cache harness retained offline with7CPUtests (both FP32 judge-style and BF16 SGLang-parity scalar/vector references). No GPU run for corrected harness because performance stopping criterion already fired.
