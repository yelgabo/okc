# D32-ATTN-01: nonbinding source feedback, ownership unchanged

November2 local / November3,00:02UTC. Existing Decode32 Worker retains the task and implementation in [D32-ATTN-01](2026-11-02-01-full-decode-attention.md); explicit23:55UTC pickup/plan is at69cf82b3. This note supplies bounded independent source feedback, not a second implementation, cancellation or GPU allocation. Driver's MID private a6fe2769779e4299 remains separate and running/queued at this snapshot.

## Distinguish the proposed overlap from existing work

Source control ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab; attention literal22c116a45f1817fbaf27a1cc3656380affad49d2cb2367dd23aac83f20b7cef5. References below are in `muse_decode_layer/cand/g1b4_stage_release.py`.

- Cfg around250: L1flat has128consumer+128producer threads,3stages,MINB2. Current configuration calculates producer56/consumer200 setmaxnreg. These are source settings, not observed compiled register use. Do not import q_len8's112register ceiling.
- Producer segment around1201 already gathers future K/V via the independent producer and3stage ring; page-index TMA hasPD3/PS4. Merely adding future loads would duplicate existing overlap.
- Actual L1 consumer around1859 is acquire(i),load/convertK,QK,mask/softmax/rescale,PV,empty(i). It carries no next-tile QK scores. The two independent even/odd QK arithmetic chains around1903–1910 already exist.

One credible fixed schedule for the owner to assess is peel QK0, retain separate current/next score banks, and interleave original QK(i+1) with current exponential/packing/PV work. Carry only eight useful FP32 scores/thread rather than unused16value arrays/rawK vectors. This changes cross-tile dependencies; it is not the old within-tile two-chain split. Software issue interleaving may not create useful hardware overlap. Actual instruction order, live ranges, spills and admission remain required evidence; no20us forecast follows.

## Mandatory lifetime and recurrence obligations

1. Original ws_acquire/publication and conditional consumer join must complete for tilei+1 before any nextK read. Do not speculate past the full barrier or weaken fences.
2. Current stagei remains owned through final Vfragment/PV use. Preserve original syncwarp/all-thread empty arrivals around1956. FutureK completion does not retire currentV readers.
3. Retain stage/phase arithmetic `(tb+i)%STAGES`. Explicitly model simultaneous ownership of two slots, ring wrap and producer progress; two live tiles do not permit early overwrite.
4. Only future QK/masking is independent. Apply online-softmax in original tile order: newmax,alpha,`l=l*alpha+sum(P)`, output rescale,currentPV accumulation. Future scores cannot update current max/sum/output early or replace current probabilities. Preserve FP16 QK/PV computation/FP32accum and BF16 output boundary.
5. Stop lookahead at each request segment, where Qfragments,lengths/masks change. Drain into original half-merge/finalization that reuses S.k storage. This schedule requires no CONT toggle/merge-storage change.

## Closures to reuse at their exact scope

- e912a5f4 / implementation19ab26cd: cross-request ring continuity, source-only TARGET HOLD, not next-tile score carry. [Owning handoff](../2026-11-02-worker-attention-continuity.md).
- c8e99e038d9f4575: owned flatL1 QK2/PE16 failed its performance gate. CurrentG already has the two arithmetic chains; do not repackage that change.
- fe75e566c1ae4d51: per-requestL1 modes11/13 QK2 null/negative diagnostic, not this flat cross-tile schedule.
- 5575e7d15d3e49a8: L8mode51 WGMMA XPIPE2 cold loss; XPIPE1's specific register ceiling is not proof against L1mma.sync score carry. [Owning schedule/lifetime record](../../runs-attention.md#L1870).

These are constraints on mechanism selection, not a veto of the newer human-owned isolated task. If the owner has a better complete source-backed schedule, document its difference and proceed within the original one-candidate assignment. No generic profiler/census or duplicate generator is requested.

## Current profile context, not removable-time bounds

Fresh unchanged private control94ef698e91964dfa is valid13tests/12benchmarks/12profiles/native56replays. D32 sliding profile: expert75.4us,total147.0us; full: attention171.3us,expert59.3us,total294.1us. Adjusted endpoint timings differ from profiler sums:157.620500us/317.105412us,C32=12.324230ms. Do not add activity improvements to forecast endpoint gain or call those durations removable slack. Complete current profile/receipt is [archived here](../evidence/2026-11-02-driver-current-incumbent-control/terminal-analysis.json).

The worker's endpoint remains one isolated reproducible candidate plus focused source/selector/lifetime/compiler evidence, or a concrete blocker. Root alone qualifies/dispatches any later ordinary private allocation. This note is published in the watched inbox; actual reading/pickup of this feedback has not been observed or claimed.

## Existing secondary-compiler support

Driver verified the already-installed WSL cache /root/.cache/okc-cuda130-88 contains nvidia/cu13/lib/libnvrtc.so.13 and nvidia/cu13/bin/ptxas. Existing tools/attention_continuity_compile.py and tools/attention_layer_continuity_compile.py document the exact device-prefix/declaration-header approach and emitted layout/resource probes. If the isolated worker lacks an installed compiler, publish its exact source/defines and Driver can run a bounded secondary compile using this existing installation; no new toolchain installation or duplicate implementation is needed. Do not copy a compiler adapter or different standard/options into the target runtime flags. Device-onlyNVRTC13/PTXAS13 success leaves complete target host build, actual target compiler/version/options, launch/admission, numerical/cold-replay/output checks open. Do not reopen the old continuity candidate merely to reuse its compiler driver.
