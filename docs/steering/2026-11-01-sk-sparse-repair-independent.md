# Independent SK sparse exponent-repair proof

November1,2026. Scope: the projection owner's new SK10 consumer-side sparse E7-XOR repair, not MoE tuning or adoption of an external kernel. No GPU or public action is authorized by this report. This is an independent proof/oracle checkpoint; candidate-source clearance is pending the emitted implementation and compiler evidence.

## Distinct algorithm and precision contract

Existing compressed SK stores a512-byte low plane and256-byte code plane per8-row×64-K unit. The4-bit code retains sign and a clamped3-bit exponent-window offset. Dedicated escape warps read the ring X and per-CTA FP32 deltas, compute delta*X outside the HMMA chain, release ring credits, and join through cbar/correction storage. The proposal leaves the768-byte stream and main geometry intact, appends a shared unit header, and restores each escaped BF16 B-fragment's exponent bits before the existing `mma.sync`. The correction-reader threads and their count/buffer/join can then disappear only because their work is replaced by exact weight reconstruction.

This differs from failed pair7/denser-code consumers (different weight encoding), failed cgemv integrated-escape warp0 (still FP32 correction work), and existing register-direct RD (different16-row/K-lane ownership and global register-prefetch schedule). It is not evidence that those previous kernels were mismeasured, nor does format correctness forecast a speedup.

For raw BF16 bits b and window start0<=base<=120, let e=(b>>8)&127, d=clamp(e-base,0,7)+base. The12-bit decoder produces `(b&0x80ff)|(d<<8)`. XOR with `(e^d)<<8` returns b bit-for-bit. Sign bit15 and low bits0..7 are invariant. The xor is E7, not the full IEEE exponent (whose low bit already lives in the preserved low byte). All65536 patterns×121valid windows pass, including signed zeros, subnormals, infinities and NaN payloads AS BITS. This does not claim finite numerical behavior for NaN weights; the proposed production offer remains finite-only.

Moving correction products into HMMA changes FP32 accumulation order compared with the old separate FP32-delta sum. Therefore bit-exact reconstructed weights do NOT imply bitwise SK10 output parity. Require the existing independent FP64 projection oracle and BF16 output boundary gates, plus same-input control comparison for performance. Do not weaken thresholds to make the new path pass.

## Independent lane mapping

For a row/K location, let cta=row//R and local row r=row%R. Tile=r//8, g=r%8, wr=tile//NT, nt=tile%NT. Unit=Kcol//64; stage=unit//(WSK*UPW), wk=(unit%(WSK*UPW))//UPW, uu=unit%UPW. Within a64-K unit:

- lane=4*g+((Kcol%8)//2)
- register=2*((Kcol%64)//16)+((Kcol%16)//8)
- half=Kcol%2
- consumer warp=wr*WSK+wk

The inverse returns exactly the original row/K. This is derived from native `tile_values` and mma B-fragment order. RD's row-pair mapping cannot substitute for it.

The proposed uint16 patch record is lane bits0..4, register5..7, half8, E7-XOR9..15. Exhausting32*8*2*128 records yields every uint16 value uniquely. A header carries10-bit count (valid0..512) and22-bit global record start. Independent boundary tests cover count0/1/511/512 and maximum start. Runtime packer must additionally prove start+count within its actual list and reject overflow before launch. Each unit's lane/register/half address must appear at most once; repeated XOR on duplicate metadata would cancel.

`tools/test_sk_sparse_repair_independent.py` contains the independent scalar oracle and4tests, passing3.623s locally. Geometry coverage includes qkv/o/pool shapes with splitK, NT and UPW variations, two CTAs and two stages. Empty metadata, all512 possible escaped elements and last stage/CTA address ownership are covered by the bit/address domain, not yet an execution test of the emitted packer.

## Ordering obligations before source clearance

1. Every header needed for a unit, compressed lw/cw value, and X fragment must be loaded before that consumer's existing early empty arrive. Immutable global patch records may be loaded after that arrive; no additional ring load may do so.
2. Header bytes must be included in the stage bulk-copy address, stride, expected transaction size, alignment and allocation. Header slots must not overlap X, weight planes, barriers or epilogue aliasing.
3. Header count is warp-uniform; the record loop is warp-uniform and only the XOR is lane-masked. MMA participation remains converged.
4. Removing NESC readers changes EMPTY_COUNT to the actual NW remaining readers and thread count to NW+producer. Keep every remaining consumer's proxy fence/syncwarp/arrive, all producer acquire/phase ordering, final reduction barriers and cluster exits.
5. Retain pbar or remove every dependent original table read. New patch-list/rowmap packing must conditionalize all old2*pmax address uses and preserve original balanced-row permutation. All metadata is immutable and owned for the full graph lifetime.
6. No source-level proof substitutes for actual selected CUBIN HGMMA/HMMA ordering, spills, dynamic shared and admission. No check is eligible before source/host review and root approval.

## Release/stop scope

Pending exact implementation review; this checkpoint is not clearance. Root owns any check after independent source/CPU/compiler gates. The owner must compare actual full30-cell projection score (and layer transfer separately), with private independent controls, fresh inputs and FP64/BF16 gates. A reduced correction buffer or thread count alone does not establish latency improvement. Fail any pointer/count/ownership/finite/replay gate, worse unsupported fallback, or whole-score/noise gate rather than promote an isolated kernel win.

## Exact emitted prototype review — 2026-11-01 addendum

Source/API review now finds **no remaining algorithm, packing or reader-lifetime blocker** for the bounded prototype at exact artifact SHA256 `4907db9cd12e47269f79c2d1e08e5362c537225ee654cccaeb1176cb9d1de3d7`,309983bytes, projection-owner `.build/today-proj/sparse-xor.py`, generated by `tools/proj_sparse_xor.py`. Runtime CUDA literal SHA256 is `8cdbb4a99f226beb3f96ca662ae9d260bb7e002ef697337d8f80c075ccaa5068` (88703bytes). Original C2 source is pinned `5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4`. Normal entry remains original C2; this separately exposed prototype is not a promoted candidate.

The first emitted prototype had a concrete host bug: variant0 expected3values from pinned C2 `pack_layer['tabs']`, which actually returns2. My actual-control fixture reproduced ValueError on every offered geometry; the owner fixed it to `(tab,pmax)` before the final tests. Initial fixture setup was also updated for the owner's new `_xor_pack_bits` helper and a sparse population safely below.005; these fixture errors were not candidate math failures.

Final seven independent tests pass in14.483s on Torch2.10CPU. Four are the scalar format/ownership tests above. Three execute actual AST-extracted packer functions from the exact emitted artifact and independently interpret the returned byte stream rather than call its decoder:

- All ten named geometries (`qkv8F`, `qkv16F`, `qkv32`, `o16F`, `o32F`, `pool16F`, `pool32F`, `unp16F`, `unp32F`, `slice8F`) reconstruct every BF16 weight through the independent SK row/K inverse, for2stages and at least2CTAs. Original control and repaired qkv row maps agree exactly, including balancing. Every patch index is consumed once; no lane/register/half collision, header overflow or dropped element is accepted.
- Actual internal bit-transport core reconstructs a dense matrix containing all65536 BF16 bit patterns and units with512patches. This intentionally bypasses the public finite/sparse offer, testing representation bounds rather than proposing dense/nonfinite GPU execution.
- Public packing accepts empty patch lists (pmax0), and both arms reject NaN weights. Signed-zero/subnormal patterns are included in sparse finite tests. Empty headers and odd patch-list padding are covered by actual packing.

Reproduction (WSL; replace ARTIFACT with the exact generated source):

```sh
SK_SPARSE_ARTIFACT=ARTIFACT /root/.cache/okc-host-validate/bin/python -m unittest discover -s tools -p 'test_sk_sparse_repair*.py' -v
```

The actual packer test explicitly skips if no artifact is provided; the archived run sets it and executes all7tests with no skips. Transcript: `evidence/2026-11-01-sk-sparse-repair-independent/cpu-tests.log`.

### Actual source/lifetime findings

Header count/start and record layout match the scalar proof. `nonzero` gives one record per unique escaped element, lexicographically grouped by unit, and cumulative counts point into the immutable global uint16 list. Total record count is below2^22; unit counts are<=512. An even16-bit padding length permits an int32 table prefix; `pmax` becomes its word offset, followed by the exact original-row map. The kernel reads the row map from global `ptab[pmax+rb*R+row]`; all old shared-table/correction readers are compiled out for SX. SX-specific host resource/allocation calculations exclude the original shared table consistently.

The final owner revision moves SX decode/repair BEFORE `load_a`, reducing live A registers. The sequence is: original full-barrier wait; load compressed lw/cw and every unit header; restore B from immutable global patch records; load A; retain the original proxy fence/syncwarp/empty arrive at its original early/late location; original MMA calls. No SX ring or header read follows the old release, and all main-reader fences remain. This is stronger lifetime ordering than the initial after-A design and requires no new producer dependency.

Stage payload appends16-byte-aligned headers, and the enlarged WBY participates in wsrc/wstride, bulk-copy byte count and full expected transaction size; X transport is unchanged. SX NESC is0, actual launched threads are `(NW+1)*32`, and empty count isNW, exactly the remaining consumer warps. No removed warp can still hold a ring credit. Original consumer joins, final reduction barrier and producer progress path remain. The cbar storage/init is retained but its wait/correction work is compiled out, so no new unmatched wait exists. Pbar table transfers/waits are inactive in SX. SX is statically restricted to compressed, non-HL, unclustered, unfused, non-global-escape configurations.

The chosen4096flag is unused elsewhere in the EXACT pinned C2 CUDA body; its TU is unconditionally768for compressed mode. This matters because another tracked development source uses4096as a codec-factorial bit; that other source must not be substituted under this proof. The original C2 literal pin and fail-closed transform anchors are therefore part of the clearance.

The owner reports a successful sm90a device-only compile at `/root/.cache/okc-sparse-xor-before-a`; I inspected the emitted source/manifest identity but did not independently rerun that compiler or CUDA host link. Local compilation is not actual H200 admission, capture, correctness or latency. The current separate returned runner owns a private diagnostic tensor and out allocation; packed weight/table storage is immutable and may be shared read-only. A future matched harness must keep writable diagnostics/outputs/captures independent and preserve rejection/fallback coverage. Root owns host integration and any actual check decision.

**Scope of clearance:** exact prototype algorithm, actual pack/stream mapping and reader-lifetime source proof only. No speedup, H200 correctness, resource-admission, release, or public-submit clearance. The frozen full-score gates and independent numerical/replay tests remain prerequisites; any further emitted-source change requires a delta review.

## Local RTX exact-helper execution — bounded follow-up

Root authorized one no-quota local GPU bit-operation check. `tools/sk_sparse_repair_local_gpu.py` extracts the exact `sx_lds32`/`sx_repair` helper slice from the emitted CUDA literal and requires helper SHA256 `61920951c9e7734be27dfc032b0760d0d5fc465e444a55fc1883db27968d1a3e`, independently matched to the retained4907 local-compile source. It supplies a one-warp-per-unit uint32 wrapper, immutable uint16 record array and unit headers. No SK/TMA kernel or matrix arithmetic runs.

**PASS on local NVIDIA GeForce RTX5070, capability12.0, PyTorch2.10.0+cu128/CUDA12.8.** NVRTC12.8 compiled sm120 CUBIN; CUDA driver loaded and launched it using Torch's current context/stream. No host compiler, new toolchain or environment migration was needed. Three eager banks and20CUDA-graph replays with changed input/header/record contents produced exact uint32 equality on all23observations. Output is poisoned inside the captured graph before the helper launch.

Each bank contains128units covering all65536BF16 bit patterns, plus four explicit units with0/512/511/1patch counts (the singleton targets last lane31/register7/high half). Every lane/register/half field is exercised. Three cyclically permuted bit banks change exponent XOR values, unit headers and record contents while reusing fixed pointers/capture; every replay's final output is checked against its own CPU reference. This checks finite/subnormal/NaN payload BIT transport without interpreting those values numerically.

The first execution correctly stopped before CUDA on the old whole-artifact pin because the owner had regenerated host code. The final probe pins the exact helper independently; input artifact SHA is `24440c2d36469858da564aab3d59fd50c35e428dd758ff59b2e338304c626258`, while the helper is byte-identical to the reviewed4907 source. This result does not clear unrelated24440changes. An initial CUDA-driver load returned201(no current context); creating an ordinary Torch CUDA allocation before module loading fixed initialization. Neither initial failure was a helper output mismatch.

Raw device source, sm120CUBIN, NVRTC log and result are retained under `evidence/2026-11-01-sk-sparse-repair-local-gpu/`. Device source SHA `ef46390cbe0f7cd57798694b6239ef43b4a48c5017e3b043851d1daa23d5c8ee`; CUBIN SHA `53dd6eb6edc841e0c2cdf870f3e8dcf0061a94aa728a984bda25a9f68fb10944`. The successful script's compile/setup/validation wall time1.867s is recorded solely for reproducibility, not a kernel performance result.

Reproduce in the existing WSL CUDA environment:

```sh
/root/.cache/okc-graph-cu128/bin/python tools/sk_sparse_repair_local_gpu.py ARTIFACT OUTPUT_DIR
```

**Limits unchanged:** no full projection/FP64 numerical test, Hopper shared-ring ordering, H200 capture/admission, throughput, latency or architecture-transfer claim. No H200 quota or public action used. Root retains all official measurement and release decisions.

## 2b537518091248f1 dependency wording correction

See `2026-11-01-sk-xor-release-terminal.md` for the actual losing H200 result and explicit correction to the4907addendum's “requires no new producer dependency” phrase. Safety/acyclicity does not mean free pipeline overlap: immutable global-record repair before A/empty can defer stage credit and potentially add producer backpressure. This is a source possibility, not causal stall quantification. MT2/UPW2 late-arrive cells retain release after MMA, so an after-A relocation is not universally after-release. No retry or layer port follows from the old wording.
