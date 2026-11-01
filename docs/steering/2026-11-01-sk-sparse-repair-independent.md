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
