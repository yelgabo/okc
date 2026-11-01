# Independent stage-local sparse record review

November1,2026. Projection owner requested source/CPU review of the distinct stage-local representation after global-record job2b537518091248f1 lost. Exact prototype `.build/today-proj/stage-xor.py`:438663B SHA256`b726aebd42c0749d3ff736fdb72d667d1143dce90819fc45b785580a1832bc2e`; native literal SHA`998b5efbf26f762cefdcc308b23629fa51dbbdc36569d0a466077aa44a4344da`. Generator `tools/proj_stage_xor.py`, based on exact d23aa17b global prototype/C2 pin5610b8a6. Normal C2 entry remains unchanged. No GPU, public action, production edit or stage-port by this reviewer.

**No remaining source/format/reader-lifetime blocker found for this exact offline prototype.** This is not H200 resource, capture, math or performance qualification and does not reopen the closed global-record variant for retries.

## Actual packing/inverse proof

`sxs_repack` reads each old stage/CTA's unit headers, preserving their actual physical stream order. It constructs each stage-local prefix from that row's unit counts, copies the identical uint16 record sequence selected by each old global start/count into the local arena, and rewrites only the header start field. Header count remains unchanged. Capacity is the maximum ACTUAL sum of unit counts over the complete stage/CTA population; it is neither a tuned bound nor a truncation cap. Record arena bytes are16-byte-aligned2*capacity. Payload bytes and original balanced-row map remain exact; pmax now carries stage capacity and ptab contains rowmap only.

Independent `tools/test_sk_stage_repair_independent.py` executes the actual emitted `sxs_repack` and original pack functions, then independently checks old/new headers, payload, records and rowmap. All ten geometries each pass sparse, empty and one-full512-record-unit fixtures, across7stages and2CTAs. A dense fixture covers all BF16 patterns and large within-stage prefix offsets. Every unit record sequence is byte-equal to the old global slice; every count/start fits the exact max arena; unused padding is zero; source old packs are not mutated. Empty capacity0 uses no arena bytes and no record read. A zero-count unit may point to its stage's one-past-last record, but executes no loop iteration.

Three focused actual-TorchCPU tests PASS in.554s, no skips. Transcript under `evidence/2026-11-01-sk-stage-repair-independent/cpu-tests.log`. Initial discovery also ran the three imported old-global pack tests (six total,12.048s, all pass); the final fixture imports that module without automatically rerunning its TestCase. The earlier full old-global bit/lane proof remains applicable because the record bits, per-unit order and repaired B fragments do not change.

Reproduce with `SK_SPARSE_ARTIFACT=EXACT_STAGE_ARTIFACT /root/.cache/okc-host-validate/bin/python -m unittest discover -s tools -p test_sk_stage_repair_independent.py -v`. Missing artifact explicitly skips; the archived run supplies the exact source.

## Actual shared lifetime and dynamic extent

The appended arena is part of the weight bulk transfer. Runtime WBY includes padded arena bytes; wsrc/wstride, stage bulk-copy bytes and full expected transaction size use that WBY. Runtime SST rounds X reservation+WBY to1024B. Kernel barrier/correction/zrow offsets use the runtime stride, while host allocation and resource queries add NSTAGE*(newSST-oldSST) to the original fixed extent. C::STAGE=false preserves old C2/global extents; static STAGE implies SX and inherited SX restrictions prohibit clustered/HL/fused paths.

Every new `sxs_lds16` is an asm-volatile/memory-clobber shared load inside the warp-uniform current-unit repair loop, before A loading and the existing empty release. Full-barrier completion covers the arena, so a consumer does not depend on a future-stage producer to obtain its metadata. The new immutable arena is not read after that warp arrives empty. Same NW readers, per-lane proxy fences, syncwarp, empty arrivals, final named reduction barrier and producer phase/wait logic remain; source counts and manually inspected addresses/lifetimes are unchanged apart from correctly enlarged offsets. Removal of the old global list does not permit a fence removal.

The repaired register values remain identical to global SX because each unit sees the exact same record order and masks; this preserves its BF16 reconstruction/math contract. Publication/acylcity safety does NOT imply free overlap: shared record loops still occur before stage release, can delay credit, and actual full completion now waits for padded metadata bytes too. MT2/UPW2 late-arrive cells still release after MMA. No latency/stall benefit is claimed.

The maximum possible units*512 count bounds header offsets and capacity; the native host checks it before a valid stage launch. Runtime shared budget remains<=227KiB or the launch is rejected. More arena bytes can cross stride boundaries, lower admission, or exceed capacity; actual H200 weights/resources must decide. There is no permission to truncate metadata, reduce ring depth, change geometry or substitute a different cap to make a case pass.

## Remaining gates and limits

Owner reports successful sm90a compile with zero stack/spills and exact C2/global control opcode/operand identities. This reviewer did not rerun that compiler and makes no actual runtime-resource assertion. Owner's separately generated CPU natural-weight capacities are not the inaccessible2bH200 bank and do not establish real H200 stage sizes or slack.

Before any measured experiment: independently frozen native/resource/host harness, exact original C2 and global-control preservation, same actual supplied weights and fresh inputs, zero local memory and runtime admission/extent checks, original BF16/FP64/one-hot/outlier/nonfinite/fallback/replay gates, full30-cell/six-load score and null/regression rules, complete transport and preparation<=300s. No public layer proposal or source adoption is authorized by this review. Root owns any experiment decision. This is a genuinely different metadata transport, not an after-A placement sweep; the exact global BEFORE-A production loss remains closed.
