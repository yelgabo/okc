# Lane attention-port: checkpoint

Started 2026-11-04 18:35 PST. Hard stop 21:00 PT.

## Base

- Standalone public file: `.build/repeats/attention_base.py`, SHA256 aa3535d0f1fbb78c92da7e8628d264392c440a17ba756a2297d7566ee46e27e1 (292,419 bytes), kernel `_ONE_SRC` (attn_one.cu), one launch per call, no PDL attribute.
- Layer file: `muse_decode_layer/cand/ly_stk7c.py`, SHA256 920587d5b89153f10addfc50627008611f5f873d440eed89db7e950374887c5e, kernel `ATT7_CUDA_SRC`.

## Step 1: what changed in the layer's attention since 11-03

Comment-stripped diff of ATT7_CUDA_SRC, g1b4_pdl8d (11-03 base) against ly_stk7c: the only code change is ly_a16o2
(`OKC_A16SM 81920`: the q_len 1 plain per-request launch `attn_kernel<1,false,1>`, decode 16 / 32 sliding,
reserves 80 KB dynamic shared memory so two CTAs share an SM). Python-side attention lines: only PDL edges
(`_bfq_attn_edge`, the "un" edge), which are layer-only and skipped. ly_aqd was held off the stack; every other
ly_a* row in docs/ruled-out.md lost.

Mechanism caveat: in the layer the 3-per-SM packing came from the PDL early launch while qkv still held SMs.
The standalone kernel launches with no predecessor (no PDL attribute), so the cause may be absent. The standalone
analogue is `attn_one_kernel<1, 1>` (mode 1; decode 16: 16 splits x 16 = 256 CTAs, decode 32: 8 x 32 = 256 CTAs,
`_ONE_TARGET` 264), the only mode-1 L = 1 build used by the scored shapes.

## Next

Build muse_attention_decode/cand/attn_port.py with tools/dl_attp_patch.py (Instance<1,1> launch bytes 81,920),
print per_sm for both files on the box, run the 16 tests, then paired A/B against attention_base.py.
