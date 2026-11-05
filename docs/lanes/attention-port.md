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

## Step 2: candidate built (18:43 PST)

- `tools/dl_attp_patch.py` (7 edits, each anchor exactly once, pinned to base aa3535d0): Instance<L, MODE> gains
  `LBYTES = (L == 1 && MODE == 1 && ONE_A16SM > BYTES) ? ONE_A16SM : BYTES` (ONE_A16SM 81920) and uses it in the
  smem attribute, the occupancy query and both launch paths. Mode 1 is not cooperative, so the plan is unchanged.
- `muse_attention_decode/cand/attn_port.py`: 292,648 bytes, SHA256 341c0bd976fb4e040e5d0aa6340eef290c23bc86db7e10193de4959ae7f77334.
- `tools/attp_tests.py`: the 16 TASK.md tests plus the residency table (per_sm) for one file, GPU lock in-process.
- Box: /root/okc/muse_attention_decode/cand/attn_port.py (sha matches), logs in /root/okc/.build/box/attp/.

## Step 3: box run in progress (started 18:48 PST)

First attempt failed: the venv's ninja is not on PATH when /root/venv/bin/python is called directly, so every
extension was "unavailable"; those runs were killed and their logs deleted. `/root/okc/.build/box/attp/run.sh`
(PATH=/root/venv/bin first) runs the 16 tests on port then base, then 4 rounds of bx_box_run (base, port
alternating, --runs 60, --lock), JSONs ab_<side>_<r>.json, progress in ab.progress.

Tests (tools/attp_tests.py, seeds 1000+i): attn_port.py 16/16 PASS, attention_base.py 16/16 PASS. Residency of
attn_one_kernel<1, 1>: base per_sm 3 (resident 396), port per_sm 2 (resident 264), so the edit is live in the
standalone launch too. Mode 1 ran in 3 of the 16 tests (plan (32, 1, 1)).

## Next

After ALLDONE in /root/okc/.build/box/attp/ab.progress: `python3 tools/attp_ab_table.py /root/okc/.build/box/attp`
on the box; decide on decode 16 / 32 sliding (the only cells the edit touches; the other 10 are controls).
