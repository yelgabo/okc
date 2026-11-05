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

## Step 4: paired A/B (box 02:54-03:05 UTC, 4 rounds, base then port each round, --runs 60, every check PASS)

Net us, median of 4 runs (tools/attp_ab_table.py /root/okc/.build/box/attp):

| cell | base | base range | port | port range | port - base |
|---|---:|---|---:|---|---:|
| decode 1 sliding | 4.67 | 4.58-5.06 | 4.82 | 4.54-4.90 | +0.14 |
| decode 1 full | 11.79 | 11.68-11.87 | 11.76 | 11.49-12.22 | -0.03 |
| decode 16 sliding | 8.10 | 7.86-8.14 | 8.02 | 7.68-8.14 | -0.08 |
| decode 16 full | 87.17 | 86.77-87.47 | 87.24 | 87.10-87.46 | +0.07 |
| decode 32 sliding | 10.34 | 10.21-10.50 | 10.34 | 10.19-10.56 | -0.01 |
| decode 32 full | 167.32 | 166.80-167.70 | 167.49 | 167.26-167.79 | +0.17 |
| verify 1 sliding | 6.67 | 6.40-6.88 | 7.07 | 6.66-7.17 | +0.40 |
| verify 1 full | 14.52 | 14.46-15.09 | 14.86 | 14.56-14.94 | +0.34 |
| verify 16 sliding | 11.41 | 11.36-11.92 | 11.80 | 11.31-12.11 | +0.39 |
| verify 16 full | 100.54 | 99.92-100.56 | 100.70 | 100.11-100.98 | +0.16 |
| verify 32 sliding | 15.05 | 14.91-15.57 | 15.54 | 15.04-15.65 | +0.50 |
| verify 32 full | 182.72 | 182.66-183.38 | 182.53 | 182.45-182.66 | -0.19 |

Only decode 16 / 32 sliding run the edited instance (L = 1, mode 1). Their change is -0.08 and -0.01 us, inside
the run ranges. The verify cells (L = 8 modes, byte-identical code paths) moved +0.3 to +0.5 us, so the A/B's
shared drift is larger than the target effect. Verdict: no measurable standalone gain; not a submission
candidate. The layer's 1.19 us came from PDL early-launch crowding, which the standalone launch lacks, even
though the standalone base also packs 3 CTAs per SM (per_sm 3 -> 2 confirmed).

## Next

None for this port. Closed; add a ruled-out row if the lead agrees.
