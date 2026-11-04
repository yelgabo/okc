#!/usr/bin/env python3
"""ly_stk6 = ly_wsrel1_sc (ly_mpdl + leader-only doneA release) + ly_gbm's two MAIN-build MFIN lines.

The two gbm edits are the ones EAT's solo agent shipped as best_1 on ly_mpdl (okc-eat tools/dl_solo_gbm_patch.py,
box verify 16 / 32 -1.2 to -1.9 us, stress PASS) and that moe-gbm proved on ly_stk4. They touch MOE_CUDA_SRC's
MFIN gate and final-pass poll only; ly_wsrel1_sc changes the ws_helper credit, a different site.
"""
import hashlib, pathlib, sys

SRC = pathlib.Path("muse_decode_layer/cand/ly_wsrel1_sc.py")
OUT = pathlib.Path("muse_decode_layer/cand/ly_stk6.py")
BASE_SHA = "ef68943633ee037e16a4bab0bd4a971b3d7fce74782eec092ba937720e6d07a3"
EDITS = [
    ("#if OKC_WGS && !OKC_B4FIN\n",
     "#if OKC_WGS && !OKC_B4FIN && !OKC_RFIN  // gbm: the RFIN (MAIN) build keeps MFIN; its finisher share releases the expert counter\n"),
    ("          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }\n",
     "          while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }\n"),
]
raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "base SHA mismatch"
text = raw.decode("utf-8")
for old, new in EDITS:
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)
text = "# ly_stk6: ly_wsrel1_sc + ly_gbm (MAIN-build MFIN, unbounded acquire poll).\n" + text
OUT.write_text(text, encoding="utf-8")
print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
