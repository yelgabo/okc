#!/usr/bin/env python3
"""Consumer-side pooler + router timeline: tools/dl_ly_prtl_patch.py with the stage clocks taken by consumer warp 0.

Slots 36 + s: consumer warp 0's full-barrier return for stage s (the later of the landing and the warp being ready);
slots 16 + s: consumer warp 0 done with stage s (pooler stages at the end of the body, interleaved FP32 router stages
before their continue). The producer's issue stamps and escape warp 0's landing stamps are dropped. Everything else
(entry, wait, x issue, loop exits, router partial, epilogue, SM id) is as in dl_ly_prtl_patch.py. The stamps sit in
the consumer loop, so this build's loop SASS differs from the base by a predicated globaltimer read and store per
stage in warp 0: a diagnostic, never a candidate.

    python3 tools/dl_ly_prtlc_patch.py --src muse_decode_layer/cand/ly_fabq.py --out muse_decode_layer/cand/ly_fabq_prtlc.py
"""
import hashlib
import pathlib
import py_compile
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dl_ly_prtl_patch as tlp  # noqa: E402

DROP = ("SK2_TLS(16 + st);\n    for (int st", "if (st < 20) SK2_TLS(16 + st);", "SK2_TLS(36 + st);\n          __syncwarp", "SK2_TLS(36 + st);\n        if ((st % NESC)")
K_EDITS = [(o, n) for o, n in tlp.K_EDITS if not any(d in n for d in DROP)]
K_EDITS.append(("    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();\n  }\n",
                "    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();\n"
                "    if (warp == 0 && lane == 0 && st < 20) SK2_TLS(16 + st);\n  }\n"))
# the gle transform's Python literals (applied to bfp_SK2_SRC at runtime): the wait-return stamp after the main
# loop's full wait and the router-stage done stamp before its continue
F_EDITS = list(tlp.F_EDITS) + [
    ('         "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\\n    const int slot = st % NSTAGE;\\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\\n"\n',
     '         "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\\n    const int slot = st % NSTAGE;\\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\\n"\n'
     '         "    if (warp == 0 && lane == 0 && st < 20) SK2_TLS(36 + st);\\n"\n', 1),
    ('         "      __syncwarp();\\n"\n         "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\\n"\n         "      continue;\\n"\n',
     '         "      __syncwarp();\\n"\n         "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\\n"\n'
     '         "      if (warp == 0 && lane == 0 && st < 20) SK2_TLS(16 + st);\\n"\n         "      continue;\\n"\n', 1),
]


def main():
    src = pathlib.Path(sys.argv[sys.argv.index("--src") + 1])
    dst = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])
    raw = src.read_bytes()
    print("base", src.name, hashlib.sha256(raw).hexdigest())
    out = tlp.patch(raw.decode(), K_EDITS, F_EDITS)
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
