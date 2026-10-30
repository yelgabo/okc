"""Bring an already gated M1 file (tools/dl_gate_patch.py --m1 output, e.g. layer_m1rt51.py) to
the current M1 gate (census skip, user decision 2026-10-30) without touching anything else:
its M1 gate region is replaced by dl_gate_patch.M1_SRC and the two runner lines get the PDL
launch counter. Rebuilding from 6377b5a would also bring every later change of the tool.

    python3 tools/dl_m1_census_patch.py IN.py OUT.py "HEADER LINE 2"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dl_gate_patch as gp  # noqa: E402

src, out, header = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src).read()
start = "# The single-element negative control in the M1 gate is drawn among elements where the"
if text.count(start) != 1 or "_M1_PDL_LAUNCHES" in text:
    raise SystemExit("IN is not a gated M1 file without the census")
text = gp.cut(text, start, "def _m1_enable(state):", gp.M1_SRC)
for old, new in gp.CENSUS_REPS:
    if text.count(old) != 1:
        raise SystemExit(f"the census runner line is not where expected: {old[:60]!r}")
    text = text.replace(old, new)
lines = text.split("\n")
if lines[0] != "# okc: problem=muse_decode_layer":
    raise SystemExit("line 1 is not the okc header")
lines[1] = header
open(out, "w").write("\n".join(lines))
print("wrote", out, len(text))
