#!/usr/bin/env python3
"""ly_f1f (tools/dl_ly_f1f_patch.py in okc-eat, EAT best_5) ported to the standalone MoE moe_port.py (5542c8a8):
the same moe_helper0 edits, and -DOKC_F1FAST=1 appended to MOE_TINY_DEFINES (decode 1 only).

    python3 tools/moe_f1f_port_patch.py SRC OUT
"""
import hashlib, importlib.util, py_compile, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("f1f", Path.home() / "GitHub/okc-eat/tools/dl_ly_f1f_patch.py")
f1f = importlib.util.module_from_spec(spec)
sys.argv[1:] = ["--base", "x", "--out", "x"]
spec.loader.exec_module(f1f)
text = src.read_text()
a = text.index("_MA_ALT_SRCS = {'moe_helper0': r'''")
b = text.index("'''", a + 40)
seg = text[a:b]
for old, new in f1f.SEG_EDITS:
    seg = f1f.sub(seg, old, new)
text = text[:a] + seg + text[b:]
old = "'-DOKC_FINPF=1', '-DOKC_NOSCL=1')\n"
line = [l for l in text.splitlines(True) if l.startswith("MOE_TINY_DEFINES = (")]
assert len(line) == 1 and line[0].endswith(old), line
text = text.replace(line[0], line[0][: -len(old)] + "'-DOKC_FINPF=1', '-DOKC_NOSCL=1', '-DOKC_F1FAST=1')\n")
out.write_text(text)
py_compile.compile(str(out), doraise=True)
print(out, hashlib.sha256(text.encode()).hexdigest(), len(text.encode()))
