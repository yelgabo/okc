#!/usr/bin/env python3
"""One-knob variants of the public layer file for box sweeps: each NAME OLD NEW triple writes
muse_decode_layer/cand/rv_k_NAME.py from g1b4_rs.py (01426b89) with OLD replaced once (it must occur exactly once).

    python3 tools/rv_knob.py NAME 'OLD' 'NEW' [NAME 'OLD' 'NEW' ...]
    python3 tools/rv_knob.py --base FILE NAME 'OLD' 'NEW' ...   (combinations: start from another variant)
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RS_PIN = "01426b891ea12af13e8f713c9ca3a9f47900a07798212e53f583ed0005ff7f07"
args = sys.argv[1:]
base = ROOT / "muse_decode_layer/cand/g1b4_rs.py"
if args[:1] == ["--base"]:
    base, args = pathlib.Path(args[1]), args[2:]
src = base.read_text()
if base.name == "g1b4_rs.py" and hashlib.sha256(src.encode()).hexdigest() != RS_PIN:
    raise SystemExit("g1b4_rs.py is not 01426b89")
edits = {}
for i in range(0, len(args), 3):
    edits.setdefault(args[i], []).append((args[i + 1], args[i + 2]))
for name, pairs in edits.items():
    out = src
    for old, new in pairs:
        if out.count(old) != 1:
            raise SystemExit(f"{name}: {old!r} occurs {out.count(old)} times")
        out = out.replace(old, new)
    dst = ROOT / f"muse_decode_layer/cand/rv_k_{name}.py"
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.relative_to(ROOT), hashlib.sha256(out.encode()).hexdigest())
