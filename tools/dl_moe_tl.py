#!/usr/bin/env python3
"""Layer TINY2 timeline: writes muse_decode_layer/cand/mo_tl8.py from rv_k_combo2.py (7f0a3e36).

Ports MOE-TL-01's per-CTA globaltimer instrument (tools/moe_tiny_timeline_cta.py, the standalone's Tiny / Tiny2
timeline) into the layer: the same CUDA edits, applied to the layer's helper0 source, and `-DOKC_TLX=1` on the
layer's TINY2 build only (verify 1, 8 tokens). The stamps are taken only when a launch passes a fourth scratch
tensor; the layer's own launches pass three, so the timed path runs every stamp as a not-taken uniform branch.
tools/box_moe_tl8.py makes the extra instrumented launches. MID (helper0df) gets the same source text with
OKC_TLX 0. Fences, barriers, ring and final pass are unchanged.

    python3 tools/dl_moe_tl.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/rv_k_combo2.py"
DST = ROOT / "muse_decode_layer/cand/mo_tl8.py"
PIN = "7f0a3e360f085c4f21c1b760d10eaf251d3d379464a1f0556d52828cf92d5f37"
STANDALONE = ROOT / "tools/moe_tiny_timeline_cta.py"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
TINY2_ADD = 'MOE_TINY2_DEFINES += ("@moe_helper0",)\n'


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PIN:
        sys.exit("rv_k_combo2.py is not 7f0a3e36")
    whole = raw.decode()
    assert whole.count(MARK) == 1
    head, rest = whole.split(MARK)
    end = rest.index('\n"""')
    helper, tail = rest[:end], rest[end:]
    tool = STANDALONE.read_text()
    cuda = tool[tool.index("# ---------------------------------------------------------------- CUDA"):
                tool.index("# ---------------------------------------------------------------- host")]
    box = {"src": helper}

    def rep(old, new, count=1):
        n = box["src"].count(old)
        if n != count:
            sys.exit(f"helper0 anchor found {n} times, {count} expected: {old[:100]!r}")
        box["src"] = box["src"].replace(old, new)
    exec(cuda, {"rep": rep})
    if tail.count(TINY2_ADD) != 1:
        sys.exit("TINY2 helper define anchor")
    tail = tail.replace(TINY2_ADD, TINY2_ADD + 'MOE_TINY2_DEFINES += ("-DOKC_TLX=1",)\n')
    out = head + MARK + box["src"] + tail
    DST.write_text(out)
    py_compile.compile(str(DST), doraise=True)
    print(DST.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
