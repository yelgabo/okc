#!/usr/bin/env python3
"""Layer MID timeline: writes muse_decode_layer/cand/ly_tl16.py from rv_k_combo2n.py (3ac83fb4).

tools/dl_moe_tl.py with the base changed to the slack-free combo2n and `-DOKC_TLX=1` on the MID build
(decode 16, helper0df, the same helper0 text) instead of TINY2. Original notes follow.


Ports MOE-TL-01's per-CTA globaltimer instrument (tools/moe_tiny_timeline_cta.py, the standalone's Tiny / Tiny2
timeline) into the layer: the same CUDA edits, applied to the layer's helper0 source, and `-DOKC_TLX=1` on the
layer's TINY2 build only (verify 1, 8 tokens). The stamps are taken only when a launch passes a fourth scratch
tensor; the layer's own launches pass three, so the timed path runs every stamp as a not-taken uniform branch.
tools/box_moe_tl8.py makes the extra instrumented launches. MID (helper0df) gets the same source text with
OKC_TLX 0. Fences, barriers, ring and final pass are unchanged.

    python3 tools/ly_moe_tl16.py [--base FILE --pin SHA256 --out FILE]   (default: combo2n -> ly_tl16.py)
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/rv_k_combo2n.py"
DST = ROOT / "muse_decode_layer/cand/ly_tl16.py"
PIN = "3ac83fb437d05308c2871301a5a10c537d9b2ec504d55305403641beec2f5d27"
STANDALONE = ROOT / "tools/moe_tiny_timeline_cta.py"
MARK = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"\n"
TINY2_ADD = 'MOE_MID_DEFINES += ("@moe_helper0",)\n'


def main():
    src, pin, dst = SRC, PIN, DST
    if "--base" in sys.argv:
        src = ROOT / sys.argv[sys.argv.index("--base") + 1]
    if "--pin" in sys.argv:
        pin = sys.argv[sys.argv.index("--pin") + 1]
    if "--out" in sys.argv:
        dst = ROOT / sys.argv[sys.argv.index("--out") + 1]
    raw = src.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        sys.exit(f"{src.name} is not {pin[:8]}")
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
        sys.exit("MID helper define anchor")
    tail = tail.replace(TINY2_ADD, TINY2_ADD + 'MOE_MID_DEFINES += ("-DOKC_TLX=1",)\n')
    out = head + MARK + box["src"] + tail
    dst.write_text(out)
    py_compile.compile(str(dst), doraise=True)
    print(dst.name, hashlib.sha256(out.encode()).hexdigest(), len(out))


if __name__ == "__main__":
    main()
