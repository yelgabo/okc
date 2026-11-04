#!/usr/bin/env python3
"""ly_a16sp: split-count arm for the decode 16 / 32 sliding attention (attn_kernel<1, false, 1>): 8 splits per request
at decode 16 (16 in the base) and 4 at decode 32 (8), one CTA per SM (dynamic shared memory raised above half an SM),
so 128 CTAs land one per SM at both loads. Host only (_TUNED7 entries, nc 1, and the OKC_A16SM size). Stamp arm, not a
candidate unless it wins.

    python3 tools/dl_ly_a16sp_patch.py [--d16 8] [--d32 4] [--smem 122880] [--base F] [--out F]
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "muse_decode_layer/cand/ly_stk3.py"
DST = ROOT / "muse_decode_layer/cand/ly_a16sp.py"
PIN = "a1c488104e48368defce6734f8f87fd29a13acf325a0d92ef05582057cb135f6"


def arg(k, d):
    return type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d


D16, D32, SM = arg("--d16", 8), arg("--d32", 4), arg("--smem", 122880)
if "--base" in sys.argv:
    SRC, PIN = ROOT / sys.argv[sys.argv.index("--base") + 1], None
DST = ROOT / arg("--out", str(DST.relative_to(ROOT)))

EDITS = [
    ("#define OKC_A16SM 81920 ", f"#define OKC_A16SM {SM} "),
    ("_TUNED7 = {\n",
     f"_TUNED7 = {{\n    (16, 1, True, 1): dict(n_splits={D16}, nc=1),   # ly_a16sp\n"
     f"    (32, 1, True, 1): dict(n_splits={D32}, nc=1),   # ly_a16sp\n"),
]


def main():
    text = SRC.read_text()
    if PIN and hashlib.sha256(text.encode()).hexdigest() != PIN:
        sys.exit("base SHA256 mismatch")
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"anchor found {text.count(old)} times: {old!r}")
        text = text.replace(old, new)
    text = (f"# ly_a16sp: ly_stk3 with {D16} / {D32} splits per request at decode 16 / 32 sliding, {SM} B of dynamic "
            "shared memory on that launch (tools/dl_ly_a16sp_patch.py). Stamp arm.\n") + text
    DST.write_text(text)
    py_compile.compile(str(DST), doraise=True)
    print(DST, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
