#!/usr/bin/env python3
"""ly_asp: the verify 1 sliding attention (one request, q_len 8, one KV pair) launched with N splits instead of the
plan's 33, through a _TUNED7 entry (nc 1, coop 1). With N < 33 every split holds more than one 64-key tile, so the host
picks the multi-tile speculative build attn_kernel<8, false, 11> (the verify 1 full instance) instead of the one-tile
<8, false, 15>. No CUDA source changes.

    python3 tools/dl_ly_asp_patch.py --splits N --src muse_decode_layer/cand/ly_stk1.py --out muse_decode_layer/cand/X.py
"""
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    n = int(arg("--splits"))
    src = ROOT / arg("--src", "muse_decode_layer/cand/ly_stk1.py")
    dst = ROOT / arg("--out")
    text = src.read_text()
    old = "_TUNED7 = {\n"
    if text.count(old) != 1:
        sys.exit(f"_TUNED7 anchor found {text.count(old)} times")
    text = text.replace(old, old + f"    (1, 8, True, 1): dict(n_splits={n}, nc=1, coop=1),   # asp: verify 1 sliding split count\n")
    text = f"# ly_asp: {src.name} with the verify 1 sliding attention at {n} splits (tools/dl_ly_asp_patch.py).\n" + text
    dst.write_text(text)
    py_compile.compile(str(dst), doraise=True)
    print(dst, hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
