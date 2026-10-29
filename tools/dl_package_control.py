"""Write a package control of the layer: muse_decode_layer/submission.py with one ported
package's generated regions taken back from a base layer source (the scored incumbent), so a
check can attribute a combined candidate's change to its packages.

    python3 tools/dl_package_control.py OUT.py BASE.py --minus attn|moe|proj [...]

Packages and the regions they own (each must be found exactly once in both files):
  attn  _ONE_SRC, _ONE_HOST_SRC (attn_one.cu and its layer wrapper: DCARRY, PK16, VHOIST)
  moe   _MOE1_SRC, _MOE1_HOST_SRC, _MOE1_VARIANTS, _MOE1_MAIN_MIN_M (modes 37, sk8w3s,
        sk8w4s-er and the skinny preamble's copied init block)
  proj  _SKP_SRC, _SKP_GEMV_SRC, _SKP_CPP_SRC, _SKP_WD_SRC, _SKP_WD_GEOM, _SKP_CONFIGS and the
        @SKP_PY@ region (qkv8 geometry, the reworked m = 1 cgemv and its packer)
Everything else (the route kernel, the tuners, the self-tests) stays the candidate's. The
header comment names the control, so the file differs from the candidate and from other
controls. OUT.py belongs in the session scratchpad.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGIONS = {
    "attn": [r'^_ONE_SRC = r""".*?"""', r'^_ONE_HOST_SRC = r""".*?"""'],
    "moe": [r'^_MOE1_SRC = r""".*?"""', r'^_MOE1_HOST_SRC = r""".*?"""',
            r"^_MOE1_VARIANTS = \(.*?\n\)$", r"^_MOE1_MAIN_MIN_M = \d+$"],
    "proj": [r'^_SKP_SRC = r""".*?"""', r'^_SKP_GEMV_SRC = r""".*?"""',
             r'^_SKP_CPP_SRC = r""".*?"""', r'^_SKP_WD_SRC = r""".*?"""',
             r"^_SKP_WD_GEOM = \{.*?\n\}$", r"^_SKP_CONFIGS = \(.*?\n\)$",
             r"# @SKP_PY_BEGIN@\n.*?# @SKP_PY_END@\n"],
}


def region(text, pattern, what):
    found = list(re.finditer(pattern, text, flags=re.S | re.M))
    if len(found) != 1:
        raise SystemExit(f"{what}: {pattern[:40]!r} found {len(found)} times, need 1")
    return found[0]


def main():
    args = sys.argv[1:]
    minus = [args[i + 1] for i, a in enumerate(args) if a == "--minus"]
    pos = [a for i, a in enumerate(args) if a != "--minus" and (i == 0 or args[i - 1] != "--minus")]
    if len(pos) != 2 or not minus or not set(minus) <= set(REGIONS):
        raise SystemExit(__doc__)
    out, base = Path(pos[0]), Path(pos[1]).read_text()
    cand = (ROOT / "muse_decode_layer/submission.py").read_text()
    for pkg in minus:
        for pattern in REGIONS[pkg]:
            old = region(cand, pattern, "candidate")
            new = region(base, pattern, "base")
            cand = cand[:old.start()] + new.group(0) + cand[old.end():]
    first, rest = cand.split("\n", 1)
    out.write_text(first + f"\n# Package control: candidate minus {'+'.join(minus)} (regions from the "
                   "incumbent source), written by tools/dl_package_control.py.\n" + rest)
    print(f"wrote {out}: candidate minus {minus}")


if __name__ == "__main__":
    main()
