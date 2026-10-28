"""Copy muse_attention_decode/attn_one.cu into submission.py's _ONE_SRC string.

The judge sees only submission.py, so the one-launch kernel lives inline there; the .cu file
is the copy that gets edited (and compiled by tools/attn_ws_quick.sh -s
muse_attention_decode/attn_one.cu, tools/attn_host_check.sh muse_attention_decode/attn_one.cu).
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_attention_decode/submission.py"
CU = ROOT / "muse_attention_decode/attn_one.cu"


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else SUB
    src = target.read_text()
    cu = CU.read_text()
    new, n = re.subn(r'_ONE_SRC = r""".*?"""', lambda _: f'_ONE_SRC = r"""\n{cu}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit(f"no _ONE_SRC block in {target}")
    target.write_text(new)
    print(f"inlined {len(cu.splitlines())} lines into {target}")


if __name__ == "__main__":
    main()
