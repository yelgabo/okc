"""Copy muse_attention_decode/attn_ws.cu into submission.py's _WS_SRC string.

The judge sees only submission.py, so the warp-specialized q_len 8 kernel lives inline there;
the .cu file is the copy that gets edited.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_attention_decode/submission.py"
CU = ROOT / "muse_attention_decode/attn_ws.cu"


def main():
    src = SUB.read_text()
    cu = CU.read_text()
    new, n = re.subn(r'_WS_SRC = r""".*?"""', lambda _: f'_WS_SRC = r"""\n{cu}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit("no _WS_SRC block in submission.py")
    SUB.write_text(new)
    print(f"inlined {len(cu.splitlines())} lines")


if __name__ == "__main__":
    main()
