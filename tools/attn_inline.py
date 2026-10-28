"""Copy muse_attention_decode/attn_kernel.cu into submission.py's _ATTN_SRC string.

The judge sees only submission.py, so the kernel source lives inline there; the .cu file is the
copy that gets edited.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_attention_decode/submission.py"
CU = ROOT / "muse_attention_decode/attn_kernel.cu"


def main():
    src = SUB.read_text()
    cu = CU.read_text()
    new, n = re.subn(r'_ATTN_SRC = r""".*?"""', lambda _: f'_ATTN_SRC = r"""\n{cu}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit("no _ATTN_SRC block in submission.py")
    SUB.write_text(new)
    print(f"inlined {len(cu.splitlines())} lines")


if __name__ == "__main__":
    main()
