"""Keep verify-attention symbols isolated inside the shared layer extension."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
CU = ROOT / "muse_attention_decode/attn_ws.cu"


def device_code(cu):
    head, sep, _ = cu.rpartition("}  // namespace")
    if not sep:
        raise SystemExit("no '}  // namespace' line in the kernel source")
    lines = [line for line in head.splitlines() if "// INSTANTIATE:" not in line]
    body = "\n".join(lines).rstrip() + "\n"
    body, n = re.subn(r"^namespace \{$", "namespace {\nnamespace verify {", body, count=1,
                      flags=re.M)
    if n != 1:
        raise SystemExit("no 'namespace {' line in the kernel source")
    return body + "\n}  // namespace verify\n}  // namespace\n"


def main():
    cu = Path(sys.argv[1]) if len(sys.argv) > 1 else CU
    src = SUB.read_text()
    code = device_code(cu.read_text())
    new, n = re.subn(r'_WS_SRC = r""".*?"""', lambda _: f'_WS_SRC = r"""\n{code}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit("no _WS_SRC block in submission.py")
    SUB.write_text(new)
    print(f"inlined {len(code.splitlines())} lines from {cu}")


if __name__ == "__main__":
    main()
