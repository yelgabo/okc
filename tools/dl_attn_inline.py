"""Copy the fused attention kernel from muse_attention_decode/attn_kernel.cu into
muse_decode_layer/submission.py's _ATTN_SRC string.

The decode layer builds one extension, so the attention device code is wrapped in a nested
`attn` namespace inside the anonymous one: its names (STAGES, cp_async16, ...) would otherwise
collide with the projection kernels'. The torch host function and the template instantiation
notes are dropped; the decode layer has its own host wrappers in _ATTN_HOST_SRC.

    python3 tools/dl_attn_inline.py [path/to/attn_kernel.cu]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
CU = ROOT / "muse_attention_decode/attn_kernel.cu"


def device_code(cu):
    head, sep, _ = cu.rpartition("}  // namespace")
    if not sep:
        raise SystemExit("no '}  // namespace' line in the kernel source")
    lines = [line for line in head.splitlines() if "// INSTANTIATE:" not in line]
    body = "\n".join(lines).rstrip() + "\n"
    body, n = re.subn(r"^namespace \{$", "namespace {\nnamespace attn {", body, count=1,
                      flags=re.M)
    if n != 1:
        raise SystemExit("no 'namespace {' line in the kernel source")
    return body + "\n}  // namespace attn\n}  // namespace\n"


def main():
    cu = Path(sys.argv[1]) if len(sys.argv) > 1 else CU
    src = SUB.read_text()
    code = device_code(cu.read_text())
    new, n = re.subn(r'_ATTN_SRC = r""".*?"""', lambda _: f'_ATTN_SRC = r"""\n{code}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit("no _ATTN_SRC block in submission.py")
    SUB.write_text(new)
    print(f"inlined {len(code.splitlines())} lines from {cu}")


if __name__ == "__main__":
    main()
