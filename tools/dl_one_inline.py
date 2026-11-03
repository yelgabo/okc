"""Keep attention names isolated and order PDL reads after producer publication."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
CU = ROOT / "muse_attention_decode/attn_one.cu"

WAIT = 'asm volatile("griddepcontrol.wait;" ::: "memory");'


PATCHES = [
    (
        "        // Query rows, the block's new key/value rows (if this range may hold them),\n",
        "        // Query rows and the epoch depend on the producer's writes.\n"
        "        // PDL requires the wait before either read.\n"
        "\n"
        f"        {WAIT}\n",
    ),
    (
        "        // The previous launch wrote this epoch (stream order makes it visible); a coherent\n",
        "        // Under PDL the previous launch's writes are visible only after the wait (on a\n"
        "        // plain launch stream order already makes them so).\n"
        f"        {WAIT}\n",
    ),
]


def device_code(cu):
    head, sep, _ = cu.rpartition("}  // namespace")
    if not sep:
        raise SystemExit("no '}  // namespace' line in the kernel source")
    lines = [line for line in head.splitlines() if "// INSTANTIATE:" not in line]
    body = "\n".join(lines).rstrip() + "\n"
    body, n = re.subn(r"^namespace \{$", "namespace {\nnamespace one {", body, count=1,
                      flags=re.M)
    if n != 1:
        raise SystemExit("no 'namespace {' line in the kernel source")
    for anchor, insert in PATCHES:
        if body.count(anchor) != 1:
            raise SystemExit(f"anchor found {body.count(anchor)} times, need 1: {anchor!r}")
        body = body.replace(anchor, insert + anchor)
    if body.count(WAIT) != len(PATCHES):
        raise SystemExit("the kernel source already carries a griddepcontrol.wait; drop PATCHES")
    return body + "\n}  // namespace one\n}  // namespace\n"


def main():
    cu = Path(sys.argv[1]) if len(sys.argv) > 1 else CU
    sub = Path(sys.argv[2]) if len(sys.argv) > 2 else SUB
    src = sub.read_text()
    code = device_code(cu.read_text())
    new, n = re.subn(r'_ONE_SRC = r""".*?"""', lambda _: f'_ONE_SRC = r"""\n{code}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit(f"no _ONE_SRC block in {sub}")
    sub.write_text(new)
    print(f"inlined {len(code.splitlines())} lines from {cu}")


if __name__ == "__main__":
    main()
