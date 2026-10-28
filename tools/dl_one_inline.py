"""Copy the one-launch attention kernel from muse_attention_decode/attn_one.cu into
muse_decode_layer/submission.py's _ONE_SRC string.

The device code goes into a nested `one` namespace inside the anonymous one, so its names (D,
Params, cp_async16, Instance, ...) cannot collide with the other attention kernels' if the block
is ever compiled into the layer's main extension. The INSTANTIATE notes and the torch host
functions after the namespace are dropped; the layer's own host wrapper (_ONE_HOST_SRC) builds
only the instances its plan uses.

The layer launches attention after a qkv projection that triggers its dependents early, so the
copy also gets the two `griddepcontrol.wait` lines the port guide asks for (before the query
group, whose rows are the projection's output and whose constants group reads the epoch, and
before the flat kernel's epoch read). Each insertion needs its anchor verbatim; a changed
kernel fails here instead of compiling without the wait. Without the PDL launch attribute the
instruction returns at once, so the plain launch is unchanged.

    python3 tools/dl_one_inline.py [path/to/attn_one.cu]
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
CU = ROOT / "muse_attention_decode/attn_one.cu"

WAIT = 'asm volatile("griddepcontrol.wait;" ::: "memory");'

# (anchor, lines inserted before it); the anchors are the first lines of the blocks that must
# follow the wait.
PATCHES = [
    (
        "        // Query rows, the block's new key/value rows (if this range may hold them),\n",
        "        // Everything above reads inputs. The query rows are the qkv projection's output\n"
        "        // and the constants group reads the epoch the previous launch wrote, so under PDL\n"
        "        // both follow the wait (which returns at once on a plain launch).\n"
        f"        {WAIT}\n",
    ),
    (
        "        // The previous launch wrote this epoch (stream order makes it visible); a coherent\n",
        "        // Under PDL the previous launch's writes are visible only after the wait.\n"
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
    src = SUB.read_text()
    code = device_code(cu.read_text())
    new, n = re.subn(r'_ONE_SRC = r""".*?"""', lambda _: f'_ONE_SRC = r"""\n{code}"""', src,
                     flags=re.S)
    if n != 1:
        raise SystemExit("no _ONE_SRC block in submission.py")
    SUB.write_text(new)
    print(f"inlined {len(code.splitlines())} lines from {cu}")


if __name__ == "__main__":
    main()
