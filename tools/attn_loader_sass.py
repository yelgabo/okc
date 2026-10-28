"""Count the instructions of attn_ws_kernel's loader-warp loop in a cuobjdump -sass dump.

    python3 tools/attn_loader_sass.py .build/ws/k.sass <kernel tag> [<kernel tag> ...]

A kernel tag is the template part of the mangled name, e.g. ILi7ELi3ELi0ELi4ELi16E for
attn_ws_kernel<7, 3, 0, 4, 16>. The loader loop is the largest backward branch whose body holds
LDGSTS (one outer iteration covers PF = 4 tiles). For each kernel it prints two counts by class:

- static: every instruction in the loop body (a loop with no EXIT and no wgmma, so the math
  warps' code is never counted);
- path: the instructions one steady-state outer iteration executes with page size 1 and full
  tiles. That walk takes a forward branch only if it skips a division (its body holds I2F.RP,
  the generic page-size path) or follows a successful mbarrier try-wait (SYNCS.PHASECHK), and
  falls through every other forward branch (first-NS-tiles, tail, dead-CTA and divergence
  checks). The taken branches are listed so the walk can be audited.
"""

import re
import sys
from collections import Counter

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
CLASSES = [
    ("int/addr", r"^(IMAD|IADD3|LEA|SHF(?!L)|LOP3|IABS|VIADD|IMNMX|SEL|PRMT|MOV|UMOV|ULDC|LDC|S2R|"
                 r"CS2R|ISETP|UISETP|UIADD3|UIMAD|USHF|ULOP3|ULEA|USEL|R2UR|PLOP3)"),
    ("i2f/f2i/mufu", r"^(I2F|F2I|MUFU)"),
    ("cp.async", r"^(LDGSTS|LDGDEPBAR|DEPBAR)"),
    ("ldg (page table)", r"^LDG"),
    ("shfl", r"^SHFL"),
    ("lds/sts", r"^(LDS|STS)"),
    ("f16 convert", r"^F2FP"),
    ("mbarrier/fence", r"^(SYNCS|MEMBAR|FENCE)"),
    ("branch", r"^(BRA|BSSY|BSYNC|WARPSYNC|YIELD|EXIT|CALL|RET)"),
]


def kernel_lines(path, tag):
    out, on = [], False
    for line in open(path):
        if "Function :" in line:
            on = "attn_ws_kernel" in line and tag in line
            continue
        if on:
            m = LINE.search(line)
            if m:
                out.append((int(m.group(1), 16), m.group(2)))
    return out


def strip_pred(text):
    return re.sub(r"^@!?U?P[T0-9]+\s+", "", text)


def opcode(text):
    return strip_pred(text).split()[0]


def classify(op):
    for name, pat in CLASSES:
        if re.match(pat, op):
            return name
    return "other"


def loader_loop(ins):
    best = None
    for k, (addr, text) in enumerate(ins):
        m = re.match(r"(@!?P\d\s+)?BRA\s+(0x[0-9a-f]+)$", text)
        if not m:
            continue
        target = int(m.group(2), 16)
        if target >= addr:
            continue
        ops = [opcode(t) for a, t in ins if target <= a <= addr]
        if (any(o.startswith("LDGSTS") for o in ops) and "EXIT" not in ops
                and not any("GMMA" in o for o in ops)):
            if best is None or addr - target > best[1] - best[0]:
                best = (target, addr)
    return [(a, t) for a, t in ins if best[0] <= a <= best[1]]


def steady_path(loop):
    index = {a: k for k, (a, _) in enumerate(loop)}
    k, out, taken = 0, [], []
    while k < len(loop):
        addr, text = loop[k]
        out.append(text)
        m = re.match(r"(@!?P\d\s+)?BRA(\.DIV\s+\S+,)?\s+(0x[0-9a-f]+)$", text)
        if m and m.group(2) is None and k + 1 < len(loop):
            target = int(m.group(3), 16)
            if target > addr and target in index:
                body = loop[k + 1:index[target]]
                skipped = " ".join(t for _, t in body)
                prev = loop[k - 1][1] if k else ""
                division = ("I2F.RP" in skipped and "LDG" not in skipped
                            and "SYNCS" not in skipped)
                if m.group(1) is None or division or "SYNCS.PHASECHK" in prev:
                    taken.append(f"{addr:#x} -> {target:#x} ({len(body)} skipped)")
                    k = index[target]
                    continue
        k += 1
    return out, taken


def counts(texts):
    c = Counter(classify(opcode(t)) for t in texts)
    c["total"] = len(texts)
    return c


def main():
    path, tags = sys.argv[1], sys.argv[2:]
    rows = {}
    for tag in tags:
        loop = loader_loop(kernel_lines(path, tag))
        path_texts, taken = steady_path(loop)
        rows[tag] = (counts([t for _, t in loop]), counts(path_texts))
        print(f"{tag}: loop {loop[0][0]:#x}..{loop[-1][0]:#x}; path takes " + "; ".join(taken))
        ops = Counter(opcode(t) for t in path_texts)
        print("  path opcodes: " + ", ".join(f"{o} {n}" for o, n in sorted(ops.items())))
    names = [n for n, _ in CLASSES] + ["other", "total"]
    print(f"{'class':18s}" + "".join(f" {t[-8:]:>8s}s {t[-8:]:>8s}p" for t in tags))
    for n in names:
        print(f"{n:18s}" + "".join(f" {rows[t][0][n]:9d} {rows[t][1][n]:9d}" for t in tags))


if __name__ == "__main__":
    main()
