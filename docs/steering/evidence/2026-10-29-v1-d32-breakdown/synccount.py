"""Per-kernel counts of synchronization and memory opcodes (predicates stripped) between two builds.

    python3 synccount.py <dir> <base> <other> <opcode regex>
"""
import re
import sys
from collections import Counter

d = sys.argv[1]


def funcs(name):
    text = open(f"{d}/{name}.sass").read()
    out = {}
    for f in re.split(r"\n\s+Function : ", text)[1:]:
        fname = f.split("\n", 1)[0].strip()
        key = re.search(r"[a-z_]+_kernel\S*", fname).group(0)
        c = Counter()
        for line in f.split("\n"):
            m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", line)
            if m:
                op = re.sub(r"^@!?U?P\w+\s+", "", m.group(1).strip()).split()[0]
                c[op] += 1
        out[key] = c
    return out


keys = re.compile(sys.argv[4])
a, b = funcs(sys.argv[2]), funcs(sys.argv[3])
bad = 0
for k in a:
    ca = {o: n for o, n in a[k].items() if keys.match(o)}
    cb = {o: n for o, n in b[k].items() if keys.match(o)}
    if ca != cb:
        bad += 1
        print(k[:70], {o: (ca.get(o, 0), cb.get(o, 0)) for o in sorted(set(ca) | set(cb)) if ca.get(o, 0) != cb.get(o, 0)})
print("kernels with differing counts:", bad, "of", len(a))
