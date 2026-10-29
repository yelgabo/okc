"""Per-kernel SASS identity and ptxas resources between builds in one gate directory.

    python3 cmp.py <dir> <base> <other> [<other> ...]
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
        ins = []
        for line in f.split("\n"):
            m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(.*?);", line)
            if m:
                ins.append(m.group(1).strip())
        out[key] = ins
    return out


def res(name):
    text = open(f"{d}/{name}.ptxas").read()
    out = {}
    cur = None
    for line in text.split("\n"):
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = re.search(r"[a-z_]+_kernel\S*", m.group(1)).group(0)
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur:
            out.setdefault(cur, {})["spill"] = tuple(int(x) for x in m.groups())
        m = re.search(r"Used (\d+) registers", line)
        if m and cur:
            out.setdefault(cur, {})["regs"] = int(m.group(1))
    return out


warn = {n: [l for l in open(f"{d}/{n}.ptxas").read().split("\n") if re.search(r"C75\d\d|warning.*ptxas|ptxas.*warning", l)]
        for n in sys.argv[2:]}
for n, w in warn.items():
    print(f"{n}: ptxas C75xx/ptxas warnings: {len(w)}", *w[:5], sep="\n  ")
base = funcs(sys.argv[2])
rbase = res(sys.argv[2])
for other in sys.argv[3:]:
    o = funcs(other)
    ro = res(other)
    same = sum(1 for k in base if k in o and o[k] == base[k])
    print(f"{other} vs {sys.argv[2]}: {len(base)} / {len(o)} kernels, {same} SASS-identical")
    for k in base:
        if k not in o:
            print("  missing", k[:100])
            continue
        if o[k] == base[k]:
            continue
        cb = Counter(i.split()[0] for i in base[k])
        co = Counter(i.split()[0] for i in o[k])
        diff = {op: (cb[op], co[op]) for op in set(cb) | set(co) if cb[op] != co[op]}
        print(f"  {k[:110]}\n    instrs {len(base[k])} -> {len(o[k])}, regs {rbase.get(k, {}).get('regs')} -> "
              f"{ro.get(k, {}).get('regs')}, spill {rbase.get(k, {}).get('spill')} -> {ro.get(k, {}).get('spill')}\n"
              f"    opcode count changes: {dict(sorted(diff.items()))}")
    spills = {k: v["spill"] for k, v in ro.items() if v.get("spill", (0, 0, 0))[1:] != (0, 0)}
    print(f"  {other}: kernels with spill stores/loads: {len(spills)}", *[f"{k[:90]} {v}" for k, v in spills.items()],
          sep="\n    ")
