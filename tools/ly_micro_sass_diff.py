#!/usr/bin/env python3
"""Compare two tools/ly_micro_sass.py reports: per build, registers, local bytes, kernel instruction count, STG
count by width, MUFU count, and the innermost MMA loops (instruction count and the opcodes whose count changed).

    python3 tools/ly_micro_sass_diff.py BASE_DIR VARIANT_DIR
"""
import collections
import json
import os
import re
import sys

K = "_ZN3okc9moe_fusedILb0EEEvNS_6ParamsE"


def stg(sass_path):
    c = collections.Counter()
    on = False
    for line in open(sass_path):
        if "Function : " in line:
            on = K in line
        if on:
            m = re.search(r"\b(STG\.E[.A-Z0-9]*|MUFU\.[A-Z0-9]+|FSETP|SHFL\.[A-Z]+|BSSY)\b", line)
            if m:
                c[m.group(1)] += 1
    return c


def main():
    a, b = sys.argv[1], sys.argv[2]
    ra, rb = (json.load(open(os.path.join(d, "report.json"))) for d in (a, b))
    for k in ra:
        if k not in rb:
            continue
        ka, kb = ra[k]["kernels"][K], rb[k]["kernels"][K]
        sa, sb = stg(os.path.join(a, f"{k}.sass")), stg(os.path.join(b, f"{k}.sass"))
        ds = {o: sb[o] - sa[o] for o in set(sa) | set(sb) if sb[o] != sa[o]}
        print(f"{k:5s} reg {ka['reg']}->{kb['reg']} local {ka['local']}->{kb['local']} ins {ka['n']}->{kb['n']} "
              f"ops {ds}")
        la = sorted((l for l in ka["loops"] if l["n"] < 800), key=lambda l: int(l["start"], 16))
        lb = sorted((l for l in kb["loops"] if l["n"] < 800), key=lambda l: int(l["start"], 16))
        for x, y in zip(la, lb):
            dm = {o: y["mix"].get(o, 0) - x["mix"].get(o, 0) for o in set(x["mix"]) | set(y["mix"])
                  if y["mix"].get(o, 0) != x["mix"].get(o, 0)}
            if x["n"] != y["n"] or dm:
                print(f"      loop {x['start']} n {x['n']}->{y['n']} mma {x['mma']} {dm}")


if __name__ == "__main__":
    main()
