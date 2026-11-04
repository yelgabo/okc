#!/usr/bin/env python3
"""Instruction-level diff of one kernel's first HMMA loop between two SASS dumps from tools/ly_prtl_sass.py.

    python3 tools/ly_prtl_loopdiff.py OUTDIR KEY_SUFFIX [kernel-substring]
e.g. python3 tools/ly_prtl_loopdiff.py .build/prtl/sass_prtl gle Lb1ELb1EEv
Register names are kept; addresses and the gle md5 tag are stripped.
"""
import difflib
import json
import os
import re
import sys


def loop_text(sass_path, report, key, sub):
    kern = [n for n in report[key] if sub in n][0]
    lp = report[key][kern]["loops"][0]
    lo, hi = int(lp["start"], 16), int(lp["end"], 16)
    out, cur = [], None
    for line in open(sass_path):
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = re.sub(r"(sk2_kernel_gle)[0-9a-f]{12}", r"\1", m.group(1))
            continue
        if cur != kern:
            continue
        mi = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?;)", line)
        if mi and lo <= int(mi.group(1), 16) <= hi:
            out.append(re.sub(r"0x[0-9a-f]+", "X", mi.group(2)))
    return out


def main():
    d, k = sys.argv[1], sys.argv[2]
    sub = sys.argv[3] if len(sys.argv) > 3 else "Lb1ELb1EEv"
    rep = json.load(open(os.path.join(d, "report.json")))
    a = loop_text(os.path.join(d, f"ctl_{k}.sass"), rep, f"ctl_{k}", sub)
    b = loop_text(os.path.join(d, f"cand_{k}.sass"), rep, f"cand_{k}", sub)
    print(len(a), len(b))
    for line in difflib.unified_diff(a, b, lineterm="", n=1):
        print(line)


if __name__ == "__main__":
    main()
