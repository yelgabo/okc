#!/usr/bin/env python3
"""Per-load medians and per-kernel profile rows of muse_decode_layer receipts (okc status --json), side by side.
   python3 tools/dl_f3_receipt.py NEW.json [BASE.json ...]"""
import json
import re
import sys

SHORT = ("decode 1 sliding", "decode 1 full", "verify 1 sliding", "verify 1 full")


def loads(d):
    return {f"{b['case']['load']} {b['case']['layer']}": b for b in d.get("benchmarks", [])}


def rows(d):
    out = {}
    for p in d.get("profiles", []):
        key = f"{p['case']['load']} {p['case']['layer']}"
        r = {}
        for line in p["report"].splitlines():
            m = re.match(r"^(.*?)\s+([0-9.]+) µs\s+[0-9.]+%\s+(\d+)$", line)
            if m:
                r[m.group(1).strip()] = float(m.group(2))
        out[key] = r
    return out


def main():
    files = sys.argv[1:]
    ds = [json.load(open(f)) for f in files]
    s = ds[0].get("summary", {})
    print("summary", {k: s.get(k) for k in ("status", "passed", "tests", "tests_passed", "score_us", "speedup", "speedup_ci", "error")})
    ls = [loads(d) for d in ds]
    print("\nload median (us)      " + "  ".join(f"{f[-20:]:>20}" for f in files))
    for key in ls[0]:
        vals = [l[key]["median_ns"] / 1000 if key in l else None for l in ls]
        print(f"{key:22} " + "  ".join(f"{v:20.2f}" if v is not None else f"{'-':>20}" for v in vals))
    rs = [rows(d) for d in ds]
    for key in SHORT:
        if key not in rs[0]:
            continue
        print(f"\nprofile {key}")
        names = list(rs[0][key])
        for r in rs[1:]:
            names += [n for n in r.get(key, {}) if n not in names]
        for n in names:
            vals = [r.get(key, {}).get(n) for r in rs]
            print(f"  {n[:58]:58} " + "  ".join(f"{v:7.1f}" if v is not None else f"{'-':>7}" for v in vals))


main()
