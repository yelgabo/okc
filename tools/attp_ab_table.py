#!/usr/bin/env python3
"""Per-shape median of bx_box_run net times over the paired A/B runs in a directory
(ab_base_<r>.json, ab_port_<r>.json), printed as a markdown table with port - base and the run spread.

    python3 tools/attp_ab_table.py /root/okc/.build/box/attp
"""
import glob
import json
import os
import statistics
import sys

d = sys.argv[1]
side = {"base": {}, "port": {}}
checks = {"base": set(), "port": set()}
for s in side:
    for p in sorted(glob.glob(os.path.join(d, f"ab_{s}_*.json"))):
        for r in json.load(open(p))["rows"]:
            c = r["case"] if "case" in r else r
            key = f"{c.get('load', '')} {c.get('layer', '')}".strip()
            side[s].setdefault(key, []).append(r["median_net_ns"] / 1e3)
            if "check" in r:
                checks[s].add(bool(r["check"][0]))
n = {s: len(glob.glob(os.path.join(d, f"ab_{s}_*.json"))) for s in side}
print(f"runs: base {n['base']}, port {n['port']}; checks base {checks['base']}, port {checks['port']}")
print("| cell | base median | base range | port median | port range | port - base |")
print("|---|---:|---|---:|---|---:|")
for key in side["base"]:
    b, q = side["base"][key], side["port"].get(key, [])
    if not q:
        continue
    mb, mq = statistics.median(b), statistics.median(q)
    print(f"| {key} | {mb:.2f} | {min(b):.2f}-{max(b):.2f} | {mq:.2f} | {min(q):.2f}-{max(q):.2f} | {mq - mb:+.2f} |")
