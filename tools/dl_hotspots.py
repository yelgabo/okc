#!/usr/bin/env python3
"""Rank decode-layer kernels by their weight in the score, from a judge profile .txt.

A kernel's raw GPU time misleads: the score is the geometric mean over six loads of
baseline/ours, where a load's time is 46 * sliding + 16 * full. So d(ln score) / d(t) for a
shape is (1/6) * w / C_load, and a microsecond at decode 1 counts several times one at verify 32.

usage: tools/dl_hotspots.py muse_decode_layer/profiles/submission.<job>.txt [--top N] [--family]
"""
import collections
import re
import sys


def family(name):
    if "moe" in name:
        return "MoE"
    if "attn_one" in name:
        return "attention"
    if "boundary" in name or "norm_residual" in name:
        return "boundaries / norm-residual"
    if "router" in name or "routing" in name:
        return "router / routing / front"
    if "pre_norm" in name:
        return "pre-expert norm"
    return "dense projections"


def main():
    args = sys.argv[1:]
    top = int(args[args.index("--top") + 1]) if "--top" in args else 25
    by_family = "--family" in args
    text = open(args[0]).read()
    ours = {}
    for m in re.finditer(r"layer=(sliding|full) load=(\w+ \d+)\S*.*?\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x",
                         text):
        ours[(m.group(2), m.group(1))] = float(m.group(4))
    load = collections.Counter()
    for (name, kind), t in ours.items():
        load[name] += (46 if kind == "sliding" else 16) * t
    weight = collections.Counter()
    for block in re.split(r"\nprofile \d+: ", text)[1:]:
        head = re.search(r"layer=(sliding|full) load=(\w+ \d+)", block.split("\n")[0])
        kind, name = head.group(1), head.group(2)
        w = 46 if kind == "sliding" else 16
        for line in block.split("\n")[2:]:
            m = re.match(r"(.+?)\s+([\d.]+) µs\s+[\d.]+%", line)
            if not m:
                continue
            kernel = m.group(1).replace("void ", "").replace("(anonymous namespace)::", "")
            key = family(kernel) if by_family else (kernel[:60], f"{name} {kind}")
            weight[key] += w * float(m.group(2)) / load[name] / 6
    total = sum(weight.values())
    for key, v in weight.most_common(top):
        label = key if by_family else f"{key[1]:18s} {key[0]}"
        print(f"{100 * v / total:5.1f}%  {label}")


if __name__ == "__main__":
    main()
