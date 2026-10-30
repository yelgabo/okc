"""Per-kernel and per-owner gaps from a same-process bench log (tools/dl_make_bench_pub.py ABK and
ABKN lines): for each shape, each arm's kernels with names and their median summed durations per
step, then per owner (MoE, attention, projections, glue) ours minus the other arm, as raw us,
load us ((46 sliding + 16 full) / 62) and share of the score (a us at one shape moves the log
score by weight / (6 x the load's 46 s + 16 f time)).

    python3 tools/dl_kernel_gap.py LOG [--arms=cand,pub] [--kernels]

Durations are kernel lifetimes: a PDL secondary counts from launch, so its wait on the
predecessor is in its own time, and overlapping kernels make a shape's sum exceed its span.
An owner's sum is therefore an upper bound of its critical-path cost, and the comparison is
fair only between kernels launched the same way; the span column is the step itself.
"""

import re
import sys

OWNERS = (
    ("moe", r"moe|marlin|w4_|silu|topk|expert|fused_moe|finaliz"),
    ("attn", r"attn|flash|fwd_|split_kv|combine|decode_kernel|merge"),
    ("proj", r"sk_kernel|cgemv|wd_kernel|nvjet|gemv|gemm|cutlass|sm90_xmma|splitK"),
    ("glue", r"."),
)
WEIGHT = {"s": 46, "f": 16}


def owner(name):
    for nm, pat in OWNERS:
        if re.search(pat, name, re.I):
            return nm
    return "glue"


def parse(path):
    names, rows, spans = {}, {}, {}
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        if ln.startswith("ABKN "):
            _, code, name = ln.split(" ", 2)
            names[code] = name
        elif ln.startswith("ABK "):
            head, _, body = ln.partition(": ")
            parts = head.split()
            label, arm, span = parts[1], parts[2], float(parts[4])
            ks = []
            toks = body.split()
            for code, val in zip(toks[::2], toks[1::2]):
                t, _, n = val.partition("x")
                ks.append((code, float(t), int(n) if n else 1))
            rows[(label, arm)] = ks
            spans[(label, arm)] = span
    return names, rows, spans


def main():
    path = sys.argv[1]
    arms = ("cand", "pub")
    show = "--kernels" in sys.argv
    for a in sys.argv[2:]:
        if a.startswith("--arms="):
            arms = tuple(a.split("=", 1)[1].split(","))
    names, rows, spans = parse(path)
    labels = sorted({lab for lab, _ in rows}, key=lambda s: (int(re.sub(r"\D", "", s)),
                                                             s[0], s[-1]))
    loads = {}
    tot = {}
    for lab in labels:
        if any((lab, a) not in rows for a in arms):
            continue
        per = {}
        for a in arms:
            per[a] = {}
            for code, t, n in rows[(lab, a)]:
                o = owner(names.get(code, code))
                per[a][o] = per[a].get(o, 0.0) + t
        if show:
            for a in arms:
                print(f"{lab} {a} span {spans[(lab, a)]:.1f}: " + "; ".join(
                    f"{names.get(c, c)[:44]} [{owner(names.get(c, c))}] {t:.1f}"
                    + (f"x{n}" if n > 1 else "") for c, t, n in rows[(lab, a)]))
        diff = {o: per[arms[0]].get(o, 0.0) - per[arms[1]].get(o, 0.0)
                for o in ("moe", "attn", "proj", "glue")}
        dspan = spans[(lab, arms[0])] - spans[(lab, arms[1])]
        print(f"{lab} span {arms[0]} {spans[(lab, arms[0])]:.1f} {arms[1]} "
              f"{spans[(lab, arms[1])]:.1f} diff {dspan:+.1f} | owner sums {arms[0]}-{arms[1]}: "
              + " ".join(f"{o} {d:+.1f}" for o, d in diff.items()))
        tot[lab] = (diff, dspan)
        load = lab[:-1]
        loads.setdefault(load, {})[lab[-1]] = spans[(lab, arms[1])]
    print("weighted (load us = weight x diff / 62; score % = weight x diff / (6 x C_other) x 100, "
          "C from the other arm's spans):")
    agg = {}
    for lab, (diff, dspan) in tot.items():
        load = lab[:-1]
        if len(loads.get(load, {})) < 2:
            continue
        c = 46 * loads[load]["s"] + 16 * loads[load]["f"]
        w = WEIGHT[lab[-1]]
        for o, d in list(diff.items()) + [("span", dspan)]:
            lu = w * d / 62
            sc = 100 * w * d / (6 * c)
            agg.setdefault(o, [0.0, 0.0])
            agg[o][0] += lu
            agg[o][1] += sc
            if o == "span" or abs(d) >= 1.0:
                print(f"  {lab} {o}: {d:+.1f} us -> {lu:+.2f} load us, {sc:+.3f}% score")
    print("totals over shapes: " + " ".join(f"{o} {v[0]:+.1f} load us / {v[1]:+.2f}%"
                                            for o, v in agg.items()))


if __name__ == "__main__":
    main()
