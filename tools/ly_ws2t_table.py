#!/usr/bin/env python3
"""Per-cell table of the tools/ly_ws2t_ab.sh runs (ab_c_a, ab_w_a, ab_c_b, ab_w_b, ab_c_c JSONs in DIR).

    python3 tools/ly_ws2t_table.py DIR

Control spread = max - min over the three control runs; threshold = max(2 x spread, 1% of the control mean);
pairs are each candidate run against its neighbouring control runs.
"""
import json
import os
import sys

NAMES = ("c_a", "w_a", "c_b", "w_b", "c_c")


def cells(d):
    return {(r["case"]["load"], r["case"]["layer"]): (r["median_net_ns"] / 1e3, r["check"][0]) for r in d["rows"]}


def main(root):
    runs = {n: json.load(open(os.path.join(root, f"ab_{n}.json"))) for n in NAMES}
    C = {n: cells(d) for n, d in runs.items()}
    print("cell".ljust(20) + "".join(n.rjust(9) for n in NAMES)
          + "   spread  w - c mean   w_a-c_a w_a-c_b w_b-c_b w_b-c_c   thr  verdict")
    for k in C["c_a"]:
        c = [C[n][k][0] for n in ("c_a", "c_b", "c_c")]
        w = [C[n][k][0] for n in ("w_a", "w_b")]
        sp = max(c) - min(c)
        thr = max(2 * sp, 0.01 * sum(c) / 3)
        d = sum(w) / 2 - sum(c) / 3
        pairs = [C["w_a"][k][0] - C["c_a"][k][0], C["w_a"][k][0] - C["c_b"][k][0],
                 C["w_b"][k][0] - C["c_b"][k][0], C["w_b"][k][0] - C["c_c"][k][0]]
        verdict = "win" if d < -thr else "loss" if d > thr else "inside"
        ok = all(C[n][k][1] for n in NAMES)
        print(f"{k[0]} {k[1]}".ljust(20) + "".join(f"{C[n][k][0]:9.2f}" for n in NAMES)
              + f"  {sp:7.2f}  {d:+10.2f}   " + " ".join(f"{x:+7.2f}" for x in pairs)
              + f"  {thr:5.2f}  {verdict}{'' if ok else ' CHECK FAIL'}")
    for n, d in runs.items():
        print(n, "timed PDL", d.get("pdl8", {}).get("submission", {}).get("timed", {}).get("on"))


if __name__ == "__main__":
    main(sys.argv[1])
