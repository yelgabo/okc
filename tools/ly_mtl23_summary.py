#!/usr/bin/env python3
"""Medians over inputs of the decode 1 MoE credit path (ly_mtl2 JSON) and prologue (ly_mtl3 JSON) stamps, us.

ly_mtl2 slots: 0 entry, 1 helper lane 0 with the CTA's last phase A partials in, 2 thread 0 phase A end, 4 helper's
doneA credit issued (after __threadfence), 6 first slot, 7 first phase B unit past the doneA wait.
ly_mtl3 slots: 0 entry, 3 FAST1 copies issued, 4 generation stored, 1 route() done, 5 WS_JOIN passed, 7 first job's
x in registers, 6 first slot full, 2 phase A end. Times from the first CTA's entry; "_max" is the latest CTA.

    python3 tools/ly_mtl23_summary.py --mtl2 A.json --mtl3 B.json
"""
import json
import statistics as st
import sys


def med(agg):
    return " ".join(f"{k} {st.median(v):.2f}" for k, v in agg.items())


def mtl2(path):
    for row in json.load(open(path))["rows"]:
        agg = {}
        for x in row["inputs"]:
            moe = x["raw"]["moe"]
            t0 = min(m[0] for m in moe)
            aend, pin, cred, bst = ([(m[k] - t0) / 1e3 for m in moe] for k in (2, 1, 4, 7))
            last = max(range(len(moe)), key=lambda i: cred[i])
            vals = dict(aend=st.mean(aend), aend_max=max(aend),
                        partials_after_aend=st.mean([p - a for p, a in zip(pin, aend)]),
                        credit_after_partials=st.mean([c - p for c, p in zip(cred, pin)]),
                        credit_after_partials_max=max(c - p for c, p in zip(cred, pin)),
                        credit_after_aend=st.mean([c - a for c, a in zip(cred, aend)]), credit_max=max(cred),
                        aend_of_last_credit=aend[last], bstart_min=min(bst), bstart=st.mean(bst), bstart_max=max(bst),
                        detect=min(bst) - max(cred))
            for k, v in vals.items():
                agg.setdefault(k, []).append(v)
        print("mtl2", row["case"]["layer"], med(agg))


def mtl3(path):
    names = {3: "fast1", 4: "gen", 1: "route", 5: "join", 7: "x", 6: "slot", 2: "aend"}
    for row in json.load(open(path))["rows"]:
        agg = {}
        for x in row["inputs"]:
            moe = x["raw"]["moe"]
            t0 = min(m[0] for m in moe)
            agg.setdefault("entry_spread", []).append(max((m[0] - t0) / 1e3 for m in moe))
            for k, n in names.items():
                v = [(m[k] - t0) / 1e3 for m in moe if m[k]]
                agg.setdefault(n, []).append(st.mean(v))
                agg.setdefault(n + "_max", []).append(max(v))
        print("mtl3", row["case"]["layer"], med(agg))


if __name__ == "__main__":
    if "--mtl2" in sys.argv:
        mtl2(sys.argv[sys.argv.index("--mtl2") + 1])
    if "--mtl3" in sys.argv:
        mtl3(sys.argv[sys.argv.index("--mtl3") + 1])
