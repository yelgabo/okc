#!/usr/bin/env python3
"""Finish-program placement at T = 8 from a tools/ly_box_v1st.py --raw JSON (v1st instrument, finish slot 5 = SM id).

Per input: how many finish programs share an SM with a slice CTA, sit alone on an SM without one, or share an SM
with other finish programs; programs per SM; entry, wait return and end of the finish programs relative to the
slice's last store; then medians over the inputs per cell. Times in us.

    python3 tools/ly_v1fs_fin.py st.json [st2.json ...]
"""
import collections
import gzip
import json
import statistics as S
import sys


def load(path):
    return json.load(gzip.open(path) if path.endswith(".gz") else open(path))


def one(raw):
    sl, fin = raw["sl"], raw["fin"]
    sl_sms = {int(r[24]) for r in sl}
    sl_end = max(r[23] for r in sl)
    per_sm = collections.Counter(int(r[5]) for r in fin)
    rel = lambda x: (x - sl_end) / 1e3  # noqa: E731
    beside = [r for r in fin if int(r[5]) in sl_sms]
    free = [r for r in fin if int(r[5]) not in sl_sms]
    out = dict(
        beside_slice=len(beside),
        on_free_sm=len(free),
        stacked=sum(1 for r in fin if per_sm[int(r[5])] > 1),
        max_per_sm=max(per_sm.values()),
        sms_used=len(per_sm),
        entry_p50=S.median(rel(r[0]) for r in fin),
        entry_max=max(rel(r[0]) for r in fin),
        wait_p50=S.median(rel(r[1]) for r in fin),
        wait_max=max(rel(r[1]) for r in fin),
        P_p50=S.median((r[2] - r[1]) / 1e3 for r in fin),
        P_max=max((r[2] - r[1]) / 1e3 for r in fin),
        end_p50=S.median(rel(r[4]) for r in fin),
        fin_end=max(rel(r[4]) for r in fin),
    )
    sl_exit = sorted(rel(r[23]) for r in sl)
    out["slice_store_p50"] = S.median(sl_exit)
    out["slice_64th_store"] = sl_exit[63] if len(sl_exit) > 63 else float("nan")
    return out


def main(paths):
    for path in paths:
        d = load(path)
        print(path)
        for row in d["rows"]:
            agg = collections.defaultdict(list)
            for inp in row["inputs"]:
                if inp.get("raw"):
                    for k, v in one(inp["raw"]).items():
                        agg[k].append(v)
            n = len(agg["fin_end"])
            print(f"  {row['case']['load']} {row['case']['layer']}: {n} inputs, medians (min / max over inputs)")
            for k, v in agg.items():
                print(f"    {k:18s} {S.median(v):7.2f}   ({min(v):.2f} / {max(v):.2f})")


if __name__ == "__main__":
    main(sys.argv[1:])
