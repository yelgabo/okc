#!/usr/bin/env python3
"""Finish-program placement at T = 8 from a tools/ly_box_v1st.py --raw JSON (ly_v1st instrument, finish SM ids).

For each input: the SMs that hold no slice CTA, the SMs that hold more than one finish program, and per group the
finish's P stage (wait return to inv formed), carry stage and end after the slice's last store; then medians over
inputs per cell.

    python3 tools/ly_v1st_fin.py st2.json[.gz]
"""
import collections
import gzip
import json
import statistics as S
import sys


def main(path):
    d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    for row in d["rows"]:
        agg = collections.defaultdict(list)
        for inp in row["inputs"]:
            raw = inp.get("raw")
            if not raw:
                continue
            sl, fin, moe = raw["sl"], raw["fin"], raw["moe"]
            t0 = min(r[0] for r in moe)
            us = lambda x: (x - t0) / 1e3  # noqa: E731
            sl_sms = {int(r[24]) for r in sl}
            per_sm = collections.Counter(int(r[5]) for r in fin)
            sl_end = max(us(r[23]) for r in sl)
            agg["free_sms"].append(132 - len(sl_sms))
            agg["stacked_sms"].append(sum(1 for n in per_sm.values() if n > 1))
            agg["stacked_on_free"].append(sum(1 for s, n in per_sm.items() if n > 1 and s not in sl_sms))
            for name, keep in (("stacked", lambda r: per_sm[int(r[5])] > 1), ("single", lambda r: per_sm[int(r[5])] == 1)):
                rows = [r for r in fin if keep(r)]
                if not rows:
                    continue
                agg[name + "_n"].append(len(rows))
                agg[name + "_P"].append(S.median((r[2] - r[1]) / 1e3 for r in rows))
                agg[name + "_carry"].append(S.median((r[3] - r[2]) / 1e3 for r in rows))
                agg[name + "_end_p50"].append(S.median(us(r[4]) - sl_end for r in rows))
                agg[name + "_end_max"].append(max(us(r[4]) - sl_end for r in rows))
            agg["fin_end"].append(max(us(r[4]) for r in fin) - sl_end)
        print(f"{row['case']['load']} {row['case']['layer']}: {len(agg['fin_end'])} inputs, medians")
        for k, v in agg.items():
            print(f"  {k:18s} {S.median(v):7.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
