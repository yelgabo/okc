#!/usr/bin/env python3
"""Stamp table of tools/ly_box_qtl.py runs (QTL median lines): one column per file and cell.

    python3 tools/ly_q32_sttab.py st_base_16.txt st_q32n_16.txt ...
"""
import sys

KEYS = ("entry_last", "issue0", "wait_x", "c_first", "land0_mean", "lat_ring", "lat_late", "slot_refill",
        "refill_after_c0", "land_ivl", "rate_tbs", "c_proc", "c_wait_mid", "c_wait_sum", "loop_end", "loop_end_max",
        "esc_end_after_loop", "cbar_after_loop", "epi", "store", "fin_mean", "fin_max", "fin_spread", "graph_plain")


def main():
    cols = []
    for path in sys.argv[1:]:
        for line in open(path):
            if line.startswith("QTL ") and " medians " in line:
                head, body = line.split(" medians ")
                t = body.split()
                d = dict(zip(t[0::2], t[1::2]))
                cols.append((path.rsplit("/", 1)[-1].replace(".txt", "") + " " + head.split(" n ")[0][4:].replace("decode ", "d"), d))
    print("item".ljust(20) + "".join(c[0].rjust(26) for c in cols))
    for k in KEYS:
        vals = []
        for _, d in cols:
            v = d.get(k, "")
            try:
                f = float(v)
                vals.append("-" if abs(f) > 1e6 or f != f else f"{f:.3f}")
            except ValueError:
                vals.append(v)
        print(k.ljust(20) + "".join(v.rjust(26) for v in vals))


if __name__ == "__main__":
    main()
