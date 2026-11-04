#!/usr/bin/env python3
"""Decode 1 MoE ramp / handoff / tail from tools/ly_box_pmt.py --raw JSONs of ly_mtl instruments (dl_ly_mtl_patch.py).

Per input, us from the first MoE CTA's entry, over the 132 CTAs (thread 0's DBG stamps): prologue = after route(),
first_slot = first phase A unit's ring slot full, aend = phase A end, b_start = first phase B unit past its doneA
wait, bend = phase B end, bar = past the grid barrier, exit = last CTA's exit. Prints medians over inputs.

    python3 tools/ly_mtl_summary.py A.json [B.json ...]
"""
import json
import statistics as st
import sys


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        print(path)
        for row in d["rows"]:
            agg = {}
            for x in row["inputs"]:
                if "raw" not in x:
                    continue
                moe = x["raw"]["moe"]
                t0 = min(m[0] for m in moe)

                def u(k):
                    return [(m[k] - t0) / 1e3 for m in moe if m[k]]
                pro, aend, bend, bar, ex, fs, fb = (u(k) for k in (1, 2, 3, 4, 5, 6, 7))
                vals = dict(prologue=st.mean(pro), first_slot=st.mean(fs), first_slot_max=max(fs),
                            aend=st.mean(aend), aend_max=max(aend), b_start=st.mean(fb), b_start_max=max(fb),
                            handoff=st.mean([b - a for a, b in zip(aend, fb)]), bend=st.mean(bend),
                            bend_max=max(bend), bar=st.mean(bar), exit=max(ex), plain=x["graph_plain"])
                for k, v in vals.items():
                    agg.setdefault(k, []).append(v)
            print(" ", row["case"]["layer"], len(agg.get("plain", [])), "inputs",
                  " ".join(f"{k} {st.median(v):.2f}" for k, v in agg.items()))


if __name__ == "__main__":
    main()
