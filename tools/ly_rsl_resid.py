#!/usr/bin/env python3
"""Residency of the decode 1 slice CTAs before the MoE ends, from tools/ly_box_pmt.py JSON written with --raw.

python3 tools/ly_rsl_resid.py pmt_w.json [pmt_c.json ...]

Per input (us from the first MoE CTA's entry): slice CTAs, distinct SMs, CTAs entered before the last MoE CTA's exit,
CTAs whose held codes had landed (slot 13) before that exit, the last such landing relative to the exit, and the
slice / finish ends after the exit. Then medians per cell.
"""
import json
import statistics as st
import sys


def per_input(raw):
    moe, cg = raw["moe"], [r for r in raw["cg"] if r[0]]
    t0 = min(r[0] for r in moe)
    us = lambda v: (v - t0) / 1e3  # noqa: E731
    pa_end = max(us(r[2]) for r in moe)
    exit_ = max(us(r[5]) for r in moe)
    ent = [us(r[0]) for r in cg]
    land = [us(r[13]) for r in cg]
    wait = [us(r[2]) for r in cg]
    end = [us(r[9]) for r in cg]
    return dict(
        n=len(cg), sms=len({int(r[10]) for r in cg}), pa_end=pa_end, exit=exit_,
        entered_before_exit=sum(e < exit_ for e in ent), landed_before_exit=sum(v < exit_ for v in land),
        entry_first=min(ent) - pa_end, entry_last=max(ent) - pa_end, landed_last_vs_exit=max(land) - exit_,
        wait_p50=st.median(wait) - exit_, wait_max=max(wait) - exit_, slice_end=max(end) - exit_,
    )


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        for row in d["rows"]:
            res = [per_input(i["raw"]) for i in row["inputs"] if "raw" in i]
            lab = f"{row['case']['load']} {row['case']['layer']}"
            for j, r in enumerate(res):
                print(f"{path} {lab} in {j} " + " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                                                        for k, v in r.items()))
            if res:
                print(f"MEDIAN {path} {lab} " + " ".join(
                    f"{k} {st.median(r[k] for r in res):.2f}" for k in res[0]) + f" | min landed_before_exit "
                      f"{min(r['landed_before_exit'] for r in res)} of {min(r['n'] for r in res)}")


if __name__ == "__main__":
    main()
