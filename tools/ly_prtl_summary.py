#!/usr/bin/env python3
"""Markdown stamp table from tools/ly_box_prtl.py JSONs: medians over each cell's inputs, us.

    python3 tools/ly_prtl_summary.py tl.json [tl_cand.json]

Columns: predecessor done (first CTA's griddepcontrol.wait return), first stage landed after it, main loop end
(mean CTA) after it, loop-end spread across CTAs, router piece after the loop (partial stores, counter, last quarter),
pooler epilogue, tail (last final store minus the latest loop end), post-wait span (last final store minus the first
wait return), byte floor, excess, the producer's per-stage issue interval (pooler stages; floor 0.708 us per stage at
128 CTAs x 24,576 B / 4.443 TB/s), and the plain and stamped graph times.
"""
import json
import statistics
import sys


def med(xs):
    return statistics.median(xs)


def rows(path):
    d = json.load(open(path))
    out = []
    for r in d["rows"]:
        ins = r["inputs"]
        g = lambda k: med([x[k] for x in ins])  # noqa: E731
        out.append(dict(
            cell=f"{r['case']['load']} {r['case']['layer']}", T=r["T"], n=len(ins), wait=g("wait_min"),
            land=g("land0_after_wait"), loop=g("loop_from_wait"), lspread=med([x["loop_end_max"] - x["loop_end_min"] for x in ins]),
            router=med([x["router_part"] + x["router_atom"] + (x["router_out"] if x["router_out"] == x["router_out"] else 0) for x in ins]),
            epi=med([x["epi_start"] + x["epi_red"] + x["epi_store"] for x in ins]),
            tail=med([x["fin_max"] - x["loop_end_max"] for x in ins]), post=g("post_wait"), floor=g("floor"),
            iv=g("iv_pool"), ivr=g("iv_rt"), gp=g("graph_plain"), gi=g("graph_inst"), entry=g("entry_spread")))
    return out


def main():
    for path in sys.argv[1:]:
        print(f"\n{path}\n")
        print("| cell | n | wait | land0 | loop end | loop-end spread | router | epilogue | tail | post-wait span | floor |"
              " excess | issue interval pool / router | graph plain / stamped |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for x in rows(path):
            print(f"| {x['cell']} | {x['n']} | {x['wait']:.2f} | {x['land']:.2f} | {x['loop']:.2f} | {x['lspread']:.2f} |"
                  f" {x['router']:.2f} | {x['epi']:.2f} | {x['tail']:.2f} | {x['post']:.2f} | {x['floor']:.2f} |"
                  f" {x['post'] - x['floor']:+.2f} | {x['iv']:.2f} / {x['ivr']:.2f} | {x['gp']:.2f} / {x['gi']:.2f} |")


if __name__ == "__main__":
    main()
