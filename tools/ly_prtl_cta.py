#!/usr/bin/env python3
"""Per-CTA view of the raw pooler + router stamps in a tools/ly_box_prtl.py JSON (--raw N inputs per cell).

    python3 tools/ly_prtl_cta.py tl.json [cell-substring]

For each raw input: the CTAs that finish last, with entry, wait, first land, loop end, router pieces, epilogue and
final store (us from the first CTA's entry), whether each is its router tile's last quarter, and the correlation of
loop end with entry; then the stage-landing clock of the median CTA.
"""
import json
import statistics
import sys


def main(path, sub=""):
    d = json.load(open(path))
    for row in d["rows"]:
        lab = f"{row['case']['load']} {row['case']['layer']}"
        if sub not in lab:
            continue
        for inp in row["inputs"]:
            if "raw" not in inp:
                continue
            v = inp["raw"]
            t0 = min(r[0] for r in v)
            u = lambda x: (x - t0) / 1e3 if x else float("nan")  # noqa: E731
            rows = []
            for c, r in enumerate(v):
                le = max(r[56 + w] for w in range(8))
                rows.append(dict(cta=c, sm=r[11], entry=u(r[0]), wait=u(r[3]), land0=u(r[36]), le=u(le),
                                 part=u(r[5]), atom=u(r[6]), rout=u(r[7]), epi=u(r[8]), fin=u(r[10]),
                                 last=r[7] != 0))
            fins = sorted(rows, key=lambda x: -x["fin"])
            print(f"== {lab} seed {inp['seed']} span {inp['span']:.2f} post_wait {inp['post_wait']:.2f}"
                  f" last-quarter CTAs {sum(x['last'] for x in rows)}")
            print("  cta  sm  entry  wait  land0  loopend  part  atom  rout   epi   fin  last")
            for x in fins[:6] + [None] + sorted(rows, key=lambda x: x["fin"])[:2]:
                if x is None:
                    print("  ...")
                    continue
                print(f"  {x['cta']:3d} {x['sm']:3d} {x['entry']:6.2f} {x['wait']:5.2f} {x['land0']:6.2f} {x['le']:8.2f}"
                      f" {x['part']:5.2f} {x['atom']:5.2f} {x['rout']:5.2f} {x['epi']:5.2f} {x['fin']:5.2f}  {int(x['last'])}")
            late = [x for x in rows if x["entry"] > 1.0]
            early = [x for x in rows if x["entry"] <= 1.0]
            if late:
                print(f"  late-entry CTAs {len(late)}: loop end {statistics.mean(x['le'] for x in late):.2f}"
                      f" vs others {statistics.mean(x['le'] for x in early):.2f}; fin {statistics.mean(x['fin'] for x in late):.2f}"
                      f" vs {statistics.mean(x['fin'] for x in early):.2f}")
            lastq = [x for x in rows if x["last"]]
            other = [x for x in rows if not x["last"]]
            print(f"  last-quarter fin {statistics.mean(x['fin'] for x in lastq):.2f} vs others"
                  f" {statistics.mean(x['fin'] for x in other):.2f}; loop end {statistics.mean(x['le'] for x in lastq):.2f}"
                  f" vs {statistics.mean(x['le'] for x in other):.2f}")
            med = sorted(range(len(v)), key=lambda c: rows[c]["le"])[len(v) // 2]
            r = v[med]
            iss = [u(r[16 + s]) for s in range(20)]
            land = [u(r[36 + s]) for s in range(20)]
            print(f"  median CTA {med}: issue " + " ".join(f"{x:.2f}" for x in iss))
            print(f"  median CTA {med}: land  " + " ".join(f"{x:.2f}" for x in land))


if __name__ == "__main__":
    main(*sys.argv[1:])
