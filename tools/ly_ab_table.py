#!/usr/bin/env python3
"""Per-cell box net us and the weighted six-load sum (46 x sliding + 16 x full per load) for bx_box_run.py JSONs.

    python3 tools/ly_ab_table.py CONTROL.json ARM.json [ARM.json ...]

The first file is the reference arm; every other arm prints its per-cell difference and six-load delta against it.
Also prints each run's timed PDL enable set, so an arm whose gate turned a T off is visible.
"""
import json
import sys

LOADS = ("decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32")


def load(path):
    d = json.load(open(path))
    cells = {}
    for r in d["rows"]:
        c = r["case"]
        cells[(c["load"], c["layer"])] = (r["median_net_ns"] / 1e3, float(c["weight"]), r["check"][1][0])
    timed = d.get("pdl8", {}).get("submission", {}).get("timed", {})
    on = sorted(int(t) for t, v in timed.get("on", {}).items() if v)
    return cells, on, len(timed.get("edges", {}).get("8", []))


def six(cells):
    return sum(v * w for v, w, _ in cells.values())


def main():
    paths = sys.argv[1:]
    runs = [(p.rsplit("/", 1)[-1].removesuffix(".json"), *load(p)) for p in paths]
    names = [n for n, *_ in runs]
    print("cell".ljust(20) + "".join(n.rjust(12) for n in names))
    for ld in LOADS:
        for layer in ("sliding", "full"):
            k = (ld, layer)
            row = [r[1][k][0] for r in runs]
            print(f"{ld} {layer}".ljust(20) + "".join(f"{v:12.2f}" for v in row)
                  + "   d: " + " ".join(f"{v - row[0]:+.2f}" for v in row[1:]))
    tot = [six(r[1]) for r in runs]
    print("six-load sum".ljust(20) + "".join(f"{v:12.1f}" for v in tot) + "   d: "
          + " ".join(f"{v - tot[0]:+.1f}" for v in tot[1:]))
    for ld in LOADS:
        lt = [sum(r[1][(ld, la)][0] * r[1][(ld, la)][1] for la in ("sliding", "full")) for r in runs]
        print(f"load {ld}".ljust(20) + "".join(f"{v:12.1f}" for v in lt) + "   d: "
              + " ".join(f"{v - lt[0]:+.1f}" for v in lt[1:]))
    print("timed PDL on".ljust(20) + "".join(("{" + ",".join(map(str, r[2])) + f"}}/{r[3]}e").rjust(12) for r in runs))
    print("all checks pass".ljust(20) + "".join(str(all(c[2] for c in r[1].values())).rjust(12) for r in runs))
    groups = {}
    for n, cells, *_ in runs:
        groups.setdefault(n.rsplit("_", 1)[0], []).append(cells)
    if all(len(g) == 1 for g in groups.values()):
        return
    print("\nper arm (runs grouped by name before the last _): mean per cell, delta of means against the first arm")
    gn = list(groups)
    mean = {g: {k: sum(c[k][0] for c in groups[g]) / len(groups[g]) for k in groups[g][0]} for g in gn}
    print("cell".ljust(20) + "".join(f"{g} (n={len(groups[g])})".rjust(16) for g in gn))
    for ld in LOADS:
        for layer in ("sliding", "full"):
            k = (ld, layer)
            print(f"{ld} {layer}".ljust(20) + "".join(f"{mean[g][k]:16.2f}" for g in gn) + "   d: "
                  + " ".join(f"{mean[g][k] - mean[gn[0]][k]:+.2f}" for g in gn[1:]))
    for g in gn:
        t = [six(c) for c in groups[g]]
        m = sum(t) / len(t)
        print(f"{g}: six-load runs {' '.join(f'{v:.1f}' for v in t)}; mean {m:.1f}; spread (max - min) {max(t) - min(t):.1f}; "
              f"mean minus {gn[0]} mean {m - sum(six(c) for c in groups[gn[0]]) / len(groups[gn[0]]):+.1f}")


if __name__ == "__main__":
    main()
