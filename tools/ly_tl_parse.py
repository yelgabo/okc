"""Per-replay kernel timeline from a tools/ly_tl_run.py cell trace.

Prints, per kernel position in the replay, the median start (from the first layer kernel), duration,
end and the gap to the previous kernel's end over the profiled replays (negative gap: PDL overlap).
The 1 GiB flush (reduce_kernel) separates the replays and is dropped.
"""
import json
import statistics
import sys


def replays(path):
    ev = [e for e in json.load(open(path))["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    ev.sort(key=lambda e: e["ts"])
    out, cur = [], []
    for e in ev:
        if "reduce_kernel" in e["name"]:
            if cur:
                out.append(cur)
            cur = []
            continue
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def main(path):
    reps = replays(path)
    n = min(len(r) for r in reps)
    reps = [r for r in reps if len(r) == n]
    med = statistics.median
    print(f"{path}: {len(reps)} replays of {n} kernels")
    print(f"{'#':>2} {'start':>7} {'dur':>7} {'end':>7} {'gap':>6}  kernel")
    t0 = [r[0]["ts"] for r in reps]
    for i in range(n):
        st = med(r[i]["ts"] - a for r, a in zip(reps, t0))
        du = med(r[i]["dur"] for r in reps)
        en = med(r[i]["ts"] + r[i]["dur"] - a for r, a in zip(reps, t0))
        gp = med(r[i]["ts"] - (r[i - 1]["ts"] + r[i - 1]["dur"]) for r in reps) if i else 0.0
        print(f"{i:>2} {st:7.2f} {du:7.2f} {en:7.2f} {gp:6.2f}  {reps[0][i]['name'][:80]}")
    span = med(r[-1]["ts"] + r[-1]["dur"] - r[0]["ts"] for r in reps)
    busy = med(sum(e["dur"] for e in r) for r in reps)
    print(f"span {span:.2f} us, sum of durations {busy:.2f} us")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
