"""Per-case frontier map for muse_bf16_proj from judge receipts.

Usage: python3 tools/proj_frontier.py <receipt.json> [...]

Prints, per case, the median judged time over the receipts, the profiled kernel time, the
baseline, a bandwidth floor from stored bytes and the score value of each microsecond.
"""
import json
import re
import statistics as st
import sys

# Bytes as stored by the current picks: 12-bit code (1.5 B per weight) for the sk / cgemv picks
# at m <= 32 (escape tables are < 0.1% of the stream), BF16 at m >= 128 (cuBLASLt / wd), the
# router as BF16 hi + lo (4 B per weight) everywhere. The unpooler at m <= 8 is the 1,024-row
# slice (TASK.md). Escape counts from the judge's pack lines: 2583 / 936 / 4553 / 3265.
SHAPES = {
    "qkv_gate": (2304, 8192),
    "o": (8192, 1024),
    "pooler": (4096, 8192),
    "unpooler": (8192, 4096),
    "router": (256, 8192),
}
LOADS = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]


def records(path):
    d = json.load(open(path))
    if "result" in d and "records" in d["result"]:
        return d["result"]["records"]
    return d.get("benchmarks", []) + d.get("profiles", [])


# The judge's streaming line, t = 1.3 us + bytes / 4.05 TB/s, measured with cold L2
# (docs/runs-moe.md:24, docs/audit-2026-10-28.md 5.4). 700 TFLOP/s is what the m = 256 cuBLASLt
# picks reach (docs/runs.md, P2 / P3 work model); 989 is the datasheet dense BF16 peak.
FIXED_US = 1.3


def main(paths, hbm_tbs=4.05, flops=700e12, peak=989e12):
    rows = {}
    for p in paths:
        for r in records(p):
            c = r.get("case")
            if not c or "load" not in c:
                continue
            e = rows.setdefault((c["load"], c["layer"], c["m"]), {"ours": [], "base": [], "kern": []})
            if r.get("event") == "benchmark":
                e["ours"].append(r["median_ns"] / 1e3)
                e["base"].append(r["baseline_ns"] / 1e3)
                e["sol"] = r["speed_of_light_ns"] / 1e3
            elif r.get("event") == "profile":
                tot = re.search(r"total\s+([\d.]+) µs", r["report"])
                if tot:
                    e["kern"].append(float(tot.group(1)))
                names = re.findall(r"^(\S.{0,40})", r["report"], re.M)
                e["names"] = [n.strip() for n in names[1:-1]]
    out = []
    ct = {}
    bt = {}
    for ld in LOADS:
        ct[ld] = sum(st.median(rows[k]["ours"]) for k in rows if k[0] == ld)
        bt[ld] = sum(st.median(rows[k]["base"]) for k in rows if k[0] == ld)
    for (ld, layer, m), e in rows.items():
        n, k = SHAPES[layer]
        if layer == "unpooler" and m <= 8:
            n = 1024
        if layer == "router":
            wb = 4.0
        elif m <= 32:
            wb = 1.5
        else:
            wb = 2.0
        obytes = 4 if layer == "router" else 2
        byts = n * k * wb + m * k * 2 + m * n * obytes
        t_bw = byts / (hbm_tbs * 1e12) * 1e6
        t_math = 2 * m * n * k * (2 if layer == "router" else 1) / flops * 1e6
        t_peak = t_math * flops / peak
        floor = FIXED_US + max(t_bw, t_math)
        ours = st.median(e["ours"])
        kern = st.median(e["kern"]) if e["kern"] else float("nan")
        val = 100.0 / (6 * ct[ld])
        out.append((ld, layer, m, ours, min(e["ours"]), max(e["ours"]), kern, st.median(e["base"]),
                    e.get("sol"), t_bw, t_peak, t_math, floor, ours - floor, (ours - floor) * val, val,
                    ",".join(e.get("names", []))[:60]))
    print("score from medians: %.4f" % (
        st.geometric_mean([bt[ld] / ct[ld] for ld in LOADS])))
    for ld in LOADS:
        print("C_T %-9s ours %6.2f base %6.2f ratio %.3f  %%/us %.3f" % (
            ld, ct[ld], bt[ld], bt[ld] / ct[ld], 100.0 / (6 * ct[ld])))
    hdr = "load|layer|m|ours|min|max|kernel|sglang|judge_sol|bw|math989|math700|floor|head_us|head_%|%/us|kernels"
    print(hdr)
    for r in sorted(out, key=lambda r: -r[14]):
        print("|".join(x if isinstance(x, str) else ("%.2f" % x if isinstance(x, float) else str(x))
                       for x in r))


if __name__ == "__main__":
    main(sys.argv[1:])
