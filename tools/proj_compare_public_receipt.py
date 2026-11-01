"""Compare all scored projection shapes to a public v18 result extract."""
import json
import math
import statistics
import sys
from pathlib import Path


def shape_key(case):
    return case["load"], case["m"], case["layer"], case["weight"]


def comparison(external, receipts):
    public = {shape_key(row): row["solution_us"] for row in external["benchmarks"]}
    assert len(public) == len(external["benchmarks"]) == 30
    runs = []
    for receipt in receipts:
        result = receipt.get("result", receipt)
        rows = result.get("benchmarks") or [r for r in result["records"] if r["event"] == "benchmark"]
        times = {shape_key(r["case"]): r["median_ns"]/1000 for r in rows}
        assert len(rows) == 30 and times.keys() == public.keys()
        runs.append(times)
    print("| Load | Projection | Public v18 µs | C3 range µs | C3 median minus v18 µs |")
    print("|---|---|---:|---:|---:|")
    for key, other in public.items():
        values = [run[key] for run in runs]
        print(f"| {key[0]} | {key[2]} | {other:.3f} | {min(values):.3f}–{max(values):.3f} | {statistics.median(values)-other:+.3f} |")
    print()
    for run, receipt in zip(runs, receipts):
        load_ratios = []
        for load in sorted({key[0] for key in public}):
            keys = [k for k in public if k[0] == load]
            original = sum(k[3]*run[k] for k in keys)
            mixed = sum(k[3]*min(run[k], public[k]) for k in keys)
            load_ratios.append(original/mixed)
        print(f"- {receipt.get('id') or receipt['job_id']} retrospective factor {math.prod(load_ratios)**(1/6):.6f}")


if __name__ == "__main__":
    data = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:]]
    comparison(data[0], data[1:])
