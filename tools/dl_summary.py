"""Per-shape SGLang vs ours from okc run JSONs (profiles/*.json or `okc status --json`), with
an optional earlier run to compare speedups against.

    python3 tools/dl_summary.py muse_decode_layer/profiles/submission.<job>.json [<earlier>.json]
"""

import json
import sys


def load(path):
    d = json.load(open(path))
    return d.get("result", d) if "benchmarks" not in d else d


def rows(run):
    out = {}
    for b in run["benchmarks"]:
        c = b["case"]
        out[(c["load"], c["layer"])] = (b["baseline_ns"] / 1e3, b["median_ns"] / 1e3, b["speedup"])
    return out


def main():
    run = load(sys.argv[1])
    prev = rows(load(sys.argv[2])) if len(sys.argv) > 2 else {}
    s = run["summary"]
    lo, hi = s["speedup_ci"]
    print(f"job {run['job_id']} problem {run['problem']}: {s['speedup']:.3f}x "
          f"({lo:.3f}-{hi:.3f}), tests {s['tests_passed']}/{s['tests']}")
    print(f"{'load':10} {'layer':8} {'SGLang us':>10} {'ours us':>9} {'speedup':>8} {'prev':>7}")
    for (load_, layer), (base, ours, sp) in rows(run).items():
        before = prev.get((load_, layer))
        tail = f"{before[2]:7.3f}" if before else ""
        print(f"{load_:10} {layer:8} {base:10.1f} {ours:9.1f} {sp:8.3f} {tail}")


if __name__ == "__main__":
    main()
