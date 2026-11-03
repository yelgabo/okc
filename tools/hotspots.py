"""Where the time goes, per shape: SGLang vs ours vs the hardware floor, with each side's kernels.

Reads okc's saved run JSONs (profiles/*.json, or the JSON from `okc status <job> --json`).
Our kernels come from the submission run's profile; SGLang's come from a check run of
baseline.py submitted as the kernel (<problem>/baseline_profile/), since a normal run profiles
only the submission.

    python3 tools/hotspots.py muse_bf16_proj/profiles/submission.<job>.json \
        [--baseline muse_bf16_proj/baseline_profile/<file>.json] >> docs/hotspots.md
"""

import argparse
import json
import re


def load(path):
    with open(path, encoding="utf-8") as f:
        envelope = json.load(f)
    d = dict(envelope.get("result", envelope))
    # `okc status` returns an envelope with event records, whereas the
    # profile JSON saves normalized benchmark/profile lists.
    if "benchmarks" not in d:
        records = d.get("records", [])
        d["benchmarks"] = [r for r in records if r.get("event") == "benchmark"]
        d["profiles"] = [r for r in records if r.get("event") == "profile"]
    d.setdefault("job_id", envelope.get("id"))
    d.setdefault("problem", envelope.get("board"))
    d.setdefault("summary", {k: d[k] for k in ("speedup", "speedup_ci") if k in d})
    return d


def case_key(case):
    return tuple(sorted((k, v) for k, v in case.items() if k not in ("seed", "repeats")))


def label(case):
    parts = [case.get("load", "")]
    parts += [f"{k}={v}" for k, v in sorted(case.items()) if k not in ("load", "seed", "weight")]
    return " ".join(p for p in parts if p)


def kernels(report, top=3):
    """'name  12.3 µs  45.6%  1' rows -> ['name 12.3us 46%', ...], largest first."""
    rows = []
    for line in report.splitlines()[1:]:
        m = re.match(r"(.+?)\s+([\d.]+)\s*µs\s+([\d.]+)%\s+(\d+)\s*$", line)
        if m and not m.group(1).startswith("total"):
            name = re.sub(r"^void\s+|\(anonymous namespace\)::", "", m.group(1))
            name = name.split("(")[0].strip()
            rows.append((float(m.group(2)), f"{name[:48]} {m.group(2)}us {float(m.group(3)):.0f}%"))
    return [r for _, r in sorted(rows, reverse=True)[:top]]


def profiles_by_case(run):
    return {case_key(p["case"]): p["report"] for p in run.get("profiles", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline")
    args = ap.parse_args()
    run = load(args.run)
    ours = profiles_by_case(run)
    sglang = profiles_by_case(load(args.baseline)) if args.baseline else {}
    s = run["summary"]
    print(f"### {run['problem']}: job {run['job_id']}, {s['speedup']:.3f}x "
          f"({s['speedup_ci'][0]:.3f}-{s['speedup_ci'][1]:.3f})\n")
    if args.baseline:
        print(f"SGLang kernels from the baseline-as-submission run in `{args.baseline}`.\n")
    print("| shape | weight | SGLang us | ours us | floor us | SGLang/floor | ours/floor "
          "| our top kernels | SGLang top kernels |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in run["benchmarks"]:
        key = case_key(b["case"])
        base_us = b["baseline_ns"] / 1000
        our_us = b["median_ns"] / 1000
        floor_us = b["speed_of_light_ns"] / 1000
        print(f"| {label(b['case'])} | {b['case'].get('weight', '')} | {base_us:.1f} | "
              f"{our_us:.1f} | {floor_us:.1f} | {base_us / floor_us:.1f}x | "
              f"{our_us / floor_us:.1f}x | {'<br>'.join(kernels(ours.get(key, '')))} | "
              f"{'<br>'.join(kernels(sglang.get(key, '')))} |")
    print()


if __name__ == "__main__":
    main()
