"""Build one HTML page that shows every saved run's profile: where each shape's time goes, by
kernel family, next to SGLang's breakdown and the floor, plus each problem's score history.

    python3 tools/profile_viz.py [--out .build/profiles.html]

Reads <problem>/profiles/*.json and <problem>/baseline_profile/profiles/*.json, and the change
column of the docs/runs*.md row that names each job id.
"""

import argparse
import glob
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROW = re.compile(r"(.+?)\s+([\d.]+)\s*µs\s+([\d.]+)%\s+(\d+)\s*$")
TOTAL = re.compile(r"total\s+([\d.]+)\s*µs")
MORE = re.compile(r"\.\.\. (\d+) more")

FAMILIES = [
    ("moe", re.compile(r"marlin|moe_wna16|fused_moe|grouped_gemm", re.I)),
    ("attn", re.compile(r"flash|attn_kernel|combine_kernel|prepare_varlen", re.I)),
    ("router", re.compile(r"router_kernel", re.I)),
    ("gemm", re.compile(r"nvjet|cublas|gemv|dot_kernel|proj_kernel|gemm|splitK", re.I)),
    ("norm", re.compile(r"norm|rms", re.I)),
    ("moeglue", re.compile(r"finalize|align|act_and_mul|topk|moe_sum|silu", re.I)),
]


def family(name):
    for fam, pattern in FAMILIES:
        if pattern.search(name):
            return fam
    return "misc"


def short_name(raw):
    name = re.sub(r"^void\s+|\(anonymous namespace\)::|sglang::|device::", "", raw.strip())
    return name.split("(")[0].strip()


def parse_report(report):
    kernels, total, more = [], None, 0
    for line in report.splitlines()[1:]:
        if m := TOTAL.match(line):
            total = float(m.group(1))
        elif m := MORE.match(line):
            more = int(m.group(1))
        elif (m := ROW.match(line)) and not line.startswith("GPU busy"):
            kernels.append({
                "name": short_name(m.group(1)),
                "ours": bool(re.match(r"(void\s+)?\(anonymous namespace\)", m.group(1))),
                "fam": family(m.group(1)),
                "us": float(m.group(2)),
                "calls": int(m.group(4)),
            })
    listed = sum(k["us"] for k in kernels)
    return {"kernels": kernels, "total": total or listed,
            "unlisted": max((total or listed) - listed, 0), "more": more}


def case_key(case):
    return " ".join(f"{k}={v}" for k, v in sorted(case.items()) if k != "seed")


def label(case):
    skip = ("load", "seed", "weight", "sharding", "ctx", "heads", "page", "spread")
    rest = [f"{v}" if k in ("layer", "dist") else f"{k}={v}" for k, v in sorted(case.items())
            if k not in skip]
    return " ".join([case.get("load", "")] + rest)


def commit_time(path):
    out = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%ct", "--", path],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out else int(os.path.getmtime(path))


def run_notes():
    notes = {}
    for path in glob.glob(os.path.join(ROOT, "docs", "runs*.md")):
        change_col = None
        for line in open(path):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "change" in cells:
                change_col = cells.index("change")
            elif change_col is not None and len(cells) > change_col:
                for job in re.findall(r"\b[0-9a-f]{16}\b", line):
                    notes.setdefault(job, cells[change_col])
    return notes


def load_run(path, kind, notes):
    d = json.load(open(path))
    d = d.get("result", d) if "benchmarks" not in d else d
    if not d or "benchmarks" not in d:
        return None
    profiles = {case_key(p["case"]): parse_report(p["report"]) for p in d.get("profiles", [])}
    shapes = []
    for b in d["benchmarks"]:
        case = b["case"]
        shapes.append({
            "key": case_key(case),
            "label": label(case),
            "weight": case.get("weight", 1),
            "base": b["baseline_ns"] / 1e3,
            "ours": b["median_ns"] / 1e3,
            "floor": b["speed_of_light_ns"] / 1e3,
            "profile": profiles.get(case_key(case)),
        })
    s = d.get("summary", {})
    return {
        "job": d["job_id"],
        "kind": kind,
        "file": os.path.relpath(path, ROOT),
        "time": commit_time(path),
        "score": s.get("speedup"),
        "ci": s.get("speedup_ci"),
        "passed": s.get("passed"),
        "note": notes.get(d["job_id"], ""),
        "shapes": shapes,
    }


def collect():
    notes = run_notes()
    problems = {}
    for pdir in sorted(glob.glob(os.path.join(ROOT, "muse_*"))):
        runs = []
        for path in glob.glob(os.path.join(pdir, "profiles", "*.json")):
            kind = os.path.basename(path).split(".")[0]
            runs.append(load_run(path, kind, notes))
        for path in glob.glob(os.path.join(pdir, "baseline_profile", "profiles", "*.json")):
            runs.append(load_run(path, "baseline", notes))
        runs = sorted((r for r in runs if r), key=lambda r: r["time"])
        if runs:
            problems[os.path.basename(pdir)] = runs
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, ".build", "profiles.html"))
    args = ap.parse_args()
    template = open(os.path.join(ROOT, "tools", "profile_viz.html")).read()
    data = json.dumps(collect(), separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(template.replace("/*DATA*/null", data))
    print(args.out)


if __name__ == "__main__":
    main()
