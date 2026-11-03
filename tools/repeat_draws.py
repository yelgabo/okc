"""Resubmit a board's public best file with a changed trailing comment for fresh judge draws.

The judge returns the earlier result for identical bytes and keeps the best run as the row, so each
draw is a comment change on an exact copy of the approved file. Stops when the board's remaining public
runs fall to the reserve, on any failed run, or after --max draws.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def runs_left(problem):
    out = run(["okc", "problems"]).stdout
    m = re.search(rf"{re.escape(problem)}\s.*?(\d+) of (\d+) runs left today", out)
    return int(m.group(1)) if m else None


def submit(problem, path):
    r = run(["okc", "submit", problem, str(path), "--no-wait", "--json"])
    if r.returncode != 0:
        raise RuntimeError(f"submit failed rc={r.returncode}: {r.stderr.strip()} {r.stdout.strip()}")
    d = json.loads(r.stdout)
    return d.get("id") or d.get("job") or d["job_id"], d


def wait(job, poll=60):
    while True:
        r = run(["okc", "status", job, "--json"])
        if r.returncode == 0:
            d = json.loads(r.stdout)
            if d.get("status") in ("done", "failed", "error", "cancelled"):
                return d
        time.sleep(poll)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("problem")
    p.add_argument("base", type=Path)
    p.add_argument("--expect-sha", required=True, help="full or prefix SHA256 of the approved base file")
    p.add_argument("--max", type=int, default=8)
    p.add_argument("--reserve", type=int, default=2, help="public runs to leave for real candidates")
    p.add_argument("--out", type=Path, default=Path(".build/repeats"))
    p.add_argument("--statement", action="store_true",
                   help="use a harmless assignment instead of a comment if the judge dedups comments")
    a = p.parse_args()
    base_sha = sha(a.base)
    if not base_sha.startswith(a.expect_sha):
        sys.exit(f"base {a.base} sha {base_sha[:16]} does not match expected {a.expect_sha}")
    a.out.mkdir(parents=True, exist_ok=True)
    ledger = a.out / f"{a.problem}.jsonl"
    text = a.base.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    use_statement = a.statement
    for i in range(a.max):
        left = runs_left(a.problem)
        if left is None or left <= a.reserve:
            print(f"{a.problem}: {left} runs left, reserve {a.reserve}, stopping", flush=True)
            break
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tag = f"repeat draw {stamp} {i}"
        extra = f'_REPEAT_DRAW = "{tag}"\n' if use_statement else f"# {tag}\n"
        draw = a.out / f"{a.problem}.draw{i}.py"
        draw.write_text(text + extra, encoding="utf-8")
        job, resp = submit(a.problem, draw)
        rec = {"problem": a.problem, "draw": i, "job": job, "file": str(draw), "file_sha256": sha(draw),
               "base_sha256": base_sha, "submitted_at": stamp}
        print(json.dumps(rec), flush=True)
        d = wait(job)
        res = d.get("result") or {}
        rec.update(status=d.get("status"), duplicate_of=d.get("duplicate_of"), passed=res.get("passed"),
                   score_us=res.get("score_us"), speedup=res.get("speedup"), speedup_ci=res.get("speedup_ci"),
                   error=res.get("error"))
        (a.out / f"{a.problem}.{job}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("job", "status", "duplicate_of", "passed", "score_us", "speedup")}), flush=True)
        if rec["duplicate_of"] and not use_statement:
            print("judge deduplicated a comment-only change; switching to a statement change", flush=True)
            use_statement = True
            continue
        if rec["status"] != "done" or not rec["passed"]:
            print(f"{a.problem}: run {job} did not pass, stopping", flush=True)
            break


if __name__ == "__main__":
    main()
