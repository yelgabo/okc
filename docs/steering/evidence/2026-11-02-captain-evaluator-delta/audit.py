import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from dl_compare_receipts import validate

OUT = Path(__file__).resolve().parent
INPUTS = (
    ("muse_decode_layer/profiles/scored.607eb52bdaf24160.json", "607eb52bdaf24160", "run"),
    ("docs/steering/evidence/2026-11-02-captain-fresh-incumbent-profile/control.cc474883117947c0.json", "cc474883117947c0", "check"),
    ("muse_decode_layer/baseline_profile/status.7d42771c3384449d.json", "7d42771c3384449d", "check"),
)
receipts = [json.loads((ROOT / path).read_bytes()) for path, _, _ in INPUTS]
checked = [validate(receipt, job, mode) for receipt, (_, job, mode) in zip(receipts, INPUTS)]
historical, fresh, baseline = [receipt["result"]["evaluator"] for receipt in receipts]
assert historical["files"] == baseline["files"]
changed = {path: {"historical": value, "fresh": fresh["files"].get(path)}
           for path, value in historical["files"].items() if fresh["files"].get(path) != value}
assert set(changed) == {"configs/muse-spark-open.yaml"}
assert set(historical["files"]) == set(fresh["files"])
assert checked[0]["content_hash"] == checked[1]["content_hash"]
cases = [sorted((json.dumps({key: value for key, value in row["case"].items() if key != "seed"}, sort_keys=True) for row in receipt["shapes"].values()))
         for receipt in checked]
assert cases[0] == cases[1] == cases[2]
result = {
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "inputs": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path, _, _ in INPUTS},
    "jobs": [job for _, job, _ in INPUTS],
    "receipt_validation": "PASS: owning identity, complete official records and score reconstruction",
    "benchmark_cases_equal_excluding_seed": True,
    "excluded_case_fields": ["seed"],
    "benchmark_case_count": len(cases[0]),
    "benchmark_cases": [json.loads(case) for case in cases[0]],
    "manifests": {name: receipt["result"]["evaluator"] for name, receipt in zip(("public", "fresh", "baseline"), receipts)},
    "changed_manifest_paths": changed,
    "unchanged_manifest_paths": sorted(set(historical["files"]) - set(changed)),
    "semantic_config_delta": "UNKNOWN: no config bytes in owning receipts",
    "not_established": ["same sampled tensors or sequence lengths", "same dependency versions",
                        "same baseline kernel selection", "causal performance effect of config or image"],
}
OUT.joinpath("audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: result[key] for key in ("jobs", "receipt_validation", "benchmark_case_count", "changed_manifest_paths", "semantic_config_delta")}))
