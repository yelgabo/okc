#!/usr/bin/env python3
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path


def normalized(result):
    result = deepcopy(result)
    result["environment"].pop("executed_at_utc")
    for row in result["rows"]:
        for strategy in ("before", "after"):
            attempt = row[strategy]
            attempt["provider"]["results"].sort()
            for command in attempt["commands"]:
                if command["argv"][0] == "rg":
                    command["stdout"] = sorted(command["stdout"].splitlines())
    return result


def main():
    here = Path(__file__).resolve().parent
    saved = json.loads((here / "results.json").read_text())
    candidate = json.loads(Path(sys.argv[1]).read_text())
    initial = json.loads((here / "results-initial.json").read_text())
    checks = {"repeat_matches": normalized(saved) == normalized(candidate),
              "acceptance_passed": saved["acceptance_passed"] and candidate["acceptance_passed"]}
    for label, result, runner, design in (
        ("final", saved, "evaluate.py", "design.json"),
        ("initial", initial, "evaluate-initial.py.txt", "design-initial.json"),
    ):
        for kind, name in (("runner", runner), ("design", design)):
            digest = hashlib.sha256((here / name).read_bytes()).hexdigest()
            checks[f"{label}_{kind}_hash_matches"] = (
                digest == result["environment"][f"{kind}_sha256"])
    print(json.dumps({"checks": checks, "all_passed": all(checks.values()),
                      "totals": saved["totals"], "case_count": saved["case_count"]}, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
