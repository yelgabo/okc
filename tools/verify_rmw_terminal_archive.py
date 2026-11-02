import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "docs/steering/evidence/2026-11-01-rmw-common-pair-v4"
    manifest = json.loads((evidence / "terminal-archive-manifest.json").read_text())
    for relative, expected in manifest.items():
        raw = (root / relative).read_bytes()
        actual = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        assert actual == expected, (relative, actual, expected)
    pins = {
        ".build/rmw-pair-v4/check.py": (924005, "785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66"),
        "tools/g_common_rmw_pair_read.py": (10920, "745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d"),
        "tools/g_common_rmw_pair_manifest.json": (4821, "c5f0e3975e30f4c012be8320615c7a19bbf9a7b7a63381906c10f3a3b3d4c574"),
    }
    for relative, expected in pins.items():
        raw = (root / relative).read_bytes()
        assert (len(raw), hashlib.sha256(raw).hexdigest()) == expected, relative
    output = root / ".build/rmw-pair-v4/replacement-reader-observation.json"
    command = [sys.executable, "tools/g_common_rmw_pair_read.py", str(evidence / "own-terminal-receipt.json"), "--job", "c5e4f89672f743bb", "--source", ".build/rmw-pair-v4/check.py", "--log", "muse_decode_layer/profiles/g-common-rmw-pair-v4.c5e4f89672f743bb.log", "--archive-attestation", str(evidence / "terminal-log-attestation.json"), "--output", str(output)]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(output.read_text())
    archived = json.loads((evidence / "terminal-reader-observation.json").read_text())
    numeric_differences = []
    def compare(actual, expected, location="root"):
        if isinstance(actual, float) and isinstance(expected, float):
            assert math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15), location
            if actual != expected:
                numeric_differences.append({"path": location, "archived": expected, "local": actual})
        elif isinstance(actual, dict):
            assert actual.keys() == expected.keys(), location
            for key in actual:
                compare(actual[key], expected[key], location + "." + key)
        elif isinstance(actual, list):
            assert len(actual) == len(expected), location
            for index, value in enumerate(actual):
                compare(value, expected[index], location + "." + str(index))
        else:
            assert actual == expected, location
    compare(observed, archived)
    assert observed["complete"] is True and observed["private_raw_gate"] is False
    result = {
        "job": "c5e4f89672f743bb",
        "archive_files_verified": len(manifest),
        "verified_paths": list(manifest),
        "frozen_pins_verified": list(pins),
        "reader_exitcode": completed.returncode,
        "reader_stdout": completed.stdout.strip(),
        "reader_stderr": completed.stderr,
        "nonfloat_fields_exact_and_floats_within_1e14_relative_1e15_absolute": True,
        "cross_platform_numeric_differences": numeric_differences,
        "complete": observed["complete"],
        "private_raw_gate": observed["private_raw_gate"],
        "scope": "Local CPU archive verification only; no judge, GPU, quota, or job-status requests",
    }
    (evidence / "replacement-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "verified_paths"}, indent=2))


if __name__ == "__main__":
    main()
