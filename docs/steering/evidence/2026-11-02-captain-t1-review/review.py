import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from t1_moe_harness import build
from t1_moe_read import read

destination = Path(__file__).resolve().parent
held, held_manifest = build()
proposal, manifest = build(target_proposal=True)
assert held.replace("H_HOLD = True\n", "H_HOLD = False\n", 1) == proposal
assert held.count("H_HOLD = True\n") == 1
assert held.encode() == (ROOT / "docs/steering/evidence/2026-11-02-worker-t1-moe/payload.py").read_bytes()
assert {key for key in manifest if manifest[key] != held_manifest[key]} == {"source_sha256", "bytes", "hold"}
assert manifest["hold"] is False and manifest["dispatch_ready"] is False
assert manifest["native_candidate_sha256"] == "dbdd3381cc9b823afb9e7d8678bba536cd5d6e3731896153aee9908652237447"
assert manifest["affected_tokens"] == [1]
reader = (ROOT / "tools/t1_moe_read.py").read_text(encoding="utf8").encode()
assert hashlib.sha256(reader).hexdigest() == manifest["reader_canonical_lf_sha256"]
destination.joinpath("target-proposal.py").write_bytes(proposal.encode())
destination.joinpath("target-proposal.manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
result = {"source_sha256": manifest["source_sha256"], "bytes": manifest["bytes"], "reader_canonical_lf_sha256": manifest["reader_canonical_lf_sha256"], "hold_only_source_change": True, "manifest_changed_keys": ["source_sha256", "bytes", "hold"], "worker_held_exact": True, "independent_host_tests": 13, "host_test_seconds": 28.882, "target_qualified": False}
destination.joinpath("review.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
print(json.dumps(result))
