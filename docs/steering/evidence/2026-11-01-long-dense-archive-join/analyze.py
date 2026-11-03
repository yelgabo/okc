import ast
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
SOURCES = {
    "607eb52bdaf24160": ("muse_decode_layer/cand/g1b4_stage_release.py", "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"),
    "c1931f0c9e834485": (".build/rmw-v2/short-rmw.py", "978b29f955dbbab3a67aad0cc06a04148aba9d233ac95bdcac45d1efd141fc2a"),
    "c5e4f89672f743bb": (".build/rmw-pair-v4/check.py", "785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66"),
    "9adfd4dbc98f4998": ("docs/steering/evidence/2026-11-01-worker-outgoing-pair/check.py.gz", "f25d714e928835fba07741ca56fa8783137032f268652c22ec4f5511d771bc9a"),
}
PREFIXES = {
    "607eb52bdaf24160": "g1-stage-release",
    "c1931f0c9e834485": "g-common-rmw-v2",
    "c5e4f89672f743bb": "g-common-rmw-pair-v4",
    "9adfd4dbc98f4998": "g-outgoing-pair",
}
FUNCTIONS = ("dense", "_rq_gemm", "_graft_lt_plan", "_graft_lt_run", "_graft_layouts", "_graft_boundary", "_policy_variant", "_graft_after_attention")


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def matching_keys(value, path=""):
    matches = []
    if isinstance(value, dict):
        for key, item in value.items():
            location = path + "/" + key
            if any(term in key.lower() for term in ("compose", "graft_lt", "mm_pick")):
                matches.append({"path": location, "value": str(item)[:300]})
            matches.extend(matching_keys(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(matching_keys(item, path + "/" + str(index)))
    return matches


records = {}
for job, (source_path, expected_hash) in SOURCES.items():
    raw = (ROOT / source_path).read_bytes()
    source_raw = gzip.decompress(raw) if source_path.endswith(".gz") else raw
    assert identity(source_raw)["sha256"] == expected_hash
    source = source_raw.decode()
    nodes = ast.parse(source).body
    constants, functions = {}, {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("_GRAFT_LT_PICKS", "_GRAFT_QKV256", "_GRAFT_LT_T"):
                    constants[target.id] = {"line": node.lineno, "text": ast.get_source_segment(source, node)}
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            functions[node.name] = {"line": node.lineno, "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(), "text": ast.get_source_segment(source, node)}
    prefix = PREFIXES[job]
    receipt_path = "muse_decode_layer/profiles/" + prefix + "." + job + ".json"
    log_path = "muse_decode_layer/profiles/" + prefix + "." + job + ".log"
    receipt_raw = (ROOT / receipt_path).read_bytes()
    receipt = json.loads(receipt_raw)
    log_raw = (ROOT / log_path).read_bytes()
    log_lines = log_raw.decode(errors="replace").splitlines()
    accepted = [{"line": index + 1, "text": line} for index, line in enumerate(log_lines) if "Lt plans" in line]
    result_records = (receipt.get("result") or {}).get("records", [])
    all_profiles = receipt.get("profiles", []) or [item for item in result_records if item.get("event") == "profile"]
    all_benchmarks = receipt.get("benchmarks", []) or [item for item in result_records if item.get("event") == "benchmark"]
    profiles = [item for item in all_profiles if item["case"]["load"] in ("verify 16", "verify 32")]
    records[job] = {
        "source": {"path": source_path, **identity(source_raw)}, "constants": constants, "functions": functions,
        "receipt": {"path": receipt_path, **identity(receipt_raw), "job_id": receipt.get("job_id", receipt.get("id")), "summary": receipt.get("summary"), "result": {key: value for key, value in (receipt.get("result") or {}).items() if key in ("passed", "returncode", "speedup", "status", "score_us")}, "profile_count": len(all_profiles), "benchmark_count": len(all_benchmarks)},
        "log": {"path": log_path, **identity(log_raw), "lines": len(log_lines), "accepted_plan_lines": accepted, "composition_lines": [{"line": index + 1, "text": line[:500]} for index, line in enumerate(log_lines) if any(term in line for term in ("compose_selection", "compose_graft_enabled", "compose_boundary_warps", "graft_lt_status"))]},
        "long_profiles": profiles, "receipt_selection_keys": matching_keys(receipt),
        "source_composition_lines": [{"line": index + 1, "text": line} for index, line in enumerate(source.splitlines()) if any(term in line for term in ("modes=[(original", "compose_selection", "compose_graft_enabled", "compose_boundary_warps"))],
    }
    if profiles:
        assert len(profiles) == 4
        assert all("_graft_row_boundary" in item["report"] and "_graft_route" in item["report"] for item in profiles)
        assert "modes=[(original,16,()),(True,16,()),(False,16,()),(False,8,()),(False,4,())]" in source
        records[job]["long_composition_inference"] = {"graft_enabled": True, "mode": [True, 16, []], "basis": "All four owning profiles name graft boundary/route; source's sole enabled mode has fallback warps16 and no plan drops.", "actual_graft_boundary_warps": {"128": 32, "256": 16}}
    else:
        records[job]["long_composition_inference"] = None
capsule_path = "docs/steering/evidence/2026-11-01-rmw-common-pair-v4/terminal-reader-observation.json"
capsule_raw = (ROOT / capsule_path).read_bytes()
capsule = json.loads(capsule_raw)
donor_path = "docs/steering/evidence/2026-11-01-ruwwww-complete-qualification/analysis.json"
donor = json.loads((ROOT / donor_path).read_text())
comparison = {}
for name in FUNCTIONS:
    hashes = {job: item["functions"][name]["ast_sha256"] for job, item in records.items()}
    comparison[name] = {"hashes": hashes, "all_identical": len(set(hashes.values())) == 1}
output = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "scope": "Existing source/archive AST and text only. No source execution, instrumentation, compiler, GPU, CLI or network.",
    "interface": {"path": "muse_decode_layer/baseline.py", **identity((ROOT / "muse_decode_layer/baseline.py").read_bytes()), "weight_shapes": {"qkv": [2304, 8192], "o": [8192, 1024], "pr_pool_hi_lo": [4608, 8192], "un": [8192, 4096]}},
    "jobs": records, "function_comparison": comparison,
    "c5e4_private_capsule": {"path": capsule_path, **identity(capsule_raw), "job": capsule["job"], "complete": capsule["complete"], "selection_keys": matching_keys(capsule)},
    "ruwwww": {"evidence": donor_path, "pin": donor["remote_pin"], "source": donor["source"], "job": donor["receipt"]["job_id"], "long_profiles": [item for item in donor["receipt"]["profiles"] if item["case"]["load"] in ("verify 16", "verify 32")]},
    "constant_texts_identical": {name: len({item["constants"][name]["text"] for item in records.values()}) == 1 for name in ("_GRAFT_LT_PICKS", "_GRAFT_QKV256")},
    "decision": "SCOPED STOP: no independent-material candidate. Plan-key acceptance recovered for607/c5e4/9ad; c193 only profile-compatible.607/c193/c5e4 long composition inferred enabled from unique graft boundaries and source modes.9ad has no official long profiles or composition state. Aggregated library rows do not give operation identity or exposure; no unmatched timing subtraction qualifies a replacement.",
}
Path(__file__).with_name("analysis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"jobs": {job: {"accepted_plan_log": bool(item["log"]["accepted_plan_lines"]), "long_profiles": len(item["long_profiles"]), "composition_log": bool(item["log"]["composition_lines"])} for job, item in records.items()}, "function_comparison": {name: item["all_identical"] for name, item in comparison.items()}, "capsule_selection_keys": output["c5e4_private_capsule"]["selection_keys"], "decision": output["decision"]}, indent=2))
