"""Reconcile six owning receipts and audit the CPU/native coverage boundary.

Offline only. Does not import submission code, dispatch jobs, or call CUDA.
"""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CLASSIFICATION = {
    "1fad70bbe3854cbc": "external_infrastructure_refunded",
    "3fd1317143304bd5": "our_setup_budget_failure",
    "c1931f0c9e834485": "completed_qualifier_not_matched_promotion",
    "16b4db27421d4714": "our_setup_failure_invalid_fixture_found",
    "c5e4f89672f743bb": "completed_matched_subthreshold",
    "9adfd4dbc98f4998": "our_setup_failure_graph_query",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cohort = json.loads((HERE / "cohort.json").read_text())
    rows = []
    for item in cohort["rows"]:
        job = item["job_id"]
        path = HERE / (job + ".json")
        # The archived JSON was serialized as UTF-8/LF. Git may check it out
        # as CRLF on Windows; verify that serialization rather than checkout
        # newline conversion. This does not normalize receipt field values.
        archived = path.read_text(encoding="utf-8").encode("utf-8")
        assert hashlib.sha256(archived).hexdigest() == item["receipt_sha256"]
        data = json.loads(path.read_text())
        assert (data["id"], data["board"], data["team"]) == (
            job, "muse_decode_layer", "yelgabo"
        )
        result = data.get("result") or {}
        kind = CLASSIFICATION[job]
        if kind.startswith("our_setup"):
            assert result["passed"] is False and result["returncode"] == 112
            assert result["speedup"] is None
            assert [r["event"] for r in result["records"]] == ["setup"]
            assert result["records"][0]["status"] == "fail"
        elif kind.startswith("completed"):
            assert result["passed"] is True and result["returncode"] == 0
            events = [r["event"] for r in result["records"]]
            assert events.count("test") == 13
            assert events.count("benchmark") == events.count("profile") == 12
        else:
            assert result["status"] == "infra_error"
            assert data.get("refunded", result.get("refunded")) is True
        rows.append(dict(job_id=job, classification=kind))

    # These are concrete source facts, not a simulation of CUDA semantics.
    query = ROOT / "tools/g_outgoing_seam_query.py"
    query_tree = ast.parse(query.read_text())
    source = next(
        ast.literal_eval(n.value)
        for n in query_tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SECONDARY" for t in n.targets)
    )
    assert "cudaGraphKernelNodeGetParams(node,&params)" in source
    assert "for(auto node:nodes)" in source

    tests = ROOT / "tools/test_g_outgoing_pair.py"
    tree = ast.parse(tests.read_text())
    mock_method = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "seam_graph_edges"
    )
    assert "graph_rows(address - 1000)" in ast.unparse(mock_method)
    old_mock = ROOT / "tools/proj_g_short_rmw_host_test.py"
    tree = ast.parse(old_mock.read_text())
    old_native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Native")
    old_native_source = ast.unparse(old_native)
    assert "zero_()" in old_native_source
    assert "bincount" not in old_native_source and "unique" not in old_native_source

    output = {
        "scope": cohort["scope"],
        "receipt_evidence_commit": cohort["evidence_commit"],
        "classifications": rows,
        "counts": dict(our_setup_failures=3, external_infrastructure=1, completed=2),
        "source_facts": {
            "failed_query_uses_runtime_accessor_on_every_kernel_node": True,
            "graph_cpu_mock_returns_synthetic_rows_without_executing_cpp": True,
            "old_native_mock_zeroes_output_without_route_contract_checks": True,
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in (query, tests, old_mock)
        },
        "causality_limits": [
            "Invalid fixture is established; the original 719 device instruction was not localized.",
            "Graph binding failure is established; exact inner API/node/arm and runtime-driver cause remain unproved.",
            "Three setup failures do not prove all were completely preventable without target execution.",
        ],
    }
    (HERE / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
