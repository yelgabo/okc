import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from prod_size_compile_g import bindings

source_path = ROOT / ".build/rmw-pair-v4/check.py"
raw = source_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66"
source = raw.decode("utf8")
environment = bindings(source)
selected = {"main_b4": environment["MOE_CUDA_SRC"], **environment["_MA_ALT_SRCS"]}
factory_path = ROOT / "docs/steering/evidence/2026-11-01-rmw-common-pair-v4/factory-invariance.json"
factories = json.loads(factory_path.read_text())["factories"]
result = {
    "scope": "Source/CPU inspection only, no submission import, compilation, CUDA, GPU, judge or numerical SiLU evaluation",
    "decision": "HOLD_RULE_ELIGIBILITY_BEFORE_IMPLEMENTATION",
    "source": {"path": str(source_path.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
    "binding_evaluator_sha256": hashlib.sha256((ROOT / "tools/prod_size_compile_g.py").read_bytes()).hexdigest(),
    "factory_receipt_sha256": hashlib.sha256(factory_path.read_bytes()).hexdigest(),
    "selected_sources": {},
    "factories": factories,
    "domain_counts": {"raw_words": 65536, "finite": 0, "infinity": 0, "nan": 0, "subnormal": 0},
    "table_bytes": 65536 * 4,
    "source_binding_limit": "Existing pure-assignment evaluator does not execute augmented assignments/comprehensions; actual final build flags come from the separately pinned seven-factory receipt, not evaluator define tuples.",
}
for raw_word in range(65536):
    exponent = (raw_word >> 7) & 255
    fraction = raw_word & 127
    if exponent == 255:
        result["domain_counts"]["infinity" if fraction == 0 else "nan"] += 1
    else:
        result["domain_counts"]["finite"] += 1
        if exponent == 0 and fraction != 0:
            result["domain_counts"]["subnormal"] += 1
for name, cuda_source in selected.items():
    sites = []
    last_function = ""
    for number, line in enumerate(cuda_source.splitlines(), 1):
        if "__device__" in line and "(" in line:
            last_function = line.strip()
        if "SILU_DIV(gt," in line:
            sites.append({"literal_line": number, "nearest_device_declaration": last_function, "text": line.strip()})
    result["selected_sources"][name] = {"bytes": len(cuda_source.encode()), "sha256": hashlib.sha256(cuda_source.encode()).hexdigest(), "gate_site_superset": sites}
source_hashes = {entry["sha256"] for entry in result["selected_sources"].values()}
assert len(factories) == 7
assert {entry["cuda_sha256"] for entry in factories} == source_hashes
assert all("--use_fast_math" not in entry["flags"] for entry in factories)
assert result["domain_counts"] == {"raw_words": 65536, "finite": 65280, "infinity": 2, "nan": 254, "subnormal": 254}
result["source_assignments"] = [{"line": node.lineno, "target": ast.unparse(node.targets[0]), "bytes": len(node.value.value.encode()), "sha256": hashlib.sha256(node.value.value.encode()).hexdigest()} for node in ast.parse(source).body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and "SILU_DIV(gt," in node.value.value]
output = Path(__file__).with_name("source-screen.json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"decision": result["decision"], "source": result["source"], "domain_counts": result["domain_counts"], "table_bytes": result["table_bytes"], "sources": {name: {"sha256": entry["sha256"], "sites": len(entry["gate_site_superset"])} for name, entry in result["selected_sources"].items()}, "factories": len(factories)}, indent=2))

