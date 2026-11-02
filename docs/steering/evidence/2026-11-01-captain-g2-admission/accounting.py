import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    design_path = root / "docs/steering/evidence/2026-11-01-expert-pipeline-next/analyze.py"
    design = design_path.read_bytes()
    tree = ast.parse(design)
    layout_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "layout")
    namespace = {"ctypes": ctypes}
    exec(compile(ast.Module(body=[layout_node], type_ignores=[]), str(design_path), "exec"), namespace)
    baseline = namespace["layout"](16, 88)
    proposed = namespace["layout"](8, 168)
    assert baseline == {"sizeof": 22240, "ring_start": 22272}
    assert proposed == {"sizeof": 23776, "ring_start": 23808}
    dynamic_bytes = proposed["ring_start"] + 3 * 18432 + 3 * 12288
    assert dynamic_bytes == 115968
    header_bytes = args.header.read_bytes()
    header = header_bytes.decode()
    granularity_body = header.split("static __OCC_INLINE cudaOccError cudaOccSMemAllocationGranularity", 1)[1].split("\n}", 1)[0]
    assert "case 9:" in granularity_body
    assert "value = 128;" in granularity_body
    assert "staticSmemSize = attributes->sharedSizeBytes + properties->reservedSharedMemPerBlock;" in header
    assert "totalSmemUsagePerCTA = staticSmemSize + dynamicSmemSize;" in header
    assert "smemAllocatedPerCTA = __occRoundUp((int)totalSmemUsagePerCTA, (int)allocationGranularity);" in header
    capacity = 233472
    granularity = 128
    cases = []
    for reserved_bytes in (0, 512, 768, 1024):
        allocated = ((dynamic_bytes + reserved_bytes + granularity - 1) // granularity) * granularity
        cases.append({"reserved_per_block_assumption": reserved_bytes, "static_per_block_assumption": 0,
                      "allocated_per_block": allocated, "two_block_bytes": 2 * allocated,
                      "capacity": capacity, "shared_only_block_limit": capacity // allocated})
    assert cases[0]["shared_only_block_limit"] == 2
    assert cases[-1]["shared_only_block_limit"] == 1
    assert cases[-1]["two_block_bytes"] - capacity == 512
    max_dynamic_at_1024 = (capacity // 2 // granularity) * granularity - 1024
    assert dynamic_bytes - max_dynamic_at_1024 == 256
    print(json.dumps({"kind": "CPU shared-memory necessary-condition accounting, NOT native occupancy execution",
                      "design_script_sha256": hashlib.sha256(design).hexdigest(),
                      "official_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                      "official_header": str(args.header), "proposed_layout": proposed,
                      "dynamic_bytes": dynamic_bytes, "sm90_header_allocation_granularity": granularity,
                      "cases": cases, "max_dynamic_for_two_at_1024_reservation_and_zero_static": max_dynamic_at_1024,
                      "minimum_per_block_reduction_under_these_assumptions": 256,
                      "target_reserved_shared_memory": None,
                      "target_runtime_occupancy": None,
                      "source_binding_assertions": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
