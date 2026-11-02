import ast
import copy
import ctypes
import hashlib
import json
import re
from pathlib import Path


def body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def check_order(source):
    route = body(source, "__device__ void route(")
    assert route.rstrip().endswith("CTA_SYNC();\n}"), "routing end barrier missing"
    assert source.count("S.wc[") == route.count("S.wc["), "wc access outside routing"
    assert "S.wcp" not in route, "prefix accessed during routing"
    early = body(source, "__device__ void early_issue(const Params& p, Smem& S) {")
    assert "#if FAST1_ON || OKC_WGS\n  return;" in early, "WGS early-issue no longer returns"
    assert source.count("wgs_prefix(S,") == 0, "unexpected prefix caller"
    assert source.count("wgs_ranges(p, S);") == 1, "unexpected range caller"
    producer = body(source, "__device__ void wgs_xproducer(")
    assert "bar.sync 7" in producer, "producer barrier missing"
    assert producer.index("bar.sync 7") < producer.index("wgs_ranges(p, S);"), "prefix begins before producer barrier"
    kernel = body(source, "__global__ void __launch_bounds__(NTL, 1) moe_fused(")
    assert kernel.index("route(p, S);") < kernel.index("bar.arrive 7"), "routing handoff missing"
    assert "__threadfence_block();" in kernel[kernel.index("route(p, S);"):kernel.index("bar.arrive 7")]
    assert '#define OKC_WGSPR 1' in source
    assert '#define CTA_SYNC() asm volatile("bar.sync 1, %0;' in source
    return {"wc_accesses": route.count("S.wc["), "prefix_range_callers": 1,
            "routing_end_barrier": True, "producer_barrier7_handoff": True,
            "scope": "pinned WGS/WGSPR selected path; source assertions, not GPU proof"}


def main():
    root = Path(__file__).resolve().parents[4]
    raw = (root / "muse_decode_layer/cand/g1b4_stage_release.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"
    tree = ast.parse(raw)
    source = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(getattr(target, "id", "") == "MOE_CUDA_SRC" for target in node.targets))
    assert hashlib.sha256(source.encode()).hexdigest() == "c7c201ae145c6a33f2d5e6b1f80d2a6e56e8d75502f9fd09ace40927ad0f7c71"
    ordering = check_order(source)
    negatives = []
    for name, old, new in (
        ("no_route_barrier", "  CTA_SYNC();\n}\n", "}\n"),
        ("no_producer_barrier", "bar.sync 7", "bar.sync 6"),
        ("late_wc_read", "wgs_ranges(p, S);", "wgs_ranges(p, S); int late = S.wc[0][0];"),
    ):
        if name == "no_route_barrier":
            route = body(source, "__device__ void route(")
            changed_route = route.removesuffix("  CTA_SYNC();\n}") + "}"
            changed = source.replace(route, changed_route, 1)
        else:
            changed = source.replace(old, new, 1)
        try:
            check_order(changed)
        except AssertionError:
            negatives.append(name)
        else:
            raise AssertionError("negative accepted: " + name)
    layout_path = root / "docs/steering/evidence/2026-11-01-expert-pipeline-next/analyze.py"
    layout_tree = ast.parse(layout_path.read_bytes())
    layout_node = copy.deepcopy(next(node for node in layout_tree.body if isinstance(node, ast.FunctionDef) and node.name == "layout"))
    layout_node.body[-1] = ast.Return(value=ast.Name(id="Header", ctx=ast.Load()))
    module = ast.fix_missing_locations(ast.Module(body=[layout_node], type_ignores=[]))
    namespace = {"ctypes": ctypes}
    exec(compile(module, str(layout_path), "exec"), namespace)
    original = namespace["layout"](8, 168)
    field_types = dict(original._fields_)

    class RoutePrefix(ctypes.Union):
        _fields_ = [("wc", field_types["wc"]), ("wcp", field_types["wcp"])]

    class Aliased(ctypes.Structure):
        _anonymous_ = ("route_prefix",)
        _fields_ = [("route_prefix", RoutePrefix) if name == "wcp" else (name, kind)
                    for name, kind in original._fields_ if name != "wc"]

    assert ctypes.sizeof(original) == 23776
    assert ctypes.sizeof(Aliased) == 23096
    original_ring = (ctypes.sizeof(original) + 127) // 128 * 128
    aliased_ring = (ctypes.sizeof(Aliased) + 127) // 128 * 128
    dynamic = aliased_ring + 3 * 18432 + 3 * 12288
    allocated = (dynamic + 1024 + 127) // 128 * 128
    assert dynamic == 115328
    assert 233472 - 2 * allocated == 768
    assert Aliased.wc.offset == Aliased.wcp.offset
    assert Aliased.wcpB.offset >= Aliased.wc.offset + ctypes.sizeof(RoutePrefix)
    old_prefix = "  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job"
    new_prefix = "  union {\n    int wc[NW][32];\n    int wcp[JMAX + 1];\n  };"
    old_wc = "  int wc[NW][32];"
    assert source.count(old_prefix) == 1 and source.count(old_wc) == 1
    patched = source.replace(old_wc, "#if !OKC_WGS\n" + old_wc + "\n#endif", 1).replace(old_prefix, new_prefix, 1)
    reverse = patched.replace(new_prefix, old_prefix, 1).replace("#if !OKC_WGS\n" + old_wc + "\n#endif", old_wc, 1)
    assert reverse == source
    assert check_order(patched) == ordering
    print(json.dumps({"status": "SOURCE_ONLY_REOPENING_CONDITION", "ordering": ordering,
                      "negative_checks_rejected": negatives, "original_sizeof": ctypes.sizeof(original),
                      "alias_sizeof": ctypes.sizeof(Aliased), "original_ring_start": original_ring,
                      "alias_ring_start": aliased_ring, "dynamic_request": dynamic,
                      "allocated_per_block_at_assumed_1024_reserve_zero_static": allocated,
                      "two_block_bytes": 2 * allocated, "shared_capacity_assumption": 233472,
                      "remaining_shared_bytes_per_sm": 768, "ring_depths_unchanged": [3, 3],
                      "token_tile_unchanged": 24, "source_patch_exact_reverse": True,
                      "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "patched_cuda_sha256": hashlib.sha256(patched.encode()).hexdigest(),
                      "target_reserved_shared": None, "compiled_cpp_union": None,
                      "target_occupancy": None, "numerical_and_timing_results": None}, indent=2))


if __name__ == "__main__":
    main()
