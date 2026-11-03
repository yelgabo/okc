"""OFF source-only composition of the two inherited G common-control repairs.

Both leaf builders independently require exact607; this composes their disjoint
changes from that pin, rather than weakening either leaf's input verification.
No kernel import, GPU dispatch, runtime qualification, or timing selection.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from attention_g_cooperative_host import build as attention, REPAIR, PIN
from proj_g_preexpert_gamma import build as preexpert, FIELD


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def attention_literal(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "ATT7_CUDA_SRC")


def build(source, enabled=False):
    assert sha(source) == PIN, "exact607 input required"
    result = source
    if enabled:
        host = attention(source, True)
        coefficient, coefficient_manifest = preexpert(source, True)
        base_node = attention_literal(source)
        host_node = attention_literal(host)
        coefficient_node = attention_literal(coefficient)
        assert ast.literal_eval(coefficient_node.value) == ast.literal_eval(base_node.value)
        assert ast.literal_eval(host_node.value).replace(REPAIR, "") == ast.literal_eval(base_node.value)
        lines = coefficient.splitlines(keepends=True)
        lines[coefficient_node.lineno - 1:coefficient_node.end_lineno] = [
            ast.get_source_segment(host, host_node) + "\n"]
        result = "".join(lines)
        assert result.count(REPAIR) == 1 and result.replace(REPAIR, "") == coefficient
        assert FIELD in result and coefficient_manifest["source_reverse_proof"]
        ast.parse(result)
    return result, dict(input_sha256=PIN, source_sha256=sha(result),
                        bytes=len(result.encode()), enabled=enabled, default_off=True,
                        scope="Common inherited attention admission and preexpert coefficient repairs; "
                              "not an upstream-to607 regression treatment",
                        required_common_to_all_ablation_arms=True,
                        H200_qualified=False, GPU=False, dispatch_enabled=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    result, manifest = build(source, args.enable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.encode())
    args.output.with_suffix(".manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps(manifest))
