import ast
import difflib
import hashlib
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from g_integration_common import build


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(source):
    tree = ast.parse(source)
    literals = {}
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            key = ast.unparse(node.targets[0])
            try:
                literals[key] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = ast.get_source_segment(source, node)
    return literals, functions


def main():
    paths = [".build/today/kel/kernels/muse_decode_layer.py",
             "muse_decode_layer/cand/g1b4_stage_release.py"]
    raw_sources = [(ROOT / path).read_bytes() for path in paths]
    sources = [raw.decode().replace("\r\n", "\n") for raw in raw_sources]
    common, manifest = build(sources[1], True)
    assert digest(common.encode()) == "4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280"
    sources.append(common)
    items = [inventory(source) for source in sources]
    key = "_MA_ALT_SRCS['moe_helper0']"
    assert items[1][0][key] == items[2][0][key]
    assert items[1][0]["bfp_SK2_SRC"] == items[2][0]["bfp_SK2_SRC"]
    policies = ["PDL", "MOE_HAS_PDL", "GLE_T", "GLE_FRT", "MOE_TINY_DEFINES", "MOE_TINY2_DEFINES", "bfp_SK2_PICK"]
    result = {
        "evidence_checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "upstream_commit": "8e30e1be994db4310794b8d04e90d18d895deb84",
        "sources": [{"path": path, "raw_bytes": len(raw), "raw_sha256": digest(raw), "lf_sha256": digest(source.encode())}
                    for path, raw, source in zip(paths, raw_sources, sources)],
        "common": manifest,
        "policy": [{name: repr(values.get(name)) for name in policies} for values, functions in items],
        "helper_and_sk2_common_equal_607": True,
        "scope": "CPU AST/source only; no imports of kernel files, compilation, GPU, timing, or legal qualification"
    }
    (OUT / "source-census.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    diff = list(difflib.unified_diff(items[0][0][key].splitlines(), items[1][0][key].splitlines(), fromfile="G0.helper0", tofile="607.helper0", lineterm=""))
    (OUT / "helper0.diff.txt").write_text("\n".join(diff) + "\n", encoding="utf-8")
    excerpts = []
    for literal in [key, "bfp_SK2_SRC"]:
        lines = items[1][0][literal].splitlines()
        selected = set()
        for index, line in enumerate(lines):
            needles = ["griddepcontrol.", "programmaticStreamSerializationAllowed", "fence.proxy.async.global", "cudaLaunchCooperativeKernel", "const int old="]
            if any(needle in line for needle in needles):
                selected.update(range(max(0, index - 4), min(len(lines), index + 9)))
        excerpts.append("\nLITERAL " + literal + " (1-based decoded-literal lines)\n")
        excerpts.extend(f"{index+1}: {lines[index]}" for index in sorted(selected))
    for name in ["kernel_fn", "_after_attention", "_gle_after_attention", "_pdl", "_pcx_val", "_bfp_pdl", "_bfp_gemm", "_fork_native_moe", "_sl_tile_after_attention"]:
        for label, (values, functions) in zip(["G0", "607", "common"], items):
            excerpts.append("\n" + label + " FUNCTION " + name + "\n" + functions.get(name, "NOT DEFINED"))
    for label, source in zip(["G0", "607", "common"], sources):
        excerpts.append("\n" + label + " ordered top-level bindings / dispatch definitions")
        for node in ast.parse(source).body:
            text = ast.get_source_segment(source, node)
            if isinstance(node, ast.Assign) and len(text) < 450 and any(token in text for token in ["_after_attention", "_moe =", "_bfp_pdl =", "PDL =", "GLE_T ="]):
                excerpts.append(str(node.lineno) + ": " + text)
            if isinstance(node, ast.FunctionDef) and node.name == "_after_attention":
                excerpts.append(str(node.lineno) + ": " + text)
    (OUT / "selected-source.txt").write_text("\n".join(excerpts) + "\n", encoding="utf-8")
    print(json.dumps({"common_sha256": digest(common.encode()), "helper_diff_lines": len(diff), "validated": True}))


if __name__ == "__main__":
    main()
