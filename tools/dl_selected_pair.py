"""Package two explicit private layer slices with an ordinary SGLang judge entry.

Module sources remain readable multiline Python strings. There is no encoding,
compression, external import of a competitor or runtime source-file access.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from dl_g_verify_slice import build as build_g
from dl_owned_main_slice import specialize
from dl_owned_slice import build as build_owned
from dl_g1_layer_gate import NAMES, PRELUDE


def literal(source):
    text = '"""\\\n' + source.replace("\\", "\\\\").replace('"""', '\\"\\"\\"') + '"""'
    assert ast.literal_eval(text) == source
    return text


LOADER = '''
import linecache
import types
import torch
import json

def _pair_module(name, source):
    module = types.ModuleType(name)
    filename = "<" + name + ">"
    module.__file__ = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module
'''


def build(root):
    g, g_manifest = build_g((root / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8"))
    owned, _ = build_owned((root / "muse_decode_layer/cand/layer_lt_boundary.py").read_text(encoding="utf8"))
    owned, _, _ = specialize(owned)
    baseline = (root / "muse_decode_layer/baseline.py").read_text(encoding="utf8")
    sources = {"G": g, "owned": owned, "baseline": baseline}
    pins = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    # The reference helpers are the independently restated FP32 judge oracle.
    original = (root / "tools/dl_make_bench_pub.py").read_text(encoding="utf8")
    body = next(ast.literal_eval(n.value) for n in ast.parse(original).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BODY")
    helpers = {n.name: ast.get_source_segment(body, n) for n in ast.parse(body).body if isinstance(n, ast.FunctionDef)}
    text = "# okc: problem=muse_decode_layer\n# PRIVATE selected-cell comparison; never submit publicly.\n" + LOADER
    text += "\n_PAIR_MODULES = {}\n"
    for name, source in sources.items():
        text += "\n_PAIR_SOURCE_" + name + " = " + literal(source) + "\n"
        text += "_PAIR_MODULES[" + repr(name) + "] = _pair_module(" + repr("_okc_selected_" + name) + ", _PAIR_SOURCE_" + name + ")\n"
    text += "\n_PAIR_PINS = " + repr(pins) + "\n" + PRELUDE
    text += "\nHIDDEN=8192\nD_EXPERT=4096\nLOCAL_EXPERTS=32\nN_EXPERTS=256\nTOP_K=8\nROPE_THETA=500000.\n"
    text += "\n\n".join(helpers[name] for name in NAMES) + "\n"
    text += (root / "tools/dl_g1_layer_owners.py").read_text(encoding="utf8") + "\n"
    text += (root / "tools/dl_selected_pair_body.py").read_text(encoding="utf8")
    ast.parse(text)
    assert len(text.encode()) < 1_000_000
    return text, {"bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "arm_pins": pins, "G_manifest": g_manifest,
                  "scope": "verify32 sliding only, all official cases use SGLang",
                  "H200_qualified": False, "prepare_budget_measured": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, manifest = build(args.root)
    args.output.write_bytes(source.encode())
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(json.dumps(manifest, indent=2))
