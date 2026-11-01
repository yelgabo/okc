"""Build a transparent, fixed-mask production verify hybrid; dispatch no GPU jobs.

Default mask zero returns the byte-identical scored607e file. Nonzero files
remain unqualified until the selected override wins the registered private test
and the complete production file passes an official H200 check.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from dl_g_verify_slice import PIN as G_PIN
from dl_selected_pair import literal

CELLS = ((16, "sliding"), (16, "full"), (32, "sliding"), (32, "full"))

LOADER = '''
import linecache
import types

def _verify_hybrid_module(name, source):
    module = types.ModuleType(name)
    filename = "<" + name + ">"
    module.__file__ = filename
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    exec(compile(source, filename, "exec"), module.__dict__)
    return module
'''


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def selected_cells(mask):
    if not isinstance(mask, int) or isinstance(mask, bool) or not 0 <= mask < 16:
        raise ValueError("verify mask must be an integer from zero through fifteen")
    return [cell for bit, cell in enumerate(CELLS) if mask & (1 << bit)]


def backends(root):
    from dl_g_literal_factor import factor
    from dl_g_production_slice import build as g_production
    from dl_owned_verify_production import build as owned_production
    g_source, g_manifest = g_production((root / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8"))
    g_source, factoring = factor(g_source)
    owned = {batches: owned_production(root, batches) for batches in ((16,), (32,), (16, 32))}
    return g_source, owned, {"G_slice": g_manifest, "G_factoring": factoring,
        "G_factored_bytes": len(g_source.encode()), "G_factored_sha256": sha(g_source),
        "owned": {str(k): value[1] for k, value in owned.items()}}


def budget_matrix(root):
    g_source, owned, backend_manifest = backends(root)
    rows = []
    for mask in range(16):
        batches = tuple(sorted({batch for batch, _ in selected_cells(mask)}))
        _, manifest = build(root, mask, g_backend=g_source,
            owned_backend=owned[batches][0] if batches else None)
        rows.append(manifest)
    return {"variants": rows, "backends": backend_manifest,
            "new_GPU_runs": [], "new_performance_claim": False,
            "all_overrides_fit": all(row["fits_one_megabyte"] for row in rows),
            "default_job": "607eb52bdaf24160"}


def build(root, mask=0, *, g_backend=None, owned_backend=None):
    cells = selected_cells(mask)
    original = (root / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8")
    assert sha(original) == G_PIN, "incumbent source pin drift"
    if not mask:
        source = original
        pins = {"G": G_PIN}
    else:
        if g_backend is None or owned_backend is None:
            raise ValueError("nonzero mask requires reviewed production backends")
        ast.parse(g_backend)
        ast.parse(owned_backend)
        pins = {"G": sha(g_backend), "owned": sha(owned_backend)}
        source = "# okc: problem=muse_decode_layer\n"
        source += "# Fixed-mask production dispatcher; H200 qualification still required.\n"
        source += LOADER
        source += "\n_VERIFY_OVERRIDE_MASK = " + repr(mask) + "\n"
        for key, backend in (("G", g_backend), ("OWNED", owned_backend)):
            source += "\n_VERIFY_HYBRID_SOURCE_" + key + " = " + literal(backend) + "\n"
            source += "_VERIFY_HYBRID_" + key + " = _verify_hybrid_module(" + repr("_okc_verify_production_" + key.lower())
            source += ", _VERIFY_HYBRID_SOURCE_" + key + ")\n"
        source += "\n" + (root / "tools/dl_verify_hybrid_body.py").read_text(encoding="utf8")
    ast.parse(source)
    manifest = {"bytes": len(source.encode()), "sha256": sha(source),
                "mask": mask, "selected_cells": cells, "backend_pins": pins,
                "incumbent_job": "607eb52bdaf24160", "incumbent_sha256": G_PIN,
                "default_byte_identical": not mask,
                "fits_one_megabyte": len(source.encode()) < 1_000_000,
                "H200_qualified": not mask, "new_performance_measurement": False,
                "source_transport": "readable multiline Python literals; no encoding/compression"}
    return source, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=int, default=0)
    parser.add_argument("--g-backend", type=Path)
    parser.add_argument("--owned-backend", type=Path)
    parser.add_argument("--matrix", action="store_true", help="write the sixteen-mask source budget, including oversized results")
    args = parser.parse_args()
    if args.matrix:
        matrix = budget_matrix(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf8", newline="\n")
        print(json.dumps({"all_overrides_fit": matrix["all_overrides_fit"],
            "variants": [{k: row[k] for k in ("mask", "bytes", "fits_one_megabyte")} for row in matrix["variants"]]}, indent=2))
        return
    source, manifest = build(args.root, args.mask,
        g_backend=args.g_backend.read_text(encoding="utf8") if args.g_backend else None,
        owned_backend=args.owned_backend.read_text(encoding="utf8") if args.owned_backend else None)
    if not manifest["fits_one_megabyte"]:
        raise ValueError(f"hybrid exceeds one-file limit: {manifest['bytes']} bytes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(source.encode())
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
