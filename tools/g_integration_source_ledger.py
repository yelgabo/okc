"""Exhaustive, source-only G0/607 top-level ledger. Never imports either kernel.

This inventory is not an execution trace, a legality certificate, or a cost model.
Occurrence numbers preserve repeated prepare/kernel_fn wrappers in source order.
Native literals are compared as values, including their embedded source comments.
"""
import argparse
import ast
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path


PINS = {
    "G0": "82c098ae7534b20b5d80f59d547650cb0853fc7d5e313fe29e1e9d9373e1bb19",
    "G607": "ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab",
}
FILES = {
    "G0": ".build/today/kel/kernels/muse_decode_layer.py",
    "G607": "muse_decode_layer/cand/g1b4_stage_release.py",
}


def digest(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atoms(source):
    seen = Counter()
    result = []
    for node in ast.parse(source).body:
        kind = type(node).__name__
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            name = " = ".join(ast.unparse(t) for t in node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            name = ast.unparse(node.target)
        elif isinstance(node, ast.Import):
            name = ",".join(x.name for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
        else:
            # Keep every top-level statement, including wrapper-registration calls.
            name = "<statement>"
        label = kind + ":" + name
        seen[label] += 1
        row = dict(label=label, occurrence=seen[label], line=node.lineno,
                   end_line=node.end_lineno,
                   ast_sha256=digest(ast.dump(node, include_attributes=False)))
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            text = value.value
            row["string_value"] = dict(bytes=len(text.encode()), sha256=digest(text),
                lines=len(text.splitlines()),
                native_source=("__global__" in text or "#include <" in text))
        result.append(row)
    return result


def compare(left, right):
    # Ordered alignment accounts for inserted module statements; nothing is dropped.
    matcher = difflib.SequenceMatcher(a=[r["label"] for r in left],
                                     b=[r["label"] for r in right], autojunk=False)
    rows = []
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == "equal":
            for old, new in zip(left[a:b], right[c:d]):
                rows.append(dict(status="unchanged" if old["ast_sha256"] == new["ast_sha256"]
                                 else "changed", before=old, after=new))
        else:
            # Do not invent pairings between differently named code units.
            rows.extend(dict(status="removed", before=r) for r in left[a:b])
            rows.extend(dict(status="added", after=r) for r in right[c:d])
    assert sum("before" in r for r in rows) == len(left)
    assert sum("after" in r for r in rows) == len(right)
    return rows


def inventory(root):
    sources = {}
    units = {}
    for key, relative in FILES.items():
        raw = (root / relative).read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")
        if digest(text) != PINS[key]:
            raise ValueError("unrecognized " + key + " source: " + digest(text))
        units[key] = atoms(text)
        sources[key] = dict(path=relative, physical_sha256=digest(raw),
                            physical_bytes=len(raw), lf_sha256=digest(text),
                            lf_bytes=len(text.encode()), top_level_statements=len(units[key]))
    rows = compare(units["G0"], units["G607"])
    return dict(schema=1, scope="All top-level AST statements; source only, no kernel import/execution. "
                "Hashes do not identify active runtime paths, compliance, or latency. "
                "Whole native-literal hashes include embedded comments; deletion is not necessarily lost behavior.",
                sources=sources, counts=dict(Counter(r["status"] for r in rows)), changes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(dict(counts=report["counts"], sources=report["sources"])))


if __name__ == "__main__":
    main()
