"""Specialize the private verify-32 owned arm to its selected main MoE family.

The arm already registers only launch_main<51>. Its four skinny namespaces cannot
be reached through that entry. This removes their complete named definitions,
without changing the main family, remaining literals or Python preparation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from dl_owned_slice import build as build_owned

SKINNY_NAMES = ("sk_8_4_0", "sk_16_4_0", "sk_8_3_0", "sk_8_4_3")


def specialize(source):
    tree = ast.parse(source)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and getattr(n.targets[0], "id", "") == "_MOE1_SRC")
    original = ast.literal_eval(assignment.value)
    device = original
    removed = {}
    for name in SKINNY_NAMES:
        opening = "namespace " + name + " {\n"
        closing = "}  // namespace " + name + "\n"
        assert device.count(opening) == device.count(closing) == 1
        start = device.index(opening)
        end = device.index(closing, start) + len(closing)
        removed[name] = device[start:end]
        device = device[:start] + device[end:]
        assert name not in device
    host = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_MOE1_HOST_SRC")
    assert all(name not in host for name in SKINNY_NAMES)
    dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_moe1_source")
    assert 'case 10: launch_main<51>(p, sms, stream); break;' in ast.unparse(dispatch)
    lines = source.splitlines(keepends=True)
    lines[assignment.lineno - 1:assignment.end_lineno] = [
        "_MOE1_SRC = r'''\n" + device + "'''\n"]
    output = "".join(lines)
    revised = ast.parse(output)
    # All Python preparation, registration, dispatch and other literals survive.
    left = [n for n in tree.body if n is not assignment]
    right = [n for n in revised.body if not (isinstance(n, ast.Assign)
             and getattr(n.targets[0], "id", "") == "_MOE1_SRC")]
    assert [ast.dump(n) for n in left] == [ast.dump(n) for n in right]
    return output, device, removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owned_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    arm, _ = build_owned(args.owned_source.read_text(encoding="utf8"))
    output, device, removed = specialize(arm)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "owned-main-arm.py").write_bytes(output.encode())
    (args.output / "original.cu").write_bytes(next(ast.literal_eval(n.value)
        for n in ast.parse(arm).body if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "_MOE1_SRC").encode())
    (args.output / "reduced.cu").write_bytes(device.encode())
    report = {
        "original_arm_bytes": len(arm.encode()),
        "main_arm_bytes": len(output.encode()),
        "saved_bytes": len(arm.encode()) - len(output.encode()),
        "main_arm_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "removed_namespaces": {k: len(v.encode()) for k, v in removed.items()},
        "python_and_other_literals_unchanged": True,
        "selected_dispatch": "launch_main<51>",
        "gpu_execution": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
