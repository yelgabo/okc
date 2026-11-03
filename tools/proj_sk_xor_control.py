"""Restore the exact scored C2 control at generation time, never in a payload."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

SOURCE = "3044ee7:muse_bf16_proj/submission.py"
SOURCE_SHA256 = "b2c4c096410e2d282d5e1eb3f2afa083f61e68d26d085deede43915bfc7efbdc"
CONTROL_SHA256 = "5610b8a64e25f70c639ac0523fe7c14754a6c54d349eb005210123b4e19fb3b4"


def restore(root):
    source = subprocess.check_output(["git", "show", SOURCE], cwd=root)
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "C2 owning source blob drift"
    text = source.decode("utf8") + "\n# Authorized scored control 2026-10-31 repeat1.\n"
    assert ast.dump(ast.parse(text)) == ast.dump(ast.parse(source)), "control comment changed executable code"
    raw = text.replace("\n", "\r\n").encode()
    assert hashlib.sha256(raw).hexdigest() == CONTROL_SHA256, "scored C2 raw pin drift"
    return raw


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    raw = restore(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"Restored C2 {len(raw)} bytes SHA256 {CONTROL_SHA256} from {SOURCE}")
