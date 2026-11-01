"""Fixed single-arm short-RMW source on the reviewed common-control epoch.

This is a source arm, not a startup qualifier or a dispatch instruction.
No comparison, timing selection, kernel import, or GPU execution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from g_integration_common import build as common, PIN
from g_short_final_rmw import build as short_rmw


COMMON = "4cbb13a9a359c9fe870a94ebcabc4706447778849f71d4aa54e47d69abbd1280"
COMMENT_OLD = "# Private OFF-prototype opt-in: T1/T8 only, no runtime timing selector."
COMMENT_NEW = "# Fixed Tiny/Tiny2 RMW policy; startup qualification required before measurement."


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def common_delta(base, treatment, target):
    """Apply exact, disjoint common-control units; reject any overlapping edit."""
    left, right = ast.parse(base).body, ast.parse(treatment).body
    assert len(left) == len(right), "common transform changed module unit count"
    original = target
    rows = []
    for before, after in zip(left, right):
        if ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False):
            continue
        old, new = ast.get_source_segment(base, before), ast.get_source_segment(treatment, after)
        assert old != new and target.count(old) == 1, "overlapping or ambiguous common delta"
        target = target.replace(old, new, 1)
        rows.append(dict(kind=type(before).__name__, line=before.lineno,
                         before_sha256=sha(old), after_sha256=sha(new), old=old, new=new))
    restored = target
    for row in reversed(rows):
        assert restored.count(row["new"]) == 1
        restored = restored.replace(row["new"], row["old"], 1)
    assert restored == original, "common delta escaped its exact units"
    return target, [{k: v for k, v in row.items() if k not in ("old", "new")} for row in rows]


def build(source, enabled=False):
    assert sha(source) == PIN, "exact607 source required"
    control, _ = common(source, True)
    assert sha(control) == COMMON
    result = control
    changes = []
    if enabled:
        rmw, rmw_manifest = short_rmw(source, True)
        result, changes = common_delta(source, control, rmw)
        assert result.count(COMMENT_OLD) == 1
        result = result.replace(COMMENT_OLD, COMMENT_NEW, 1)
        assert rmw_manifest["enabled"]
    ast.parse(result)
    return result, dict(base_sha256=PIN, common_control_sha256=COMMON,
                        source_sha256=sha(result), bytes=len(result.encode()),
                        short_rmw_enabled=bool(enabled), common_unit_deltas=changes,
                        affected_families=["tiny", "tiny2"],
                        fixed_policy=True, comparative_timings=False,
                        startup_qualified=False, H200_qualified=False,
                        dispatch_enabled=False,
                        scope="Fixed single-arm source; still needs prepare-time all12-cell qualifier "
                              "and runtime admission/correctness before measurement")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    result, manifest = build(args.source.read_text(encoding="utf-8"), args.enable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.encode())
    args.output.with_suffix(".manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode())
    print(json.dumps({k: manifest[k] for k in ("source_sha256", "bytes", "short_rmw_enabled")}))
