"""Generate the mode 51 timeline check (CHECK ONLY, never a submission).

Arms, all timed on verify 16 / 32 full only (the flat q_len 8 plan, mode 51): ctrl (the judged
source), ctrl2 (independent null), stamp_probe (ONE_STAMP=2: per-CTA timeline of every fourth
tile, see attn_one.cu TL_BASE), hs (ONE_HSPLIT=1: one mbarrier pair per KV head and stage,
correct output, a candidate arm) and stamphs_probe (both). Builds other than the control
compile only instance 8/51 to fit the preparation budget, as tools/attn_make_q8_attrib.py does.

    python3 tools/attn_make_q8_tl.py OUT.py
"""
from pathlib import Path
import ast
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

ONLY = ('#if ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n',
        '#if defined(ONE_ATTRIB_ONLY) && ONE_ATTRIB_ONLY\n#define ONE_INSTANCES(X) X(8, 51)\n'
        '#elif ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n')

GATE = r'''
def _q8_resource_gate():
    mods = {"ctrl": _ONE, **_AB_MODS}
    want = {"ctrl": 107104, "ctrl2": 107104, "stamp_probe": 107104, "hs": 107168,
            "stamphs_probe": 107168}
    assert set(mods) == set(want), f"arm set {sorted(mods)}"
    for name, mod in mods.items():
        lim = mod.attn_one_prepare()
        rows = [lim[i:i + 5] for i in range(0, len(lim), 5)]
        found = [r for r in rows if r[:2] == [8, 51]]
        assert len(found) == 1, (name, rows)
        row = found[0]
        assert row[3:] == [want[name], 2], f"Q8 resource mismatch {name}: {row}"
        print(f"Q8RESOURCE {name} L,mode,resident,dynamic_bytes,blocks_per_sm={row}", flush=True)
_q8_resource_gate()

'''


def patch_generated_source(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_ONE_SRC' for t in n.targets)]
    assert len(nodes) == 1, 'exactly one inlined source'
    node = nodes[0]
    body = ast.literal_eval(node.value)
    assert body.count(ONLY[0]) == 1
    body = body.replace(ONLY[0], ONLY[1], 1)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['_ONE_SRC = ' + repr(body) + '\n']
    return ''.join(lines)


def main():
    out = Path(sys.argv[1])
    subprocess.run([sys.executable, str(ROOT / "tools/attn_make_bench_ab.py"), str(out),
                    "--null", "--shapes", "full", "--inputs", "2", "--rounds", "120",
                    "--seed", "311301",
                    "stamp_probe@own=-DONE_STAMP=2,-DONE_ATTRIB_ONLY=1",
                    "hs@own=-DONE_HSPLIT=1,-DONE_ATTRIB_ONLY=1",
                    "stamphs_probe@own=-DONE_STAMP=2,-DONE_HSPLIT=1,-DONE_ATTRIB_ONLY=1"],
                   check=True, cwd=ROOT)
    src = patch_generated_source(out.read_text())
    first, rest = src.split("\n", 1)
    src = first + "\n# CHECK ONLY: mode 51 timeline and per-head ring arms; never submit.\n" + rest
    marker = "        yield batch, layer, q_len"
    assert src.count(marker) == 1
    src = src.replace(marker, "        if q_len != 8:\n            continue\n" + marker)
    extra = "    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]"
    assert src.count(extra) == 1
    src = src.replace(extra, "    # q8 timeline: verify 16 / 32 full only.")
    marker = "# Own-state arms: a private _ONE_STATE dict per arm, swapped in around each call."
    assert src.count(marker) == 1
    src = src.replace(marker, GATE + marker)
    marker = "            _ab_stsum()"
    assert src.count(marker) == 1
    src = src.replace(marker, "            _q8_resource_gate()\n" + marker)
    out.write_text(src)


if __name__ == "__main__":
    main()
