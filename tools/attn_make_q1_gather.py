"""Generate the mode 49 gather attribution check (CHECK ONLY, never a submission).

Arms, all timed on decode 16 / 32 full only (the flat q_len 1 plan, mode 49): ctrl (the judged
source), ctrlb (the control's code in its own instance-49-only build and state: the null),
nc_probe (ONE_PROBE=1: no cache reads, compute kept), g6_probe (6: page-table rows read, no
compute), g7_probe (7: sequential rows, compute kept), g8_probe (8: sequential rows, no compute),
g9_probe (9: one cp.async.bulk per 128-byte row, no compute). Builds other than the control
compile only instance 1/49 to fit the preparation budget.

    python3 tools/attn_make_q1_gather.py OUT.py
"""
from pathlib import Path
import ast
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent

ONLY = ('#if ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)\n',
        '#if (defined(ONE_ATTRIB_ONLY) && ONE_ATTRIB_ONLY) || ONE_DETM\n'
        '#define ONE_INSTANCES(X) X(1, 49)\n')

ARMS = {
    "ctrlb": "",
    "nc_probe": "-DONE_PROBE=1",
    "g6_probe": "-DONE_PROBE=6",
    "g7_probe": "-DONE_PROBE=7",
    "g8_probe": "-DONE_PROBE=8",
    "g9_probe": "-DONE_PROBE=9",
}

# Printed, not asserted: a mismatch is read from the log instead of costing the whole check.
GATE = r'''
def _q1_resource_gate():
    mods = {"ctrl": _ONE, **_AB_MODS}
    for name, mod in mods.items():
        try:
            lim = mod.attn_one_prepare()
            rows = [lim[i:i + 5] for i in range(0, len(lim), 5)]
            found = [r for r in rows if r[:2] == [1, 49]]
            ok = len(found) == 1 and found[0][3:] == [58960, 2]
            print(f"Q1RESOURCE {name} ok={ok} L,mode,resident,dynamic_bytes,blocks_per_sm="
                  f"{found}", flush=True)
        except Exception as exc:
            print(f"Q1RESOURCE {name} failed {type(exc).__name__}: {exc}", flush=True)
_q1_resource_gate()

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
    arms = [f"{n}@own={(f + ',' if f else '')}-DONE_ATTRIB_ONLY=1" for n, f in ARMS.items()]
    subprocess.run([sys.executable, str(ROOT / "tools/attn_make_bench_ab.py"), str(out),
                    "--shapes", "full", "--inputs", "2", "--rounds", "120",
                    "--seed", "311401", *arms], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "tools/attn_inline_one.py"), str(out)],
                   check=True, cwd=ROOT)
    src = patch_generated_source(out.read_text())
    first, rest = src.split("\n", 1)
    src = first + "\n# CHECK ONLY: mode 49 gather attribution probes; never submit.\n" + rest
    marker = "        yield batch, layer, q_len"
    assert src.count(marker) == 1
    src = src.replace(marker, "        if q_len != 1:\n            continue\n" + marker)
    extra = "    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]"
    assert src.count(extra) == 1
    src = src.replace(extra, "    # q1 gather probes: decode 16 / 32 full only.")
    marker = "# Own-state arms: a private _ONE_STATE dict per arm, swapped in around each call."
    assert src.count(marker) == 1
    src = src.replace(marker, GATE + marker)
    out.write_text(src)


if __name__ == "__main__":
    main()
