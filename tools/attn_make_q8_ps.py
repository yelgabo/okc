"""Generate the mode 51 run-time pair split check (CHECK ONLY, never a submission).

Arms, timed on verify 16 / 32 full only (the flat q_len 8 plan, mode 51): ctrl (the judged
source), ctrl2 (independent null), ps (ONE_PSTEAL=1: each pair of CTAs splits its two shares at
run time, walk loop unrolled twice) and ps2 (ONE_PSTEAL=2, the same without the unroll).
Non-control builds compile only instance 8/51, as tools/attn_make_q8_tl.py does. ABCHECK adds
the two tp8 test shapes that reach mode 51 (batch 2 at 100K, page 1; batch 4 at 6K, page 64).

    python3 tools/attn_make_q8_ps.py OUT.py
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from attn_make_q8_tl import patch_generated_source  # noqa: E402

GATE = r'''
def _q8_resource_gate():
    mods = {"ctrl": _ONE, **_AB_MODS}
    want = {"ctrl": (107104, 107104), "ctrl2": (107104, 107104),
            "ps": (107104 + 256, 107104 + 256 + 15), "ps2": (107104 + 256, 107104 + 256 + 15)}
    assert set(mods) == set(want), f"arm set {sorted(mods)}"
    for name, mod in mods.items():
        lim = mod.attn_one_prepare()
        rows = [lim[i:i + 5] for i in range(0, len(lim), 5)]
        found = [r for r in rows if r[:2] == [8, 51]]
        assert len(found) == 1, (name, rows)
        row = found[0]
        lo, hi = want[name]
        assert lo <= row[3] <= hi and row[4] == 2, f"Q8 resource mismatch {name}: {row}"
        print(f"Q8RESOURCE {name} L,mode,resident,dynamic_bytes,blocks_per_sm={row}", flush=True)
_q8_resource_gate()

'''


def main():
    out = Path(sys.argv[1])
    subprocess.run([sys.executable, str(ROOT / "tools/attn_make_bench_ab.py"), str(out),
                    "--null", "--shapes", "full", "--inputs", "2", "--rounds", "120",
                    "--seed", "311302",
                    "ps@own=-DONE_PSTEAL=1,-DONE_ATTRIB_ONLY=1",
                    "ps2@own=-DONE_PSTEAL=2,-DONE_ATTRIB_ONLY=1"],
                   check=True, cwd=ROOT)
    src = patch_generated_source(out.read_text())
    first, rest = src.split("\n", 1)
    src = first + "\n# CHECK ONLY: mode 51 run-time pair split arms; never submit.\n" + rest
    marker = "        yield batch, layer, q_len"
    assert src.count(marker) == 1
    src = src.replace(marker, "        if q_len != 8:\n            continue\n" + marker)
    extra = "    shapes += [_OB_TESTS[i] for i in (0, 1, 2, 3, 7, 9, 10)]"
    assert src.count(extra) == 1
    src = src.replace(extra, "    shapes += [_OB_TESTS[i] for i in (5, 7)]")
    marker = "# Own-state arms: a private _ONE_STATE dict per arm, swapped in around each call."
    assert src.count(marker) == 1
    src = src.replace(marker, GATE + marker)
    marker = "            _ab_stsum()"
    assert src.count(marker) == 1
    src = src.replace(marker, "            _q8_resource_gate()\n" + marker)
    out.write_text(src)


if __name__ == "__main__":
    main()
