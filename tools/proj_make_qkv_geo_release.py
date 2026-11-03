"""Full-board confirmation file: exact C2 bytes plus one wd geometry offered only to qkv_gate m = 128.

The new geometry goes through C2's own prepare: FP64 self-test at m = 65 / 100 / 128, cold-L2
autotune against cuBLAS and the other wd configs (WD_MARGIN), then the stable Lt pick. No other
layer or bucket sees it, so every other pick is C2's.

    python3 tools/proj_make_qkv_geo_release.py NAME WG BN CLK CLR NST out.py
"""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import proj_sk_xor_control  # noqa: E402


def sub(src, old, new):
    assert src.count(old) == 1, old[:80]
    return src.replace(old, new)


def build(name, wg, bn, clk, clr, nst):
    src = proj_sk_xor_control.restore(ROOT).decode("utf8").replace("\r\n", "\n")
    src = sub(src, '    "b1n128c3s8h": (1, 128, 3, 1, False, 8),\n',
              '    "b1n128c3s8h": (1, 128, 3, 1, False, 8),\n'
              f'    # qkv_gate m = 128 only (WD_ONLY): QKV128-GEO-01 bench 361d5130d70b48e1.\n'
              f'    "{name}": ({wg}, {bn}, {clk}, {clr}, False, {nst}),\n')
    src = sub(src, "WD_POL = {nm for nm in WD_GEOM if nm.endswith(\"h\")}\n",
              "WD_POL = {nm for nm in WD_GEOM if nm.endswith(\"h\")}\n"
              f"WD_ONLY = {{\"{name}\": {{(\"qkv_gate\", 128)}}}}\n")
    src = sub(src, "                if hl != is_router or not bn_ok or not wd_fits(nm, n, k) or WD_VAR.get(nm, 0) & WV_NOEMPTY:\n",
              "                if nm in WD_ONLY and (layer, b) not in WD_ONLY[nm]:\n"
              "                    continue\n"
              "                if hl != is_router or not bn_ok or not wd_fits(nm, n, k) or WD_VAR.get(nm, 0) & WV_NOEMPTY:\n")
    ast.parse(src)
    return src


if __name__ == "__main__":
    name, wg, bn, clk, clr, nst, out = sys.argv[1:8]
    assert name.endswith("h"), "the L2 policy follows the name suffix, as on the production pick"
    text = build(name, int(wg), int(bn), int(clk), int(clr), int(nst))
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out}: C2 + {name} on qkv_gate m = 128")
