"""Build a CHECK-ONLY M1/control replay diagnostic; judged path always disables M1.

Same weights/input bank/reference, repeat off and on to distinguish control variability
from a PDL-only failure. Existing comparator thresholds and all CUDA code stay unchanged.
Usage: python tools/dl_make_m1_matched_diag.py INPUT_LAYER OUTPUT_CHECK
"""
import ast
from pathlib import Path
import sys

WRAPPER = '''
_md_original_fresh = _m1_fresh
def _m1_fresh(state, data, policy, bank, refs, replays):
    first = _md_original_fresh(state, data, policy, bank, refs, replays)
    if not policy:
        state["_md_control"] = first
        return first
    off2 = _md_original_fresh(state, data, {}, bank, refs, replays)
    on2 = _md_original_fresh(state, data, policy, bank, refs, replays)
    ctrl = state.pop("_md_control")
    label = (data["config"]["batch"], data["config"]["q_len"], data["config"]["window"])
    for name, a, b in (("off2/off1", off2, ctrl), ("on1/off1", first, ctrl),
                       ("on2/off2", on2, off2), ("on2/on1", on2, first)):
        ok, text, classes = _gate_verdict(a, b)
        line = f"M1MATCH {label} {name} {ok} {classes} {text}"
        state.setdefault("_md_lines", []).append(line)
        print(line, flush=True)
    return first

_md_original_prepare = prepare
def prepare(weights):
    state = _md_original_prepare(weights)
    # The judge retains only its final log tail; emit these AFTER the normal summary.
    for line in state.pop("_md_lines", []):
        print(line, flush=True)
    return state
'''

TUNE = '''def _m1_tune(state):
    # Diagnostic-only subset, no timing/promotional selection. Gate itself is unchanged.
    state["m1_report"] = []
    state["m1_ok"] = True
    try:
        checked = _m1_selftest(state)
        print("M1MATCH selftest", checked, flush=True)
        for line in _m1_stress(state):
            print("M1MATCH gate", line, flush=True)
    finally:
        state["m1_ok"] = False
        state["m1_policy"] = {}
        state.pop("_md_control", None)
'''

def main():
    text = Path(sys.argv[1]).read_text()
    old = '''        for batch, length, layer in ((1, 1, "sliding"), (1, 1, "full"), (1, 8, "sliding"),
                                     (1, 8, "full"), (16, 1, "sliding"), (16, 1, "full"),
                                     (32, 1, "sliding"), (32, 1, "full")):'''
    assert text.count(old) == 1
    text = text.replace(old, '        for batch, length, layer in ((16, 1, "sliding"), (1, 8, "sliding")):')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == '_m1_tune')
    lines = text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [TUNE + '\n']
    text = ''.join(lines) + WRAPPER
    text = text.replace('# okc: problem=muse_decode_layer\n',
                        '# okc: problem=muse_decode_layer\n# CHECK ONLY: M1 matched replay diagnostic; judged M1 OFF.\n', 1)
    ast.parse(text)
    Path(sys.argv[2]).write_text(text)

if __name__ == '__main__':
    main()
