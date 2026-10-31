"""Diagnostic build (not a candidate) for the illegal memory access of check 36f93b13d4c14d5b:
layer_m1rt51s78 crashed in the front tuner at 32 tokens, after the MoE tuner and stress had
run. This prefers mode 78 at 32 tokens, so the front tuner's front + MoE chains run it, and
prints a synchronized, flushed `D78` line before and after each step of the front tuner at 32
tokens (captures, first replays, the route variants' fresh-input gate, each timing round), so
the last D78 line in the log tail names the step that faults. Writes OUT with the header line.

    python3 tools/dl_moe78_diag.py layer_m1rt51s78.py OUT.py "HEADER LINE 2"
"""
import sys

HELPER = '''

def _d78(tokens, what):
    """Synchronizes and prints one flushed line at 32 tokens (check 36f93b13d4c14d5b)."""
    if tokens != 32:
        return
    try:
        torch.cuda.synchronize()
        print(f"D78 ok {what}", flush=True)
    except Exception as exc:
        print(f"D78 FAULT at {what}: {type(exc).__name__}", flush=True)
        raise


def _front_tune(state):'''

REPS = [
    ('\n\ndef _front_tune(state):', HELPER),
    ('''            fres, cres = {}, {}
            fgraph = _graph_build(lambda: front(fres))
            fgraph.replay()
            torch.cuda.synchronize()''',
     '''            fres, cres = {}, {}
            _d78(tokens, f"{variant} before front capture; moe pick "
                 + _moe1_name(_moe_opts(tokens, state)["fused"]))
            fgraph = _graph_build(lambda: front(fres))
            _d78(tokens, f"{variant} front captured")
            fgraph.replay()
            torch.cuda.synchronize()
            _d78(tokens, f"{variant} front replayed")'''),
    ('''                cgraph = _graph_build(lambda: front_moe(cres))
                cgraph.replay()
                torch.cuda.synchronize()''',
     '''                cgraph = _graph_build(lambda: front_moe(cres))
                _d78(tokens, f"{variant} front+moe captured")
                cgraph.replay()
                torch.cuda.synchronize()
                _d78(tokens, f"{variant} front+moe replayed")'''),
    ('''                        f"{variant}@{tokens}", state))
            graphs[variant] = (fgraph, cgraph)''',
     '''                        f"{variant}@{tokens}", state))
                    _d78(tokens, f"{variant} fresh-input gate done")
            graphs[variant] = (fgraph, cgraph)'''),
    ('''                rounds[v].append(_graph_median(graphs[v][1 if with_moe else 0], flush, 20))''',
     '''                rounds[v].append(_graph_median(graphs[v][1 if with_moe else 0], flush, 20))
                _d78(tokens, f"round {r} {v}")'''),
]


def main():
    src, out, header = sys.argv[1:4]
    text = open(src).read()
    if "STABLE_PREF = {" not in text or "_MOE1_MAIN_MAX_M" not in text:
        raise SystemExit("IN must be layer_m1rt51s78")
    a = text.index("STABLE_PREF = {")
    b = text.index("\n}\n", a)
    block = text[a:b]
    if '32: "main-lean-grp-finss64"' not in block:
        raise SystemExit("no mode 51 preference at 32 tokens")
    block = block.replace('32: "main-lean-grp-finss64"', '32: "main-lean-grp-fin4ps"', 1)
    text = text[:a] + block + text[b:]
    for old, new in REPS:
        if text.count(old) != 1:
            raise SystemExit(f"expected one {old[:60]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    lines = text.split("\n")
    lines[1] = header
    open(out, "w").write("\n".join(lines))
    print("wrote", out, len(text))


if __name__ == "__main__":
    main()
