"""Build the M1 determinism diagnostic from the follow-up file: _m1_tune runs the bit-16
self-test, then per stress shape measures M1-off against M1-off (eager repeats, two graph
stresses) and M1-on against M1-off at the stress's bound, and attention alone; never raises
on a difference, and leaves M1 off for the timed path. Run as check 6372154fcd7a4fa9.

    python3 tools/dl_m1_diag.py LAYER_WITH_M1.py OUT.py [--fresh]   (the layer file at 6377b5a)
"""
import sys
from pathlib import Path

src, out = sys.argv[1], sys.argv[2]
text = open(src).read()
start = text.index("def _m1_tune(state):")
end = text.index("def _m1_enable(state):")
new = '''def _m1_diag_shape(state, batch, length, layer):
    data = _timeline_case(state, batch, length, layer, 9300 + batch * length)

    def eager(policy):
        state["m1_policy"] = policy
        return [t.clone() for t in kernel_fn(data, state)]

    def stress(policy):
        state["m1_policy"] = policy
        res = {}

        def run():
            res["o"] = kernel_fn(data, state)

        graph = _graph_build(run)
        for _ in range(M1_STRESS_REPLAYS):
            graph.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in res["o"]]
        del graph
        return got

    on = {b: "o+front" for b in M1_BUCKETS}
    ref = eager({})

    def cmp(got):
        same = torch.equal(got[2], ref[2])
        bit = torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        return (f"{_rel(got[0], ref[0]):.1e}/{_rel(got[1], ref[1]):.1e}"
                + ("=" if bit else "") + ("" if same else "!ids"))

    eoff = [cmp(eager({})) for _ in range(6)]
    goff = [cmp(stress({})) for _ in range(2)]
    gon = [cmp(stress(on)) for _ in range(2)]
    eon = [cmp(eager(on)) for _ in range(2)]
    state["m1_policy"] = {}
    cfg = data["config"]
    qkv = _proj(data["attn_input"], "qkv_gate", state["dense"]["qkv_gate"].shape[0], state)
    path = _attn_path(cfg, state)
    a0 = path(qkv, data, state).clone()
    arel, aeq = [], 0
    for _ in range(6):
        a = path(qkv, data, state)
        arel.append(_rel(a, a0))
        aeq += int(torch.equal(a, a0))
    torch.cuda.synchronize()
    return (f"{batch}x{length}{layer[0]} eoff {' '.join(eoff)} | goff {' '.join(goff)} | gon "
            f"{' '.join(gon)} | eon {' '.join(eon)} | attn {path.__name__} max {max(arel):.1e} "
            f"eq {aeq}/6")


def _m1_tune(state):
    """Diagnostic build: the bit-16 self-test, then per stress shape hidden/stream relative
    differences (= when bitwise equal) against one eager M1-off call: six more eager M1-off
    calls, two M1-off graph stresses and two M1-on graph stresses (2000 replays each), two
    eager M1-on calls; and attention alone, six eager calls against a first. M1 stays off."""
    state["m1_report"] = []
    rep = state["m1_report"]
    checked = _m1_selftest(state)
    rep.append(f"selftest PDL = plain bitwise on {len(checked)} launches")
    state["m1_ok"] = True
    try:
        for batch, length, layer in ((1, 8, "full"), (1, 8, "sliding"), (1, 1, "full"),
                                     (1, 1, "sliding"), (16, 1, "full"), (32, 1, "full")):
            try:
                rep.append("diag " + _m1_diag_shape(state, batch, length, layer))
            except Exception as exc:
                rep.append(f"diag {batch}x{length}{layer[0]} error {type(exc).__name__}: "
                           f"{str(exc)[:150]}")
        if "_m1_fresh" in globals():
            rep.extend(_m1_fresh(state))
    finally:
        state["m1_ok"] = False
        state["m1_policy"] = {}


'''
if "--fresh" in sys.argv:
    # The fresh-input stress (check 3e2aa83358dd4f60): new rows every replay, each replay
    # checked on the device, so a read of the previous replay's rows shows as order 1.
    new = open(Path(__file__).with_name("dl_m1_fresh_block.py")).read() + new
text = text[:start] + new + text[end:]
text = text.replace("# okc: problem=muse_decode_layer\n",
                    "# okc: problem=muse_decode_layer\n# M1 determinism diagnostic (not a candidate)\n", 1)
open(out, "w").write(text)
print("wrote", out, len(text))
