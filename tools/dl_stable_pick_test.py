"""Offline test of _stable_pick (tools/dl_stable_tuner_patch.py) on synthetic round times: a true
tie stays on the preferred pick across many noise draws, a consistent win beyond the margin
switches, a win with three disagreeing rounds does not, a noisy preferred pick widens the
margin, and an absent preferred pick falls back to the lowest median.

    python3 tools/dl_stable_pick_test.py tools [LAYER.py]
"""
import random
import sys

sys.path.insert(0, sys.argv[1])
import dl_stable_tuner_patch as sp  # noqa: E402

ns = {}
exec(sp.HELPER, ns)
pick = ns["_stable_pick"]
R = ns["STABLE_ROUNDS"]
rng = random.Random(7)


def draw(mean, sd):
    return [mean + rng.gauss(0, sd) for _ in range(R)]


bad = 0
flips = argmin = 0
for _ in range(2000):
    rounds = {"a": draw(100, 0.4), "b": draw(100, 0.4)}
    flips += pick(rounds, "a")[0] != "a"
    med = {k: sorted(v)[R // 2] for k, v in rounds.items()}
    argmin += min(med, key=med.get) != "a"
ok = flips <= 100
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} true tie, noise 0.4 us: {flips}/2000 off the preferred pick "
      f"(lowest median of the same rounds: {argmin}/2000)")
wins = sum(pick({"a": draw(100, 0.3), "b": draw(98, 0.3)}, "a")[0] == "b" for _ in range(500))
ok = wins >= 475
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} 2 us real win, noise 0.3 us: switched {wins}/500")
base = [100.0] * R
other = [97.0] * (R - 3) + [101.0] * 3
got, note = pick({"a": base, "b": other}, "a")
ok = got == "a"
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} 3 us median win, 3 rounds disagree: {note}")
noisy = [100, 103, 97, 104, 96, 100, 102, 98, 100][:R]
got, note = pick({"a": noisy, "b": [x - 1.5 for x in noisy]}, "a")
ok = got == "a" and "margin 4.00" in note
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} 1.5 us win under a 2 x MAD = 4 us margin: {note}")
got, note = pick({"b": [10.0] * R, "c": [9.0] * R}, "a")
ok = got == "c" and "absent" in note
bad += not ok
print(f"{'ok  ' if ok else 'FAIL'} preferred absent: {note}")

if len(sys.argv) > 2:
    import ast
    layer = open(sys.argv[2]).read()
    tree = ast.parse(layer)
    keep = {"STABLE_ROUNDS", "STABLE_MARGIN_US", "STABLE_PREF", "_stable_pick", "_m1_rounds"}
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in keep) or (
        isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in keep)]
    fns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), sys.argv[2], "exec"), fns)
    same = ast.unparse(next(n for n in body if getattr(n, "name", None) == "_stable_pick")) == \
        ast.unparse(next(n for n in ast.parse(sp.HELPER).body
                         if getattr(n, "name", None) == "_stable_pick"))
    bad += not same
    print(f"{'ok  ' if same else 'FAIL'} {sys.argv[2]}: _stable_pick equals the tool's")
    pref = fns["STABLE_PREF"]
    print(f"     preference table: {pref}")
    if "_m1_rounds" in fns:
        # M1's whole-step loads as 4d4ea5db46db4a10's tuner era measured them: off 2-3 us slower
        # than o and o+front, which sit within about 0.5 us of each other at 16 and 32 tokens.
        lp = fns["_stable_pick"]
        for bucket, gaps in ((8, {"off": 4.0, "o": 2.5, "o+front": 0.0}),
                             (16, {"off": 2.5, "o": 0.0, "o+front": 0.2}),
                             (32, {"off": 2.8, "o": 0.0, "o+front": 0.0})):
            want = pref["m1"][bucket]
            # A preferred mode the measured era did not have (M-B's o+front+unp) ties the best.
            gaps = dict(gaps, **{want: gaps.get(want, 0.0)})
            stay = 0
            for _ in range(500):
                rounds = {m: draw(200 + g, 0.4) for m, g in gaps.items()}
                stay += lp(rounds, want)[0] == want
            ok = stay >= 475 and want != "off"
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} m1 T={bucket}: preferred {want} kept {stay}/500 "
                  f"(gaps {gaps}, noise 0.4 us)")
        rounds = {m: draw(200, 0.4) for m in ("off", "o")}
        rounds["o+front"] = draw(197, 0.4)
        ok = lp(rounds, "o")[0] == "o+front"
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} m1: a 3 us win over the preferred mode switches")
        calls = []

        class G:
            def __init__(self, key):
                self.key = key

        fns["_graph_median"] = lambda g, flush, n: calls.append(g.key) or float(len(calls))
        keys = [(m, la) for m in ("off", "o", "o+front") for la in ("sliding", "full")]
        times = fns["_m1_rounds"]({k: G(k) for k in keys}, None)
        firsts = [calls[6 * r][0] for r in range(R)]
        pairs = all(calls[i][0] == calls[i + 1][0] for i in range(0, len(calls), 2))
        ok = (all(len(v) == R for v in times.values()) and len(calls) == 6 * R and pairs
              and firsts[:3] == ["off", "o", "o+front"])
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} _m1_rounds: {R} rounds, both layers of a mode back to "
              f"back, first mode per round {firsts}")
print("all cases as expected" if not bad else f"{bad} cases failed")
sys.exit(1 if bad else 0)
