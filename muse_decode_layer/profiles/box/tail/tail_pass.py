import json, sys, statistics, math
d = json.load(open(sys.argv[1]))
med = statistics.median
for row in d["rows"]:
    c = row["case"]; label = f'{c["load"]} {c["layer"]}'
    agg = {k: [] for k in ("wait", "after_last", "maxmean", "p1", "p2", "pass", "p1min", "p1max", "p2min", "p2max", "t2wait", "t2pass", "exit_after_p2", "span_minus_maxbend", "late_group")}
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        v = inp["raw"]; n = len(v)
        t0 = min(r[0] for r in v)
        us = lambda x: (x - t0) / 1e3
        bend = [us(r[3]) for r in v]; maxb = max(bend)
        fin = [us(r[4]) if r[4] else math.nan for r in v]
        ex = [us(r[5]) for r in v]
        p1 = [us(r[333]) if r[333] else math.nan for r in v]
        p2 = [us(r[334]) if r[334] else math.nan for r in v]
        t2w = [us(r[22]) if r[22] else math.nan for r in v]
        t2p = [us(r[23]) if r[23] else math.nan for r in v]
        ok = lambda *xs: all(not math.isnan(x) for x in xs)
        agg["wait"].append(statistics.mean([f - b for f, b in zip(fin, bend) if ok(f)]))
        agg["after_last"].append(statistics.mean([f for f in fin if ok(f)]) - maxb)
        agg["maxmean"].append(maxb - statistics.mean(bend))
        d1 = [a - f for a, f in zip(p1, fin) if ok(a, f)]; d2 = [b - a for a, b in zip(p1, p2) if ok(a, b)]
        agg["p1"].append(statistics.mean(d1)); agg["p1min"].append(min(d1)); agg["p1max"].append(max(d1))
        agg["p2"].append(statistics.mean(d2)); agg["p2min"].append(min(d2)); agg["p2max"].append(max(d2))
        agg["pass"].append(statistics.mean([b - f for b, f in zip(p2, fin) if ok(b, f)]))
        w2 = [a - b for a, b in zip(t2w, p2) if ok(a, b)]; q2 = [b - a for a, b in zip(t2w, t2p) if ok(a, b)]
        agg["t2wait"].append(statistics.mean(w2) if w2 else math.nan); agg["t2pass"].append(statistics.mean(q2) if q2 else math.nan)
        last = [b if ok(b) else a for a, b in zip(p2, t2p)]
        agg["exit_after_p2"].append(statistics.mean([e - l for e, l in zip(ex, last) if ok(l)]))
        agg["span_minus_maxbend"].append(max(ex) - maxb)
        # the late group: CTAs within 2 us of the last B end: their index range and stage counts
        rngB = [((r[21] >> 32) & 0xFFFF, (r[21] >> 48) & 0xFFFF) for r in v]
        late = [i for i in range(n) if bend[i] > maxb - 2.0]
        agg["late_group"].append((len(late), min(late), max(late), med([rngB[i][1] - rngB[i][0] for i in late]), med([b - a for a, b in rngB])))
    print(f"{label}: n {len(agg['wait'])}; medians over inputs: wait(fin-bend) {med(agg['wait']):.2f} = max(bend)-mean(bend) {med(agg['maxmean']):.2f} + fin-max(bend) {med(agg['after_last']):.2f}")
    print(f"  first token: zeroing+pass1 {med(agg['p1']):.2f} (per-CTA min {med(agg['p1min']):.2f} max {med(agg['p1max']):.2f}), pass2 {med(agg['p2']):.2f} (min {med(agg['p2min']):.2f} max {med(agg['p2max']):.2f}), total {med(agg['pass']):.2f}; second token: wait {med(agg['t2wait']):.2f} pass {med(agg['t2pass']):.2f}; exit after last pass {med(agg['exit_after_p2']):.2f}; span - max(bend) {med(agg['span_minus_maxbend']):.2f}")
    lg = agg["late_group"]
    print(f"  late group (B end within 2 us of the last): n {med([x[0] for x in lg]):.0f}, idx {med([x[1] for x in lg]):.0f}..{med([x[2] for x in lg]):.0f}, phase B stages {med([x[3] for x in lg]):.0f} vs all-CTA median {med([x[4] for x in lg]):.0f}")
