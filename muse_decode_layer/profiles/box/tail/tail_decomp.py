import json, sys, statistics, math
import numpy as np
d = json.load(open(sys.argv[1]))
for row in d["rows"]:
    c = row["case"]; label = f'{c["load"]} {c["layer"]}'
    for inp in row["inputs"]:
        if "raw" not in inp: continue
        v = inp["raw"]; n = len(v)
        t0 = min(r[0] for r in v)
        us = lambda x: (x - t0) / 1e3
        bend = [us(r[3]) for r in v]; fin = [us(r[4]) if r[4] else math.nan for r in v]; ex = [us(r[5]) for r in v]
        aend = [us(r[2]) for r in v]
        maxb = max(bend); meanb = statistics.mean(bend)
        finv = [f for f in fin if not math.isnan(f)]
        wait = [f - b for f, b in zip(fin, bend) if not math.isnan(f)]
        after_last = [f - maxb for f in finv]
        rng = [((r[21] >> 32) & 0xFFFF, (r[21] >> 48) & 0xFFFF) for r in v]
        nB = [b - a for a, b in rng]
        rngA = [(r[21] & 0xFFFF, (r[21] >> 16) & 0xFFFF) for r in v]
        nA = [b - a for a, b in rngA]
        final = [e - f for e, f in zip(ex, fin) if not math.isnan(f)]
        order = sorted(range(n), key=lambda i: bend[i])
        late = order[-5:]; early = order[:5]
        print(f"{label} seed {inp['seed']} distinct {inp['distinct']} nj {inp['nj']} span {max(ex):.1f}")
        print(f"  B end: mean {meanb:.1f} min {min(bend):.1f} max {maxb:.1f} sd {statistics.pstdev(bend):.2f}; A end mean {statistics.mean(aend):.1f} min {min(aend):.1f} max {max(aend):.1f}")
        print(f"  fin: mean {statistics.mean(finv):.1f} min {min(finv):.1f} max {max(finv):.1f}; mean wait (fin-bend) {statistics.mean(wait):.2f}; fin - max(bend): mean {statistics.mean(after_last):.2f} min {min(after_last):.2f} max {max(after_last):.2f}")
        print(f"  final (exit-fin): mean {statistics.mean(final):.2f} min {min(final):.2f} max {max(final):.2f}; exit mean {statistics.mean(ex):.1f} max {max(ex):.1f}")
        print(f"  CTAs with fin < max(bend): {sum(1 for f in finv if f < maxb)} of {len(finv)}; fin > max(bend)+1us: {sum(1 for f in finv if f > maxb + 1)}")
        print(f"  stages A per CTA min/max {min(nA)}/{max(nA)}, B {min(nB)}/{max(nB)}; late CTAs (idx, bend, nA, nB, A-end): " + " ".join(f"({i},{bend[i]:.1f},{nA[i]},{nB[i]},{aend[i]:.1f})" for i in late))
        print(f"  early CTAs: " + " ".join(f"({i},{bend[i]:.1f},{nA[i]},{nB[i]},{aend[i]:.1f})" for i in early))
        X = np.array([[a, b, 1.0] for a, b in zip(nA, nB)], float); Y = np.array(bend)
        cf, *_ = np.linalg.lstsq(X, Y, rcond=None); res = Y - X @ cf
        print(f"  bend = {cf[0]:.3f} nA + {cf[1]:.3f} nB + {cf[2]:.1f}, resid sd {res.std():.2f}; corr(bend, aend) {np.corrcoef(bend, aend)[0,1]:.2f}; B time (bend-aend) sd {np.std(np.array(bend)-np.array(aend)):.2f}")
